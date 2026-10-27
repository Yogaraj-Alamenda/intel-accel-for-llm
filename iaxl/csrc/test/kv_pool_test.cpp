// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "kv_pool.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <atomic>
#include <string>
#include <thread>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, msg)                                                                           \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "[FAIL] %s (%s:%d)\n", (msg), __FILE__, __LINE__);                \
            ++g_failures;                                                                          \
        } else {                                                                                   \
            std::printf("[ok]   %s\n", (msg));                                                     \
        }                                                                                          \
    } while (0)

static char *dup_buf(const std::string &s) {
    char *p = static_cast<char *>(std::malloc(s.size()));
    std::memcpy(p, s.data(), s.size());
    return p;
}

static bool buf_eq(const char *p, size_t n, const std::string &s) {
    return p != nullptr && n == s.size() && std::memcmp(p, s.data(), n) == 0;
}

int main(int argc, char **argv) {
    const std::string dir = (argc > 1) ? argv[1] : "/tmp/kv_pool_test_dir";
    const std::string db_path = (argc > 2) ? argv[2] : "/tmp/kv_pool_test.db";

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::remove(db_path.c_str());
    std::remove((db_path + "-wal").c_str());
    std::remove((db_path + "-shm").c_str());

    {
        kv_pool::Storage storage(dir);
        const std::string label = "kv:abc123:0";
        const std::string payload = "hello-storage";
        CHECK(storage.save(label, payload.data(), payload.size()), "storage.save() ok");
        auto [buf, n] = storage.load(label);
        CHECK(buf_eq(buf, n, payload), "storage.load() returns saved bytes");
        std::free(buf);
    }

    {
        kv_pool::Mem mem(0);
        const std::vector<std::string> keys = {"kv:0:k", "kv:0:v", "kv:1:k"};
        const std::string d0 = "AAAA", d1 = "BBBBBB", d2 = "CC";

        std::vector<char *> ptrs = {dup_buf(d0), dup_buf(d1), dup_buf(d2)};
        std::vector<size_t> sizes = {d0.size(), d1.size(), d2.size()};
        mem.put(keys, std::move(ptrs), sizes);

        CHECK(mem.size() == 3, "size() == 3 after put");
        CHECK(mem.group_count() == 2, "group_count() == 2 (kv:0, kv:1)");
        CHECK(mem.current_bytes() == d0.size() + d1.size() + d2.size(),
              "current_bytes() sums buffers");

        auto got = mem.get(keys);
        CHECK(buf_eq(got[0].first, got[0].second, d0), "get() key0 matches");
        CHECK(buf_eq(got[1].first, got[1].second, d1), "get() key1 matches");
        CHECK(buf_eq(got[2].first, got[2].second, d2), "get() key2 matches");
        CHECK(mem.hits() == 3, "hits() == 3 after 3 gets");

        auto present = mem.has({"kv:0:k", "kv:9:x"});
        CHECK(present.size() == 2 && present[0] && !present[1], "has() reports presence");

        size_t freed = mem.remove("kv:0:k");
        CHECK(freed == d0.size() + d1.size(), "remove() frees whole group bytes");
        CHECK(mem.group_count() == 1, "group_count() == 1 after remove");
        CHECK(mem.size() == 1, "size() == 1 after remove");

        auto miss = mem.get({"kv:0:k"});
        CHECK(miss[0].first == nullptr, "get() after remove is a miss");
        CHECK(mem.misses() >= 1, "misses() incremented");
    }

    {
        kv_pool::Mem mem(0);
        std::vector<std::string> keys;
        std::vector<char *> ptrs;
        std::vector<size_t> sizes;
        for (int i = 0; i < 4; ++i) {
            keys.push_back("kv:" + std::to_string(i) + ":k");
            std::string d(100, 'x');
            ptrs.push_back(dup_buf(d));
            sizes.push_back(d.size());
        }
        mem.put(keys, std::move(ptrs), sizes);
        CHECK(mem.current_bytes() == 400, "current_bytes() == 400 before evict");

        size_t evicted = mem.evict_to_size(150);
        CHECK(evicted >= 250, "evict_to_size() freed oldest groups");
        CHECK(mem.current_bytes() <= 150, "current_bytes() <= target after evict");
        CHECK(mem.evictions() >= 1, "evictions() incremented");
    }

    {
        kv_pool::Storage storage(dir);
        kv_pool::Record record(db_path, true);
        kv_pool::Mem mem(0, &storage, &record);

        const std::vector<std::string> keys = {"kv:aa:k", "kv:bb:k"};
        const std::string da = "payload-A", db = "payload-B";
        std::vector<char *> ptrs = {dup_buf(da), dup_buf(db)};
        std::vector<size_t> sizes = {da.size(), db.size()};

        record.submit("kv", {"aa", "bb"});
        record.sync();
        mem.put(keys, std::move(ptrs), sizes);
        CHECK(mem.unpersisted_count() == 2, "unpersisted_count() == 2 after put");

        auto persisted = mem.persist_groups(10);
        CHECK(persisted.size() == 2, "persist_groups() persisted 2 groups");
        CHECK(mem.unpersisted_count() == 0, "unpersisted_count() == 0 after persist");

        size_t evicted = mem.force_evict_to_size(0);
        CHECK(evicted == da.size() + db.size(), "force_evict_to_size(0) cleared memory");
        CHECK(mem.group_count() == 0, "group_count() == 0 after full evict");

        auto reloaded = mem.get(keys, true);
        CHECK(buf_eq(reloaded[0].first, reloaded[0].second, da), "get() reloads key0 from storage");
        CHECK(buf_eq(reloaded[1].first, reloaded[1].second, db), "get() reloads key1 from storage");
        CHECK(mem.hits_in_storage() == 2, "hits_in_storage() == 2 after reload");
    }

    // Groups are recorded only once all entries are stored, and over-budget puts evict LRU
    // groups until the pool fits again, dropping them from the Record.
    {
        std::remove(db_path.c_str());
        kv_pool::Storage storage(dir);
        kv_pool::Record record(db_path, true);
        kv_pool::Mem mem(1000, &storage, &record);
        mem.set_entries_per_group(2);

        auto put_layer = [&](const std::string &chunk, const std::string &layer) {
            std::string d(100, 'x');
            std::vector<char *> ptrs = {dup_buf(d)};
            mem.put({"kv:" + chunk + ":" + layer}, std::move(ptrs), {d.size()});
        };

        put_layer("c0", "k");
        CHECK(!record.has("kv", {"c0"})[0], "partially stored group is not recorded");
        put_layer("c0", "v");
        CHECK(record.has("kv", {"c0"})[0], "group recorded once all entries are stored");

        for (int i = 1; i <= 4; ++i) {
            put_layer("c" + std::to_string(i), "k");
            put_layer("c" + std::to_string(i), "v");
        }
        CHECK(mem.current_bytes() == 1000, "pool exactly at capacity is not evicted");
        CHECK(mem.evictions() == 0, "no eviction while within capacity");

        put_layer("c5", "k");
        CHECK(mem.current_bytes() == 900, "over-budget put evicts LRU groups to a batch below capacity");
        CHECK(mem.evictions() == 1, "only the least-recently-used group was evicted");
        CHECK(mem.evicted_bytes() == 200, "evicted_bytes() counts freed bytes");
        auto rec = record.has("kv", {"c0", "c1", "c2", "c3", "c4", "c5"});
        CHECK(!rec[0], "evicted group was removed from the Record");
        CHECK(rec[1] && rec[2] && rec[3] && rec[4], "surviving complete groups stay recorded");
        CHECK(!rec[5], "group written by the evicting put is kept but not yet recorded");
        auto present = mem.has({"kv:c5:k", "kv:c0:k"});
        CHECK(present[0] && !present[1], "the just-written group survives eviction");
    }

    // A single put larger than the whole pool must not overflow it either.
    {
        std::remove(db_path.c_str());
        kv_pool::Record record(db_path, true);
        kv_pool::Mem mem(250, nullptr, &record);
        mem.set_entries_per_group(1);
        std::vector<std::string> keys;
        std::vector<char *> ptrs;
        std::vector<size_t> sizes;
        std::string d(100, 'w');
        for (int i = 0; i < 5; ++i) {
            keys.push_back("kv:b" + std::to_string(i) + ":k");
            ptrs.push_back(dup_buf(d));
            sizes.push_back(d.size());
        }
        mem.put(keys, std::move(ptrs), sizes);
        CHECK(mem.current_bytes() <= 250, "oversized put is trimmed to capacity");
        auto present = mem.has(keys);
        CHECK(!present[0] && !present[1] && !present[2] && present[3] && present[4],
              "oversized put keeps its newest groups");
        auto rec = record.has("kv", {"b0", "b1", "b2", "b3", "b4"});
        CHECK(!rec[0] && !rec[1] && !rec[2] && rec[3] && rec[4],
              "Record matches the groups kept from the oversized put");
    }

    // acquire() refs stay readable after their entry is evicted or replaced.
    {
        std::remove(db_path.c_str());
        kv_pool::Record record(db_path, true);
        kv_pool::Mem mem(300, nullptr, &record);
        mem.set_entries_per_group(1);

        auto put_group = [&](const std::string &chunk, char fill) {
            std::string d(100, fill);
            std::vector<char *> ptrs = {dup_buf(d)};
            mem.put({"kv:" + chunk + ":k"}, std::move(ptrs), {d.size()});
        };
        put_group("p0", 'a');
        put_group("p1", 'b');

        auto refs = mem.acquire({"kv:p0:k", "kv:zz:k"});
        CHECK(refs[0].data && !refs[1].data, "acquire() hit and miss");
        CHECK(mem.evict_groups(10).size() == 2, "evict_groups() evicts acquired groups too");
        CHECK(mem.current_bytes() == 0 && !record.has("kv", {"p0"})[0],
              "evicted groups leave the pool and the Record");
        CHECK(buf_eq(refs[0].data.get(), refs[0].size, std::string(100, 'a')),
              "acquired buffer stays readable after eviction");

        put_group("p2", 'c');
        auto old = mem.acquire({"kv:p2:k"});
        put_group("p2", 'd');
        CHECK(buf_eq(old[0].data.get(), old[0].size, std::string(100, 'c')),
              "acquired buffer stays readable after the entry is replaced");
        auto now = mem.acquire({"kv:p2:k"});
        CHECK(buf_eq(now[0].data.get(), now[0].size, std::string(100, 'd')),
              "a new acquire() sees the replacement");
        CHECK(mem.current_bytes() == 100, "a replaced entry is accounted once");
    }

    // Persisted groups keep their Record entry on eviction and reload from storage.
    {
        std::remove(db_path.c_str());
        kv_pool::Storage storage(dir);
        kv_pool::Record record(db_path, true);
        kv_pool::Mem mem(250, &storage, &record);
        mem.set_entries_per_group(1);

        std::string d(100, 'z');
        std::vector<char *> ptrs = {dup_buf(d)};
        mem.put({"kv:s0:k"}, std::move(ptrs), {d.size()});
        CHECK(mem.persist_groups(10).size() == 1, "group persisted");
        for (const char *chunk : {"s1", "s2"}) {
            std::vector<char *> p = {dup_buf(d)};
            mem.put({std::string("kv:") + chunk + ":k"}, std::move(p), {d.size()});
        }
        CHECK(!mem.has({"kv:s0:k"})[0], "persisted group evicted from DRAM");
        CHECK(record.has("kv", {"s0"})[0], "persisted group stays recorded after eviction");
        auto reloaded = mem.acquire({"kv:s0:k"}, true);
        CHECK(buf_eq(reloaded[0].data.get(), reloaded[0].size, d),
              "evicted group reloads from storage");
        CHECK(mem.current_bytes() <= 250, "reload kept the pool within capacity");

        // A key repeated in one reload batch: the second load replaces the first buffer.
        mem.force_evict_to_size(0);
        auto dup = mem.acquire({"kv:s0:k", "kv:s0:k"}, true);
        CHECK(buf_eq(dup[0].data.get(), dup[0].size, d) && buf_eq(dup[1].data.get(), dup[1].size, d),
              "duplicate keys in a reload both resolve to live buffers");
    }

    // Only complete groups are persisted; remove() drops a group and its Record row.
    {
        std::remove(db_path.c_str());
        kv_pool::Storage storage(dir);
        kv_pool::Record record(db_path, true);
        kv_pool::Mem mem(1000, &storage, &record);
        mem.set_entries_per_group(2);

        auto put_entry = [&](const std::string &label) {
            std::string d(100, 'q');
            std::vector<char *> ptrs = {dup_buf(d)};
            mem.put({label}, std::move(ptrs), {d.size()});
        };
        put_entry("kv:g0:k");
        CHECK(mem.persist_groups(10).empty(), "incomplete group is not persisted");
        put_entry("kv:g0:v");
        CHECK(mem.persist_groups(10).size() == 1, "complete group is persisted");
        mem.set_capacity_bytes(100);
        put_entry("kv:g1:k");
        CHECK(!mem.has({"kv:g0:k"})[0] && record.has("kv", {"g0"})[0],
              "evicted persisted group stays recorded");

        mem.set_capacity_bytes(1000);
        put_entry("kv:g2:k");
        put_entry("kv:g2:v");
        CHECK(record.has("kv", {"g2"})[0], "complete group recorded");
        CHECK(mem.remove("kv:g2:k") == 200, "remove() drops a whole group");
        CHECK(!record.has("kv", {"g2"})[0], "remove() also drops the Record row");
    }

    // A label can need a different entry count (e.g. hybrid models' mamba states).
    {
        std::remove(db_path.c_str());
        kv_pool::Storage storage(dir);
        kv_pool::Record record(db_path, true);
        kv_pool::Mem mem(0, &storage, &record);
        mem.set_entries_per_group(3);
        mem.set_entries_per_group(1, "mamba");
        CHECK(mem.entries_per_group() == 3 && mem.entries_per_group("mamba") == 1 &&
                  mem.entries_per_group("other") == 3,
              "entries_per_group() resolves per-label counts");

        auto put_entry = [&](const std::string &label) {
            std::string d(10, 'm');
            std::vector<char *> ptrs = {dup_buf(d)};
            mem.put({label}, std::move(ptrs), {d.size()});
        };
        put_entry("mamba:b0:l0");
        CHECK(record.has("mamba", {"b0"})[0], "label group recorded at its own entry count");
        put_entry("kv:b0:l1");
        put_entry("kv:b0:l2");
        CHECK(!record.has("kv", {"b0"})[0], "default-label group waits for the default count");
        put_entry("kv:b0:l3");
        CHECK(record.has("kv", {"b0"})[0], "default-label group recorded once complete");
    }

    // Puts that evict, lock-free readers, and manual evictions run concurrently: every acquired
    // buffer must stay intact while it is read, and the pool must end within capacity.
    {
        std::remove(db_path.c_str());
        kv_pool::Record record(db_path, true);
        kv_pool::Mem mem(64 * 1024, nullptr, &record);
        mem.set_entries_per_group(2);
        constexpr int kBlocks = 256;
        constexpr size_t kSize = 1024;
        auto key = [](int b, int layer) {
            return "kv:b" + std::to_string(b) + ":l" + std::to_string(layer);
        };
        std::atomic<bool> stop{false};
        std::atomic<int> corrupt{0};
        std::atomic<long> reads{0};

        std::thread writer([&] {
            for (int round = 0; round < 20; ++round) {
                for (int b = 0; b < kBlocks; ++b) {
                    for (int layer = 0; layer < 2; ++layer) {
                        std::string d(kSize, static_cast<char>('A' + b % 26));
                        std::vector<char *> ptrs = {dup_buf(d)};
                        mem.put({key(b, layer)}, std::move(ptrs), {kSize});
                    }
                }
            }
            stop = true;
        });
        auto reader = [&](int seed) {
            int b = seed;
            while (!stop) {
                b = (b * 7 + 3) % kBlocks;
                auto refs = mem.acquire({key(b, 0), key(b, 1)});
                for (const auto &ref : refs) {
                    if (ref.data && !buf_eq(ref.data.get(), ref.size,
                                            std::string(kSize, static_cast<char>('A' + b % 26))))
                        corrupt++;
                }
                reads++;
            }
        };
        std::thread r1(reader, 1), r2(reader, 2);
        std::thread evictor([&] {
            while (!stop) {
                mem.evict_groups(4);
                std::this_thread::yield();
            }
        });
        writer.join();
        r1.join();
        r2.join();
        evictor.join();
        CHECK(corrupt == 0 && reads > 0, "concurrent readers always see intact buffers");
        CHECK(mem.current_bytes() <= mem.capacity_bytes(), "pool within capacity after the race");
    }

    std::filesystem::remove_all(dir, ec);
    std::remove(db_path.c_str());
    std::remove((db_path + "-wal").c_str());
    std::remove((db_path + "-shm").c_str());

    if (g_failures == 0) {
        std::printf("\nAll kv_pool checks passed.\n");
        return 0;
    }
    std::fprintf(stderr, "\n%d kv_pool check(s) failed.\n", g_failures);
    return 1;
}
