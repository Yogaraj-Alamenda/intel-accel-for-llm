// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "iaxl_common.h"
#include "env.h"
#include "kv_pool.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>

namespace kv_pool {

namespace {

double now_timestamp() {
    const auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
}

struct FreeDeleter {
    void operator()(char *p) const { free(p); }
};
using Buffer = std::unique_ptr<char, FreeDeleter>;

struct SubEntry {
    Buffer data;
    size_t size = 0;
};

struct EntryGroup {
    EntryGroup *lru_prev = nullptr;
    EntryGroup *lru_next = nullptr;

    EntryGroup *unpersisted_prev = nullptr;
    EntryGroup *unpersisted_next = nullptr;
    bool is_persisted = false;
    // A copy was written to / read from storage. Unlike is_persisted it survives re-puts: keys are
    // content-addressed, so the disk copy stays valid and the Record entry may be kept on eviction.
    bool on_disk = false;
    // Set once the group key has been submitted to the Record (all entries stored).
    bool committed = false;

    const std::string *group_key_ptr = nullptr;

    double created_at = 0.0;
    double last_access = 0.0;

    std::unordered_map<std::string, SubEntry> entries;
    // Buffers replaced while a reader held this group; freed with the group or the next put.
    std::vector<Buffer> retired;

    size_t total_bytes() const {
        size_t total = 0;
        for (const auto &[_, sub] : entries)
            total += sub.size;
        return total;
    }

    size_t entry_count() const { return entries.size(); }

    SubEntry *find(std::string_view tensor_key) {
        auto it = entries.find(std::string(tensor_key));
        return it != entries.end() ? &it->second : nullptr;
    }

    const SubEntry *find(std::string_view tensor_key) const {
        auto it = entries.find(std::string(tensor_key));
        return it != entries.end() ? &it->second : nullptr;
    }

    SubEntry &get_or_create(const std::string &tensor_key) { return entries[tensor_key]; }

    ~EntryGroup() = default;

    EntryGroup() { entries.reserve((size_t)envs.IAXL_CACHE_CACHEGROUP_SIZE); }

    // Held by shared_ptr (readers share it) and linked into intrusive lists: never copied or moved.
    EntryGroup(const EntryGroup &) = delete;
    EntryGroup &operator=(const EntryGroup &) = delete;
};

} // namespace

class Mem::Impl {
    // Groups are shared: a reader (Mem::acquire) holds the groups it reads, so a group evicted or
    // cleared meanwhile is freed only when the last reader releases it. Eviction never waits.
    using GroupMap = std::unordered_map<std::string, std::shared_ptr<EntryGroup>>;
    using CacheNode = GroupMap::node_type;

    // Frees evicted groups on its own thread, so eviction does not stall the caller (the IAXL
    // worker, which also compresses and decompresses). It only ever destroys what it is given.
    class Reclaimer {
      public:
        Reclaimer() : thread_([this] { run(); }) {}
        ~Reclaimer() {
            {
                std::lock_guard<std::mutex> lk(mutex_);
                stop_ = true;
            }
            cv_.notify_one();
            thread_.join();
        }
        void push(std::vector<CacheNode> &nodes) {
            if (nodes.empty())
                return;
            {
                std::lock_guard<std::mutex> lk(mutex_);
                for (auto &node : nodes)
                    queue_.push_back(std::move(node));
            }
            nodes.clear();
            cv_.notify_one();
        }

      private:
        void run() {
            std::unique_lock<std::mutex> lk(mutex_);
            while (true) {
                cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
                if (queue_.empty())
                    return;
                std::vector<CacheNode> batch;
                batch.swap(queue_);
                lk.unlock();
                batch.clear();
                lk.lock();
            }
        }
        std::mutex mutex_;
        std::condition_variable cv_;
        std::vector<CacheNode> queue_;
        bool stop_ = false;
        std::thread thread_;
    };

    // Groups removed from cache_ under mutex_. Declared before the lock in each caller, so they
    // reach the reclaimer after mutex_ is released.
    struct Dropped {
        explicit Dropped(Reclaimer &reclaimer) : reclaimer(reclaimer) {}
        ~Dropped() { reclaimer.push(nodes); }
        Reclaimer &reclaimer;
        std::vector<CacheNode> nodes;
        std::vector<std::string> record_removals;
    };

    // Over budget, eviction frees this much below capacity, so it runs once per batch of puts
    // (and issues one Record removal per batch) instead of on every put.
    static constexpr size_t kEvictBatchBytes = 64ull << 20;

  public:
    explicit Impl(size_t capacity_bytes, Storage *storage, Record *record)
        : capacity_bytes_(capacity_bytes), storage_(storage), record_(record), current_bytes_(0),
          total_entries_(0), evict_enabled_(true) {
        cache_.reserve((size_t)envs.IAXL_CACHE_CACHEGROUP_NUM);

        lru_head_.lru_next = &lru_head_;
        lru_head_.lru_prev = &lru_head_;

        unpersisted_head_.unpersisted_next = &unpersisted_head_;
        unpersisted_head_.unpersisted_prev = &unpersisted_head_;
    }

    ~Impl() { cache_.clear(); }

    void lock() { mutex_.lock(); }
    void unlock() { mutex_.unlock(); }

    void acquire_deletion_guard() { deletion_guard_count_.fetch_add(1, std::memory_order_acq_rel); }

    void release_deletion_guard() {
        if (deletion_guard_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lk(deletion_cv_mutex_);
            deletion_cv_.notify_all();
        }
    }

    bool is_deletion_guarded() const {
        return deletion_guard_count_.load(std::memory_order_acquire) > 0;
    }

    void wait_deletion_guard() {
        if (deletion_guard_count_.load(std::memory_order_acquire) == 0)
            return;
        std::unique_lock<std::mutex> lk(deletion_cv_mutex_);
        deletion_cv_.wait(
            lk, [this] { return deletion_guard_count_.load(std::memory_order_acquire) == 0; });
    }

    void put(const std::vector<std::string> &keys, std::vector<char *> &&data_ptrs,
             const std::vector<size_t> &sizes, const std::vector<size_t> &unzip_sizes) {
        IAXL_CHECK(keys.size() == data_ptrs.size(),
                   "Mem::put: keys and data pointers must have the same length");
        IAXL_CHECK(keys.size() == sizes.size(),
                   "Mem::put: keys and sizes must have the same length");
        IAXL_CHECK(unzip_sizes.empty() || keys.size() == unzip_sizes.size(),
                   "Mem::put: unzip sizes must be empty or match keys length");
        Dropped dropped(reclaimer_); // handed to the reclaimer after the lock is released
        std::lock_guard<std::mutex> lock(mutex_);
        put_unlocked(keys, std::move(data_ptrs), sizes, unzip_sizes);
        enforce_capacity_unlocked(dropped);
    }

    std::vector<std::pair<const char *, size_t>> get(const std::vector<std::string> &keys,
                                                     bool allow_using_omp) {
        std::lock_guard<std::mutex> lock(mutex_);
        return get_unlocked(keys, allow_using_omp);
    }

    std::vector<Mem::BufferRef> acquire(const std::vector<std::string> &keys,
                                        bool allow_using_omp) {
        Dropped dropped(reclaimer_);
        std::lock_guard<std::mutex> lock(mutex_);
        auto refs = lookup_unlocked(keys, allow_using_omp);
        // Storage reloads may have pushed the pool over budget; the returned refs stay valid.
        enforce_capacity_unlocked(dropped);
        return refs;
    }

    std::vector<bool> has(const std::vector<std::string> &keys) {
        std::lock_guard<std::mutex> lock(mutex_);
        return has_unlocked(keys);
    }

    std::vector<std::tuple<std::string, size_t>> persist_groups(size_t max_groups) {
        std::lock_guard<std::mutex> lock(mutex_);
        return persist_groups_unlocked(max_groups);
    }

    std::vector<std::tuple<std::string, size_t>> evict_groups(size_t max_groups) {
        wait_deletion_guard();
        Dropped dropped(reclaimer_);
        std::lock_guard<std::mutex> lock(mutex_);
        return evict_groups_unlocked(max_groups, dropped);
    }

    void put_unlocked(const std::vector<std::string> &keys, std::vector<char *> &&data_ptrs,
                      const std::vector<size_t> &sizes, const std::vector<size_t> &unzip_sizes) {
        double now = now_timestamp();
        for (size_t i = 0; i < keys.size(); i++) {

            size_t unzip_size = (i < unzip_sizes.size()) ? unzip_sizes[i] : 0;
            put_one_unlocked(keys[i], data_ptrs[i], sizes[i], unzip_size, now);
            data_ptrs[i] = nullptr;
        }
        commit_complete_groups_unlocked(keys);
    }

    // Raw pointers stay valid only until the entry is evicted or replaced; see acquire().
    std::vector<std::pair<const char *, size_t>> get_unlocked(const std::vector<std::string> &keys,
                                                              bool allow_using_omp) {
        auto refs = lookup_unlocked(keys, allow_using_omp);
        std::vector<std::pair<const char *, size_t>> results(refs.size());
        for (size_t i = 0; i < refs.size(); ++i)
            results[i] = {refs[i].data.get(), refs[i].size};
        return results;
    }

    std::vector<Mem::BufferRef> lookup_unlocked(const std::vector<std::string> &keys,
                                                bool allow_using_omp) {
        std::vector<Mem::BufferRef> results(keys.size());
        double now = now_timestamp();

        std::vector<size_t> miss_indices;
        miss_indices.reserve(keys.size());

        for (size_t i = 0; i < keys.size(); ++i) {
            results[i] = get_one_unlocked(keys[i], now);
            if (!results[i].data) {
                miss_indices.push_back(i);
            }
        }

        if (miss_indices.empty()) {
            return results;
        }

        if (!storage_) {
            misses_.fetch_add(miss_indices.size(), std::memory_order_relaxed);
            return results;
        }

        std::vector<size_t> storage_indices;
        storage_indices.reserve(miss_indices.size());

        if (record_) {
            std::vector<std::string> check_keys;
            check_keys.reserve(miss_indices.size());
            for (size_t idx : miss_indices) {
                auto [group_key_sv, _] = parse_full_label(keys[idx]);
                check_keys.emplace_back(group_key_sv);
            }

            std::vector<bool> is_persisted = record_->is_persisted(check_keys);

            for (size_t i = 0; i < miss_indices.size(); ++i) {
                if (is_persisted[i]) {
                    storage_indices.push_back(miss_indices[i]);
                } else {
                }
            }
            misses_.fetch_add(miss_indices.size() - storage_indices.size(),
                              std::memory_order_relaxed);
        } else {

            misses_.fetch_add(miss_indices.size(), std::memory_order_relaxed);
            return results;
        }

        if (storage_indices.empty()) {
            return results;
        }

        struct LoadedData {
            size_t index;
            char *buffer;
            size_t size;
        };
        std::vector<LoadedData> loaded_data(storage_indices.size());

        if (allow_using_omp) {
#pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < storage_indices.size(); ++i) {
                size_t key_index = storage_indices[i];
                auto [buffer, size] = storage_->load(keys[key_index]);
                loaded_data[i] = {key_index, buffer, size};
            }
        } else {
            for (size_t i = 0; i < storage_indices.size(); ++i) {
                size_t key_index = storage_indices[i];
                auto [buffer, size] = storage_->load(keys[key_index]);
                loaded_data[i] = {key_index, buffer, size};
            }
        }

        for (const auto &item : loaded_data) {
            if (item.buffer) {

                put_one_unlocked(keys[item.index], item.buffer, item.size, 0, now);

                mark_persisted_unlocked({keys[item.index]});

                hits_in_storage_.fetch_add(1, std::memory_order_relaxed);

                auto [group_key_sv, _] = parse_full_label(keys[item.index]);
                auto it = cache_.find(std::string(group_key_sv));
                // Reloaded only because the Record marks it persisted, so it is already recorded.
                it->second->committed = true;
            } else {

                misses_.fetch_add(1, std::memory_order_relaxed);
            }
        }

        // Resolve after every insert: a key repeated in `keys` is loaded twice, and the second
        // put replaces the first buffer.
        for (const auto &item : loaded_data) {
            if (item.buffer) {
                auto [group_key_sv, tensor_key_sv] = parse_full_label(keys[item.index]);
                const auto &group = cache_.find(std::string(group_key_sv))->second;
                results[item.index] = ref(group, group->find(tensor_key_sv));
            }
        }

        return results;
    }

    std::vector<bool> has_unlocked(const std::vector<std::string> &keys) const {
        std::vector<bool> results;
        results.reserve(keys.size());
        for (const auto &key : keys) {
            auto [group_key_sv, tensor_key_sv] = parse_full_label(key);
            auto it = cache_.find(std::string(group_key_sv));
            if (it != cache_.end()) {
                if (tensor_key_sv.empty()) {
                    results.push_back(true);
                } else {
                    const SubEntry *sub = it->second->find(tensor_key_sv);
                    results.push_back(sub != nullptr && sub->data != nullptr);
                }
            } else {
                results.push_back(false);
            }
        }
        return results;
    }

    std::vector<std::tuple<std::string, const char *, size_t>>
    get_unpersisted_unlocked(size_t max_count) {
        std::vector<std::tuple<std::string, const char *, size_t>> results;
        EntryGroup *group = unpersisted_head_.unpersisted_next;
        size_t groups_returned = 0;
        while (group != &unpersisted_head_ && groups_returned < max_count) {
            const std::string &group_key = *group->group_key_ptr;
            for (const auto &[tensor_key, sub] : group->entries) {
                std::string full_label = make_full_label(group_key, tensor_key);
                results.emplace_back(std::move(full_label), sub.data.get(), sub.size);
            }
            groups_returned++;
            group = group->unpersisted_next;
        }
        return results;
    }

    std::vector<std::tuple<std::string, const char *, size_t>>
    get_lru_oldest_unlocked(size_t max_count) {
        std::vector<std::tuple<std::string, const char *, size_t>> results;
        EntryGroup *group = lru_head_.lru_next;
        size_t groups_returned = 0;
        while (group != &lru_head_ && groups_returned < max_count) {
            const std::string &group_key = *group->group_key_ptr;
            for (const auto &[tensor_key, sub] : group->entries) {
                std::string full_label = make_full_label(group_key, tensor_key);
                results.emplace_back(std::move(full_label), sub.data.get(), sub.size);
            }
            groups_returned++;
            group = group->lru_next;
        }
        return results;
    }

    void mark_persisted(const std::vector<std::string> &keys) {
        std::lock_guard<std::mutex> lock(mutex_);
        mark_persisted_unlocked(keys);
    }

    void mark_persisted_unlocked(const std::vector<std::string> &keys) {
        for (const auto &key : keys) {
            auto [group_key_sv, _] = parse_full_label(key);
            auto it = cache_.find(std::string(group_key_sv));
            if (it != cache_.end()) {
                EntryGroup &group = *it->second;
                group.on_disk = true;
                if (!group.is_persisted) {
                    group.is_persisted = true;
                    unpersisted_remove(&group);
                }
            }
        }
    }

    size_t unpersisted_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return unpersisted_count_unlocked();
    }

    size_t unpersisted_count_unlocked() const {
        size_t count = 0;
        const EntryGroup *group = unpersisted_head_.unpersisted_next;
        while (group != &unpersisted_head_) {
            count++;
            group = group->unpersisted_next;
        }
        return count;
    }

    void set_evict_enabled(bool enabled) {
        evict_enabled_.store(enabled, std::memory_order_release);
    }

    bool is_evict_enabled() const { return evict_enabled_.load(std::memory_order_acquire); }

    void set_entries_per_group(size_t count, const std::string &label) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (label.empty()) {
            entries_per_group_ = count;
        } else {
            validate_label_component(label);
            label_entries_per_group_[label] = count;
        }
    }

    size_t entries_per_group(const std::string &label) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!label.empty()) {
            auto it = label_entries_per_group_.find(label);
            if (it != label_entries_per_group_.end())
                return it->second;
        }
        return entries_per_group_;
    }

    size_t evict_to_size(size_t target_bytes) {
        if (!evict_enabled_.load(std::memory_order_acquire)) {
            return 0;
        }
        wait_deletion_guard();
        Dropped dropped(reclaimer_);
        std::lock_guard<std::mutex> lock(mutex_);
        return evict_unlocked(target_bytes, dropped);
    }

    size_t force_evict_to_size(size_t target_bytes) {
        wait_deletion_guard();
        Dropped dropped(reclaimer_);
        std::lock_guard<std::mutex> lock(mutex_);
        return evict_unlocked(target_bytes, dropped);
    }

    size_t remove(const std::string &key) {
        wait_deletion_guard();
        Dropped dropped(reclaimer_);
        std::lock_guard<std::mutex> lock(mutex_);
        auto [group_key_sv, _] = parse_full_label(key);
        auto it = cache_.find(std::string(group_key_sv));
        if (it == cache_.end())
            return 0;
        size_t bytes = drop_group_unlocked(it->second.get(), dropped);
        finish_eviction_unlocked(dropped);
        return bytes;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return total_entries_;
    }

    size_t group_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.size();
    }

    size_t current_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_bytes_;
    }

    size_t capacity_bytes() const { return capacity_bytes_; }
    void set_capacity_bytes(size_t capacity) { capacity_bytes_ = capacity; }

    uint64_t hits() const { return hits_.load(std::memory_order_relaxed); }
    uint64_t hits_in_storage() const { return hits_in_storage_.load(std::memory_order_relaxed); }
    uint64_t misses() const { return misses_.load(std::memory_order_relaxed); }
    uint64_t puts() const { return puts_.load(std::memory_order_relaxed); }
    uint64_t evictions() const { return evictions_.load(std::memory_order_relaxed); }
    uint64_t evicted_bytes() const { return evicted_bytes_.load(std::memory_order_relaxed); }

    uint64_t total_unzip_bytes() const {
        return total_unzip_bytes_.load(std::memory_order_relaxed);
    }
    uint64_t total_zip_bytes() const { return total_zip_bytes_.load(std::memory_order_relaxed); }

    double compression_ratio() const {
        uint64_t unzip = total_unzip_bytes_.load(std::memory_order_relaxed);
        uint64_t zip = total_zip_bytes_.load(std::memory_order_relaxed);
        return (zip > 0) ? static_cast<double>(unzip) / static_cast<double>(zip) : 0.0;
    }

    void reset_stats() {
        hits_.store(0, std::memory_order_relaxed);
        hits_in_storage_.store(0, std::memory_order_relaxed);
        misses_.store(0, std::memory_order_relaxed);
        puts_.store(0, std::memory_order_relaxed);
        evictions_.store(0, std::memory_order_relaxed);
        evicted_bytes_.store(0, std::memory_order_relaxed);
        total_unzip_bytes_.store(0, std::memory_order_relaxed);
        total_zip_bytes_.store(0, std::memory_order_relaxed);
    }

    void clear() {
        GroupMap old; // freed outside mutex_ (groups held by readers outlive it)
        std::lock_guard<std::mutex> lock(mutex_);
        old.swap(cache_);
        cache_.reserve(old.bucket_count());
        current_bytes_ = 0;
        total_entries_ = 0;
        lru_head_.lru_next = &lru_head_;
        lru_head_.lru_prev = &lru_head_;
        unpersisted_head_.unpersisted_next = &unpersisted_head_;
        unpersisted_head_.unpersisted_prev = &unpersisted_head_;
    }

  private:
    void put_one_unlocked(const std::string &full_label, char *data, size_t size, size_t unzip_size,
                          double now) {
        if (!data) {
            throw std::runtime_error(
                "[Mem] memory allocation failed for '" + full_label + "' (" + std::to_string(size) +
                " bytes). DDR pool usage: " + std::to_string(current_bytes_ / (1024 * 1024)) +
                " MB / " + std::to_string(capacity_bytes_ / (1024 * 1024)) +
                " MB capacity. Increase IAXL_DDR_POOL_SIZE_GB or enable cache eviction.");
        }

        Buffer buffer(data);

        if (unzip_size > 0) {
            total_unzip_bytes_.fetch_add(unzip_size, std::memory_order_relaxed);
            total_zip_bytes_.fetch_add(size, std::memory_order_relaxed);
        }
        auto [group_key_sv, tensor_key_sv] = parse_full_label(full_label);
        std::string group_key(group_key_sv);
        std::string tensor_key(tensor_key_sv);

        auto it = cache_.find(group_key);
        if (it != cache_.end()) {

            EntryGroup &group = *it->second;
            // Reader refs are only taken under mutex_, so a count of 1 means no reader holds it.
            const bool read_by_others = it->second.use_count() > 1;
            if (!read_by_others)
                group.retired.clear();
            SubEntry *sub = group.find(tensor_key_sv);
            if (sub) {
                current_bytes_ -= sub->size;
                if (read_by_others)
                    group.retired.push_back(std::move(sub->data));
                sub->data = std::move(buffer);
                sub->size = size;
            } else {
                SubEntry &new_sub = group.get_or_create(tensor_key);
                new_sub.data = std::move(buffer);
                new_sub.size = size;
                total_entries_++;

                if (group.entries.size() > (size_t)envs.IAXL_CACHE_CACHEGROUP_SIZE) {
                    fprintf(stderr,
                            "[Mem] WARN: EntryGroup '%s' entries (%zu) "
                            "exceeded reserve (%zu), rehash triggered\n",
                            group_key.c_str(), group.entries.size(),
                            (size_t)envs.IAXL_CACHE_CACHEGROUP_SIZE);
                }
            }
            current_bytes_ += size;
            group.last_access = now;

            lru_remove(&group);
            lru_push_back(&group);

            if (group.is_persisted) {
                group.is_persisted = false;
                unpersisted_push_back(&group);
            }
        } else {

            decltype(cache_.begin()) new_it;
            try {
                new_it = cache_.emplace(group_key, std::make_shared<EntryGroup>()).first;
            } catch (const std::bad_alloc &) {
                throw std::runtime_error(
                    "[Mem] failed to allocate new cache group '" + group_key +
                    "'. DDR pool usage: " + std::to_string(current_bytes_ / (1024 * 1024)) +
                    " MB / " + std::to_string(capacity_bytes_ / (1024 * 1024)) + " MB capacity, " +
                    std::to_string(cache_.size()) +
                    " groups. System memory exhausted. "
                    "Increase IAXL_DDR_POOL_SIZE_GB or enable cache eviction.");
            }
            EntryGroup &group = *new_it->second;
            group.group_key_ptr = &new_it->first;

            SubEntry &sub = group.get_or_create(tensor_key);
            sub.data = std::move(buffer);
            sub.size = size;
            group.created_at = now;
            group.last_access = now;
            group.is_persisted = false;

            current_bytes_ += size;
            total_entries_++;

            lru_push_back(&group);
            unpersisted_push_back(&group);

            if (cache_.size() > (size_t)envs.IAXL_CACHE_CACHEGROUP_NUM) {
                fprintf(stderr,
                        "[Mem] WARN: cache groups (%zu) "
                        "exceeded reserve (%zu), rehash triggered\n",
                        cache_.size(), (size_t)envs.IAXL_CACHE_CACHEGROUP_NUM);
            }
        }

        puts_.fetch_add(1, std::memory_order_relaxed);
    }

    // Entries a group ("label:chunk_id") needs before it is recorded; 0 = caller-managed record.
    size_t expected_entries_unlocked(const std::string &group_key) const {
        if (!label_entries_per_group_.empty()) {
            auto [label, _] = parse_group_key(group_key);
            auto it = label_entries_per_group_.find(std::string(label));
            if (it != label_entries_per_group_.end())
                return it->second;
        }
        return entries_per_group_;
    }

    // Record groups only once every expected entry is in memory, so a Record lookup never
    // claims a block whose (asynchronously compressed) data has not landed yet. Submits are
    // issued under mutex_, so their order on the Record's FIFO queue matches eviction removals.
    void commit_complete_groups_unlocked(const std::vector<std::string> &keys) {
        if (!record_ || (entries_per_group_ == 0 && label_entries_per_group_.empty()))
            return;
        std::vector<std::string> ready;
        for (const auto &key : keys) {
            auto [group_key_sv, _] = parse_full_label(key);
            auto it = cache_.find(std::string(group_key_sv));
            if (it == cache_.end())
                continue;
            EntryGroup &group = *it->second;
            const size_t expected = expected_entries_unlocked(it->first);
            if (expected > 0 && !group.committed && group.entries.size() >= expected) {
                group.committed = true;
                ready.push_back(it->first);
            }
        }
        if (!ready.empty())
            record_->submit_group_keys(ready);
    }

    // Removes one group from memory. Groups not on disk are queued for Record removal so the
    // lookup stops reporting them; on-disk groups stay recorded and reload from storage.
    size_t drop_group_unlocked(EntryGroup *group, Dropped &dropped) {
        std::string group_key = *group->group_key_ptr;
        size_t bytes = group->total_bytes();

        current_bytes_ -= bytes;
        total_entries_ -= group->entry_count();
        lru_remove(group);
        unpersisted_remove(group);
        if (!group->on_disk)
            dropped.record_removals.push_back(group_key);

        dropped.nodes.push_back(cache_.extract(group_key));
        return bytes;
    }

    size_t evict_group_unlocked(EntryGroup *group, Dropped &dropped) {
        size_t bytes = drop_group_unlocked(group, dropped);
        evictions_.fetch_add(1, std::memory_order_relaxed);
        evicted_bytes_.fetch_add(bytes, std::memory_order_relaxed);
        return bytes;
    }

    // Queues the Record removals while still under mutex_, so they stay ordered with submits.
    void finish_eviction_unlocked(Dropped &dropped) {
        if (record_ && !dropped.record_removals.empty())
            record_->remove(dropped.record_removals, /*wait=*/false);
        dropped.record_removals.clear();
    }

    // Once the pool exceeds capacity_bytes_, evicts least-recently-used groups until it is a
    // small batch (kEvictBatchBytes, at most 1/16 of capacity) below it. The groups a put just
    // wrote are the most recently used and go last.
    void enforce_capacity_unlocked(Dropped &dropped) {
        if (capacity_bytes_ == 0 || current_bytes_ <= capacity_bytes_)
            return;
        if (evict_enabled_.load(std::memory_order_acquire) &&
            deletion_guard_count_.load(std::memory_order_acquire) == 0) {
            evict_unlocked(capacity_bytes_ - std::min(kEvictBatchBytes, capacity_bytes_ / 16),
                           dropped);
            return;
        }
        static std::atomic<uint64_t> warn_count{0};
        if ((warn_count.fetch_add(1) % 10000) == 0) {
            fprintf(stderr, "[Mem] WARNING: Over budget! %zu MB > %zu MB capacity (eviction off)\n",
                    current_bytes_ / (1024 * 1024), capacity_bytes_ / (1024 * 1024));
        }
    }

    // A ref shares ownership of the whole group, whose count sits next to the group itself.
    static Mem::BufferRef ref(const std::shared_ptr<EntryGroup> &group, const SubEntry *sub) {
        return {std::shared_ptr<const char>(group, sub->data.get()), sub->size};
    }

    Mem::BufferRef get_one_unlocked(const std::string &full_label, double now) {
        auto [group_key_sv, tensor_key_sv] = parse_full_label(full_label);
        auto it = cache_.find(std::string(group_key_sv));
        if (it != cache_.end()) {
            EntryGroup &group = *it->second;
            SubEntry *sub = group.find(tensor_key_sv);
            if (sub && sub->data) {
                group.last_access = now;
                lru_remove(&group);
                lru_push_back(&group);
                hits_.fetch_add(1, std::memory_order_relaxed);
                return ref(it->second, sub);
            }
        }
        return {};
    }

    std::vector<std::tuple<std::string, size_t>> persist_groups_unlocked(size_t max_groups) {
        if (!storage_) {
            std::cerr << "\033[1;31m*** FATAL: No storage backend configured, cannot persist "
                         "groups.\033[0m"
                      << std::endl;
            std::abort();
        }

        // Only complete groups are persisted, and a group counts as persisted only if every
        // entry was saved; otherwise its Record row would outlive the DRAM copy on eviction
        // while storage lacks some layers.
        std::vector<EntryGroup *> saved;
        std::vector<std::tuple<std::string, size_t>> persisted_groups;
        EntryGroup *group = unpersisted_head_.unpersisted_next;
        while (group != &unpersisted_head_ && saved.size() < max_groups) {
            EntryGroup *next = group->unpersisted_next;
            if (group->committed || expected_entries_unlocked(*group->group_key_ptr) == 0) {
                const std::string &group_key = *group->group_key_ptr;
                bool all_saved = true;
                size_t bytes = 0;
                for (const auto &[tensor_key, sub] : group->entries) {
                    if (!storage_->save(make_full_label(group_key, tensor_key), sub.data.get(),
                                        sub.size)) {
                        all_saved = false;
                        break;
                    }
                    bytes += sub.size;
                }
                if (all_saved) {
                    saved.push_back(group);
                    persisted_groups.emplace_back(group_key, bytes);
                }
            }
            group = next;
        }
        if (saved.empty()) {
            return {};
        }

        std::vector<std::string> group_keys;
        group_keys.reserve(saved.size());
        for (EntryGroup *g : saved) {
            g->is_persisted = true;
            g->on_disk = true;
            unpersisted_remove(g);
            group_keys.push_back(*g->group_key_ptr);
        }
        if (record_) {
            record_->mark_persisted(group_keys);
        }

        return persisted_groups;
    }

    std::vector<std::tuple<std::string, size_t>> evict_groups_unlocked(size_t max_groups,
                                                                       Dropped &dropped) {
        if (!evict_enabled_) {
            return {};
        }

        std::vector<std::tuple<std::string, size_t>> evicted_groups;
        while (lru_head_.lru_next != &lru_head_ && evicted_groups.size() < max_groups) {
            EntryGroup *group = lru_head_.lru_next;
            std::string group_key = *group->group_key_ptr;
            size_t bytes = evict_group_unlocked(group, dropped);
            evicted_groups.emplace_back(std::move(group_key), bytes);
        }
        finish_eviction_unlocked(dropped);
        return evicted_groups;
    }

    size_t evict_unlocked(size_t target_bytes, Dropped &dropped) {
        size_t evicted = 0;
        while (current_bytes_ > target_bytes && lru_head_.lru_next != &lru_head_)
            evicted += evict_group_unlocked(lru_head_.lru_next, dropped);
        finish_eviction_unlocked(dropped);
        return evicted;
    }

    void lru_remove(EntryGroup *group) {
        if (group->lru_prev) {
            group->lru_prev->lru_next = group->lru_next;
        }
        if (group->lru_next) {
            group->lru_next->lru_prev = group->lru_prev;
        }
        group->lru_prev = nullptr;
        group->lru_next = nullptr;
    }

    void lru_push_back(EntryGroup *group) {
        group->lru_prev = lru_head_.lru_prev;
        group->lru_next = &lru_head_;
        lru_head_.lru_prev->lru_next = group;
        lru_head_.lru_prev = group;
    }

    void unpersisted_remove(EntryGroup *group) {
        if (group->unpersisted_prev) {
            group->unpersisted_prev->unpersisted_next = group->unpersisted_next;
        }
        if (group->unpersisted_next) {
            group->unpersisted_next->unpersisted_prev = group->unpersisted_prev;
        }
        group->unpersisted_prev = nullptr;
        group->unpersisted_next = nullptr;
    }

    void unpersisted_push_back(EntryGroup *group) {
        group->unpersisted_prev = unpersisted_head_.unpersisted_prev;
        group->unpersisted_next = &unpersisted_head_;
        unpersisted_head_.unpersisted_prev->unpersisted_next = group;
        unpersisted_head_.unpersisted_prev = group;
    }

    GroupMap cache_;
    Reclaimer reclaimer_;

    EntryGroup lru_head_;
    EntryGroup unpersisted_head_;

    size_t capacity_bytes_;
    size_t current_bytes_;
    size_t total_entries_;
    size_t entries_per_group_ = 0;
    std::unordered_map<std::string, size_t> label_entries_per_group_;

    std::atomic<bool> evict_enabled_;

    mutable std::mutex mutex_;

    std::atomic<int> deletion_guard_count_{0};
    std::mutex deletion_cv_mutex_;
    std::condition_variable deletion_cv_;

    std::atomic<uint64_t> hits_{0};
    std::atomic<uint64_t> misses_{0};
    std::atomic<uint64_t> hits_in_storage_{0};
    std::atomic<uint64_t> puts_{0};
    std::atomic<uint64_t> evictions_{0};
    std::atomic<uint64_t> evicted_bytes_{0};

    Storage *storage_ = nullptr;
    Record *record_ = nullptr;

    std::atomic<uint64_t> total_unzip_bytes_{0};
    std::atomic<uint64_t> total_zip_bytes_{0};
};

Mem::Mem(size_t capacity_bytes, Storage *storage, Record *record)
    : impl_(std::make_unique<Impl>(capacity_bytes, storage, record)) {}
Mem::~Mem() = default;

void Mem::put(const std::vector<std::string> &keys, std::vector<char *> &&data_ptrs,
              const std::vector<size_t> &sizes, const std::vector<size_t> &unzip_sizes) {
    impl_->put(keys, std::move(data_ptrs), sizes, unzip_sizes);
}

std::vector<std::pair<const char *, size_t>> Mem::get(const std::vector<std::string> &keys,
                                                      bool allow_using_omp) {
    return impl_->get(keys, allow_using_omp);
}

std::vector<std::pair<const char *, size_t>> Mem::get_unlocked(const std::vector<std::string> &keys,
                                                               bool allow_using_omp) {
    return impl_->get_unlocked(keys, allow_using_omp);
}

std::vector<Mem::BufferRef> Mem::acquire(const std::vector<std::string> &keys,
                                         bool allow_using_omp) {
    return impl_->acquire(keys, allow_using_omp);
}

std::vector<bool> Mem::has(const std::vector<std::string> &keys) { return impl_->has(keys); }

size_t Mem::remove(const std::string &key) { return impl_->remove(key); }

std::vector<std::tuple<std::string, size_t>> Mem::persist_groups(size_t max_groups) {
    return impl_->persist_groups(max_groups);
}

std::vector<std::tuple<std::string, size_t>> Mem::evict_groups(size_t max_groups) {
    return impl_->evict_groups(max_groups);
}

size_t Mem::evict_to_size(size_t target_bytes) { return impl_->evict_to_size(target_bytes); }
size_t Mem::force_evict_to_size(size_t target_bytes) {
    return impl_->force_evict_to_size(target_bytes);
}

void Mem::set_evict_enabled(bool enabled) { impl_->set_evict_enabled(enabled); }
bool Mem::is_evict_enabled() const { return impl_->is_evict_enabled(); }

void Mem::set_entries_per_group(size_t count, const std::string &label) {
    impl_->set_entries_per_group(count, label);
}
size_t Mem::entries_per_group(const std::string &label) const {
    return impl_->entries_per_group(label);
}

void Mem::lock() { impl_->lock(); }
void Mem::unlock() { impl_->unlock(); }

void Mem::acquire_deletion_guard() { impl_->acquire_deletion_guard(); }
void Mem::release_deletion_guard() { impl_->release_deletion_guard(); }
bool Mem::is_deletion_guarded() const { return impl_->is_deletion_guarded(); }

std::vector<std::tuple<std::string, const char *, size_t>>
Mem::get_unpersisted_unlocked(size_t max_count) {
    return impl_->get_unpersisted_unlocked(max_count);
}

std::vector<std::tuple<std::string, const char *, size_t>>
Mem::get_lru_oldest_unlocked(size_t max_count) {
    return impl_->get_lru_oldest_unlocked(max_count);
}

void Mem::mark_persisted(const std::vector<std::string> &keys) { impl_->mark_persisted(keys); }

size_t Mem::size() const { return impl_->size(); }
size_t Mem::group_count() const { return impl_->group_count(); }
size_t Mem::current_bytes() const { return impl_->current_bytes(); }
size_t Mem::capacity_bytes() const { return impl_->capacity_bytes(); }
void Mem::set_capacity_bytes(size_t capacity) { impl_->set_capacity_bytes(capacity); }

uint64_t Mem::hits() const { return impl_->hits(); }
uint64_t Mem::hits_in_storage() const { return impl_->hits_in_storage(); }
uint64_t Mem::misses() const { return impl_->misses(); }
uint64_t Mem::puts() const { return impl_->puts(); }
uint64_t Mem::evictions() const { return impl_->evictions(); }
uint64_t Mem::evicted_bytes() const { return impl_->evicted_bytes(); }
uint64_t Mem::total_unzip_bytes() const { return impl_->total_unzip_bytes(); }
uint64_t Mem::total_zip_bytes() const { return impl_->total_zip_bytes(); }
double Mem::compression_ratio() const { return impl_->compression_ratio(); }
size_t Mem::unpersisted_count() const { return impl_->unpersisted_count(); }

void Mem::reset_stats() { impl_->reset_stats(); }
void Mem::clear() { impl_->clear(); }

} // namespace kv_pool
