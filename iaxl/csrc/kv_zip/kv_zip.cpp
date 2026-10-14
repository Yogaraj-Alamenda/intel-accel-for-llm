// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// QAT, IAA and CPU workers share one task pool. Each worker claims another item when a request
// completes, so the faster backend naturally processes more of the batch. QAT and IAA workers keep
// multiple asynchronous requests in flight, while CPU workers run one synchronous raw-DEFLATE
// request each. Up to IAXL_OMP_THREAD_NUM threads (see parallel.h) drive the workers, each polling
// the slots of its workers without blocking, so a single thread can drive them all.
//
// Blocks are addressed as ChunkViews: strided slices of the inference tensor (or whole scratch
// tensors). PUT gathers a view straight into the codec's staging buffer and GET scatters the
// codec's output straight back into the view, so no intermediate host buffer is touched.
//
// CPU builds with IAXL_DSA_MEMCPY_ENABLE=1 hand those plain copies to Intel DSA: raw blocks as one
// batch, and the per-block QAT/IAA staging/scatter copies asynchronously, so a polling thread keeps
// its other slots busy while DSA moves the bytes.
//
// QAT and CPU streams are mutually compatible, IAA streams are compatible with neither, so a
// compressed block records in its header whether IAA produced it and decompression only hands it
// to a backend that can decode it.

#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

#include "env.h"
#include "iaxl_common.h"
#include "parallel.h"
#include "cpu_zip.h"
#include "iaa_zip.h"
#include "qat_zip.h"
#include "data_shuffle.h"
#include "lossy.h"
#include "kv_zip.h"

#ifdef DSA_MEMCPY_SUPPORT
#include "dsa_memcpy.h"
#endif

#define DSA_ALIGN 8u
// Per-block copies below this stay on memcpy: the descriptor round trip would dominate.
#define DSA_ASYNC_MIN_BYTES 4096u

// Header word 0 is the payload length with two flag bits: IAA produced the stream (it cannot be
// decoded by QAT/CPU), and the block was byte-plane shuffled before compression. GET must obey
// these rather than the current environment so a persisted cache survives a config change.
#define KV_ZIP_IAA_FLAG (1u << 31)
#define KV_ZIP_SHUFFLE_FLAG (1u << 30)
#define KV_ZIP_LEN_MASK (~(KV_ZIP_IAA_FLAG | KV_ZIP_SHUFFLE_FLAG))

namespace kv_zip {

enum class ZipBackend { QAT, IAA, CPU };

// Backend entry points, indexed by ZipBackend.
struct ZipOps {
    int (*decompress)(int slot, void *src, int len);
    void *(*input_buf)(int slot);
    int (*compress_staged)(int slot, int len);
    int (*wait)(int slot, void **dest, int *len, int non_block);
    int (*src_cap)(void);
    int (*queue_depth)(void);
};

static const ZipOps kZipOps[] = {
    {qat_zip_decompress, qat_zip_input_buf, qat_zip_compress_staged, qat_zip_wait,
     qat_zip_src_cap, qat_zip_queue_depth},
    {iaa_zip_decompress, iaa_zip_input_buf, iaa_zip_compress_staged, iaa_zip_wait,
     iaa_zip_src_cap, iaa_zip_queue_depth},
    {cpu_zip_decompress, cpu_zip_input_buf, cpu_zip_compress_staged, cpu_zip_wait,
     cpu_zip_src_cap, cpu_zip_queue_depth},
};

static const ZipOps &ops(ZipBackend backend) { return kZipOps[static_cast<int>(backend)]; }

// Returned by get_next once a backend has nothing left to claim.
static constexpr size_t kNoTask = static_cast<size_t>(-1);

static void ensure_zip_init() {
    static std::once_flag flag;
    std::call_once(flag, [] {
        if (envs.IAXL_QAT_ZIP_ENABLE)
            IAXL_CHECK(qat_zip_init() == 0, "kv_zip: qat_zip_init failed");
        if (envs.IAXL_IAA_ZIP_ENABLE)
            IAXL_CHECK(iaa_zip_init() == 0, "kv_zip: iaa_zip_init failed");
        if (envs.IAXL_CPU_ZIP_ENABLE)
            IAXL_CHECK(cpu_zip_init() == 0, "kv_zip: cpu_zip_init failed");
    });
}

// ---------------------------------------------------------------------------------------------
// Segment copies

#ifdef DSA_MEMCPY_SUPPORT
static std::atomic<bool> g_dsa_disabled{false};

static void dsa_disable(const char *why) {
    g_dsa_disabled.store(true, std::memory_order_relaxed);
    static std::once_flag warned;
    std::call_once(warned, [why] {
        fprintf(stderr, "[kv_zip] WARNING: %s; falling back to CPU memcpy for the rest of this "
                        "process\n", why);
    });
}

static bool dsa_copy_segments(const std::vector<CopySegment> &segments) {
    if (g_dsa_disabled.load(std::memory_order_relaxed))
        return false;
    for (const auto &s : segments) {
        if ((reinterpret_cast<uintptr_t>(s.dst) | reinterpret_cast<uintptr_t>(s.src) | s.n) &
            (DSA_ALIGN - 1))
            return false;
    }
    std::vector<void *> dst(segments.size());
    std::vector<const void *> src(segments.size());
    std::vector<size_t> n(segments.size());
    for (size_t i = 0; i < segments.size(); i++) {
        dst[i] = segments[i].dst;
        src[i] = segments[i].src;
        n[i] = segments[i].n;
    }
    if (dsa_memcpy_batch(dst.data(), src.data(), n.data(), segments.size()) == 0) {
        static std::once_flag noted;
        std::call_once(noted, [] {
            fprintf(stderr, "[kv_zip] segment copies: using Intel DSA (IAXL_DSA_WQS=%s)\n",
                    envs.IAXL_DSA_WQS());
        });
        return true;
    }
    // A failed batch may have partially copied; the caller redoes the whole batch with memcpy.
    dsa_disable("DSA memcpy failed");
    return false;
}
#endif

static bool dsa_async_enabled() {
#ifdef DSA_MEMCPY_SUPPORT
    return envs.IAXL_DSA_MEMCPY_ENABLE && !g_dsa_disabled.load(std::memory_order_relaxed);
#else
    return false;
#endif
}

// Copies owned by one codec slot. DSA moves the bytes while the polling thread keeps the other
// slots busy; anything DSA cannot take is copied on the CPU before start() returns. Movable but
// not copyable: in-flight completion records live in the heap buffer of active_, which a move
// hands over intact.
class SlotCopies {
  public:
    SlotCopies() = default;
    SlotCopies(SlotCopies &&) = default;
    SlotCopies &operator=(SlotCopies &&) = default;
    SlotCopies(const SlotCopies &) = delete;
    SlotCopies &operator=(const SlotCopies &) = delete;

    void start(const std::vector<CopySegment> &segments) {
        IAXL_CHECK(active_.empty(), "kv_zip: slot copies restarted while in flight");
#ifdef DSA_MEMCPY_SUPPORT
        // Reserved up front: a reallocation would move completion records the engine writes to.
        active_.reserve(segments.size());
        for (const auto &s : segments) {
            if (s.n >= DSA_ASYNC_MIN_BYTES && dsa_async_enabled()) {
                Copy &c = active_.emplace_back();
                c.seg = s;
                const int queued = dsa_copy_submit(&c.job, s.dst, s.src, s.n);
                if (queued == 0) {
                    static std::once_flag noted;
                    std::call_once(noted, [] {
                        fprintf(stderr, "[kv_zip] codec staging copies: using Intel DSA\n");
                    });
                    continue;
                }
                active_.pop_back();
                if (queued < 0)
                    dsa_disable("DSA is unavailable for async copies");
            }
            memcpy(s.dst, s.src, s.n);
        }
#else
        for (const auto &s : segments)
            memcpy(s.dst, s.src, s.n);
#endif
    }

    // True once every copy has landed; failed DSA copies are redone on the CPU.
    bool done() {
#ifdef DSA_MEMCPY_SUPPORT
        bool all = true;
        for (auto &c : active_) {
            if (c.landed)
                continue;
            const int state = dsa_copy_poll(&c.job);
            if (state == 0) {
                all = false;
                continue;
            }
            if (state < 0) {
                memcpy(c.seg.dst, c.seg.src, c.seg.n);
                dsa_disable("DSA async copy failed");
            }
            c.landed = true;
        }
        if (all)
            active_.clear();
        return all;
#else
        return true;
#endif
    }

  private:
#ifdef DSA_MEMCPY_SUPPORT
    struct Copy {
        dsa_copy_job job{};
        CopySegment seg{};
        bool landed = false;
    };
    std::vector<Copy> active_;
#else
    std::vector<int> active_;
#endif
};

void copy_segments(const std::vector<CopySegment> &segments, bool parallel) {
    if (segments.empty())
        return;
#ifdef DSA_MEMCPY_SUPPORT
    if (envs.IAXL_DSA_MEMCPY_ENABLE) {
        size_t total = 0;
        for (const auto &s : segments)
            total += s.n;
        if (total >= envs.IAXL_DSA_MEMCPY_MIN_BYTES && dsa_copy_segments(segments))
            return;
    }
#endif
    const size_t n = segments.size();
    if (parallel && n > 1) {
        parallel::parallel_for(n, [&](size_t i) {
            memcpy(segments[i].dst, segments[i].src, segments[i].n);
        });
    } else {
        for (size_t i = 0; i < n; i++)
            memcpy(segments[i].dst, segments[i].src, segments[i].n);
    }
}

// Appends the segments that gather `view` into (or scatter `contiguous` out of) a flat buffer.
static void view_segments(const ChunkView &view, char *contiguous, bool gather,
                          std::vector<CopySegment> &out) {
    for (int64_t o = 0; o < view.outer_dims; o++) {
        char *tensor_ptr = view.base + o * view.outer_block_size;
        char *flat_ptr = contiguous + o * view.inner_size;
        if (gather)
            out.push_back({flat_ptr, tensor_ptr, static_cast<size_t>(view.inner_size)});
        else
            out.push_back({tensor_ptr, flat_ptr, static_cast<size_t>(view.inner_size)});
    }
}

static ChunkView tensor_view(const torch::Tensor &t) {
    IAXL_CHECK(t.is_contiguous() && t.device().type() == c10::DeviceType::CPU,
               "kv_zip: tensor must be a contiguous CPU tensor");
    const int64_t nbytes = t.numel() * t.element_size();
    return ChunkView{static_cast<char *>(t.data_ptr()), 1, nbytes, nbytes,
                     static_cast<int>(t.element_size()), t.dtype() == torch::kBFloat16};
}

// ---------------------------------------------------------------------------------------------
// Fused byte-plane shuffle + strided copy
//
// data_shuffle() swaps the low byte of element i with the high byte of element n+i (n = half the
// elements). As 16-bit lanes a[] (first half) and b[] (second half) that is
//     u = (a & 0xFF00) | (b >> 8)      v = (a << 8) | (b & 0x00FF)
// which is its own inverse, so one kernel does PUT (gather+shuffle) and GET (unshuffle+scatter)
// in a single pass instead of copy-then-transform. The block is walked in runs that stay inside
// one segment on both halves, so segment boundaries need not line up with the half-way point.

typedef uint16_t __attribute__((may_alias)) u16a;

static inline void shuffle_lanes(const u16a *a, const u16a *b, u16a *u, u16a *v, size_t lanes) {
    for (size_t k = 0; k < lanes; k++) {
        const uint16_t x = a[k], y = b[k];
        u[k] = static_cast<uint16_t>((x & 0xFF00u) | (y >> 8));
        v[k] = static_cast<uint16_t>((x << 8) | (y & 0x00FFu));
    }
}

static inline std::pair<char *, size_t> view_at(const ChunkView &view, size_t off) {
    const size_t seg = off / view.inner_size, rem = off % view.inner_size;
    return {view.base + seg * view.outer_block_size + rem,
            static_cast<size_t>(view.inner_size) - rem};
}

static inline bool fused_shuffle_ok(const ChunkView &view) {
    return view.is_bf16 && view.nbytes() % 4 == 0;
}

// data_shuffle() only transforms bf16 blocks, so this is what PUT actually applies.
static inline bool put_shuffles(const ChunkView &view) {
    return data_shuffle_enabled() && view.is_bf16;
}

// to_view: flat (shuffled) -> view (plain); otherwise view (plain) -> flat (shuffled).
static void shuffle_xfer(const ChunkView &view, char *flat, bool to_view) {
    const size_t half = view.nbytes() / 2;
    for (size_t p = 0; p < half;) {
        auto [v0, rem0] = view_at(view, p);
        auto [v1, rem1] = view_at(view, half + p);
        const size_t run = std::min({half - p, rem0, rem1}) & ~size_t{1};
        IAXL_CHECK(run > 0, "kv_zip: shuffle run is not 16-bit aligned");
        char *f0 = flat + p, *f1 = flat + half + p;
        if (to_view)
            shuffle_lanes(reinterpret_cast<const u16a *>(f0), reinterpret_cast<const u16a *>(f1),
                          reinterpret_cast<u16a *>(v0), reinterpret_cast<u16a *>(v1), run / 2);
        else
            shuffle_lanes(reinterpret_cast<const u16a *>(v0), reinterpret_cast<const u16a *>(v1),
                          reinterpret_cast<u16a *>(f0), reinterpret_cast<u16a *>(f1), run / 2);
        p += run;
    }
}

// ---------------------------------------------------------------------------------------------
// Worker pipeline

// Each lane moves Staging (input copies) -> Codec -> Unstaging (output copies) -> next item.
// prepare/complete may start SlotCopies, so a thread keeps its other lanes busy while DSA copies.
template <class Next, class Prepare, class Launch, class Complete>
static void zip_pipeline(Next &&get_next, Prepare &&prepare, Launch &&launch,
                         Complete &&complete) {
    ensure_zip_init();
    const int qat_workers = envs.IAXL_QAT_ZIP_ENABLE ? envs.IAXL_QAT_INSTANCE_NUM : 0;
    const int iaa_workers = envs.IAXL_IAA_ZIP_ENABLE ? envs.IAXL_IAA_INSTANCE_NUM : 0;
    const int cpu_workers = envs.IAXL_CPU_ZIP_ENABLE ? cpu_zip_num_slots() : 0;
    const int worker_count = qat_workers + iaa_workers + cpu_workers;
    IAXL_CHECK(qat_workers == 0 || qat_workers <= qat_zip_num_slots() / qat_zip_queue_depth(),
               "kv_zip: IAXL_QAT_INSTANCE_NUM exceeds available QAT instances");
    IAXL_CHECK(iaa_workers == 0 || iaa_workers <= iaa_zip_num_slots() / iaa_zip_queue_depth(),
               "kv_zip: IAXL_IAA_INSTANCE_NUM exceeds available IAA instances");
    const int threads = std::min(envs.IAXL_OMP_THREAD_NUM, worker_count);

    enum class Stage { Staging, Codec, Unstaging };
    struct Lane {
        ZipBackend backend;
        int slot;
        size_t item;
        Stage stage;
        SlotCopies copies;
    };

    // Runs a lane forward until it has to wait; returns true once it has nothing left to claim.
    auto advance = [&](Lane &lane) {
        for (;;) {
            switch (lane.stage) {
            case Stage::Staging:
                if (!lane.copies.done())
                    return false;
                launch(lane.backend, lane.slot, lane.item);
                lane.stage = Stage::Codec;
                break;
            case Stage::Codec: {
                void *out;
                int out_len;
                const int status =
                    ops(lane.backend).wait(lane.slot, &out, &out_len, /*non_block=*/1);
                if (status == IAXL_ZIP_PENDING)
                    return false;
                IAXL_CHECK(status == 0, "kv_zip: zip wait failed");
                complete(lane.backend, lane.item, out, out_len, lane.copies);
                lane.stage = Stage::Unstaging;
                break;
            }
            case Stage::Unstaging:
                if (!lane.copies.done())
                    return false;
                lane.item = get_next(lane.backend);
                if (lane.item == kNoTask)
                    return true;
                prepare(lane.backend, lane.slot, lane.item, lane.copies);
                lane.stage = Stage::Staging;
                break;
            }
        }
    };

    // Thread `tid` owns workers tid, tid + threads, ..., so each backend instance stays on one
    // thread while the backends are spread across all of them.
    auto drive = [&](int tid) {
        std::vector<Lane> lanes;
        for (int w = tid; w < worker_count; w += threads) {
            ZipBackend backend = ZipBackend::CPU;
            int first = qat_workers + iaa_workers;
            if (w < qat_workers) {
                backend = ZipBackend::QAT;
                first = 0;
            } else if (w < qat_workers + iaa_workers) {
                backend = ZipBackend::IAA;
                first = qat_workers;
            }
            const int depth = ops(backend).queue_depth();
            for (int k = 0; k < depth; k++) {
                const size_t i = get_next(backend);
                if (i == kNoTask)
                    break;
                const int slot = (w - first) * depth + k;
                Lane &lane = lanes.emplace_back(Lane{backend, slot, i, Stage::Staging, {}});
                prepare(backend, slot, i, lane.copies);
            }
        }

        while (!lanes.empty()) {
            for (size_t s = 0; s < lanes.size();) {
                if (!advance(lanes[s])) {
                    s++;
                    continue;
                }
                if (s + 1 != lanes.size())
                    lanes[s] = std::move(lanes.back());
                lanes.pop_back();
            }
        }
    };

    parallel::run_threads(threads, drive);
}

// ---------------------------------------------------------------------------------------------
// Compression

static void pack_header(char *buf, uint32_t payload_len, ZipBackend backend, size_t orig_size,
                        bool shuffled) {
    IAXL_CHECK((payload_len & ~KV_ZIP_LEN_MASK) == 0, "kv_zip: payload length overflows header");
    reinterpret_cast<uint32_t *>(buf)[0] = payload_len |
                                           (backend == ZipBackend::IAA ? KV_ZIP_IAA_FLAG : 0u) |
                                           (shuffled ? KV_ZIP_SHUFFLE_FLAG : 0u);
    reinterpret_cast<int *>(buf)[1] = static_cast<int>(orig_size);
}

void kv_zip_compress_views(const std::vector<ChunkView> &views, std::vector<char *> &out_bufs,
                           std::vector<size_t> &out_sizes, std::vector<size_t> &orig_sizes,
                           bool compress) {
    const size_t n = views.size();

    if (!compress || !envs.IAXL_KV_COMPRESSION) {
        std::vector<CopySegment> segments;
        for (size_t i = 0; i < n; i++) {
            const size_t nbytes = views[i].nbytes();
            char *buffer = static_cast<char *>(malloc(sizeof(int) * 2 + nbytes));
            IAXL_CHECK(buffer != nullptr, "kv_zip: raw cache buffer allocation failed");
            reinterpret_cast<int *>(buffer)[0] = 0;
            reinterpret_cast<int *>(buffer)[1] = 0;
            view_segments(views[i], buffer + sizeof(int) * 2, true, segments);
            out_bufs[i] = buffer;
            out_sizes[i] = sizeof(int) * 2 + nbytes;
            orig_sizes[i] = nbytes;
        }
        copy_segments(segments, true);
        return;
    }

    std::atomic<size_t> next{0};
    zip_pipeline(
        [&](ZipBackend) {
            const size_t i = next.fetch_add(1, std::memory_order_relaxed);
            return i < n ? i : kNoTask;
        },
        [&](ZipBackend backend, int slot, size_t i, SlotCopies &copies) {
            const ChunkView &view = views[i];
            const size_t nb = view.nbytes();
            orig_sizes[i] = nb;
            IAXL_CHECK(nb <= static_cast<size_t>(INT_MAX),
                       "kv_zip: tensor byte size exceeds zip integer length range");
            IAXL_CHECK(nb <= static_cast<size_t>(ops(backend).src_cap()),
                       "kv_zip: tensor byte size exceeds zip source capacity");
            char *staging = static_cast<char *>(ops(backend).input_buf(slot));
            IAXL_CHECK(staging != nullptr, "kv_zip: backend has no staging buffer for slot");
            // Stage into the codec's own input buffer so the inference tensor is only ever read.
            if (put_shuffles(view) && fused_shuffle_ok(view) && envs.IAXL_KV_LOSSY_TRUNC == 0) {
                shuffle_xfer(view, staging, false);
                return;
            }
            std::vector<CopySegment> segments;
            view_segments(view, staging, true, segments);
            // A plain gather can land asynchronously; truncation and shuffle need the bytes now.
            if (backend != ZipBackend::CPU && !put_shuffles(view) &&
                envs.IAXL_KV_LOSSY_TRUNC == 0) {
                copies.start(segments);
                return;
            }
            copy_segments(segments, false);
            lossy_trunc(staging, nb, view.element_size);
            data_shuffle(staging, nb, view.is_bf16, data_shuffle_enabled());
        },
        [&](ZipBackend backend, int slot, size_t i) {
            const int status =
                ops(backend).compress_staged(slot, static_cast<int>(views[i].nbytes()));
            IAXL_CHECK(status == 0, "kv_zip: zip compress failed");
        },
        [&](ZipBackend backend, size_t i, void *out, int out_len, SlotCopies &) {
            char *buf = static_cast<char *>(malloc(sizeof(int) * 2 + out_len));
            IAXL_CHECK(buf != nullptr, "kv_zip: cache buffer allocation failed");
            pack_header(buf, static_cast<uint32_t>(out_len), backend, orig_sizes[i],
                        put_shuffles(views[i]));
            memcpy(buf + sizeof(int) * 2, out, out_len);
            out_bufs[i] = buf;
            out_sizes[i] = sizeof(int) * 2 + out_len;
        });
}

void kv_zip_compress_batch(const std::vector<torch::Tensor> &tensors, std::vector<char *> &out_bufs,
                           std::vector<size_t> &out_sizes, std::vector<size_t> &orig_sizes,
                           bool compress) {
    std::vector<ChunkView> views;
    views.reserve(tensors.size());
    for (const auto &t : tensors)
        views.push_back(tensor_view(t));
    kv_zip_compress_views(views, out_bufs, out_sizes, orig_sizes, compress);
}

// ---------------------------------------------------------------------------------------------
// Decompression

void kv_zip_decompress_views(const std::vector<const char *> &data_ptrs,
                             const std::vector<ChunkView> &views) {
    const size_t n = views.size();
    IAXL_CHECK(data_ptrs.size() == n, "kv_zip: decompression inputs must have matching lengths");

    // Raw blocks scatter straight from the cache payload; IAA-produced blocks must go back to
    // IAA, everything else to QAT/CPU.
    std::vector<size_t> iaa_items, other_items;
    std::vector<CopySegment> raw_segments;
    for (size_t i = 0; i < n; i++) {
        const int *header = reinterpret_cast<const int *>(data_ptrs[i]);
        if (header[1] == 0) {
            view_segments(views[i], const_cast<char *>(data_ptrs[i]) + sizeof(int) * 2, false,
                          raw_segments);
            continue;
        }
        const uint32_t encoded = static_cast<uint32_t>(header[0]);
        IAXL_CHECK((encoded & KV_ZIP_LEN_MASK) != 0,
                   "kv_zip: invalid compressed payload length");
        ((encoded & KV_ZIP_IAA_FLAG) ? iaa_items : other_items).push_back(i);
    }
    copy_segments(raw_segments, true);

    if (iaa_items.empty() && other_items.empty())
        return;
    IAXL_CHECK(iaa_items.empty() || envs.IAXL_IAA_ZIP_ENABLE,
               "kv_zip: batch holds IAA-compressed blocks but the IAA backend is disabled");
    IAXL_CHECK(other_items.empty() || envs.IAXL_QAT_ZIP_ENABLE || envs.IAXL_CPU_ZIP_ENABLE,
               "kv_zip: batch holds QAT/CPU-compressed blocks but both backends are disabled");

    auto finish = [&](ZipBackend backend, size_t i, void *out, int out_len,
                      SlotCopies &copies) {
        const ChunkView &view = views[i];
        const size_t nb = view.nbytes();
        IAXL_CHECK(out_len >= 0 && static_cast<size_t>(out_len) == nb,
                   "kv_zip: decompressed size does not match tensor byte size");
        // The header, not the environment, says whether the payload was shuffled.
        const bool shuffled =
            reinterpret_cast<const uint32_t *>(data_ptrs[i])[0] & KV_ZIP_SHUFFLE_FLAG;
        char *flat = static_cast<char *>(out);
        if (shuffled && fused_shuffle_ok(view)) {
            shuffle_xfer(view, flat, true);
            return;
        }
        data_shuffle(flat, nb, true, shuffled);
        std::vector<CopySegment> segments;
        view_segments(view, flat, false, segments);
        if (backend != ZipBackend::CPU)
            copies.start(segments);
        else
            copy_segments(segments, false);
    };

    auto payload_of = [&](size_t i) {
        const uint32_t encoded = reinterpret_cast<const uint32_t *>(data_ptrs[i])[0];
        return std::make_pair(const_cast<char *>(data_ptrs[i]) + sizeof(int) * 2,
                              static_cast<int>(encoded & KV_ZIP_LEN_MASK));
    };
    // Items whose payload prepare copied into the slot's device input buffer.
    std::vector<char> staged(n, 0);

    std::atomic<size_t> iaa_next{0}, other_next{0};
    zip_pipeline(
        [&](ZipBackend backend) {
            const bool iaa = backend == ZipBackend::IAA;
            const std::vector<size_t> &items = iaa ? iaa_items : other_items;
            std::atomic<size_t> &cursor = iaa ? iaa_next : other_next;
            const size_t k = cursor.fetch_add(1, std::memory_order_relaxed);
            return k < items.size() ? items[k] : kNoTask;
        },
        [&](ZipBackend backend, int slot, size_t i, SlotCopies &copies) {
            auto [payload, payload_len] = payload_of(i);
            // The CPU backend inflates straight from the cache; devices need their DMA buffer.
            if (backend == ZipBackend::CPU || payload_len > ops(backend).src_cap())
                return;
            char *in = static_cast<char *>(ops(backend).input_buf(slot));
            IAXL_CHECK(in != nullptr, "kv_zip: backend has no staging buffer for slot");
            copies.start({{in, payload, static_cast<size_t>(payload_len)}});
            staged[i] = 1;
        },
        [&](ZipBackend backend, int slot, size_t i) {
            auto [payload, payload_len] = payload_of(i);
            void *src = staged[i] ? ops(backend).input_buf(slot) : payload;
            const int status = ops(backend).decompress(slot, src, payload_len);
            IAXL_CHECK(status == 0, "kv_zip: zip decompress failed");
        },
        finish);
}

void kv_zip_decompress_batch(const std::vector<const char *> &data_ptrs,
                             const std::vector<torch::Tensor> &tensors) {
    std::vector<ChunkView> views;
    views.reserve(tensors.size());
    for (const auto &t : tensors)
        views.push_back(tensor_view(t));
    kv_zip_decompress_views(data_ptrs, views);
}

} // namespace kv_zip
