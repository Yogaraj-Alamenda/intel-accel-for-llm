// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "env.h"

namespace kv_zip {

inline bool data_shuffle_enabled() { return envs.IAXL_KV_DATA_SHUFFLE; }

// A 16-bit word at any byte offset.
typedef uint16_t unaligned_u16 __attribute__((aligned(1), may_alias));

// Copies `size` bytes with every 16-bit word ANDed with `mask`.
inline void masked_copy(char *__restrict dst, const char *__restrict src, size_t size,
                        uint16_t mask) {
    if (mask == 0xFFFF) {
        memcpy(dst, src, size);
        return;
    }
    const auto *s = reinterpret_cast<const unaligned_u16 *>(src);
    auto *d = reinterpret_cast<unaligned_u16 *>(dst);
    for (size_t i = 0; i < size / 2; i++)
        d[i] = s[i] & mask;
    if (size % 2)
        dst[size - 1] = static_cast<char>(src[size - 1] & mask);
}

// Copies a bf16 buffer with every element ANDed with `mask`, moving the high bytes into the first
// half and the low bytes into the second. Shuffling the result again restores the input. The
// byte order is persisted in compressed blocks, so it must not change.
// The source is split: its first size / 2 bytes at `a`, the rest at `b`.
inline void shuffle_copy(char *__restrict dst, const char *__restrict a, const char *__restrict b,
                         size_t size, uint16_t mask = 0xFFFF) {
    const size_t half = size / 2, n = size / 4;
    // With an odd half, every word of the second half straddles two elements.
    const uint16_t mask_b = half % 2 ? __builtin_bswap16(mask) : mask;
    const auto *sa = reinterpret_cast<const unaligned_u16 *>(a);
    const auto *sb = reinterpret_cast<const unaligned_u16 *>(b);
    auto *da = reinterpret_cast<unaligned_u16 *>(dst);
    auto *db = reinterpret_cast<unaligned_u16 *>(dst + half);
    for (size_t i = 0; i < n; i++) {
        const uint16_t x = sa[i] & mask, y = sb[i] & mask_b;
        da[i] = static_cast<uint16_t>((x & 0xFF00) | (y >> 8));
        db[i] = static_cast<uint16_t>((y & 0x00FF) | (x << 8));
    }
    // Bytes the loop leaves in place when size is not a multiple of 4.
    for (size_t o = 2 * n; o < half; o++)
        dst[o] = static_cast<char>(a[o] & (o % 2 ? mask >> 8 : mask));
    for (size_t o = half + 2 * n; o < size; o++)
        dst[o] = static_cast<char>(b[o - half] & (o % 2 ? mask >> 8 : mask));
}

inline void shuffle_copy(char *__restrict dst, const char *src, size_t size,
                         uint16_t mask = 0xFFFF) {
    shuffle_copy(dst, src, src + size / 2, size, mask);
}

} // namespace kv_zip
