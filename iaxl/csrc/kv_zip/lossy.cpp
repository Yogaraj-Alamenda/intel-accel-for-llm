// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "lossy.h"

#include "env.h"

namespace kv_zip {

uint16_t lossy_mask(int element_size) {
    const int bits = envs.IAXL_KV_LOSSY_TRUNC;
    if (bits <= 0)
        return 0xFFFF;
    if (element_size == 2)
        return static_cast<uint16_t>(0xFFFF << bits);
    if (element_size == 1) {
        const auto byte = static_cast<uint8_t>(0xFF << bits);
        return static_cast<uint16_t>(byte << 8 | byte);
    }
    return 0xFFFF;
}

} // namespace kv_zip
