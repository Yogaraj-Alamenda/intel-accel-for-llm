// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace kv_zip {

// IAXL_KV_LOSSY_TRUNC as an AND mask for every 16-bit word of a tensor; 0xFFFF keeps all bits.
uint16_t lossy_mask(int element_size);

}
