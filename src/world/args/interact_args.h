/*
 * Copyright (c) 2026 Chris Giles
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies.
 * Chris Giles makes no representations about the suitability
 * of this software for any purpose.
 * It is provided "as is" without express or implied warranty.
 */

#pragma once

// Pick and drag argument structs (pick.slang, drag.slang). Mirrors of the Slang structs: field order and padding are byte-for-byte, and
// avbd_test_vk_layout checks them against the compiler's offsets.

#include "vk_buffer.h"
#include "solver_args.h"

#include <cstdint>

namespace avbdvk
{

// ===========================================================================
// pick.slang / drag.slang -- the pick path and the drag path.
// Reuses FullBodySoAGpu and FullJointSoAGpu (both defined above).
// ===========================================================================

struct PickArgsGpu
{
    FullBodySoAGpu b;
    float originX = 0, originY = 0, originZ = 0;
    float dirX = 0, dirY = 0, dirZ = 0;
    VkDeviceAddress bestHit =
        0;         // uint64_t*: (tBits << 32) | bodyIndex; csPick min's the high word, csPickResolve the low
    ShapeSoAGpu s; // the ray tests every shape
};
static_assert(sizeof(PickArgsGpu) % 8 == 0, "layout");

struct DragUpdateArgsGpu
{
    FullBodySoAGpu b;
    FullJointSoAGpu j;
    int32_t slot0 = 0;
    int32_t count = 0;
    VkDeviceAddress bodies = 0;    // const int32_t*
    VkDeviceAddress targets = 0;   // const vec3*
    VkDeviceAddress locals = 0;    // const vec3*
    VkDeviceAddress stiffness = 0; // const float*
    int32_t grab = 0;
    int32_t _pad0 = 0;
};
static_assert(sizeof(DragUpdateArgsGpu) % 8 == 0, "layout");

struct DragReleaseArgsGpu
{
    FullJointSoAGpu j;
    int32_t slot0 = 0;
    int32_t count = 0;
};
static_assert(sizeof(DragReleaseArgsGpu) % 8 == 0, "layout");

} // namespace avbdvk
