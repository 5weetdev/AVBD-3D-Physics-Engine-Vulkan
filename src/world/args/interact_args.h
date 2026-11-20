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
