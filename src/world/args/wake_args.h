#pragma once

// Wake stage argument structs (wake.slang). Mirrors of the Slang structs: field order and padding are byte-for-byte, and
// avbd_test_vk_layout checks them against the compiler's offsets.

#include "vk_buffer.h"
#include "solver_args.h"

#include <cstdint>

namespace avbdvk
{

// ===========================================================================
// wake.slang -- the wake stage, the implicit velocity-gated wake
// path plus wakeAll(). Reuses FullBodySoAGpu/FullManifoldSoAGpu/
// FullJointSoAGpu/FullSpringSoAGpu (all defined above).
// ===========================================================================

struct WakeRequestArgsGpu
{
    FullBodySoAGpu b;
    FullManifoldSoAGpu m;
    int32_t slots = 0;
    float wakeLinVel = 0.0f;
    float wakeAngVel = 0.0f;
    VkDeviceAddress request = 0;
};
static_assert(sizeof(WakeRequestArgsGpu) % 8 == 0, "layout");

struct TouchClearArgsGpu
{
    VkDeviceAddress dyn = 0;
    int32_t dynCount = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress count = 0;
    VkDeviceAddress hash = 0;
};
static_assert(sizeof(TouchClearArgsGpu) % 8 == 0, "layout");

struct TouchAccumArgsGpu
{
    FullBodySoAGpu b;
    FullManifoldSoAGpu m;
    int32_t slots = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress count = 0;
    VkDeviceAddress hash = 0;
};
static_assert(sizeof(TouchAccumArgsGpu) % 8 == 0, "layout");

struct WakeLostSupportArgsGpu
{
    FullBodySoAGpu b;
    VkDeviceAddress dyn = 0;
    int32_t dynCount = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress count = 0;
    VkDeviceAddress hash = 0;
    VkDeviceAddress snapCount = 0;
    VkDeviceAddress snapHash = 0;
    VkDeviceAddress request = 0;
};
static_assert(sizeof(WakeLostSupportArgsGpu) % 8 == 0, "layout");

struct WakeApplyArgsGpu
{
    FullBodySoAGpu b;
    VkDeviceAddress dyn = 0;
    int32_t dynCount = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress request = 0;
};
static_assert(sizeof(WakeApplyArgsGpu) % 8 == 0, "layout");

struct TouchAccumJointsArgsGpu
{
    FullBodySoAGpu b;
    FullJointSoAGpu j;
    FullSpringSoAGpu sp;
    VkDeviceAddress count = 0;
    VkDeviceAddress hash = 0;
};
static_assert(sizeof(TouchAccumJointsArgsGpu) % 8 == 0, "layout");

struct WakeRequestJointsArgsGpu
{
    FullBodySoAGpu b;
    FullJointSoAGpu j;
    FullSpringSoAGpu sp;
    VkDeviceAddress request = 0;
};
static_assert(sizeof(WakeRequestJointsArgsGpu) % 8 == 0, "layout");

struct WakeIslandSeedArgsGpu
{
    FullBodySoAGpu b;
    VkDeviceAddress dyn = 0;
    int32_t dynCount = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress request = 0;
    VkDeviceAddress seed = 0;
    VkDeviceAddress any = 0;
};
static_assert(sizeof(WakeIslandSeedArgsGpu) % 8 == 0, "layout");

struct WakeAllArgsGpu
{
    FullBodySoAGpu b;
    VkDeviceAddress dyn = 0;
    int32_t dynCount = 0;
    int32_t _pad0 = 0;
};
static_assert(sizeof(WakeAllArgsGpu) % 8 == 0, "layout");

} // namespace avbdvk
