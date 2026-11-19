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

// Mutation-queue argument structs (mutate.slang). Mirrors of the Slang structs: field order and padding are byte-for-byte, and
// avbd_test_vk_layout checks them against the compiler's offsets.

#include "vk_buffer.h"
#include "solver_args.h"

#include <cstdint>

namespace avbdvk
{

// ===========================================================================
// mutate.slang -- the mutation path: the batched external
// mutation apply and the explicit, UNGATED wake flood every mutation seeds.
// Reuses FullBodySoAGpu/FullManifoldSoAGpu/FullJointSoAGpu/FullSpringSoAGpu.
// ===========================================================================

// Mirrors BodyTransform / BodyPointVector byte
// for byte -- see mutate.slang's own header comment: both are all-4-byte-aligned
// (vec3/quat, no 8-byte pointers), so no padding either side.
struct BodyTransformGpu
{
    float px, py, pz;
    float qx, qy, qz, qw;
};
static_assert(sizeof(BodyTransformGpu) == 28, "must match BodyTransform");

struct BodyPointVectorGpu
{
    float vx, vy, vz;
    float wx, wy, wz;
};
static_assert(sizeof(BodyPointVectorGpu) == 24, "must match BodyPointVector");

struct MutateApplyArgsGpu
{
    FullBodySoAGpu b;
    int32_t type = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress idx = 0;
    VkDeviceAddress data = 0;
    int32_t count = 0;
    float dt = 0.0f;
    VkDeviceAddress seed = 0;
};
static_assert(sizeof(MutateApplyArgsGpu) % 8 == 0, "layout");

// csJointWrite (mutate.slang:JointRowWrite). Scalars only: 176 bytes, asserted below.
// One record per joint row whose authored fields VkWorld::setJointRows rewrote (dual state kept).
struct JointWriteRecGpu
{
    uint32_t row = 0, _pad0 = 0;
    float rA[3] = {0, 0, 0};
    float rB[3] = {0, 0, 0};
    float restAng[4] = {0, 0, 0, 1};
    float frameA[4] = {0, 0, 0, 1};
    float frameB[4] = {0, 0, 0, 1};
    float kLin = 0, kAng = 0, frac = 0, fracAng = 0, arm = 0;
    uint32_t generic = 0;
    uint32_t axisMode[6] = {0, 0, 0, 0, 0, 0};
    float axisLo[6] = {0, 0, 0, 0, 0, 0};
    float axisHi[6] = {0, 0, 0, 0, 0, 0};
};
static_assert(sizeof(JointWriteRecGpu) == 176, "must match mutate.slang:JointRowWrite");

struct JointWriteArgsGpu
{
    FullJointSoAGpu j;
    VkDeviceAddress recs = 0;
    int32_t count = 0;
    int32_t _pad0 = 0;
};
static_assert(sizeof(JointWriteArgsGpu) % 8 == 0, "layout");

struct WakeSeedRegionArgsGpu
{
    FullBodySoAGpu b;
    VkDeviceAddress dyn = 0;
    int32_t dynCount = 0;
    int32_t _pad0 = 0;
    float loX = 0.0f, loY = 0.0f, loZ = 0.0f;
    float hiX = 0.0f, hiY = 0.0f, hiZ = 0.0f;
    VkDeviceAddress seed = 0;
    VkDeviceAddress changed = 0;
};
static_assert(sizeof(WakeSeedRegionArgsGpu) % 8 == 0, "layout");

struct WakeFloodContactsArgsGpu
{
    FullBodySoAGpu b;
    FullManifoldSoAGpu m;
    int32_t slots = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress seed = 0;
    VkDeviceAddress changed = 0;
};
static_assert(sizeof(WakeFloodContactsArgsGpu) % 8 == 0, "layout");

struct WakeFloodJointsArgsGpu
{
    FullBodySoAGpu b;
    FullJointSoAGpu j;
    FullSpringSoAGpu sp;
    VkDeviceAddress seed = 0;
    VkDeviceAddress changed = 0;
};
static_assert(sizeof(WakeFloodJointsArgsGpu) % 8 == 0, "layout");

struct WakeSeedMergeArgsGpu
{
    VkDeviceAddress dyn = 0;
    int32_t dynCount = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress seed = 0;
    VkDeviceAddress request = 0;
    VkDeviceAddress gate = 0; // skip the merge while *gate != 0; 0 = ungated
};
static_assert(sizeof(WakeSeedMergeArgsGpu) % 8 == 0, "layout");

struct BodyGatherArgsGpu
{
    FullBodySoAGpu b;
    int32_t type = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress idx = 0;
    VkDeviceAddress out = 0;
    int32_t count = 0;
    int32_t _pad1 = 0;
};
static_assert(sizeof(BodyGatherArgsGpu) % 8 == 0, "layout");

} // namespace avbdvk
