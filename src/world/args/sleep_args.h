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

// Sleep, LOD, contact report, restitution and CCD argument structs (sleep.slang, lod.slang, contact_report.slang, restitution.slang). Mirrors of the Slang structs: field order and padding are byte-for-byte, and
// avbd_test_vk_layout checks them against the compiler's offsets.

#include "vk_buffer.h"
#include "solver_args.h"

#include <cstdint>

namespace avbdvk
{

// ===========================================================================
// sleep.slang (Phase 5B) -- the sleep stage. Reuses FullBodySoAGpu,
// FullJointSoAGpu, FullManifoldSoAGpu, FullSpringSoAGpu, ConstraintGraphGpu (all
// defined above). Field order transcribed from sleep.slang at the time this was
// written; every explicit _padN below matches a gap sleep.slang's Slang compiler
// inserts implicitly for pointer alignment (see this file's header comment on why
// the C++ side names it and the Slang side does not need to).
// ===========================================================================

struct SleepBlockFractureArgsGpu
{
    FullBodySoAGpu b;
    FullJointSoAGpu j;
    float ratio = 0.0f;
    float _pad0 = 0.0f;
    VkDeviceAddress block = 0; // never 0: csSleepBlockFracture stores through it unguarded
};
static_assert(sizeof(SleepBlockFractureArgsGpu) % 8 == 0, "layout");

struct SleepBlockDragArgsGpu
{
    VkDeviceAddress dragIdx = 0;
    int32_t dragCount = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress block = 0; // never 0: csSleepBlockDrag stores through it unguarded
};
static_assert(sizeof(SleepBlockDragArgsGpu) % 8 == 0, "layout");

struct SleepMarkArgsGpu
{
    FullBodySoAGpu b;
    VkDeviceAddress dyn = 0;
    int32_t dynCount = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress candidate = 0;
    int32_t frames = 0;
    float linVel = 0.0f;
    float angVel = 0.0f;
    float disp = 0.0f;
    VkDeviceAddress block = 0; // may be 0 (no fracture joints, no active drags)
    uint32_t hasBlock = 0;
    float leak = 0.0f; // 0: reset-on-spike; > 0: leaky streak
};
static_assert(sizeof(SleepMarkArgsGpu) % 8 == 0, "layout");

struct SleepCommitArgsGpu
{
    FullBodySoAGpu b;
    ConstraintGraphGpu g;
    FullManifoldSoAGpu m;
    FullJointSoAGpu j;
    FullSpringSoAGpu sp;
    VkDeviceAddress active = 0;
    int32_t activeCount = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress candidate = 0;
    VkDeviceAddress touchCount = 0;
    VkDeviceAddress touchHash = 0;
    VkDeviceAddress snapCount = 0;
    VkDeviceAddress snapHash = 0;
    VkDeviceAddress activeCountPtr = 0; // M4 live count (0: activeCount is exact)
};
static_assert(sizeof(SleepCommitArgsGpu) % 8 == 0, "layout");

struct ActiveFlagsArgsGpu
{
    FullBodySoAGpu b;
    VkDeviceAddress dyn = 0;
    int32_t dynCount = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress flags = 0;
    // Distance LOD, both may be 0 (LOD off): the active set is then awake && not frozen,
    // and the result is written back to lodLive for the kernels that read it.
    VkDeviceAddress lodState = 0;
    VkDeviceAddress lodLive = 0;
};
static_assert(sizeof(ActiveFlagsArgsGpu) % 8 == 0, "layout");

// ---------------------------------------------------------------------------
// lod.slang (distance LOD). Not in the layout probe: every field is a pointer or a
// 4-byte scalar in declaration order, and the static_asserts below pin the sizes.
// ---------------------------------------------------------------------------
struct LodClassifyArgsGpu
{
    VkDeviceAddress pos = 0, velLin = 0, velAng = 0, mass = 0, awake = 0;
    VkDeviceAddress state = 0, timer = 0, live = 0;
    VkDeviceAddress stats = 0; // may be 0
    uint32_t count = 0;
    float focusX = 0, focusY = 0, focusZ = 0;
    float nearRadius = 0, farRadius = 0, hysteresis = 0, freezeSpeed = 0;
    uint32_t freezeFrames = 0, freezeFar = 0;
};
static_assert(sizeof(LodClassifyArgsGpu) == 9 * 8 + 10 * 4 + 0, "lod.slang LodClassifyArgs");

struct LodThawRefsGpu
{
    VkDeviceAddress velLin = 0, velAng = 0, prevVelLin = 0, mass = 0, state = 0, timer = 0, live = 0;
    float speed = 0;
    uint32_t _pad0 = 0;
};
static_assert(sizeof(LodThawRefsGpu) == 64, "lod.slang LodThawRefs");

struct LodThawContactsArgsGpu
{
    LodThawRefsGpu r;
    VkDeviceAddress bodyA = 0, bodyB = 0, contactCount = 0;
    uint32_t slots = 0, _pad0 = 0;
};
static_assert(sizeof(LodThawContactsArgsGpu) == 64 + 24 + 8, "lod.slang LodThawContactsArgs");

// contact_report.slang: one point of a reported manifold (32 bytes).
struct ContactReportPointGpu
{
    float ax = 0, ay = 0, az = 0, bx = 0, by = 0, bz = 0;
    float separation = 0, normalImpulse = 0;
};
static_assert(sizeof(ContactReportPointGpu) == 32, "contact_report.slang ContactReportPoint");

// contact_report.slang: one record per reported manifold (192 bytes: 64-byte header + 4 points).
struct ContactReportRecordGpu
{
    int32_t shapeA = 0, shapeB = 0, bodyA = 0, bodyB = 0;
    float nx = 0, ny = 0, nz = 0, approachSpeed = 0;
    float normalImpulse = 0, tangentImpulse = 0, separation = 0, _f0 = 0;
    uint32_t pointCount = 0, _pad0 = 0, _pad1 = 0, _pad2 = 0;
    ContactReportPointGpu points[4];
};
static_assert(sizeof(ContactReportRecordGpu) == 192, "contact_report.slang ContactReportRecord");

struct ContactReportArgsGpu
{
    VkDeviceAddress pos = 0, rot = 0, prevVelLin = 0;
    VkDeviceAddress bodyA = 0, bodyB = 0, shapeA = 0, shapeB = 0, basis = 0;
    VkDeviceAddress contactOffset = 0, contactCount = 0, rA = 0, rB = 0, C0 = 0, lambda = 0;
    VkDeviceAddress shapeFlags = 0, counter = 0, records = 0;
    uint32_t capacity = 0, slots = 0;
};
static_assert(sizeof(ContactReportArgsGpu) == 17 * 8 + 8, "contact_report.slang ContactReportArgs");

// restitution.slang: one manifold slot's bounce impulse (32 bytes).
struct RestitutionSlotGpu
{
    float ax = 0, ay = 0, az = 0, impulse = 0;
    float bx = 0, by = 0, bz = 0, vnPre = 0;
};
static_assert(sizeof(RestitutionSlotGpu) == 32, "restitution.slang RestitutionSlot");

// restitution.slang RestitutionArgs: shared by csRestitutionPre/Impulse/Apply.
struct RestitutionArgsGpu
{
    VkDeviceAddress pos = 0, rot = 0, velLin = 0, velAng = 0, mass = 0, moment = 0, props = 0;
    VkDeviceAddress bodyA = 0, bodyB = 0, basis = 0, contactOffset = 0, contactCount = 0;
    VkDeviceAddress rA = 0, rB = 0, lambda = 0;
    VkDeviceAddress gKey = 0, gIsA = 0, gStart = 0, gEnd = 0; // may be 0 except in csRestitutionApply
    VkDeviceAddress dyn = 0;                                  // may be 0 except in csRestitutionApply
    VkDeviceAddress out = 0;
    uint32_t slots = 0, dynCount = 0;
    float threshold = 0;
    uint32_t _pad = 0;
};
static_assert(sizeof(RestitutionArgsGpu) == 21 * 8 + 16, "restitution.slang RestitutionArgs");

// ccd.slang CcdArgs: shared by csCcdFlag / csCcdToi / csCcdApply.
struct CcdArgsGpu
{
    VkDeviceAddress pos = 0, rot = 0, velLin = 0, initLin = 0, initAng = 0, mass = 0, awake = 0, sleepTimer = 0;
    VkDeviceAddress props = 0; // may be null
    VkDeviceAddress shpBody = 0, shpLocalPos = 0, shpLocalRot = 0, shpHalf = 0, shpType = 0, shpFlags = 0;
    VkDeviceAddress shpCategory = 0, shpMask = 0, shpGroup = 0;
    VkDeviceAddress dynShape = 0, staticShape = 0;
    VkDeviceAddress ignore = 0;                   // may be null when ignoreCount == 0
    VkDeviceAddress geomHeap = 0, geomOffset = 0; // may be null (no hull or mesh in the scene)
    uint64_t geomHeapBytes = 0;
    VkDeviceAddress list = 0, counter = 0, frac = 0, scratch = 0;
    uint32_t dynCount = 0, staticCount = 0, ignoreCount = 0, scratchStride = 0;
    uint32_t lanes = 0, pass = 0;
    float safety = 1.0f;
    uint32_t _pad = 0;
};
static_assert(sizeof(CcdArgsGpu) == 28 * 8 + 32, "ccd.slang CcdArgs");

struct LodThawJointsArgsGpu
{
    LodThawRefsGpu r;
    VkDeviceAddress jBodyA = 0, jBodyB = 0, jBroken = 0, sBodyA = 0, sBodyB = 0;
    uint32_t jointSlots = 0, jointCount = 0, springCount = 0, _pad0 = 0;
};
static_assert(sizeof(LodThawJointsArgsGpu) == 64 + 40 + 16, "lod.slang LodThawJointsArgs");

} // namespace avbdvk
