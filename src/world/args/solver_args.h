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

// Body, manifold, joint and spring SoA views plus the primal/dual/iterate argument structs (primal.slang, dual.slang, forces.slang). Mirrors of the Slang structs: field order and padding are byte-for-byte, and
// avbd_test_vk_layout checks them against the compiler's offsets.

#include "solver_params_pack.h"
#include "vk_buffer.h"

#include <cstdint>

namespace avbdvk
{

// ===========================================================================
// primal.slang / dual.slang (Phase 4b) -- full BodySoA/ManifoldSoA/JointSoA/SpringSoA
// device-address mirrors (types.slang's structs, byte-for-byte), the local
// ConstraintGraph mirror (primal.slang/dual.slang declare their own copy the same way
// integrate.slang keeps a private BodySoA -- see those files' header comments), and
// the per-kernel argument structs. Field order transcribed from types.slang / the two
// .slang files at the time this was written; any drift is exactly the bug class
// tools/test_vk_layout.cpp + engine/shaders/layout_probe.slang exist to catch.
// ===========================================================================

struct FullBodySoAGpu
{
    VkDeviceAddress pos = 0, rot = 0;
    VkDeviceAddress initialLin = 0, initialAng = 0;
    VkDeviceAddress inertialLin = 0, inertialAng = 0;
    VkDeviceAddress velLin = 0, velAng = 0, prevVelLin = 0;
    VkDeviceAddress halfExtents = 0;
    VkDeviceAddress shapeType = 0;
    VkDeviceAddress moment = 0;
    VkDeviceAddress mass = 0, friction = 0;
    VkDeviceAddress awake = 0;
    VkDeviceAddress sleepTimer = 0;
    VkDeviceAddress restPos = 0, restRot = 0;
    VkDeviceAddress gyration = 0;
    VkDeviceAddress props = 0; // may be 0: no body has a non-default BodyProps
    uint32_t count = 0, _pad0 = 0;
};
static_assert(sizeof(FullBodySoAGpu) % 8 == 0, "layout");

// Mirror of types.slang ShapeSoA: the collision shapes (one or more per body).
struct ShapeSoAGpu
{
    VkDeviceAddress body = 0;
    VkDeviceAddress localPos = 0, localRot = 0, halfExtents = 0, shapeType = 0, friction = 0;
    VkDeviceAddress category = 0, mask = 0, group = 0, flags = 0, geomOffset = 0;
    uint32_t count = 0, _pad0 = 0;
};
static_assert(sizeof(ShapeSoAGpu) % 8 == 0, "layout");

struct FullManifoldSoAGpu
{
    VkDeviceAddress key = 0;
    VkDeviceAddress bodyA = 0, bodyB = 0;
    VkDeviceAddress basis = 0;
    VkDeviceAddress friction = 0;
    VkDeviceAddress contactOffset = 0, contactCount = 0;
    int32_t count = 0, capacity = 0;
    VkDeviceAddress feature = 0;
    VkDeviceAddress rA = 0, rB = 0;
    VkDeviceAddress C0 = 0;
    VkDeviceAddress penalty = 0, lambda = 0;
    VkDeviceAddress stick = 0;
    int32_t contactTotal = 0, contactCapacity = 0;
    VkDeviceAddress shapeA = 0, shapeB = 0;
};
static_assert(sizeof(FullManifoldSoAGpu) % 8 == 0, "layout");

struct FullJointSoAGpu
{
    VkDeviceAddress bodyA = 0, bodyB = 0;
    VkDeviceAddress rA = 0, rB = 0;
    VkDeviceAddress C0Lin = 0, C0Ang = 0;
    VkDeviceAddress penaltyLin = 0, penaltyAng = 0;
    VkDeviceAddress lambdaLin = 0, lambdaAng = 0;
    VkDeviceAddress stiffnessLin = 0, stiffnessAng = 0;
    VkDeviceAddress fracture = 0;
    VkDeviceAddress torqueArm = 0;
    VkDeviceAddress restAng = 0;
    VkDeviceAddress broken = 0;
    VkDeviceAddress frameA = 0, frameB = 0;
    VkDeviceAddress axisMode = 0;
    VkDeviceAddress axisLo = 0, axisHi = 0;
    VkDeviceAddress generic = 0;
    VkDeviceAddress fractureAng = 0; // may be 0 (legacy angular test)
    uint32_t count = 0, _pad0 = 0;
};
static_assert(sizeof(FullJointSoAGpu) % 8 == 0, "layout");

struct FullSpringSoAGpu
{
    VkDeviceAddress bodyA = 0, bodyB = 0;
    VkDeviceAddress rA = 0, rB = 0;
    VkDeviceAddress rest = 0, stiffness = 0;
    // Cloth extension (types.slang SpringSoA): all may be 0 in a scene without cloth,
    // where every spring is a plain one (mode 0).
    VkDeviceAddress lambda = 0, penalty = 0, C0 = 0;    // may be 0
    VkDeviceAddress fracture = 0, broken = 0, mode = 0; // may be 0
    uint32_t count = 0, _pad0 = 0;
};
static_assert(sizeof(FullSpringSoAGpu) % 8 == 0, "layout");

// Mirrors primal.slang/dual.slang's private ConstraintGraph (key, isA, start, end).
struct ConstraintGraphGpu
{
    VkDeviceAddress key = 0;
    VkDeviceAddress isA = 0;
    VkDeviceAddress start = 0;
    VkDeviceAddress end = 0;
};
static_assert(sizeof(ConstraintGraphGpu) % 8 == 0, "layout");

// Mirrors solver_params_pack.h's SolverParamsGpu / types.slang's SolverParamsGpu --
// reuse avbdvk::SolverParamsGpu (defined above) rather than a second copy.

struct PrimalArgsGpu
{
    FullBodySoAGpu b;
    ConstraintGraphGpu g;
    FullManifoldSoAGpu m;
    FullJointSoAGpu joints;
    FullSpringSoAGpu springs;
    VkDeviceAddress colorBodies = 0;
    int32_t listStart = 0;
    int32_t listCount = 0;
    SolverParamsGpu p;
    VkDeviceAddress colorStart = 0, colorEnd = 0; // M4 device colour range, see primal.slang
    int32_t color = 0, heavyDegree = 0;           // heavyDegree > 0: high-degree bodies go to csPrimalWide*
    VkDeviceAddress shapeRolling = 0;             // per shape rolling resistance; may be null (no shape has any)
};

// Mirrors CoarseArgs (primal.slang): the two-level AVBD coarse correction.
struct CoarseArgsGpu
{
    PrimalArgsGpu pa;
    VkDeviceAddress active = 0, activeCount = 0, aggOf = 0, keys = 0, acc = 0, delta = 0;
    VkDeviceAddress stats = 0; // may be null
    int32_t maxActive = 0, bodyCount = 0, tableSize = 0;
    float cell = 1.0f, damping = 0.7f;
    int32_t withForces = 0;
    int32_t color = -1, _pad0 = 0; // cell parity swept, -1 = all
};
static_assert(sizeof(CoarseArgsGpu) % 8 == 0, "layout");

struct PrimalDispatchArgsGpu
{
    VkDeviceAddress colorStart = 0, colorEnd = 0, out = 0;
    uint32_t colorCount = 0, bodiesPerGroup = 0;
};
static_assert(sizeof(PrimalDispatchArgsGpu) % 8 == 0, "layout");

struct GateDispatchArgsGpu
{
    VkDeviceAddress src = 0, dst = 0, flag = 0;
    int32_t count = 0, _pad0 = 0;
};
static_assert(sizeof(GateDispatchArgsGpu) % 8 == 0, "layout");

struct StepFlagIfSetArgsGpu
{
    VkDeviceAddress src = 0, flag = 0;
};
static_assert(sizeof(StepFlagIfSetArgsGpu) % 8 == 0, "layout");

struct StepFlagIfGreaterArgsGpu
{
    VkDeviceAddress count = 0, flag = 0;
    int32_t cap = 0, _pad0 = 0;
};
static_assert(sizeof(StepFlagIfGreaterArgsGpu) % 8 == 0, "layout");

struct StepColorBoundArgsGpu
{
    VkDeviceAddress start = 0, end = 0, flag = 0;
    int32_t bound = 0, maxColors = 0;
};
static_assert(sizeof(StepColorBoundArgsGpu) % 8 == 0, "layout");

struct JointInitArgsGpu
{
    FullBodySoAGpu b;
    FullJointSoAGpu j;
    SolverParamsGpu p;
};

struct JointDualArgsGpu
{
    FullBodySoAGpu b;
    FullJointSoAGpu j;
    SolverParamsGpu p;
};

// NOTE: no padding between `slots` and `p` here, deliberately -- dual.slang's DualArgs
// is `int slots; SolverParamsGpu p;` with nothing between them, and SolverParamsGpu is
// all 4-byte scalars (needs no 8-byte alignment), so Slang's scalar layout packs `p`
// immediately after the 4-byte `slots` with no gap. Adding a _pad0 here would shift
// every field of `p` the GPU reads by 4 bytes relative to what this struct writes --
// exactly the NarrowphaseArgsGpu bug this file's header warns about. See
// src/shaders/dual.slang's DualArgs / ContactQualityArgs for the field order this
// mirrors, and tools/test_vk_layout.cpp's probes for the field-by-field GPU proof.
struct DualArgsGpu
{
    FullBodySoAGpu b;
    FullManifoldSoAGpu m;
    int32_t slots = 0;
    SolverParamsGpu p;
};

// Mirrors IterateArgs (primal.slang): the fused small-world iteration loop. The three argument
// structs sit elsewhere in the ring and are reached through pointers, so no nested struct
// layout is involved.
struct IterateArgsGpu
{
    VkDeviceAddress pa = 0; // PrimalArgsGpu (colorStart/colorEnd set, color ignored)
    VkDeviceAddress da = 0; // DualArgsGpu
    VkDeviceAddress jd = 0; // JointDualArgsGpu
    int32_t iterations = 0;
    int32_t colorBound = 0;
    int32_t withForces = 0;
    int32_t jointSlots = 0;
};
static_assert(sizeof(IterateArgsGpu) == 40, "IterateArgs: 3 pointers + 4 ints");

} // namespace avbdvk
