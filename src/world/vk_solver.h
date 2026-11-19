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

// Vulkan host orchestration for src/shaders/primal.slang and dual.slang -- the AVBD
// iteration itself. Mirrors the launch*() seam convention:
// launchPrimalThreads/launchPrimalLanes (collapsed here into one launchPrimalColor call,
// selecting thread-per-body vs the 4-lane cooperative variant and With/NoForces via
// booleans instead of four separate the reference template instantiations), launchJointInit,
// launchJointDual, launchDual.
// Phase 4b and tools/test_vk_solve_iter.cpp for the reference-vs-Vulkan parity test that
// exercises these.
//
// Phase 4b scope (see forces.slang / primal.slang header comments): legacy weld joints
// and plain springs only; LANES=4 only (not the reference full 2/8/16/32 switch).

#include "vk_args.h"
#include "vk_buffer.h"
#include "vk_cmd.h"
#include "vk_device.h"
#include "vk_pipeline.h"

#include <cstdint>

namespace avbdvk
{

struct SolverPipelines
{
    Pipeline primalThreadNoForces, primalThreadForces;
    Pipeline primalWarp4NoForces, primalWarp4Forces;
    Pipeline primalWideNoForces, primalWideForces; // high-degree bodies (PrimalArgsGpu.heavyDegree)
    Pipeline jointInit, jointDual, dual;
    Pipeline iterateSmall; // csIterateSmall: the whole iteration loop in one workgroup (primal.slang)
    Pipeline primalDispatchArgs;
    Pipeline gateDispatchArgs;  // csGateDispatchArgs (primal.slang), see vk_gate.h
    Pipeline stepFlagIfGreater; // csStepFlagIfGreater: flag words 0 and 1 if *count > cap
    Pipeline stepFlagIfSet;     // csStepFlagIfSet: flag word |= (src[0] != 0)
    Pipeline stepColorBound;    // csStepColorBound: flag word |= any colour in [bound, max) non-empty
    // Two-level AVBD coarse correction (primal.slang csCoarse*).
    Pipeline coarseReset, coarseAssign, coarseAssemble, coarseSolve, coarseApply;

    // primalSpirv/primalWords: src/shaders/primal.slang's embedded SPIR-V.
    // dualSpirv/dualWords: src/shaders/dual.slang's embedded SPIR-V.
    void init(Device &dev, PipelineLayout &layout, const uint32_t *primalSpirv, size_t primalWords,
              const uint32_t *dualSpirv, size_t dualWords);
    void destroy();
};

// One primal sweep over ONE color's body list (args.listStart/args.listCount select the
// range within args.colorBodies -- the caller loops this once per color, exactly like
// VkWorld::step()'s per-color launchPrimal* loop). `warp4` selects the LANES=4 cooperative
// variant (csPrimalWarp4*) over thread-per-body (csPrimalThread*); `withForces` selects
// whether joints/springs are stamped (matches the reference WITH_FORCES template parameter).
void launchPrimalColor(CommandList &cmd, HostRingBuffer &ring, SolverPipelines &pl, const PrimalArgsGpu &args,
                       bool warp4, bool withForces);

// M4: the same sweep with the colour's range and group count on device. Record
// launchPrimalDispatchArgs once per step (after the colouring, fenced with
// barrierIndirect), then one launchPrimalColorIndirect per colour: args.colorStart/
// colorEnd/color select the range, and `indirect` holds a VkDispatchIndirectCommand per
// colour. `warp4` must match between the two, since it sets the bodies per group.
void launchPrimalDispatchArgs(CommandList &cmd, HostRingBuffer &ring, SolverPipelines &pl, VkDeviceAddress colorStart,
                              VkDeviceAddress colorEnd, Buffer &indirect, int colorCount, bool warp4);
void launchPrimalColorIndirect(CommandList &cmd, HostRingBuffer &ring, SolverPipelines &pl, const PrimalArgsGpu &args,
                               bool warp4, bool withForces, const Buffer &indirect);
// The wide pass for one colour: args.heavyDegree > 0, `indirect` built at 256 list
// entries per group (launchPrimalDispatchArgs with warp4 = false). Records alongside the
// colour's normal launch, no barrier between: they touch disjoint bodies.
void launchPrimalWideIndirect(CommandList &cmd, HostRingBuffer &ring, SolverPipelines &pl, const PrimalArgsGpu &args,
                              bool withForces, const Buffer &indirect);

// The fused small-world iteration loop: one single-workgroup dispatch that runs every iteration
// (each colour's primal, then the contact dual, then the joint dual). `pa`, `da` and `jd` are
// the same argument structs the per-colour / per-pass launches take (PrimalArgsGpu with
// colorStart/colorEnd set, DualArgsGpu, JointDualArgsGpu).
void launchIterateSmall(CommandList &cmd, HostRingBuffer &ring, SolverPipelines &pl, const PrimalArgsGpu &pa,
                        const DualArgsGpu &da, const JointDualArgsGpu &jd, int iterations, int colorBound,
                        bool withForces, int jointSlots);

// Step-loop flags (record BEFORE the dispatch gate is installed): set the int32 at `flag`
// if the int32 at `src` is non-zero / if any colour in [bound, maxColors) is non-empty.
void launchStepFlagIfSet(CommandList &cmd, HostRingBuffer &ring, SolverPipelines &pl, VkDeviceAddress src,
                         VkDeviceAddress flag);
void launchStepFlagIfGreater(CommandList &cmd, HostRingBuffer &ring, SolverPipelines &pl, VkDeviceAddress count,
                             VkDeviceAddress flag, int cap);
void launchStepColorBound(CommandList &cmd, HostRingBuffer &ring, SolverPipelines &pl, VkDeviceAddress colorStart,
                          VkDeviceAddress colorEnd, VkDeviceAddress flag, int bound, int maxColors);

// jointCount bodies of thread-per-joint work; args.j.count is NOT read on the host side
// (the kernel itself bounds-checks against j.count), so jointCount here only sizes the
// dispatch's group count -- pass the same value used to size args.j.count upload.
void launchJointInit(CommandList &cmd, HostRingBuffer &ring, SolverPipelines &pl, const JointInitArgsGpu &args,
                     int jointCount);

void launchJointDual(CommandList &cmd, HostRingBuffer &ring, SolverPipelines &pl, const JointDualArgsGpu &args,
                     int jointCount);

// `slots` is the manifold slot count (args.slots is also uploaded in the args struct;
// duplicated here purely to size the dispatch, same convention as jointCount above).
void launchDual(CommandList &cmd, HostRingBuffer &ring, SolverPipelines &pl, const DualArgsGpu &args, int slots);

// One csCoarse* dispatch over `threads` invocations; groupSize must match the kernel's
// numthreads (256 for reset/assign/apply, 128 for assemble/solve).
void launchCoarse(CommandList &cmd, HostRingBuffer &ring, Pipeline &pipeline, const CoarseArgsGpu &args, int threads,
                  int groupSize);

} // namespace avbdvk
