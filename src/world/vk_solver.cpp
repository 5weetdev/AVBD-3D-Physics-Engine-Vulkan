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

#include "vk_solver.h"

#include <algorithm>

namespace avbdvk
{

static uint32_t ceilDiv(int n, int d)
{
    if (n <= 0)
        return 0;
    return (uint32_t)((n + d - 1) / d);
}

void SolverPipelines::init(Device &dev, PipelineLayout &layout, const uint32_t *primalSpirv, size_t primalWords,
                           const uint32_t *dualSpirv, size_t dualWords)
{
    primalThreadNoForces.init(dev, layout, primalSpirv, primalWords, "csPrimalThreadNoForces",
                              "csPrimalThreadNoForces");
    primalThreadForces.init(dev, layout, primalSpirv, primalWords, "csPrimalThreadForces", "csPrimalThreadForces");
    primalWarp4NoForces.init(dev, layout, primalSpirv, primalWords, "csPrimalWarp4NoForces", "csPrimalWarp4NoForces");
    primalWarp4Forces.init(dev, layout, primalSpirv, primalWords, "csPrimalWarp4Forces", "csPrimalWarp4Forces");
    primalWideNoForces.init(dev, layout, primalSpirv, primalWords, "csPrimalWideNoForces", "csPrimalWideNoForces");
    primalWideForces.init(dev, layout, primalSpirv, primalWords, "csPrimalWideForces", "csPrimalWideForces");

    jointInit.init(dev, layout, dualSpirv, dualWords, "csJointInit", "csJointInit");
    jointDual.init(dev, layout, dualSpirv, dualWords, "csJointDual", "csJointDual");
    dual.init(dev, layout, dualSpirv, dualWords, "csDual", "csDual");
    iterateSmall.init(dev, layout, primalSpirv, primalWords, "csIterateSmall", "csIterateSmall");
    primalDispatchArgs.init(dev, layout, primalSpirv, primalWords, "csPrimalDispatchArgs", "csPrimalDispatchArgs");
    gateDispatchArgs.init(dev, layout, primalSpirv, primalWords, "csGateDispatchArgs", "csGateDispatchArgs");
    stepFlagIfGreater.init(dev, layout, primalSpirv, primalWords, "csStepFlagIfGreater", "csStepFlagIfGreater");
    stepFlagIfSet.init(dev, layout, primalSpirv, primalWords, "csStepFlagIfSet", "csStepFlagIfSet");
    stepColorBound.init(dev, layout, primalSpirv, primalWords, "csStepColorBound", "csStepColorBound");
    coarseReset.init(dev, layout, primalSpirv, primalWords, "csCoarseReset", "csCoarseReset");
    coarseAssign.init(dev, layout, primalSpirv, primalWords, "csCoarseAssign", "csCoarseAssign");
    coarseAssemble.init(dev, layout, primalSpirv, primalWords, "csCoarseAssemble", "csCoarseAssemble");
    coarseSolve.init(dev, layout, primalSpirv, primalWords, "csCoarseSolve", "csCoarseSolve");
    coarseApply.init(dev, layout, primalSpirv, primalWords, "csCoarseApply", "csCoarseApply");
}

void SolverPipelines::destroy()
{
    primalThreadNoForces.destroy();
    primalThreadForces.destroy();
    primalWarp4NoForces.destroy();
    primalWarp4Forces.destroy();
    primalWideNoForces.destroy();
    primalWideForces.destroy();
    primalDispatchArgs.destroy();
    gateDispatchArgs.destroy();
    stepFlagIfGreater.destroy();
    stepFlagIfSet.destroy();
    stepColorBound.destroy();
    coarseReset.destroy();
    coarseAssign.destroy();
    coarseAssemble.destroy();
    coarseSolve.destroy();
    coarseApply.destroy();
    jointInit.destroy();
    jointDual.destroy();
    dual.destroy();
    iterateSmall.destroy();
}

void launchPrimalColor(CommandList &cmd, HostRingBuffer &ring, SolverPipelines &pl, const PrimalArgsGpu &args,
                       bool warp4, bool withForces)
{
    VkDeviceAddress argsAddr = ring.write(&args, sizeof(args));

    if (!warp4)
    {
        // csPrimalThread*: numthreads(256,1,1), one body per thread.
        Pipeline &p = withForces ? pl.primalThreadForces : pl.primalThreadNoForces;
        uint32_t groups = ceilDiv(args.listCount, 256);
        if (groups == 0)
            return;
        cmd.dispatch(p, groups, argsAddr);
        return;
    }

    // csPrimalWarp4*: numthreads(256,1,1), LANES4=4 threads cooperate per body, so
    // 256/4 = 64 bodies per group.
    Pipeline &p = withForces ? pl.primalWarp4Forces : pl.primalWarp4NoForces;
    uint32_t groups = ceilDiv(args.listCount, 64);
    if (groups == 0)
        return;
    cmd.dispatch(p, groups, argsAddr);
}

void launchPrimalDispatchArgs(CommandList &cmd, HostRingBuffer &ring, SolverPipelines &pl, VkDeviceAddress colorStart,
                              VkDeviceAddress colorEnd, Buffer &indirect, int colorCount, bool warp4)
{
    if (colorCount <= 0)
        return;
    PrimalDispatchArgsGpu a{colorStart, colorEnd, indirect.address(), (uint32_t)colorCount, warp4 ? 64u : 256u};
    VkDeviceAddress argsAddr = ring.write(&a, sizeof(a));
    cmd.dispatch(pl.primalDispatchArgs, ceilDiv(colorCount, 128), argsAddr);
}

void launchPrimalColorIndirect(CommandList &cmd, HostRingBuffer &ring, SolverPipelines &pl, const PrimalArgsGpu &args,
                               bool warp4, bool withForces, const Buffer &indirect)
{
    VkDeviceAddress argsAddr = ring.write(&args, sizeof(args));
    Pipeline &p = warp4 ? (withForces ? pl.primalWarp4Forces : pl.primalWarp4NoForces)
                        : (withForces ? pl.primalThreadForces : pl.primalThreadNoForces);
    cmd.dispatchIndirect(p, indirect, (VkDeviceSize)args.color * 3 * sizeof(uint32_t), argsAddr);
}

void launchPrimalWideIndirect(CommandList &cmd, HostRingBuffer &ring, SolverPipelines &pl, const PrimalArgsGpu &args,
                              bool withForces, const Buffer &indirect)
{
    VkDeviceAddress argsAddr = ring.write(&args, sizeof(args));
    Pipeline &p = withForces ? pl.primalWideForces : pl.primalWideNoForces;
    cmd.dispatchIndirect(p, indirect, (VkDeviceSize)args.color * 3 * sizeof(uint32_t), argsAddr);
}

void launchIterateSmall(CommandList &cmd, HostRingBuffer &ring, SolverPipelines &pl, const PrimalArgsGpu &pa,
                        const DualArgsGpu &da, const JointDualArgsGpu &jd, int iterations, int colorBound,
                        bool withForces, int jointSlots)
{
    IterateArgsGpu a{};
    a.pa = ring.write(&pa, sizeof(pa));
    a.da = ring.write(&da, sizeof(da));
    a.jd = ring.write(&jd, sizeof(jd));
    a.iterations = iterations;
    a.colorBound = colorBound;
    a.withForces = withForces ? 1 : 0;
    a.jointSlots = jointSlots;
    cmd.dispatch(pl.iterateSmall, 1, ring.write(&a, sizeof(a)));
}

void launchStepFlagIfSet(CommandList &cmd, HostRingBuffer &ring, SolverPipelines &pl, VkDeviceAddress src,
                         VkDeviceAddress flag)
{
    StepFlagIfSetArgsGpu a{src, flag};
    VkDeviceAddress argsAddr = ring.write(&a, sizeof(a));
    cmd.dispatch(pl.stepFlagIfSet, 1, argsAddr);
}

void launchStepFlagIfGreater(CommandList &cmd, HostRingBuffer &ring, SolverPipelines &pl, VkDeviceAddress count,
                             VkDeviceAddress flag, int cap)
{
    StepFlagIfGreaterArgsGpu a{count, flag, cap, 0};
    VkDeviceAddress argsAddr = ring.write(&a, sizeof(a));
    cmd.dispatch(pl.stepFlagIfGreater, 1, argsAddr);
}

void launchStepColorBound(CommandList &cmd, HostRingBuffer &ring, SolverPipelines &pl, VkDeviceAddress colorStart,
                          VkDeviceAddress colorEnd, VkDeviceAddress flag, int bound, int maxColors)
{
    if (bound >= maxColors)
        return;
    StepColorBoundArgsGpu a{colorStart, colorEnd, flag, bound, maxColors};
    VkDeviceAddress argsAddr = ring.write(&a, sizeof(a));
    cmd.dispatch(pl.stepColorBound, ceilDiv(maxColors - bound, 128), argsAddr);
}

void launchJointInit(CommandList &cmd, HostRingBuffer &ring, SolverPipelines &pl, const JointInitArgsGpu &args,
                     int jointCount)
{
    VkDeviceAddress argsAddr = ring.write(&args, sizeof(args));
    uint32_t groups = ceilDiv(jointCount, 256);
    if (groups == 0)
        return;
    cmd.dispatch(pl.jointInit, groups, argsAddr);
}

void launchJointDual(CommandList &cmd, HostRingBuffer &ring, SolverPipelines &pl, const JointDualArgsGpu &args,
                     int jointCount)
{
    VkDeviceAddress argsAddr = ring.write(&args, sizeof(args));
    uint32_t groups = ceilDiv(jointCount, 256);
    if (groups == 0)
        return;
    cmd.dispatch(pl.jointDual, groups, argsAddr);
}

void launchDual(CommandList &cmd, HostRingBuffer &ring, SolverPipelines &pl, const DualArgsGpu &args, int slots)
{
    VkDeviceAddress argsAddr = ring.write(&args, sizeof(args));
    uint32_t groups = ceilDiv(slots, 256);
    if (groups == 0)
        return;
    cmd.dispatch(pl.dual, groups, argsAddr);
}

void launchCoarse(CommandList &cmd, HostRingBuffer &ring, Pipeline &pipeline, const CoarseArgsGpu &args, int threads,
                  int groupSize)
{
    VkDeviceAddress argsAddr = ring.write(&args, sizeof(args));
    cmd.dispatch(pipeline, ceilDiv(threads, groupSize), argsAddr);
}

} // namespace avbdvk
