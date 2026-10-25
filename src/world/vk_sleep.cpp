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

#include "vk_sleep.h"

namespace avbdvk
{

static uint32_t ceilDiv(int n, int d)
{
    if (n <= 0)
        return 0;
    return (uint32_t)((n + d - 1) / d);
}

// ---------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------

void SleepPipelines::init(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t spirvWords)
{
    sleepBlockFracture.init(dev, layout, spirv, spirvWords, "csSleepBlockFracture", "csSleepBlockFracture");
    sleepBlockDrag.init(dev, layout, spirv, spirvWords, "csSleepBlockDrag", "csSleepBlockDrag");
    sleepMark.init(dev, layout, spirv, spirvWords, "csSleepMark", "csSleepMark");
    sleepCommit.init(dev, layout, spirv, spirvWords, "csSleepCommit", "csSleepCommit");
    activeFlags.init(dev, layout, spirv, spirvWords, "csActiveFlags", "csActiveFlags");
}

void SleepPipelines::destroy()
{
    sleepBlockFracture.destroy();
    sleepBlockDrag.destroy();
    sleepMark.destroy();
    sleepCommit.destroy();
    activeFlags.destroy();
}

void WakePipelines::init(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t spirvWords)
{
    wakeRequest.init(dev, layout, spirv, spirvWords, "csWakeRequest", "csWakeRequest");
    touchClear.init(dev, layout, spirv, spirvWords, "csTouchClear", "csTouchClear");
    touchAccum.init(dev, layout, spirv, spirvWords, "csTouchAccum", "csTouchAccum");
    wakeLostSupport.init(dev, layout, spirv, spirvWords, "csWakeLostSupport", "csWakeLostSupport");
    wakeApply.init(dev, layout, spirv, spirvWords, "csWakeApply", "csWakeApply");
    touchAccumJoints.init(dev, layout, spirv, spirvWords, "csTouchAccumJoints", "csTouchAccumJoints");
    wakeRequestJoints.init(dev, layout, spirv, spirvWords, "csWakeRequestJoints", "csWakeRequestJoints");
    wakeIslandSeed.init(dev, layout, spirv, spirvWords, "csWakeIslandSeed", "csWakeIslandSeed");
    wakeAll.init(dev, layout, spirv, spirvWords, "csWakeAll", "csWakeAll");
}

void WakePipelines::destroy()
{
    wakeRequest.destroy();
    touchClear.destroy();
    touchAccum.destroy();
    wakeLostSupport.destroy();
    wakeApply.destroy();
    touchAccumJoints.destroy();
    wakeRequestJoints.destroy();
    wakeIslandSeed.destroy();
    wakeAll.destroy();
}

void MutatePipelines::init(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t spirvWords)
{
    mutateApply.init(dev, layout, spirv, spirvWords, "csMutateApply", "csMutateApply");
    wakeSeedRegion.init(dev, layout, spirv, spirvWords, "csWakeSeedRegion", "csWakeSeedRegion");
    wakeFloodContacts.init(dev, layout, spirv, spirvWords, "csWakeFloodContacts", "csWakeFloodContacts");
    wakeFloodJoints.init(dev, layout, spirv, spirvWords, "csWakeFloodJoints", "csWakeFloodJoints");
    wakeSeedMerge.init(dev, layout, spirv, spirvWords, "csWakeSeedMerge", "csWakeSeedMerge");
    bodyGather.init(dev, layout, spirv, spirvWords, "csBodyGather", "csBodyGather");
    jointWrite.init(dev, layout, spirv, spirvWords, "csJointWrite", "csJointWrite");
}

void MutatePipelines::destroy()
{
    mutateApply.destroy();
    wakeSeedRegion.destroy();
    wakeFloodContacts.destroy();
    wakeFloodJoints.destroy();
    wakeSeedMerge.destroy();
    bodyGather.destroy();
    jointWrite.destroy();
}

// ---------------------------------------------------------------------------
// sleep.slang launchers
// ---------------------------------------------------------------------------

void launchActiveFlags(CommandList &cmd, HostRingBuffer &ring, SleepPipelines &pl,
                        const ActiveFlagsArgsGpu &args, int dynCount)
{
    uint32_t groups = ceilDiv(dynCount, 256);
    if (groups == 0)
        return;
    VkDeviceAddress a = ring.write(&args, sizeof(args));
    cmd.dispatch(pl.activeFlags, groups, a);
}

void launchSleepCommit(CommandList &cmd, HostRingBuffer &ring, SleepPipelines &pl, SleepCommitCall &call)
{
    const bool anyBlock = call.dragCount > 0 || call.jointCount > 0;

    if (anyBlock)
    {
        // Same trick as VkConstraints::build's `counter.zero()`: a synchronous,
        // separate one-shot submit+wait that only has to land before `cmd` is later
        // submitted by the caller, which it does by construction (zero() blocks before
        // this function returns, and the caller submits `cmd` strictly after this call).
        if (call.block)
            call.block->zero();

        if (call.jointCount > 0 && call.fracture.ratio < 1.0f)
        {
            VkDeviceAddress a = ring.write(&call.fracture, sizeof(call.fracture));
            cmd.dispatch(pl.sleepBlockFracture, ceilDiv(call.jointCount, 256), a);
        }
        if (call.dragCount > 0)
        {
            VkDeviceAddress a = ring.write(&call.drag, sizeof(call.drag));
            cmd.dispatch(pl.sleepBlockDrag, ceilDiv(call.dragCount, 256), a);
        }
        if (call.jointCount > 0 || call.dragCount > 0)
            cmd.barrier();
    }

    call.mark.hasBlock = anyBlock ? 1u : 0u;
    {
        VkDeviceAddress a = ring.write(&call.mark, sizeof(call.mark));
        cmd.dispatch(pl.sleepMark, ceilDiv(call.dynCount, 256), a);
    }

    if (call.solveCount > 0)
    {
        cmd.barrier();
        VkDeviceAddress a = ring.write(&call.commit, sizeof(call.commit));
        cmd.dispatch(pl.sleepCommit, ceilDiv(call.solveCount, 256), a);
    }
}

// ---------------------------------------------------------------------------
// wake.slang launchers
// ---------------------------------------------------------------------------

void launchWakeDetect(CommandList &cmd, HostRingBuffer &ring, WakePipelines &pl, WakeDetectCall &call)
{
    {
        VkDeviceAddress a = ring.write(&call.clear, sizeof(call.clear));
        cmd.dispatch(pl.touchClear, ceilDiv(call.dynCount, 256), a);
    }
    if (call.slots > 0)
    {
        cmd.barrier();
        VkDeviceAddress a = ring.write(&call.accum, sizeof(call.accum));
        cmd.dispatch(pl.touchAccum, ceilDiv(call.slots, 256), a);
    }
    if (call.edgeCount > 0)
    {
        cmd.barrier();
        VkDeviceAddress a = ring.write(&call.accumJ, sizeof(call.accumJ));
        cmd.dispatch(pl.touchAccumJoints, ceilDiv(call.edgeCount, 256), a);
    }

    cmd.barrier();
    if (call.slots > 0)
    {
        VkDeviceAddress a = ring.write(&call.request, sizeof(call.request));
        cmd.dispatch(pl.wakeRequest, ceilDiv(call.slots, 256), a);
    }
    if (call.edgeCount > 0)
    {
        VkDeviceAddress a = ring.write(&call.requestJ, sizeof(call.requestJ));
        cmd.dispatch(pl.wakeRequestJoints, ceilDiv(call.edgeCount, 256), a);
    }

    cmd.barrier();
    {
        VkDeviceAddress a = ring.write(&call.lostSupport, sizeof(call.lostSupport));
        cmd.dispatch(pl.wakeLostSupport, ceilDiv(call.dynCount, 256), a);
    }
}

void launchWakeIslandSeed(CommandList &cmd, HostRingBuffer &ring, WakePipelines &pl,
                           const WakeIslandSeedArgsGpu &args, int dynCount)
{
    uint32_t groups = ceilDiv(dynCount, 256);
    if (groups == 0)
        return;
    VkDeviceAddress a = ring.write(&args, sizeof(args));
    cmd.dispatch(pl.wakeIslandSeed, groups, a);
}

void launchWakeApply(CommandList &cmd, HostRingBuffer &ring, WakePipelines &pl,
                      const WakeApplyArgsGpu &args, int dynCount)
{
    uint32_t groups = ceilDiv(dynCount, 256);
    if (groups == 0)
        return;
    VkDeviceAddress a = ring.write(&args, sizeof(args));
    cmd.dispatch(pl.wakeApply, groups, a);
}

void launchWakeAll(CommandList &cmd, HostRingBuffer &ring, WakePipelines &pl,
                    const WakeAllArgsGpu &args, int dynCount)
{
    uint32_t groups = ceilDiv(dynCount, 256);
    if (groups == 0)
        return;
    VkDeviceAddress a = ring.write(&args, sizeof(args));
    cmd.dispatch(pl.wakeAll, groups, a);
}

// ---------------------------------------------------------------------------
// mutate.slang launchers
// ---------------------------------------------------------------------------

void launchMutateApply(CommandList &cmd, HostRingBuffer &ring, MutatePipelines &pl,
                        const MutateApplyArgsGpu &args, int count)
{
    uint32_t groups = ceilDiv(count, 256);
    if (groups == 0)
        return;
    VkDeviceAddress a = ring.write(&args, sizeof(args));
    cmd.dispatch(pl.mutateApply, groups, a);
}

void launchJointWrite(CommandList &cmd, HostRingBuffer &ring, MutatePipelines &pl,
                      const JointWriteArgsGpu &args, int count)
{
    uint32_t groups = ceilDiv(count, 64);
    if (groups == 0)
        return;
    VkDeviceAddress a = ring.write(&args, sizeof(args));
    cmd.dispatch(pl.jointWrite, groups, a);
}

void launchWakeSeedRegion(CommandList &cmd, HostRingBuffer &ring, MutatePipelines &pl,
                           const WakeSeedRegionArgsGpu &args, int dynCount)
{
    uint32_t groups = ceilDiv(dynCount, 256);
    if (groups == 0)
        return;
    VkDeviceAddress a = ring.write(&args, sizeof(args));
    cmd.dispatch(pl.wakeSeedRegion, groups, a);
}

void launchWakeFloodJoints(CommandList &cmd, HostRingBuffer &ring, MutatePipelines &pl,
                            const WakeFloodJointsArgsGpu &args, int edgeCount)
{
    uint32_t groups = ceilDiv(edgeCount, 256);
    if (groups == 0)
        return;
    VkDeviceAddress a = ring.write(&args, sizeof(args));
    cmd.dispatch(pl.wakeFloodJoints, groups, a);
}

void launchWakeSeedMerge(CommandList &cmd, HostRingBuffer &ring, MutatePipelines &pl,
                          const WakeSeedMergeArgsGpu &args, int dynCount)
{
    uint32_t groups = ceilDiv(dynCount, 256);
    if (groups == 0)
        return;
    VkDeviceAddress a = ring.write(&args, sizeof(args));
    cmd.dispatch(pl.wakeSeedMerge, groups, a);
}

void recordWakeFloodRounds(CommandList &cmd, HostRingBuffer &ring, MutatePipelines &pl,
                           WakeFloodContactsArgsGpu contactsArgs, WakeFloodJointsArgsGpu jointsArgs,
                           int slots, int edgeCount, VkDeviceAddress changedSlots, int firstRound, int count)
{
    for (int r = 0; r < count; r++)
    {
        const VkDeviceAddress changed = changedSlots + (VkDeviceAddress)(firstRound + r) * sizeof(int32_t);
        if (slots > 0)
        {
            contactsArgs.changed = changed;
            VkDeviceAddress a = ring.write(&contactsArgs, sizeof(contactsArgs));
            cmd.dispatch(pl.wakeFloodContacts, ceilDiv(slots, 256), a);
        }
        if (edgeCount > 0)
        {
            jointsArgs.changed = changed;
            VkDeviceAddress a = ring.write(&jointsArgs, sizeof(jointsArgs));
            cmd.dispatch(pl.wakeFloodJoints, ceilDiv(edgeCount, 256), a);
        }
        cmd.barrier();
    }
}

void launchBodyGather(CommandList &cmd, HostRingBuffer &ring, MutatePipelines &pl,
                       const BodyGatherArgsGpu &args, int count)
{
    uint32_t groups = ceilDiv(count, 256);
    if (groups == 0)
        return;
    VkDeviceAddress a = ring.write(&args, sizeof(args));
    cmd.dispatch(pl.bodyGather, groups, a);
}

// launchWakeFlood: owns its own CommandList/HostRingBuffer/submit loop, exactly
// mirroring solver_kernels.cuh's launchWakeFlood (kFloodMaxRounds=64,
// kFloodSyncEvery=4, batch growth is irrelevant here since -- unlike VkConstraints's
// JP coloring -- this loop's unit of work is fixed at "one round" per readback check,
// matching the reference source exactly rather than growing a batch).
void launchWakeFlood(Device &dev, MutatePipelines &mutPl, WakePipelines &wakePl,
                      const WakeFloodContactsArgsGpu &contactsArgs,
                      const WakeFloodJointsArgsGpu &jointsArgs,
                      const WakeSeedMergeArgsGpu &mergeArgs,
                      int slots, int edgeCount, int dynCount,
                      Buffer &floodChanged)
{
    (void)wakePl; // reserved: flood kernels live in MutatePipelines, not WakePipelines

    const int kFloodMaxRounds = 64;
    const int kFloodSyncEvery = 4;

    CommandList cmd;
    cmd.init(dev, 4);
    HostRingBuffer ring;
    ring.init(dev, 1 << 16);

    for (int round = 0; round < kFloodMaxRounds; round++)
    {
        ring.reset();
        cmd.begin();
        if (slots > 0)
        {
            VkDeviceAddress a = ring.write(&contactsArgs, sizeof(contactsArgs));
            cmd.dispatch(mutPl.wakeFloodContacts, ceilDiv(slots, 256), a);
        }
        if (edgeCount > 0)
        {
            VkDeviceAddress a = ring.write(&jointsArgs, sizeof(jointsArgs));
            cmd.dispatch(mutPl.wakeFloodJoints, ceilDiv(edgeCount, 256), a);
        }
        cmd.submit();
        cmd.wait();

        if ((round + 1) % kFloodSyncEvery)
            continue;
        uint32_t changed = 0;
        floodChanged.readback(&changed, sizeof(uint32_t));
        if (!changed)
            break;
        floodChanged.zero(sizeof(int32_t));
    }

    ring.reset();
    cmd.begin();
    VkDeviceAddress a = ring.write(&mergeArgs, sizeof(mergeArgs));
    cmd.dispatch(mutPl.wakeSeedMerge, ceilDiv(dynCount, 256), a);
    cmd.submit();
    cmd.wait();

    cmd.destroy();
    ring.destroy();
}

} // namespace avbdvk
