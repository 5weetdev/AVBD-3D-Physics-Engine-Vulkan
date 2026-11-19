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
#include "vk_util.h"

namespace avbdvk
{

// ---------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------

// Every entry point of this module, named once; init and destroy walk the same table.
static std::array<PipelineEntry, 5> entries(SleepPipelines &p)
{
    return {{
        {&p.sleepBlockFracture, "csSleepBlockFracture"},
        {&p.sleepBlockDrag, "csSleepBlockDrag"},
        {&p.sleepMark, "csSleepMark"},
        {&p.sleepCommit, "csSleepCommit"},
        {&p.activeFlags, "csActiveFlags"},
    }};
}

void SleepPipelines::init(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t spirvWords)
{
    initPipelines(dev, layout, spirv, spirvWords, entries(*this));
}

void SleepPipelines::destroy()
{
    destroyPipelines(entries(*this));
}

// Every entry point of this module, named once; init and destroy walk the same table.
static std::array<PipelineEntry, 9> entries(WakePipelines &p)
{
    return {{
        {&p.wakeRequest, "csWakeRequest"},
        {&p.touchClear, "csTouchClear"},
        {&p.touchAccum, "csTouchAccum"},
        {&p.wakeLostSupport, "csWakeLostSupport"},
        {&p.wakeApply, "csWakeApply"},
        {&p.touchAccumJoints, "csTouchAccumJoints"},
        {&p.wakeRequestJoints, "csWakeRequestJoints"},
        {&p.wakeIslandSeed, "csWakeIslandSeed"},
        {&p.wakeAll, "csWakeAll"},
    }};
}

void WakePipelines::init(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t spirvWords)
{
    initPipelines(dev, layout, spirv, spirvWords, entries(*this));
}

void WakePipelines::destroy()
{
    destroyPipelines(entries(*this));
}

// Every entry point of this module, named once; init and destroy walk the same table.
static std::array<PipelineEntry, 7> entries(MutatePipelines &p)
{
    return {{
        {&p.mutateApply, "csMutateApply"},
        {&p.wakeSeedRegion, "csWakeSeedRegion"},
        {&p.wakeFloodContacts, "csWakeFloodContacts"},
        {&p.wakeFloodJoints, "csWakeFloodJoints"},
        {&p.wakeSeedMerge, "csWakeSeedMerge"},
        {&p.bodyGather, "csBodyGather"},
        {&p.jointWrite, "csJointWrite"},
    }};
}

void MutatePipelines::init(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t spirvWords)
{
    initPipelines(dev, layout, spirv, spirvWords, entries(*this));
}

void MutatePipelines::destroy()
{
    destroyPipelines(entries(*this));
}

// ---------------------------------------------------------------------------
// sleep.slang launchers
// ---------------------------------------------------------------------------

void launchActiveFlags(CommandList &cmd, HostRingBuffer &ring, SleepPipelines &pl, const ActiveFlagsArgsGpu &args,
                       int dynCount)
{
    uint32_t groups = ceilDiv(dynCount, 256);
    if (groups == 0)
        return;
    cmd.dispatchArgs(pl.activeFlags, groups, ring, args);
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
        cmd.dispatchArgs(pl.sleepCommit, ceilDiv(call.solveCount, 256), ring, call.commit);
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
        cmd.dispatchArgs(pl.wakeLostSupport, ceilDiv(call.dynCount, 256), ring, call.lostSupport);
    }
}

void launchWakeIslandSeed(CommandList &cmd, HostRingBuffer &ring, WakePipelines &pl, const WakeIslandSeedArgsGpu &args,
                          int dynCount)
{
    uint32_t groups = ceilDiv(dynCount, 256);
    if (groups == 0)
        return;
    cmd.dispatchArgs(pl.wakeIslandSeed, groups, ring, args);
}

void launchWakeApply(CommandList &cmd, HostRingBuffer &ring, WakePipelines &pl, const WakeApplyArgsGpu &args,
                     int dynCount)
{
    uint32_t groups = ceilDiv(dynCount, 256);
    if (groups == 0)
        return;
    cmd.dispatchArgs(pl.wakeApply, groups, ring, args);
}

void launchWakeAll(CommandList &cmd, HostRingBuffer &ring, WakePipelines &pl, const WakeAllArgsGpu &args, int dynCount)
{
    uint32_t groups = ceilDiv(dynCount, 256);
    if (groups == 0)
        return;
    cmd.dispatchArgs(pl.wakeAll, groups, ring, args);
}

// ---------------------------------------------------------------------------
// mutate.slang launchers
// ---------------------------------------------------------------------------

void launchMutateApply(CommandList &cmd, HostRingBuffer &ring, MutatePipelines &pl, const MutateApplyArgsGpu &args,
                       int count)
{
    uint32_t groups = ceilDiv(count, 256);
    if (groups == 0)
        return;
    cmd.dispatchArgs(pl.mutateApply, groups, ring, args);
}

void launchJointWrite(CommandList &cmd, HostRingBuffer &ring, MutatePipelines &pl, const JointWriteArgsGpu &args,
                      int count)
{
    uint32_t groups = ceilDiv(count, 64);
    if (groups == 0)
        return;
    cmd.dispatchArgs(pl.jointWrite, groups, ring, args);
}

void launchWakeSeedRegion(CommandList &cmd, HostRingBuffer &ring, MutatePipelines &pl,
                          const WakeSeedRegionArgsGpu &args, int dynCount)
{
    uint32_t groups = ceilDiv(dynCount, 256);
    if (groups == 0)
        return;
    cmd.dispatchArgs(pl.wakeSeedRegion, groups, ring, args);
}

void launchWakeFloodJoints(CommandList &cmd, HostRingBuffer &ring, MutatePipelines &pl,
                           const WakeFloodJointsArgsGpu &args, int edgeCount)
{
    uint32_t groups = ceilDiv(edgeCount, 256);
    if (groups == 0)
        return;
    cmd.dispatchArgs(pl.wakeFloodJoints, groups, ring, args);
}

void launchWakeSeedMerge(CommandList &cmd, HostRingBuffer &ring, MutatePipelines &pl, const WakeSeedMergeArgsGpu &args,
                         int dynCount)
{
    uint32_t groups = ceilDiv(dynCount, 256);
    if (groups == 0)
        return;
    cmd.dispatchArgs(pl.wakeSeedMerge, groups, ring, args);
}

void recordWakeFloodRounds(CommandList &cmd, HostRingBuffer &ring, MutatePipelines &pl,
                           WakeFloodContactsArgsGpu contactsArgs, WakeFloodJointsArgsGpu jointsArgs, int slots,
                           int edgeCount, VkDeviceAddress changedSlots, int firstRound, int count)
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
            cmd.dispatchArgs(pl.wakeFloodJoints, ceilDiv(edgeCount, 256), ring, jointsArgs);
        }
        cmd.barrier();
    }
}

void launchBodyGather(CommandList &cmd, HostRingBuffer &ring, MutatePipelines &pl, const BodyGatherArgsGpu &args,
                      int count)
{
    uint32_t groups = ceilDiv(count, 256);
    if (groups == 0)
        return;
    cmd.dispatchArgs(pl.bodyGather, groups, ring, args);
}

// launchWakeFlood: owns its own CommandList/HostRingBuffer/submit loop (kFloodMaxRounds=64,
// kFloodSyncEvery=4). Unlike VkConstraints's JP coloring, its unit of work is fixed at "one round"
// per readback check, so it does not grow a batch.
void launchWakeFlood(Device &dev, MutatePipelines &mutPl, WakePipelines &wakePl,
                     const WakeFloodContactsArgsGpu &contactsArgs, const WakeFloodJointsArgsGpu &jointsArgs,
                     const WakeSeedMergeArgsGpu &mergeArgs, int slots, int edgeCount, int dynCount,
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
    cmd.dispatchArgs(mutPl.wakeSeedMerge, ceilDiv(dynCount, 256), ring, mergeArgs);
    cmd.submit();
    cmd.wait();

    cmd.destroy();
    ring.destroy();
}

} // namespace avbdvk
