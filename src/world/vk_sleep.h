#pragma once

// Vulkan host orchestration for src/shaders/sleep.slang, wake.slang and mutate.slang
// -- sleeping, the implicit velocity-gated wake path, and the explicit ungated
// mutation-flood path. Mirrors the launch*() seam convention
// (launchActiveFlags, launchSleepCommit, launchWakeDetect, launchWakeIslandSeed,
// launchWakeApply, launchMutateApply, launchWakeFloodJoints, launchWakeSeedMerge,
// launchWakeFlood) one for one, plus launchWakeAll and launchBodyGather for the two
// Methods (wakeAll(), getBodyData()) that call kernels directly rather than
// through the launch seam. See the sleep stage, the wake stage,
// the mutation path for the reference behavior, and
// tools/test_vk_sleep_wake.cpp for the reference-vs-Vulkan parity gate.
//
// Same convention as vk_solver.h: these launchers take an externally-owned
// CommandList/HostRingBuffer (recorded into by whatever driver assembles the full step,
// e.g. vk_world.cpp) plus a small args struct, rather than owning their own -- unlike
// VkConstraints/VkBroadphase/VkNarrowphase, which do several submit+wait round trips of
// their own and so own a CommandList. Every kernel here is one dispatch with no
// intervening readback except launchWakeFlood's convergence loop, which needs its own
// submit+wait per batch like any readback-in-the-loop -- so THAT one
// function takes a Device reference to issue its own submits.

#include "vk_args.h"
#include "vk_buffer.h"
#include "vk_cmd.h"
#include "vk_device.h"
#include "vk_pipeline.h"

#include <cstdint>

namespace avbdvk
{

struct SleepPipelines
{
    Pipeline sleepBlockFracture, sleepBlockDrag, sleepMark, sleepCommit, activeFlags;

    void init(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t spirvWords);
    void destroy();
};

struct WakePipelines
{
    Pipeline wakeRequest, touchClear, touchAccum, wakeLostSupport, wakeApply;
    Pipeline touchAccumJoints, wakeRequestJoints, wakeIslandSeed, wakeAll;

    void init(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t spirvWords);
    void destroy();
};

struct MutatePipelines
{
    Pipeline mutateApply, wakeSeedRegion, wakeFloodContacts, wakeFloodJoints, wakeSeedMerge, bodyGather, jointWrite;

    void init(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t spirvWords);
    void destroy();
};

// ---------------------------------------------------------------------------
// sleep.slang launchers
// ---------------------------------------------------------------------------

void launchActiveFlags(CommandList &cmd, HostRingBuffer &ring, SleepPipelines &pl, const ActiveFlagsArgsGpu &args,
                       int dynCount);

// Groups the "sleep_commit" phase exactly like the launch seam's launchSleepCommit:
// the optional fracture/drag block guards, the per-body rest mark, and the island
// commit -- same anyBlock/solveCount>0 guards, same dispatch order. `args.sm`/`args.sc`
// carry every field csSleepMark/csSleepCommit need except the ones this function fills
// in per-call (dynCount, block pointer/hasBlock, solveCount); the caller pre-fills
// everything else (buffers, thresholds).
struct SleepCommitCall
{
    SleepBlockFractureArgsGpu fracture; // .ratio/.block/.b/.j pre-filled
    SleepBlockDragArgsGpu drag;         // .dragIdx/.dragCount/.block pre-filled
    SleepMarkArgsGpu mark;              // .block/.hasBlock filled by this function
    SleepCommitArgsGpu commit;
    int jointCount = 0;
    int dragCount = 0;
    int dynCount = 0;
    int solveCount = 0;
    // Zeroed by this function (its own synchronous Buffer::zero(), same trick
    // VkConstraints::build uses for `counter` -- see vk_sleep.cpp) when anyBlock is
    // true. Left null when jointCount==0 && dragCount==0 (nothing ever sets a bit, so
    // the memset -- and the block pointer -- would be pure overhead).
    Buffer *block = nullptr;
};

void launchSleepCommit(CommandList &cmd, HostRingBuffer &ring, SleepPipelines &pl, SleepCommitCall &call);

// ---------------------------------------------------------------------------
// wake.slang launchers
// ---------------------------------------------------------------------------

// Groups the six launches the launch seam's launchWakeDetect issues back-to-back for
// the implicit wake pass (k_touch_clear through k_wake_lost_support), same slots>0/
// edgeCount>0 early-outs.
struct WakeDetectCall
{
    TouchClearArgsGpu clear;
    TouchAccumArgsGpu accum;        // .slots pre-filled
    TouchAccumJointsArgsGpu accumJ; // .j.count/.sp.count pre-filled
    WakeRequestArgsGpu request;     // .slots/.wakeLinVel/.wakeAngVel pre-filled
    WakeRequestJointsArgsGpu requestJ;
    WakeLostSupportArgsGpu lostSupport;
    int dynCount = 0;
    int slots = 0;
    int edgeCount = 0; // joints.count + springs.count
};

void launchWakeDetect(CommandList &cmd, HostRingBuffer &ring, WakePipelines &pl, WakeDetectCall &call);

void launchWakeIslandSeed(CommandList &cmd, HostRingBuffer &ring, WakePipelines &pl, const WakeIslandSeedArgsGpu &args,
                          int dynCount);

void launchWakeApply(CommandList &cmd, HostRingBuffer &ring, WakePipelines &pl, const WakeApplyArgsGpu &args,
                     int dynCount);

void launchWakeAll(CommandList &cmd, HostRingBuffer &ring, WakePipelines &pl, const WakeAllArgsGpu &args, int dynCount);

// ---------------------------------------------------------------------------
// mutate.slang launchers
// ---------------------------------------------------------------------------

void launchMutateApply(CommandList &cmd, HostRingBuffer &ring, MutatePipelines &pl, const MutateApplyArgsGpu &args,
                       int count);
void launchJointWrite(CommandList &cmd, HostRingBuffer &ring, MutatePipelines &pl, const JointWriteArgsGpu &args,
                      int count);

void launchWakeSeedRegion(CommandList &cmd, HostRingBuffer &ring, MutatePipelines &pl,
                          const WakeSeedRegionArgsGpu &args, int dynCount);

void launchWakeFloodJoints(CommandList &cmd, HostRingBuffer &ring, MutatePipelines &pl,
                           const WakeFloodJointsArgsGpu &args, int edgeCount);

void launchWakeSeedMerge(CommandList &cmd, HostRingBuffer &ring, MutatePipelines &pl, const WakeSeedMergeArgsGpu &args,
                         int dynCount);

// `count` wake-flood rounds recorded back to back, no readback between them.
// Round r dispatches the contact flood (if slots > 0) and the joint flood (if
// edgeCount > 0), both flagging into their own int at changedSlots + 4*(firstRound+r),
// then a barrier. The flood is monotone, so a round after the fixed point changes
// nothing: over-recording is exact, and the first zero slot is the convergence round.
// The args' own `changed` fields are overwritten per round.
void recordWakeFloodRounds(CommandList &cmd, HostRingBuffer &ring, MutatePipelines &pl,
                           WakeFloodContactsArgsGpu contactsArgs, WakeFloodJointsArgsGpu jointsArgs, int slots,
                           int edgeCount, VkDeviceAddress changedSlots, int firstRound, int count);

void launchBodyGather(CommandList &cmd, HostRingBuffer &ring, MutatePipelines &pl, const BodyGatherArgsGpu &args,
                      int count);

// The whole explicit-mutation flood, exactly mirroring the launch seam's
// launchWakeFlood: contacts and joints both flood to a fixed point (round cap 64,
// readback every 4th round, early exit on convergence), then merge into the ordinary
// wake requests. Unlike every other launcher here, this one owns its submit/wait loop
// (the "dispatch until the remaining-counter reads zero" shape is inherently a host
// readback loop, same reasoning as VkConstraints's Jones-Plassmann batches) so it takes
// a Device plus its own CommandList/HostRingBuffer rather than an externally-recorded
// one. `floodChanged` must be a 4-byte device buffer the caller has already zeroed.
void launchWakeFlood(Device &dev, MutatePipelines &mutPl, WakePipelines &wakePl,
                     const WakeFloodContactsArgsGpu &contactsArgs, const WakeFloodJointsArgsGpu &jointsArgs,
                     const WakeSeedMergeArgsGpu &mergeArgs, int slots, int edgeCount, int dynCount,
                     Buffer &floodChanged);

} // namespace avbdvk
