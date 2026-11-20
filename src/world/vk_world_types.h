#pragma once

// Result, timing and view records that VkWorld hands out (aliased inside VkWorld, so callers write
// VkWorld::StepResult as before).

#include "vk_args.h"
#include "vk_rerun.h"

#include <cstdint>

namespace avbdvk
{

// Advances the world by params.dt. Phase order: integrate -> broadphase (+overflow retry) -> narrowphase (+overflow
// retry) -> sleep/wake first half (if params.sleepEnabled) -> constraints build
// -> the iteration loop (one command buffer) -> velocity -> explosion guard ->
// kill-fallen -> sleep commit second half.
// Returns the same StepResult lastStepResult() keeps; callers may ignore it.
// What the last step() did. ok == false only when a capacity ceiling forced the step
// to commit clamped results (truncated): failPhase/failResource name what ran out and
// required/capacity say by how much. Reruns are counted per cause (a pass that hit
// several causes counts each). submits is the vkQueueSubmit delta over the step.
struct StepResult : RerunCounters
{
    bool ok = true;
    bool truncated = false;
    int floodReruns = 0;    // explicit wake flood bound too small
    int islandReruns = 0;   // joint-island flood bound too small
    const char *failPhase = nullptr;
    const char *failResource = nullptr;
    // failPhase == "device": a Vulkan call failed (or a memory budget was hit). vkError is
    // the first failing VkResult, deviceLost is vkError == VK_ERROR_DEVICE_LOST. The world
    // is dead from then on (deviceError()): every later step returns this same failure
    // without touching Vulkan. Destroy it and build a new one.
    int vkError = 0;
    bool deviceLost = false;
    int64_t required = 0;
    int64_t capacity = 0;
    int totalReruns() const { return capacityReruns + floodReruns + islandReruns + jpReruns + colorReruns; }
};

// Test-only overrides, all default 0 = off. The initial* / *Bound values replace the
// initial per-step quantity (normal growth/doubling applies after); hard* lower the
// capacity ceiling that grow() clamps to. Set before build() for the initial caps.
struct DebugOverrides
{
    int initialPairCapPerBody = 0; // pair capacity = this * bodies (at build)
    int initialContactCap = 0;     // contact capacity (at build)
    int hardPairCap = 0;
    int hardContactCap = 0;
    int floodBound = 0;
    int islandBound = 0;
    int jpBound = 0;
    int colorBound = 0;
    // Force the legacy broadphase rule: grid cell = largest body, no oversize path.
    bool legacyBroadphase = false;
};

// Timings of the last step(). step() is one command-buffer submission, so the split
// is GPU timestamps (ms; -1 if unavailable) except where noted:
//   stepMs, solveMs  host wall clock of the whole step / of the final solve pass
//   buildMs          HOST time spent recording phase 5 on the final pass (not GPU)
//   broadMs/narrowMs broadphase / narrowphase, from the front pass
//   frontMs          phase 1 (mutate, integrate, wake front) up to broadphase. Recorded once,
//                    on the first pass, so it is measured there and carried over reruns
//   gpuBuildMs       phases 4-5 (wake flood, constraint graph, colouring) through the
//                    iteration start, narrowphase readouts included
//   iterMs           the whole iteration loop; primalMs/dualMs summed over iterations
//   tailMs           after the last iteration to the end of the buffer (phases 7-8)
//   gpuMs            first to last timestamp of the committed pass. If reruns were needed the
//                    committed pass has no front, so gpuMs then excludes phases 1-3
struct StepTiming
{
    float stepMs = -1, broadMs = -1, narrowMs = -1, solveMs = -1;
    float buildMs = -1, iterMs = -1, primalMs = -1, dualMs = -1;
    float frontMs = -1, gpuBuildMs = -1, tailMs = -1, gpuMs = -1;
    // Host wall time inside vkQueueSubmit and blocked in the fence wait, summed over
    // passes. stepMs - hostWaitMs is the host CPU time the step actually costs.
    float hostSubmitMs = -1, hostWaitMs = -1;
    // Host time before the pass loop (mutation staging, lists), and recording the
    // passes' command buffers (summed). The rest of stepMs - hostWaitMs is post-wait.
    float hostPreMs = -1, hostRecordMs = -1;
};

// Dispatches and compute barriers recorded into the last step's committed pass (a rerun pass
// replaces the count; fills and copies are not included), and whether that pass ran the fused
// small-world iteration loop. fusedIterPasses() counts every pass that did, since init.
struct OpStats
{
    uint32_t dispatches = 0, barriers = 0, transfers = 0;
    bool fusedIter = false;
};

// --- Render output (docs/VK_DEMO_INTERFACE.md point 2) ---------------------------
//
// Bundles the k_write_instances arguments the reference side spreads across
// writeInstances()'s parameter list plus setOverlay()'s
// state, since VkWorld has no persistent "current overlay" of its own to fold them
// into implicitly.
struct RenderInstancesParams
{
    const float *frustum = nullptr; // 6 planes (nx,ny,nz,d), or null for no culling
    OverlayViewGpu overlay;         // .mode = OVERLAY_NONE (0) for plain colouring
    bool segmentByShape = false;    // true: one region per render group (0..2 = shapes)
};

struct SnapshotView
{
    const vec3 *pos = nullptr;
    const quat *rot = nullptr;
    const vec3 *velLin = nullptr;
    const vec3 *velAng = nullptr;
    const unsigned char *awake = nullptr;
    const unsigned char *prevAwake = nullptr; // awake before the step's mutations
    int count = 0;                            // bodies the step had
    uint64_t serial = 0;
};

// The joint multipliers of the same snapshot (authored joints, in order): lambdaLin and
// lambdaAng of the step's end, copied in VRAM inside the step only while the world has
// joints (a world without joints pays nothing). They reach host memory by their own small
// DMA, started only by this call and only when the newest snapshot is not already on the
// host. Like snapshotView() it never waits for a step in flight.
struct JointSnapshotView
{
    const vec3 *lambdaLin = nullptr;
    const vec3 *lambdaAng = nullptr;
    int count = 0; // joints the step had; a later joint index has no data yet
    uint64_t serial = 0;
};

struct ContactReportView
{
    const ContactReportRecordGpu *records = nullptr; // valid until the next stepBegin
    int count = 0;                                   // records held by the last committed step (<= capacity)
    bool truncated = false; // the step reported more than the buffer held; the excess is lost
};

} // namespace avbdvk
