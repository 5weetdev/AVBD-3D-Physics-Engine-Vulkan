#pragma once

// Narrowphase argument structs (narrowphase*.slang). Mirrors of the Slang structs: field order and padding are byte-for-byte, and
// avbd_test_vk_layout checks them against the compiler's offsets.

#include "vk_buffer.h"

#include <cstdint>

namespace avbdvk
{

// ===========================================================================
// narrowphase.slang
// ===========================================================================

struct NpBodySoAGpu
{
    VkDeviceAddress pos = 0, rot = 0;                                        // per body
    VkDeviceAddress shapeBody = 0, localPos = 0, localRot = 0;               // per shape
    VkDeviceAddress halfExtents = 0, shapeType = 0, friction = 0, flags = 0; // per shape
    VkDeviceAddress mass = 0, moment = 0;                                    // per body
    VkDeviceAddress velLin = 0, velAng = 0;
    VkDeviceAddress props = 0;                         // may be 0 (no non-default BodyProps)
    VkDeviceAddress category = 0, mask = 0, group = 0; // per shape, may be 0 (no filtering)
    uint32_t count = 0, _pad0 = 0;
};
static_assert(sizeof(NpBodySoAGpu) % 8 == 0, "layout");

struct NpManifoldSoAGpu
{
    VkDeviceAddress key = 0;
    VkDeviceAddress bodyA = 0, bodyB = 0;
    VkDeviceAddress basis = 0;
    VkDeviceAddress friction = 0;
    VkDeviceAddress contactOffset = 0, contactCount = 0;
    VkDeviceAddress feature = 0;
    VkDeviceAddress rA = 0, rB = 0;
    VkDeviceAddress C0 = 0;
    VkDeviceAddress penalty = 0, lambda = 0;
    VkDeviceAddress stick = 0;
    VkDeviceAddress shapeA = 0, shapeB = 0;
};
static_assert(sizeof(NpManifoldSoAGpu) % 8 == 0, "layout");

struct SolverParamsNpGpu
{
    float dt = 0.0f;
    float alpha = 0.97f;
    float gamma = 0.99f;
    uint32_t autoKStart = 0;
    float dualDamping = 0.0f;
    float dualDampVel = 0.0f;
};
static_assert(sizeof(SolverParamsNpGpu) % 4 == 0, "layout");

struct NarrowphaseArgsGpu
{
    NpBodySoAGpu b;
    VkDeviceAddress pairs = 0;
    uint32_t pairCount = 0, _pad0 = 0;
    NpManifoldSoAGpu cur;
    NpManifoldSoAGpu prev;
    VkDeviceAddress prevKeys = 0;
    uint32_t prevCount = 0, _pad1 = 0;
    VkDeviceAddress contactCounter = 0, activeCounter = 0;
    // No padding here, deliberately: narrowphase.slang's NarrowphaseArgs has
    // `int contactCapacity; SolverParamsNp p;` back to back with nothing between them
    // (a float-first struct needs no 8-byte alignment), so an extra padding field here
    // would shift every SolverParamsNpGpu field the GPU reads by 4 bytes relative to
    // what this struct writes. That was exactly the earlier bug: `p.gamma` read back
    // what `alpha` held (matching a warm-start decay observed at ~0.97/build instead of
    // gamma's 0.99), `p.autoKStart` read `gamma`'s bit pattern (nonzero -> spuriously
    // truthy, silently enabling the auto-k-start branch even with it requested off),
    // and `p.dt` read the old pad slot (0), so `1/(w*dt*dt)` divided by zero and
    // clamped straight to PENALTY_MAX -- reproduced exactly as a stuck 1e10 penalty on
    // every contact touching the static ground body. Verified against
    // narrowphase.slang's field order; do not reintroduce padding here without also
    // adding it there, and see this header's csProbeNarrowphaseArgs in
    // layout_probe.slang, which is exactly what would have caught this the first time.
    int32_t contactCapacity = 0;
    SolverParamsNpGpu p;
    uint32_t _pad2 = 0;
    // M4: when set, pairCount is a capacity and the live count is *livePairs (see
    // narrowphase.slang). Appended last so the layout probe's prefix stays valid.
    VkDeviceAddress livePairs = 0;
    // Fidelity trade, appended for the same reason.
    int32_t contactCap = 0;
    int32_t _pad3 = 0;
    // Hull pairs (Phase 1.3): csNarrowphase appends their slots here for csNarrowphaseHull.
    VkDeviceAddress hullList = 0;    // may be 0 (no hull shape in the scene)
    VkDeviceAddress hullCounter = 0; // may be 0
    // Mesh pairs (Phase 1.5a): csNarrowphase appends mesh-vs-convex pair slots here for csMeshMidphase.
    VkDeviceAddress meshList = 0;    // may be 0 (no mesh shape in the scene)
    VkDeviceAddress meshCounter = 0; // may be 0
};

// Bytes of one hull lane's scratch slot: [B3Scratch (22268 B on device, after the triangle kernels)][B3HullAux][b3BoxHull blob], each
// 16-byte aligned (narrowphase_hull.slang). avbd_test_vk_hulls checks that a hull pair finds it large enough.
static const uint32_t kHullScratchStride = 24576;

// narrowphase_hull.slang: NarrowphaseHullArgs = the narrowphase args plus the hull inputs.
// The SAT cache is one record per manifold slot (cur written, prev read by the same slot index
// as the other cur/prev arrays), kept beside the manifold set because only this kernel uses it.
struct NarrowphaseHullArgsGpu
{
    NarrowphaseArgsGpu np;
    VkDeviceAddress geomHeap = 0;                  // base of the geometry heap
    VkDeviceAddress geomOffset = 0;                // per shape: byte offset of its blob, 0xFFFFFFFF none
    VkDeviceAddress scratch = 0;                   // hullCapacity slots of scratchStride bytes
    VkDeviceAddress satSepCur = 0, satBitsCur = 0; // per manifold slot
    VkDeviceAddress satSepPrev = 0, satBitsPrev = 0;
    uint64_t geomHeapBytes = 0;
    uint32_t hullCapacity = 0;  // scratch slots; list entries at or past it are skipped
    uint32_t scratchStride = 0; // bytes per slot
};
static_assert(sizeof(NarrowphaseHullArgsGpu) % 8 == 0, "layout");

// Writes the indirect dispatch for the hull kernel: ceil(min(*counter, capacity) / groupSize).
struct NarrowphaseHullDispatchArgsGpu
{
    VkDeviceAddress counter = 0;
    VkDeviceAddress indirect = 0; // three uints
    uint32_t capacity = 0;
    uint32_t groupSize = 0;
};
static_assert(sizeof(NarrowphaseHullDispatchArgsGpu) % 8 == 0, "layout");

// ---------------------------------------------------------------------------
// narrowphase_mesh.slang: static triangle mesh pairs.
// ---------------------------------------------------------------------------

// Lanes of the child kernel that loop over the child list, one scratch slot each. A fixed count, so
// the child dispatch is not indirect; the list is walked with a stride of kMeshWorkers.
static const uint32_t kMeshWorkers = 256;

// Bytes of one child lane's scratch slot: [B3Scratch][B3MeshAux][b3BoxHull blob], each 16-byte aligned
// (narrowphase_mesh.slang). avbd_test_vk_mesh checks that a child fits.
static const uint32_t kMeshScratchStride = 24576;

// csMeshMidphase: the narrowphase args plus the mesh inputs. Each lane walks the BVH of one listed
// mesh pair and appends child keys to childKeys (the key layout is in narrowphase_mesh.slang).
struct NarrowphaseMeshArgsGpu
{
    NarrowphaseArgsGpu np;
    VkDeviceAddress geomHeap = 0;   // base of the geometry heap
    VkDeviceAddress geomOffset = 0; // per shape: byte offset of its blob, 0xFFFFFFFF none
    uint64_t geomHeapBytes = 0;
    VkDeviceAddress childKeys = 0;    // raw child keys (unsorted)
    VkDeviceAddress childCounter = 0; // counted appends into childKeys
    uint32_t childCapacity = 0;       // childKeys entries; an append at or past it is only counted
    uint32_t meshCount = 0;           // mesh pairs to walk (the list length, clamped to the capacity)
};
static_assert(sizeof(NarrowphaseMeshArgsGpu) % 8 == 0, "layout");

// csMeshChild: the per-triangle narrowphase. Sorted child i is manifold slot childBase + i.
struct NarrowphaseMeshChildArgsGpu
{
    NarrowphaseArgsGpu np; // b, cur, prev, contact counters and the solver parameters
    VkDeviceAddress geomHeap = 0;
    VkDeviceAddress geomOffset = 0;
    VkDeviceAddress scratch = 0;      // kMeshWorkers slots of scratchStride bytes
    VkDeviceAddress childSorted = 0;  // child keys, sorted ascending
    VkDeviceAddress childCounter = 0; // device count of the sorted list (clamped to childCapacity)
    VkDeviceAddress childInfo = 0;    // per sorted child: Box3D class/feature and squared distance
    uint64_t geomHeapBytes = 0;
    uint32_t childCapacity = 0;  // child manifold slots: [childBase, childBase + childCapacity)
    uint32_t childBase = 0;      // first child manifold slot (= the pair slot capacity)
    uint32_t prevChildBase = 0;  // previous step's first child slot
    uint32_t prevChildCount = 0; // previous step's committed child count
    uint32_t workers = 0;        // lanes that loop over the list (kMeshWorkers)
    uint32_t scratchStride = 0;  // bytes per slot
};
static_assert(sizeof(NarrowphaseMeshChildArgsGpu) % 8 == 0, "layout");

// Writes an indirect dispatch for a counted list: ceil(min(*counter, capacity) / groupSize).
struct NarrowphaseMeshDispatchArgsGpu
{
    VkDeviceAddress counter = 0;
    VkDeviceAddress indirect = 0; // three uints
    uint32_t capacity = 0;
    uint32_t groupSize = 0;
};
static_assert(sizeof(NarrowphaseMeshDispatchArgsGpu) % 8 == 0, "layout");

} // namespace avbdvk
