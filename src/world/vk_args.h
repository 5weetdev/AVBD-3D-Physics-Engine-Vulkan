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

// SolverParamsGpu (used by PrimalArgsGpu/JointInitArgsGpu/JointDualArgsGpu/DualArgsGpu below) lives in solver_params_pack.h, not in this file -- pull
// it in here rather than relying on every includer to have already done so.
#include "solver_params_pack.h"

// Single shared home for every host-side "*ArgsGpu" struct dispatched to a Slang
// compute kernel via HostRingBuffer::write()+VkDeviceAddress, across broadphase.slang,
// narrowphase.slang, and {scan,compact,radix}.slang. These used to be duplicated (one
// hand-written copy per .cpp), which is exactly how the narrowphase penalty-divergence
// bug happened: an extra _pad2 field in vk_narrowphase.cpp's NarrowphaseArgsGpu that
// narrowphase.slang's NarrowphaseArgs did not have, silently shifting every
// SolverParamsNp field the GPU read by 4 bytes -- no compile error, no size-only
// assert would have caught it (the struct was still a plausible size), only a live GPU
// probe comparing field-by-field against the real Slang-compiler-computed offsets
// caught it (see tools/test_vk_layout.cpp / src/shaders/layout_probe.slang).
//
// Putting every struct here means the probe (test_vk_layout.cpp) and the real dispatch
// code (vk_broadphase.cpp, vk_narrowphase.cpp, vk_primitives.cpp) share ONE definition
// -- there is no longer a "hand mirror" for the probe to test against itself instead of
// the real thing. Every struct still gets a static_assert(sizeof(...) % 8 == 0): that
// alone would NOT have caught the _pad2 bug (adding one exchanges tail padding for an
// internal shift, same total size), but it does catch the more common accidental typo
// of an odd scalar tail with no padding at all, and documents the alignment contract
// scalar layout relies on. The GPU probe is what actually proves per-field placement.
//
// Field order in every struct below is a byte-for-byte transcription of its Slang
// counterpart (verified against src/shaders/broadphase.slang,
// src/shaders/narrowphase.slang, src/shaders/radix.slang,
// src/shaders/scan.slang, src/shaders/compact.slang at the time this header was
// written) with VkDeviceAddress (8 bytes) standing in for every Slang pointer field,
// and explicit uint32_t _padN fields standing in for the padding the Slang compiler
// inserts implicitly under scalar layout to keep pointer-typed fields 8-byte aligned
// (Slang does not tail-pad nested structs the way C++ does -- see the BodySoA gotcha
// noted in src/shaders/integrate.slang -- so every struct here is verified against
// its .slang counterpart's actual field order, not derived from a general padding rule).

#include "vk_buffer.h"

#include <cstdint>

namespace avbdvk
{

// ===========================================================================
// broadphase.slang
// ===========================================================================

struct BphBodySoAGpu
{
    VkDeviceAddress pos, rot;               // per body
    VkDeviceAddress shapeBody;              // per shape from here down
    VkDeviceAddress localPos, localRot, halfExtents, flags;
    uint32_t count, _pad0;
};
static_assert(sizeof(BphBodySoAGpu) % 8 == 0, "layout");

struct AabbBinArgsGpu
{
    BphBodySoAGpu b;
    VkDeviceAddress idx;
    uint32_t n;
    float invCellSize;
    VkDeviceAddress aabbMin, aabbMax, cellId, hash;
    VkDeviceAddress binBody;
    uint32_t mask;
    uint32_t _pad0;
};
static_assert(sizeof(AabbBinArgsGpu) % 8 == 0, "layout");

struct AabbStaticArgsGpu
{
    BphBodySoAGpu b;
    VkDeviceAddress idx;
    uint32_t n, _pad0;
    VkDeviceAddress aabbMin, aabbMax;
};
static_assert(sizeof(AabbStaticArgsGpu) % 8 == 0, "layout");

struct ResetRangesArgsGpu
{
    VkDeviceAddress cellStart, cellEnd;
    uint32_t tableSize, _pad0;
};
static_assert(sizeof(ResetRangesArgsGpu) % 8 == 0, "layout");

struct CellRangesArgsGpu
{
    VkDeviceAddress hashSorted;
    uint32_t n, _pad0;
    VkDeviceAddress cellStart, cellEnd;
};
static_assert(sizeof(CellRangesArgsGpu) % 8 == 0, "layout");

struct EmitPairsArgsGpu
{
    VkDeviceAddress binBodySorted;
    uint32_t n, _pad0;
    VkDeviceAddress cellId, aabbMin, aabbMax, cellStart, cellEnd;
    float invCellSize;
    uint32_t mask;
    VkDeviceAddress pairs, counter;
    int32_t capacity, _pad1;
    VkDeviceAddress ignore;
    int32_t ignoreCount, _pad2;
    VkDeviceAddress shapeBody;
    VkDeviceAddress sBinBodySorted; // may be null
    VkDeviceAddress sCellStart;     // may be null: no static table
    VkDeviceAddress sCellEnd;       // may be null
    uint32_t sMask, _pad3;
};
static_assert(sizeof(EmitPairsArgsGpu) % 8 == 0, "layout");

struct StaticBinArgsGpu
{
    BphBodySoAGpu b;
    VkDeviceAddress idx;
    uint32_t n;
    float invCellSize;
    VkDeviceAddress aabbMin, aabbMax, cellId, hash;
    VkDeviceAddress binBody;
    uint32_t mask;
    float cellSize;
};
static_assert(sizeof(StaticBinArgsGpu) % 8 == 0, "layout");

struct EmitCrossPairsArgsGpu
{
    VkDeviceAddress aIdx;
    uint32_t nA, _pad0;
    VkDeviceAddress bIdx;
    uint32_t nB, _pad1;
    VkDeviceAddress bStart; // may be null
    VkDeviceAddress bEnd;   // may be null
    VkDeviceAddress aabbMin, aabbMax, pairs, counter;
    int32_t capacity, _pad2;
    VkDeviceAddress ignore;
    int32_t ignoreCount, _pad3;
    VkDeviceAddress shapeBody;
};
static_assert(sizeof(EmitCrossPairsArgsGpu) % 8 == 0, "layout");

struct EmitOversizePairsArgsGpu
{
    VkDeviceAddress gridIdx;
    uint32_t nGrid, _pad0;
    VkDeviceAddress oversizeIdx;
    uint32_t nOver, _pad1;
    VkDeviceAddress aabbMin, aabbMax, pairs, counter;
    int32_t capacity, _pad2;
    VkDeviceAddress ignore;
    int32_t ignoreCount, _pad3;
    VkDeviceAddress shapeBody;
};
static_assert(sizeof(EmitOversizePairsArgsGpu) % 8 == 0, "layout");

// ===========================================================================
// narrowphase.slang
// ===========================================================================

struct NpBodySoAGpu
{
    VkDeviceAddress pos = 0, rot = 0;                 // per body
    VkDeviceAddress shapeBody = 0, localPos = 0, localRot = 0; // per shape
    VkDeviceAddress halfExtents = 0, shapeType = 0, friction = 0, flags = 0; // per shape
    VkDeviceAddress mass = 0, moment = 0;             // per body
    VkDeviceAddress velLin = 0, velAng = 0;
    VkDeviceAddress props = 0;                        // may be 0 (no non-default BodyProps)
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
    VkDeviceAddress geomHeap = 0;    // base of the geometry heap
    VkDeviceAddress geomOffset = 0;  // per shape: byte offset of its blob, 0xFFFFFFFF none
    VkDeviceAddress scratch = 0;     // hullCapacity slots of scratchStride bytes
    VkDeviceAddress satSepCur = 0, satBitsCur = 0; // per manifold slot
    VkDeviceAddress satSepPrev = 0, satBitsPrev = 0;
    uint64_t geomHeapBytes = 0;
    uint32_t hullCapacity = 0;       // scratch slots; list entries at or past it are skipped
    uint32_t scratchStride = 0;      // bytes per slot
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
// narrowphase_mesh.slang (Phase 1.5a): static triangle mesh pairs.
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
    VkDeviceAddress geomHeap = 0;     // base of the geometry heap
    VkDeviceAddress geomOffset = 0;   // per shape: byte offset of its blob, 0xFFFFFFFF none
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
    NarrowphaseArgsGpu np;            // b, cur, prev, contact counters and the solver parameters
    VkDeviceAddress geomHeap = 0;
    VkDeviceAddress geomOffset = 0;
    VkDeviceAddress scratch = 0;      // kMeshWorkers slots of scratchStride bytes
    VkDeviceAddress childSorted = 0;  // child keys, sorted ascending
    VkDeviceAddress childCounter = 0; // device count of the sorted list (clamped to childCapacity)
    VkDeviceAddress childInfo = 0;    // per sorted child: Box3D class/feature and squared distance
    uint64_t geomHeapBytes = 0;
    uint32_t childCapacity = 0;       // child manifold slots: [childBase, childBase + childCapacity)
    uint32_t childBase = 0;           // first child manifold slot (= the pair slot capacity)
    uint32_t prevChildBase = 0;       // previous step's first child slot
    uint32_t prevChildCount = 0;      // previous step's committed child count
    uint32_t workers = 0;             // lanes that loop over the list (kMeshWorkers)
    uint32_t scratchStride = 0;       // bytes per slot
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

// ===========================================================================
// constraints.slang
// ===========================================================================

struct CsResetColorsArgsGpu
{
    VkDeviceAddress color = 0;
    VkDeviceAddress dyn = 0;
    uint32_t dynCount = 0, _pad0 = 0;
    VkDeviceAddress dynCountPtr = 0; // M4 live count (0: dynCount is exact)
};
static_assert(sizeof(CsResetColorsArgsGpu) % 8 == 0, "layout");

struct CsEmitManifoldArgsGpu
{
    VkDeviceAddress mBodyA = 0, mBodyB = 0, mContactCount = 0;
    uint32_t slots = 0, _pad0 = 0;
    VkDeviceAddress mass = 0, awake = 0;
    VkDeviceAddress keys = 0, isA = 0, counter = 0;
    int32_t capacity = 0, _pad1 = 0;
};
static_assert(sizeof(CsEmitManifoldArgsGpu) % 8 == 0, "layout");

struct CsEmitForceArgsGpu
{
    VkDeviceAddress fBodyA = 0, fBodyB = 0;
    uint32_t count = 0, _pad0 = 0;
    VkDeviceAddress broken = 0; // may be 0 (springs)
    VkDeviceAddress mass = 0, awake = 0;
    VkDeviceAddress keys = 0, isA = 0, counter = 0;
    int32_t capacity = 0;
    int32_t type = 0; // ForceType: FORCE_JOINT=1, FORCE_SPRING=2
};
static_assert(sizeof(CsEmitForceArgsGpu) % 8 == 0, "layout");

struct CsRowsArgsGpu
{
    VkDeviceAddress keys = 0;
    uint32_t n = 0, _pad0 = 0;
    VkDeviceAddress start = 0, end = 0;
    VkDeviceAddress count = 0; // M4 live count; `n` is then the capacity
};
static_assert(sizeof(CsRowsArgsGpu) % 8 == 0, "layout");

struct CsJpRoundArgsGpu
{
    VkDeviceAddress gKey = 0, gIsA = 0, gStart = 0, gEnd = 0;
    VkDeviceAddress mBodyA = 0, mBodyB = 0;
    VkDeviceAddress jBodyA = 0, jBodyB = 0;
    VkDeviceAddress sBodyA = 0, sBodyB = 0;
    VkDeviceAddress mass = 0, awake = 0;
    VkDeviceAddress dyn = 0;
    uint32_t dynCount = 0, _pad0 = 0;
    VkDeviceAddress color = 0, remaining = 0;
    VkDeviceAddress dynCountPtr = 0; // M4 live count (0: dynCount is exact)
};
static_assert(sizeof(CsJpRoundArgsGpu) % 8 == 0, "layout");

struct CsColorKeysArgsGpu
{
    VkDeviceAddress color = 0, dyn = 0;
    uint32_t dynCount = 0, _pad0 = 0;
    VkDeviceAddress keys = 0, vals = 0;
    VkDeviceAddress dynCountPtr = 0; // M4 live count (0: dynCount is exact)
};
static_assert(sizeof(CsColorKeysArgsGpu) % 8 == 0, "layout");

struct CsColorRowsArgsGpu
{
    VkDeviceAddress keys = 0;
    uint32_t n = 0, _pad0 = 0;
    VkDeviceAddress start = 0, end = 0;
    VkDeviceAddress dynCountPtr = 0; // M4 live count (0: n is exact)
};
static_assert(sizeof(CsColorRowsArgsGpu) % 8 == 0, "layout");

// csBuildSmall (constraints.slang): the one-workgroup constraint build.
struct CsBuildSmallArgsGpu
{
    VkDeviceAddress mBodyA = 0, mBodyB = 0, mCount = 0;
    VkDeviceAddress jBodyA = 0, jBodyB = 0, jBroken = 0; // jBroken may be 0
    VkDeviceAddress sBodyA = 0, sBodyB = 0;
    VkDeviceAddress mass = 0, awake = 0, dyn = 0;
    VkDeviceAddress keys = 0, isA = 0, rowStart = 0, rowEnd = 0;
    VkDeviceAddress color = 0, bodyByColor = 0;
    VkDeviceAddress ranges = 0;     // start[128], end[128]
    VkDeviceAddress counter = 0, remaining = 0;
    VkDeviceAddress dynCountPtr = 0; // may be 0
    uint32_t slots = 0, jointSlots = 0, springSlots = 0, bodyCount = 0, dynCount = 0;
    int32_t capacity = 0;
    uint32_t rounds = 0, _pad0 = 0;
};
static_assert(sizeof(CsBuildSmallArgsGpu) == 21 * 8 + 8 * 4, "CsBuildSmallArgsGpu drifted from constraints.slang BuildSmallArgs");

// ===========================================================================
// scan.slang / compact.slang / radix.slang (Phase-2 primitives)
// ===========================================================================

struct ScanBlockArgs
{
    VkDeviceAddress in, out, blockSums;
    uint32_t n, _pad0;
};
static_assert(sizeof(ScanBlockArgs) == 32 && sizeof(ScanBlockArgs) % 8 == 0, "layout");

struct AddBlockSumsArgs
{
    VkDeviceAddress out, blockOffsets;
    uint32_t n, _pad0;
};
static_assert(sizeof(AddBlockSumsArgs) % 8 == 0, "layout");

struct FlagsToU32Args
{
    VkDeviceAddress flags, out;
    uint32_t n, _pad0;
};
static_assert(sizeof(FlagsToU32Args) % 8 == 0, "layout");

struct ScatterCompactArgs
{
    VkDeviceAddress items, flags, scanned, out;
    uint32_t n, _pad0;
};
static_assert(sizeof(ScatterCompactArgs) % 8 == 0, "layout");

struct NumSelectedArgs
{
    VkDeviceAddress scanned, flags, numSelected;
    uint32_t n, _pad0;
};
static_assert(sizeof(NumSelectedArgs) % 8 == 0, "layout");

struct HistArgsU32Pairs
{
    VkDeviceAddress keysIn, digitCount, skipMask;
    uint32_t n, numBlocks, pass, _pad0;
    VkDeviceAddress count = 0; // live-count pointer, 0 = n is exact
};
static_assert(sizeof(HistArgsU32Pairs) % 8 == 0, "layout");

struct ScatterArgsU32Pairs
{
    VkDeviceAddress keysIn, valsIn, digitCountScanned, skipMask, keysOut, valsOut;
    uint32_t n, numBlocks, pass, _pad0;
    VkDeviceAddress count = 0; // live-count pointer, 0 = n is exact
};
static_assert(sizeof(ScatterArgsU32Pairs) % 8 == 0, "layout");

struct HistArgsU64Pairs
{
    VkDeviceAddress keysIn, digitCount, skipMask;
    uint32_t n, numBlocks, pass, _pad0;
    VkDeviceAddress count = 0; // live-count pointer, 0 = n is exact
};
static_assert(sizeof(HistArgsU64Pairs) % 8 == 0, "layout");

struct ScatterArgsU64Pairs
{
    VkDeviceAddress keysIn, valsIn, digitCountScanned, skipMask, keysOut, valsOut;
    uint32_t n, numBlocks, pass, _pad0;
    VkDeviceAddress count = 0; // live-count pointer, 0 = n is exact
};
static_assert(sizeof(ScatterArgsU64Pairs) % 8 == 0, "layout");

struct HistArgsU64Keys
{
    VkDeviceAddress keysIn, digitCount, skipMask;
    uint32_t n, numBlocks, pass, _pad0;
    VkDeviceAddress count = 0; // live-count pointer, 0 = n is exact
};
static_assert(sizeof(HistArgsU64Keys) % 8 == 0, "layout");

struct ScatterArgsU64Keys
{
    VkDeviceAddress keysIn, digitCountScanned, skipMask, keysOut;
    uint32_t n, numBlocks, pass, _pad0;
    VkDeviceAddress count = 0; // live-count pointer, 0 = n is exact
};
static_assert(sizeof(ScatterArgsU64Keys) % 8 == 0, "layout");

struct ReduceOrAndArgsU32
{
    VkDeviceAddress keysIn, outOr, outAnd;
    uint32_t n, _pad0;
    VkDeviceAddress count = 0; // live-count pointer, 0 = n is exact
};
static_assert(sizeof(ReduceOrAndArgsU32) % 8 == 0, "layout");

struct ReduceOrAndArgsU64
{
    VkDeviceAddress keysIn, outOr, outAnd;
    uint32_t n, _pad0;
    VkDeviceAddress count = 0; // live-count pointer, 0 = n is exact
};
static_assert(sizeof(ReduceOrAndArgsU64) % 8 == 0, "layout");

// Small-world one-workgroup sorts (radix.slang section (e)).
struct SortSmallArgsU32Pairs
{
    VkDeviceAddress keysIn, valsIn, keysOut, valsOut;
    uint32_t n, _pad0;
    VkDeviceAddress count = 0; // live-count pointer, 0 = n is exact
};
static_assert(sizeof(SortSmallArgsU32Pairs) % 8 == 0, "layout");
using SortSmallArgsU64Pairs = SortSmallArgsU32Pairs; // identical field layout

struct SortSmallArgsU64Keys
{
    VkDeviceAddress keysIn, keysOut;
    uint32_t n, _pad0;
    VkDeviceAddress count = 0;
};
static_assert(sizeof(SortSmallArgsU64Keys) % 8 == 0, "layout");

struct FinalizeSkipMaskArgsU32
{
    VkDeviceAddress orAddr, andAddr, skipMask;
    uint32_t numPasses, _pad0;
};
static_assert(sizeof(FinalizeSkipMaskArgsU32) % 8 == 0, "layout");

struct FinalizeSkipMaskArgsU64
{
    VkDeviceAddress orAddr, andAddr, skipMask;
    uint32_t numPasses, _pad0;
};
static_assert(sizeof(FinalizeSkipMaskArgsU64) % 8 == 0, "layout");

// ===========================================================================
// primal.slang / dual.slang (Phase 4b) -- full BodySoA/ManifoldSoA/JointSoA/SpringSoA
// device-address mirrors (types.slang's structs, byte-for-byte), the local
// ConstraintGraph mirror (primal.slang/dual.slang declare their own copy the same way
// integrate.slang keeps a private BodySoA -- see those files' header comments), and
// the per-kernel argument structs. Field order transcribed from types.slang / the two
// .slang files at the time this was written; any drift is exactly the bug class
// tools/test_vk_layout.cpp + src/shaders/layout_probe.slang exist to catch.
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
    int32_t color = 0, heavyDegree = 0; // heavyDegree > 0: high-degree bodies go to csPrimalWide*
    VkDeviceAddress shapeRolling = 0;   // per shape rolling resistance; may be null (no shape has any)
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

// ===========================================================================
// render.slang (Phase 5C) -- reuses FullBodySoAGpu (defined above) for the body
// fields both csWriteInstances reads. Field order transcribed from
// render.slang at the time this was written.
// ===========================================================================

// Mirrors BodyInstance byte for byte (44 bytes, no
// padding needed since every field is already 4-byte aligned and the struct ends on
// a 4-byte boundary).
struct BodyInstanceGpu
{
    float pos[3];
    float rot[4];
    float half[3];
    uint32_t rgba;
};
static_assert(sizeof(BodyInstanceGpu) == 44, "must match BodyInstance/AvbdInstance");

// Mirrors render.slang's OverlayView / render_types's OverlayView, field for
// field. _pad0 sits where the compiler needs 4 bytes to bring the pointer fields that
// follow (jointDegree onward) back onto 8-byte alignment after the three trailing
// floats (jointLoadMax/degreeJointMax/penetrationMax) -- verified against the real
// Slang-computed offsets by OverlayViewProbe in layout_probe.slang.
struct OverlayViewGpu
{
    int32_t mode = 0;
    float speedMax = 0, loadMax = 0, degreeMax = 0, invCellSize = 0, sleepFrames = 0;
    int32_t colorCount = 0, colorStatics = 0;
    VkDeviceAddress graphColor = 0, load = 0, degree = 0;
    float jointLoadMax = 0, degreeJointMax = 0, penetrationMax = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress jointDegree = 0, push = 0, pull = 0, stress = 0, cone = 0, penetration = 0;
};
static_assert(sizeof(OverlayViewGpu) % 8 == 0, "layout");

// Full k_write_instances port: frustum culling, every OverlayMode, and per-shape/group
// segmentation, matching the reference parameter order (b, out, maxInstances, frustum,
// visible, ov, groupOf, shapeBase, shapeVisible) exactly.
struct WriteInstancesArgsGpu
{
    FullBodySoAGpu b;
    VkDeviceAddress out = 0;
    int32_t maxInstances = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress frustum = 0;      // const float* (24 floats), or 0
    VkDeviceAddress visible = 0;      // int* atomic counter, or 0
    OverlayViewGpu ov;
    VkDeviceAddress groupOf = 0;      // const int*, or 0
    VkDeviceAddress shapeBase = 0;    // const int*, or 0
    VkDeviceAddress shapeVisible = 0; // int*, or 0
    ShapeSoAGpu s;                    // one instance per shape; groupOf is per shape
};
static_assert(sizeof(WriteInstancesArgsGpu) % 8 == 0, "layout");

// the overlay reductions' OverlayStats: 6 int* reduction targets.
struct OverlayStatsGpu
{
    VkDeviceAddress push = 0, pull = 0, stress = 0, cone = 0, penetration = 0, jointDegree = 0;
};
static_assert(sizeof(OverlayStatsGpu) % 8 == 0, "layout");

struct OverlayStatsClearArgsGpu
{
    OverlayStatsGpu s;
    int32_t n = 0;
    int32_t _pad0 = 0;
};
static_assert(sizeof(OverlayStatsClearArgsGpu) % 8 == 0, "layout");

struct OverlayStatsContactsArgsGpu
{
    FullManifoldSoAGpu m;
    int32_t slots = 0;
    int32_t _pad0 = 0;
    OverlayStatsGpu s;
};
static_assert(sizeof(OverlayStatsContactsArgsGpu) % 8 == 0, "layout");

struct OverlayStatsForcesArgsGpu
{
    FullBodySoAGpu b;
    FullJointSoAGpu j;
    FullSpringSoAGpu sp;
    int32_t jointsAuthored = 0;
    int32_t _pad0 = 0;
    OverlayStatsGpu s;
};
static_assert(sizeof(OverlayStatsForcesArgsGpu) % 8 == 0, "layout");

struct OverlayLoadClearArgsGpu
{
    VkDeviceAddress load = 0;
    int32_t n = 0;
    int32_t _pad0 = 0;
};
static_assert(sizeof(OverlayLoadClearArgsGpu) % 8 == 0, "layout");

struct OverlayLoadAccumArgsGpu
{
    FullBodySoAGpu b;
    FullManifoldSoAGpu m;
    int32_t slots = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress load = 0;
};
static_assert(sizeof(OverlayLoadAccumArgsGpu) % 8 == 0, "layout");

// render_types's DebugDrawView.
struct DebugDrawViewGpu
{
    int32_t contactPoints = 0, contactNormals = 0, manifoldLinks = 0, tint = 0;
    float pointSize = 0, normalLength = 0, forceMax = 0;
    int32_t _pad0 = 0;
};
static_assert(sizeof(DebugDrawViewGpu) % 8 == 0, "layout");

struct WriteDebugInstancesArgsGpu
{
    FullBodySoAGpu b;
    FullManifoldSoAGpu m;
    int32_t slots = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress out = 0;
    int32_t maxInstances = 0;
    int32_t _pad1 = 0;
    VkDeviceAddress counter = 0;
    DebugDrawViewGpu d;
};
static_assert(sizeof(WriteDebugInstancesArgsGpu) % 8 == 0, "layout");

// ===========================================================================
// drag.slang (Phase 5C) -- the drag path.
// Reuses FullBodySoAGpu and FullJointSoAGpu (both defined above).
// ===========================================================================

struct DragUpdateArgsGpu
{
    FullBodySoAGpu b;
    FullJointSoAGpu j;
    int32_t slot0 = 0;
    int32_t count = 0;
    VkDeviceAddress bodies = 0;    // const int32_t*
    VkDeviceAddress targets = 0;   // const vec3*
    VkDeviceAddress locals = 0;    // const vec3*
    VkDeviceAddress stiffness = 0; // const float*
    int32_t grab = 0;
    int32_t _pad0 = 0;
};
static_assert(sizeof(DragUpdateArgsGpu) % 8 == 0, "layout");

struct DragReleaseArgsGpu
{
    FullJointSoAGpu j;
    int32_t slot0 = 0;
    int32_t count = 0;
};
static_assert(sizeof(DragReleaseArgsGpu) % 8 == 0, "layout");

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
};
static_assert(sizeof(ActiveFlagsArgsGpu) % 8 == 0, "layout");

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
    VkDeviceAddress shapeA = 0, shapeB = 0, shpRestitution = 0; // per manifold slot / per shape
    uint32_t slots = 0, dynCount = 0;
    float threshold = 0;
    uint32_t _pad = 0;
};
static_assert(sizeof(RestitutionArgsGpu) == 24 * 8 + 16, "restitution.slang RestitutionArgs");

// ccd.slang CcdArgs: shared by csCcdFlag / csCcdToi / csCcdApply.
struct CcdArgsGpu
{
    VkDeviceAddress pos = 0, rot = 0, velLin = 0, initLin = 0, initAng = 0, mass = 0, awake = 0, sleepTimer = 0;
    VkDeviceAddress props = 0; // may be null
    VkDeviceAddress shpBody = 0, shpLocalPos = 0, shpLocalRot = 0, shpHalf = 0, shpType = 0, shpFlags = 0;
    VkDeviceAddress shpCategory = 0, shpMask = 0, shpGroup = 0;
    VkDeviceAddress dynShape = 0, staticShape = 0;
    VkDeviceAddress ignore = 0; // may be null when ignoreCount == 0
    VkDeviceAddress geomHeap = 0, geomOffset = 0; // may be null (no hull or mesh in the scene)
    uint64_t geomHeapBytes = 0;
    VkDeviceAddress list = 0, counter = 0, frac = 0, scratch = 0;
    uint32_t dynCount = 0, staticCount = 0, ignoreCount = 0, scratchStride = 0;
    uint32_t lanes = 0, pass = 0;
    float safety = 1.0f;
    uint32_t _pad = 0;
};
static_assert(sizeof(CcdArgsGpu) == 28 * 8 + 32, "ccd.slang CcdArgs");

// ===========================================================================
// wake.slang (Phase 5B) -- the wake stage, the implicit velocity-gated wake
// path plus GpuSolver::wakeAll(). Reuses FullBodySoAGpu/FullManifoldSoAGpu/
// FullJointSoAGpu/FullSpringSoAGpu (all defined above).
// ===========================================================================

struct WakeRequestArgsGpu
{
    FullBodySoAGpu b;
    FullManifoldSoAGpu m;
    int32_t slots = 0;
    float wakeLinVel = 0.0f;
    float wakeAngVel = 0.0f;
    VkDeviceAddress request = 0;
};
static_assert(sizeof(WakeRequestArgsGpu) % 8 == 0, "layout");

struct TouchClearArgsGpu
{
    VkDeviceAddress dyn = 0;
    int32_t dynCount = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress count = 0;
    VkDeviceAddress hash = 0;
};
static_assert(sizeof(TouchClearArgsGpu) % 8 == 0, "layout");

struct TouchAccumArgsGpu
{
    FullBodySoAGpu b;
    FullManifoldSoAGpu m;
    int32_t slots = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress count = 0;
    VkDeviceAddress hash = 0;
};
static_assert(sizeof(TouchAccumArgsGpu) % 8 == 0, "layout");

struct WakeLostSupportArgsGpu
{
    FullBodySoAGpu b;
    VkDeviceAddress dyn = 0;
    int32_t dynCount = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress count = 0;
    VkDeviceAddress hash = 0;
    VkDeviceAddress snapCount = 0;
    VkDeviceAddress snapHash = 0;
    VkDeviceAddress request = 0;
};
static_assert(sizeof(WakeLostSupportArgsGpu) % 8 == 0, "layout");

struct WakeApplyArgsGpu
{
    FullBodySoAGpu b;
    VkDeviceAddress dyn = 0;
    int32_t dynCount = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress request = 0;
};
static_assert(sizeof(WakeApplyArgsGpu) % 8 == 0, "layout");

struct TouchAccumJointsArgsGpu
{
    FullBodySoAGpu b;
    FullJointSoAGpu j;
    FullSpringSoAGpu sp;
    VkDeviceAddress count = 0;
    VkDeviceAddress hash = 0;
};
static_assert(sizeof(TouchAccumJointsArgsGpu) % 8 == 0, "layout");

struct WakeRequestJointsArgsGpu
{
    FullBodySoAGpu b;
    FullJointSoAGpu j;
    FullSpringSoAGpu sp;
    VkDeviceAddress request = 0;
};
static_assert(sizeof(WakeRequestJointsArgsGpu) % 8 == 0, "layout");

struct WakeIslandSeedArgsGpu
{
    FullBodySoAGpu b;
    VkDeviceAddress dyn = 0;
    int32_t dynCount = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress request = 0;
    VkDeviceAddress seed = 0;
    VkDeviceAddress any = 0;
};
static_assert(sizeof(WakeIslandSeedArgsGpu) % 8 == 0, "layout");

struct WakeAllArgsGpu
{
    FullBodySoAGpu b;
    VkDeviceAddress dyn = 0;
    int32_t dynCount = 0;
    int32_t _pad0 = 0;
};
static_assert(sizeof(WakeAllArgsGpu) % 8 == 0, "layout");

// ===========================================================================
// mutate.slang (Phase 5B) -- the mutation path: the batched external
// mutation apply and the explicit, UNGATED wake flood every mutation seeds.
// Reuses FullBodySoAGpu/FullManifoldSoAGpu/FullJointSoAGpu/FullSpringSoAGpu.
// ===========================================================================

// Mirrors GpuSolver::BodyTransform / BodyPointVector (engine/host/gpu_solver.h) byte
// for byte -- see mutate.slang's own header comment: both are all-4-byte-aligned
// (vec3/quat, no 8-byte pointers), so no padding either side.
struct BodyTransformGpu
{
    float px, py, pz;
    float qx, qy, qz, qw;
};
static_assert(sizeof(BodyTransformGpu) == 28, "must match GpuSolver::BodyTransform");

struct BodyPointVectorGpu
{
    float vx, vy, vz;
    float wx, wy, wz;
};
static_assert(sizeof(BodyPointVectorGpu) == 24, "must match GpuSolver::BodyPointVector");

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
    VkDeviceAddress gate = 0; // M4: skip the merge while *gate != 0; 0 = ungated
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
