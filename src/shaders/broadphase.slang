// Phase 3b: uniform-grid broadphase
// (Broadphase::build's kernels: k_aabb_and_bin, k_aabb_static, k_reset_ranges,
// k_cell_ranges, k_emit_pairs, k_emit_static_pairs). Same cell math, same hash, same
// per-cell scan cap, so the packed cell id / hash bucket a body lands in is bit-exact
// with the reference path -- required for tools/test_vk_broadphase.cpp's exact pair-SET
// comparison. Host orchestration (sorts, buffer growth, the two-sort pipeline) is
// engine/vk/vk_broadphase.h/.cpp, using vk_primitives' radixSortPairsU32 /
// radixSortKeysU64 (same call shape as the broadphase's CUB wrappers).
//
// the reference k_emit_pairs/k_emit_static_pairs use warp-aggregated atomics (warpReserveOne,
// gpu_warp) to turn one atomicAdd per warp into the pair-slot reservation for every
// lane that found a pair that step, with __activemask() covering however many lanes
// are actually live at that point in the (divergent) neighbour scan. subgroupReserveOne
// below is the same trick over WaveActiveCountBits/WavePrefixCountBits/WaveReadLaneFirst,
// which -- unlike __activemask() -- is defined generically over the subgroup's current
// execution mask regardless of subgroup size, so it needs no size-32 assumption (AMD's
// 64-wide subgroups included).

import maths;

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

static const int MAX_CELL_SCAN = 512;          // the broadphase: MAX_CELL_SCAN
static const float COLLISION_MARGIN_BP = 0.01f; // collision.h: COLLISION_MARGIN

// ---------------------------------------------------------------------------
// Cell math -- bit-for-bit the same as the broadphase's packCell/hashCell/
// cellCoord (both host and device sides of the reference source use the same functions,
// so this is the one piece of the port that must match to the bit, not just to
// tolerance).
// ---------------------------------------------------------------------------
uint64_t packCellS(int x, int y, int z)
{
    // Literal-suffix parsing for 64-bit constants is unverified in this Slang
    // toolchain (no prior use of `UL`-suffixed literals anywhere in this shader tree),
    // so every constant here is built with an explicit uint64_t cast instead of
    // trusting a suffix -- a silently-truncated-to-32-bit hash constant would corrupt
    // every bucket assignment without producing a compile error.
    const uint64_t bias = uint64_t(1) << 20;
    uint64_t ux = (uint64_t(x) + bias) & uint64_t(0x1FFFFF);
    uint64_t uy = (uint64_t(y) + bias) & uint64_t(0x1FFFFF);
    uint64_t uz = (uint64_t(z) + bias) & uint64_t(0x1FFFFF);
    return (ux << 42) | (uy << 21) | uz;
}

uint hashCellS(uint64_t c, uint mask)
{
    c ^= c >> 33;
    c *= (uint64_t(0xff51afd7u) << 32) | uint64_t(0xed558ccdu);
    c ^= c >> 33;
    c *= (uint64_t(0xc4ceb9feu) << 32) | uint64_t(0x1a85ec53u);
    c ^= c >> 33;
    return (uint)c & mask;
}

int cellCoordS(float x, float invCellSize)
{
    return (int)floor(x * invCellSize);
}

uint64_t makePairKeyS(int a, int b)
{
    int lo = a < b ? a : b;
    int hi = a < b ? b : a;
    return ((uint64_t)(uint)lo << 32) | (uint)hi;
}

void boxAabbS(vec3 pos, quat rot, vec3 half, out vec3 lo, out vec3 hi)
{
    vec3 axis0 = rotateQ(rot, vec3(1.0f, 0.0f, 0.0f));
    vec3 axis1 = rotateQ(rot, vec3(0.0f, 1.0f, 0.0f));
    vec3 axis2 = rotateQ(rot, vec3(0.0f, 0.0f, 1.0f));
    vec3 e;
    e.x = abs(axis0.x) * half.x + abs(axis1.x) * half.y + abs(axis2.x) * half.z;
    e.y = abs(axis0.y) * half.x + abs(axis1.y) * half.y + abs(axis2.y) * half.z;
    e.z = abs(axis0.z) * half.x + abs(axis1.z) * half.y + abs(axis2.z) * half.z;
    e = add(e, vec3(COLLISION_MARGIN_BP, COLLISION_MARGIN_BP, COLLISION_MARGIN_BP));
    lo = sub(pos, e);
    hi = add(pos, e);
}

bool aabbOverlapS(vec3 loA, vec3 hiA, vec3 loB, vec3 hiB)
{
    if (!isfinite(loA.x) || !isfinite(loA.y) || !isfinite(loA.z) ||
        !isfinite(hiA.x) || !isfinite(hiA.y) || !isfinite(hiA.z) ||
        !isfinite(loB.x) || !isfinite(loB.y) || !isfinite(loB.z) ||
        !isfinite(hiB.x) || !isfinite(hiB.y) || !isfinite(hiB.z))
        return false;
    return !(hiA.x < loB.x || hiB.x < loA.x ||
             hiA.y < loB.y || hiB.y < loA.y ||
             hiA.z < loB.z || hiB.z < loA.z);
}

bool pairIgnoredS(uint64_t *ignore, int n, uint64_t k)
{
    int lo = 0, hi = n - 1;
    while (lo <= hi)
    {
        int mid = (lo + hi) >> 1;
        uint64_t v = ignore[mid];
        if (v == k)
            return true;
        if (v < k)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return false;
}

// Subgroup-aggregated atomic reservation: subgroup equivalent of gpu_warp's
// warpReserveOne, generic over subgroup size (WaveGetLaneCount(), not hardcoded 32).
int subgroupReserveOne(int *counter)
{
    uint rank = WavePrefixCountBits(true);
    uint total = WaveActiveCountBits(true);
    int base = 0;
    if (WaveIsFirstLane())
        InterlockedAdd(*counter, (int)total, base);
    base = WaveReadLaneFirst(base);
    return base + (int)rank;
}

// ---------------------------------------------------------------------------
// Minimal body SoA -- pos/rot/halfExtents only, mirroring collide.slang's
// BodyBoxSoA. Broadphase::build only ever reads these three fields (the rest of
// the reference BodySoA is dead in the broadphase kernels -- k_emit_pairs's own `BodySoA b`
// parameter is unused, dropped here rather than carried across for no reason).
// ---------------------------------------------------------------------------
struct BphBodySoA
{
    vec3 *pos;        // per BODY
    quat *rot;        // per BODY
    int *shapeBody;   // per SHAPE: owning body
    vec3 *localPos;   // per SHAPE: pose in the body frame
    quat *localRot;
    vec3 *halfExtents;
    uint *flags;      // AVBD_SHAPE_FLAG_IDENTITY (bit 0): local pose is identity, skip composing
    uint count;
    uint _pad0;
};

// The world AABB of shape `s`. An identity-flagged shape takes the body pose untouched, so
// a single-shape body computes exactly the arithmetic it did before shapes existed.
void shapeAabbS(BphBodySoA b, int s, out vec3 lo, out vec3 hi)
{
    int body = b.shapeBody[s];
    vec3 p = b.pos[body];
    quat r = b.rot[body];
    if ((b.flags[s] & 1u) == 0u)
    {
        p = add(p, rotateQ(r, b.localPos[s]));
        r = mulQQ(r, b.localRot[s]);
    }
    boxAabbS(p, r, b.halfExtents[s], lo, hi);
}

struct AabbBinArgs
{
    BphBodySoA b;
    int *idx;
    uint n;
    float invCellSize;
    vec3 *aabbMin;
    vec3 *aabbMax;
    uint64_t *cellId;
    uint *hash;
    int *binBody;
    uint mask;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csAabbAndBin(uint3 tid: SV_DispatchThreadID)
{
    AabbBinArgs *args = (AabbBinArgs *)pc.argsAddr;
    uint t = tid.x;
    if (t >= args->n)
        return;

    int i = args->idx[t];
    vec3 lo, hi;
    shapeAabbS(args->b, i, lo, hi);
    args->aabbMin[i] = lo;
    args->aabbMax[i] = hi;

    vec3 c = scale(add(lo, hi), 0.5f);
    uint64_t cell = packCellS(cellCoordS(c.x, args->invCellSize),
                              cellCoordS(c.y, args->invCellSize),
                              cellCoordS(c.z, args->invCellSize));
    args->cellId[i] = cell;
    args->hash[t] = hashCellS(cell, args->mask);
    args->binBody[t] = i;
}

// Static hash table (Appendix A1): statics are binned once, when the table is dirty,
// into their own sorted cell table, which csEmitPairs probes beside the dynamic one.
// A static wider than a cell on any axis cannot be found through the 27-cell probe, so
// it is binned to the extra bucket `mask + 1` (one past the probed range): the sort puts
// those last, and cellStart/cellEnd[mask + 1] is the range csEmitCrossPairs walks.
struct StaticBinArgs
{
    BphBodySoA b;
    int *idx;
    uint n;
    float invCellSize;
    vec3 *aabbMin;
    vec3 *aabbMax;
    uint64_t *cellId;
    uint *hash;
    int *binBody;
    uint mask;
    float cellSize;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csAabbBinStatic(uint3 tid: SV_DispatchThreadID)
{
    StaticBinArgs *args = (StaticBinArgs *)pc.argsAddr;
    uint t = tid.x;
    if (t >= args->n)
        return;

    int i = args->idx[t];
    vec3 lo, hi;
    shapeAabbS(args->b, i, lo, hi);
    args->aabbMin[i] = lo;
    args->aabbMax[i] = hi;

    vec3 c = scale(add(lo, hi), 0.5f);
    uint64_t cell = packCellS(cellCoordS(c.x, args->invCellSize),
                              cellCoordS(c.y, args->invCellSize),
                              cellCoordS(c.z, args->invCellSize));
    args->cellId[i] = cell;
    vec3 e = sub(hi, lo);
    bool big = !(e.x <= args->cellSize && e.y <= args->cellSize && e.z <= args->cellSize);
    args->hash[t] = big ? args->mask + 1u : hashCellS(cell, args->mask);
    args->binBody[t] = i;
}

struct AabbStaticArgs
{
    BphBodySoA b;
    int *idx;
    uint n;
    vec3 *aabbMin;
    vec3 *aabbMax;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csAabbStatic(uint3 tid: SV_DispatchThreadID)
{
    AabbStaticArgs *args = (AabbStaticArgs *)pc.argsAddr;
    uint t = tid.x;
    if (t >= args->n)
        return;
    int i = args->idx[t];
    vec3 lo, hi;
    shapeAabbS(args->b, i, lo, hi);
    args->aabbMin[i] = lo;
    args->aabbMax[i] = hi;
}

struct ResetRangesArgs
{
    int *cellStart;
    int *cellEnd;
    uint tableSize;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csResetRanges(uint3 tid: SV_DispatchThreadID)
{
    ResetRangesArgs *args = (ResetRangesArgs *)pc.argsAddr;
    uint i = tid.x;
    if (i >= args->tableSize)
        return;
    args->cellStart[i] = 0;
    args->cellEnd[i] = 0;
}

struct CellRangesArgs
{
    uint *hashSorted;
    uint n;
    int *cellStart;
    int *cellEnd;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csCellRanges(uint3 tid: SV_DispatchThreadID)
{
    CellRangesArgs *args = (CellRangesArgs *)pc.argsAddr;
    uint i = tid.x;
    if (i >= args->n)
        return;

    uint h = args->hashSorted[i];
    if (i == 0 || args->hashSorted[i - 1] != h)
        args->cellStart[h] = (int)i;
    if (i == args->n - 1 || args->hashSorted[i + 1] != h)
        args->cellEnd[h] = (int)i + 1;
}

struct EmitPairsArgs
{
    int *binBodySorted;
    uint n;
    uint64_t *cellId;
    vec3 *aabbMin;
    vec3 *aabbMax;
    int *cellStart;
    int *cellEnd;
    float invCellSize;
    uint mask;
    uint64_t *pairs;
    int *counter;
    int capacity;
    uint64_t *ignore;
    int ignoreCount;
    int *shapeBody; // per shape: owning body (same-body skip, ignore test)
    // Static table (csAabbBinStatic). sCellStart may be null: no static table.
    int *sBinBodySorted; // may be null
    int *sCellStart;     // may be null
    int *sCellEnd;       // may be null
    uint sMask;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csEmitPairs(uint3 tid: SV_DispatchThreadID)
{
    EmitPairsArgs *args = (EmitPairsArgs *)pc.argsAddr;
    uint t = tid.x;
    if (t >= args->n)
        return;

    int i = args->binBodySorted[t];
    vec3 loA = args->aabbMin[i], hiA = args->aabbMax[i];
    vec3 c = scale(add(loA, hiA), 0.5f);
    int cx = cellCoordS(c.x, args->invCellSize);
    int cy = cellCoordS(c.y, args->invCellSize);
    int cz = cellCoordS(c.z, args->invCellSize);

    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
            {
                uint64_t target = packCellS(cx + dx, cy + dy, cz + dz);
                uint h = hashCellS(target, args->mask);

                int start = args->cellStart[h], end = args->cellEnd[h];
                if (end - start > MAX_CELL_SCAN)
                    end = start + MAX_CELL_SCAN;
                for (int s = start; s < end; ++s)
                {
                    int j = args->binBodySorted[s];

                    if (args->cellId[j] != target)
                        continue;
                    if (j <= i)
                        continue;
                    if (!aabbOverlapS(loA, hiA, args->aabbMin[j], args->aabbMax[j]))
                        continue;
                    int bi = args->shapeBody[i], bj = args->shapeBody[j];
                    if (bi == bj)
                        continue;
                    if (args->ignoreCount > 0 &&
                        pairIgnoredS(args->ignore, args->ignoreCount, makePairKeyS(bi, bj)))
                        continue;

                    int slot = subgroupReserveOne(args->counter);
                    if (slot < args->capacity)
                        args->pairs[slot] = makePairKeyS(i, j);
                }

                if (args->sCellStart == nullptr)
                    continue;
                // Same cell in the static table: no j <= i test, a static is never
                // the probing side, so every dynamic-static pair is emitted once.
                uint sh = hashCellS(target, args->sMask);
                int sStart = args->sCellStart[sh], sEnd = args->sCellEnd[sh];
                if (sEnd - sStart > MAX_CELL_SCAN)
                    sEnd = sStart + MAX_CELL_SCAN;
                for (int s = sStart; s < sEnd; ++s)
                {
                    int j = args->sBinBodySorted[s];
                    if (args->cellId[j] != target)
                        continue;
                    if (!aabbOverlapS(loA, hiA, args->aabbMin[j], args->aabbMax[j]))
                        continue;
                    int bi = args->shapeBody[i], bj = args->shapeBody[j];
                    if (bi == bj)
                        continue;
                    if (args->ignoreCount > 0 &&
                        pairIgnoredS(args->ignore, args->ignoreCount, makePairKeyS(bi, bj)))
                        continue;

                    int slot = subgroupReserveOne(args->counter);
                    if (slot < args->capacity)
                        args->pairs[slot] = makePairKeyS(i, j);
                }
            }
}

// Brute-force cross test, list A against list B, for what the grids cannot see:
// gridded dynamics x statics too big for the static table, and oversized dynamics x
// every static. B is bIdx[0, nB), or bIdx[*bStart, *bEnd) when bStart is non-null (the
// big-static bucket of the sorted static table, whose bounds only the device knows).
struct EmitCrossPairsArgs
{
    int *aIdx;
    uint nA;
    int *bIdx;
    uint nB;
    int *bStart; // may be null
    int *bEnd;   // may be null
    vec3 *aabbMin;
    vec3 *aabbMax;
    uint64_t *pairs;
    int *counter;
    int capacity;
    uint64_t *ignore;
    int ignoreCount;
    int *shapeBody;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csEmitCrossPairs(uint3 tid: SV_DispatchThreadID)
{
    EmitCrossPairsArgs *args = (EmitCrossPairsArgs *)pc.argsAddr;
    uint t = tid.x;
    if (t >= args->nA)
        return;

    int i = args->aIdx[t];
    vec3 loA = args->aabbMin[i], hiA = args->aabbMax[i];
    int s0 = 0, s1 = (int)args->nB;
    if (args->bStart != nullptr)
    {
        s0 = *args->bStart;
        s1 = *args->bEnd;
    }

    for (int s = s0; s < s1; ++s)
    {
        int j = args->bIdx[s];
        if (!aabbOverlapS(loA, hiA, args->aabbMin[j], args->aabbMax[j]))
            continue;
        int bi = args->shapeBody[i], bj = args->shapeBody[j];
        if (bi == bj)
            continue;
        if (args->ignoreCount > 0 &&
            pairIgnoredS(args->ignore, args->ignoreCount, makePairKeyS(bi, bj)))
            continue;

        int slot = subgroupReserveOne(args->counter);
        if (slot < args->capacity)
            args->pairs[slot] = makePairKeyS(i, j);
    }
}

// Oversized bodies (too wide for the grid's 27-cell neighbourhood to be exact) are kept
// out of the grid and tested here against everything else. One thread per grid body
// loops the (few) oversized bodies; one extra thread per oversized body covers the
// oversized-vs-oversized pairs (k > own index, so each is emitted once). Same overlap
// predicate, ignore test and counter protocol as csEmitPairs; the final key sort
// canonicalises the order.
struct EmitOversizePairsArgs
{
    int *gridIdx;
    uint nGrid;
    int *oversizeIdx;
    uint nOver;
    vec3 *aabbMin;
    vec3 *aabbMax;
    uint64_t *pairs;
    int *counter;
    int capacity;
    uint64_t *ignore;
    int ignoreCount;
    int *shapeBody;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csEmitOversizePairs(uint3 tid: SV_DispatchThreadID)
{
    EmitOversizePairsArgs *args = (EmitOversizePairsArgs *)pc.argsAddr;
    uint t = tid.x;
    if (t >= args->nGrid + args->nOver)
        return;

    int j;
    uint k0;
    if (t < args->nGrid)
    {
        j = args->gridIdx[t];
        k0 = 0;
    }
    else
    {
        uint kk = t - args->nGrid;
        j = args->oversizeIdx[kk];
        k0 = kk + 1;
    }
    vec3 loJ = args->aabbMin[j], hiJ = args->aabbMax[j];

    for (uint k = k0; k < args->nOver; ++k)
    {
        int i = args->oversizeIdx[k];
        if (!aabbOverlapS(loJ, hiJ, args->aabbMin[i], args->aabbMax[i]))
            continue;
        int bi = args->shapeBody[i], bj = args->shapeBody[j];
        if (bi == bj)
            continue;
        if (args->ignoreCount > 0 &&
            pairIgnoredS(args->ignore, args->ignoreCount, makePairKeyS(bi, bj)))
            continue;

        int slot = subgroupReserveOne(args->counter);
        if (slot < args->capacity)
            args->pairs[slot] = makePairKeyS(i, j);
    }
}
