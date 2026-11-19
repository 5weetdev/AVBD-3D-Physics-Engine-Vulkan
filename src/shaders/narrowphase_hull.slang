// narrowphase for hull pairs. csNarrowphase (narrowphase.slang) does not collide a pair
// with a hull shape: it appends the pair's slot to a compacted list (a counted append, capacity =
// pair capacity, so it cannot overflow). csNarrowphaseHullArgs turns the list length into an
// indirect dispatch and csNarrowphaseHull runs one lane per listed pair through the generated Box3D
// convex collision (b3/b3_gen.slang). The box, sphere and capsule path never compiles
// this code, and a scene without a hull shape never records either kernel.
//
// Pair classes (the hull is always the "A" of the Box3D call; a pair whose A is not a hull is
// swapped, the normal negated and the anchors exchanged afterwards):
//   hull-hull     b3CollideHulls         (b3SATCache persisted per manifold slot)
//   hull-box      b3CollideHulls         the box is built as a b3BoxHull blob in per-lane scratch
//                                        (a canonical heap hull cannot be scaled; building 640 bytes
//                                        from constants is cheaper than a heap read plus a rescale).
//                                        The SAT cache is persisted too.
//   hull-sphere   b3CollideHullAndSphere
//   hull-capsule  b3CollideHullAndCapsule
// The simplex cache of the last two starts empty each step; only the SAT cache is persisted.
//
// The output is the same manifold/contact SoA csNarrowphase writes (basis, body-frame rA/rB, C0,
// feature key, friction, warm-started dual state), so everything downstream is shared. The
// warm-start tail below is a copy of csNarrowphase's, kept in the same order.
//
// Memory safety (this shader cannot be checked by GPU-AV): every heap offset is range checked
// against geomHeapBytes before a hull header is read, the hull counts are bounded by the scratch
// array sizes, and a lane whose list index is at or past hullCapacity does nothing.

import maths;
import collide;
#include "b3/b3_gen.slang"

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

static const float COLLISION_MARGIN_NH = 0.01f;   // collision.h: COLLISION_MARGIN
static const float PENALTY_MIN_NH = 1.0f;
static const float PENALTY_MAX_NH = 10000000000.0f;

// ---- mirrors of narrowphase.slang (change together; layout_probe checks NarrowphaseArgs) ----
int pairLoS(uint64_t k) { return (int)(uint)(k >> 32); }
int pairHiS(uint64_t k) { return (int)(uint)(k & uint64_t(0xFFFFFFFFu)); }

int subgroupReserve(int *counter, int n)
{
    int prefix = WavePrefixSum(n);
    int total = WaveActiveSum(n);
    int base = 0;
    if (WaveIsFirstLane())
        InterlockedAdd(*counter, total, base);
    base = WaveReadLaneFirst(base);
    return base + prefix;
}

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

int findPrevManifoldS(uint64_t *keys, int n, uint64_t want)
{
    int lo = 0, hi = n - 1;
    while (lo <= hi)
    {
        int mid = (lo + hi) >> 1;
        uint64_t k = keys[mid];
        if (k == want)
            return mid;
        if (k < want)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return -1;
}

vec3 matVec(float3x3 m, vec3 v) { return vec3(dot3(m.row0, v), dot3(m.row1, v), dot3(m.row2, v)); }
vec3 rowAtS(float3x3 m, int i) { return i == 0 ? m.row0 : (i == 1 ? m.row1 : m.row2); }
vec3 clampV(vec3 v, float lo, float hi) { return vec3(clampf(v.x, lo, hi), clampf(v.y, lo, hi), clampf(v.z, lo, hi)); }

struct NpBodySoA
{
    vec3 *pos;
    quat *rot;
    int *shapeBody;
    vec3 *localPos;
    quat *localRot;
    vec3 *halfExtents;
    uint8_t *shapeType;
    float *friction;
    uint *flags;
    float *mass;
    vec3 *moment;
    vec3 *velLin;
    vec3 *velAng;
    // Phase 2 fields, unused here (csNarrowphase filters before diverting a hull pair). They
    // must stay in step with narrowphase.slang's NpBodySoA: this is a byte copy of it.
    uint8_t *props;      // BodyProps*, may be null
    uint64_t *category;      // may be null
    uint64_t *mask;
    int *group;
    uint count;
    uint _pad0;
};

struct NpManifoldSoA
{
    uint64_t *key;
    int *bodyA;
    int *bodyB;
    float3x3 *basis;
    float *friction;
    int *contactOffset;
    int *contactCount;
    int *feature;
    vec3 *rA;
    vec3 *rB;
    vec3 *C0;
    vec3 *penalty;
    vec3 *lambda;
    uint8_t *stick;
    int *shapeA;
    int *shapeB;
};

struct SolverParamsNp
{
    float dt;
    float alpha;
    float gamma;
    uint autoKStart;
    float dualDamping;
    float dualDampVel;
};

struct NarrowphaseArgs
{
    NpBodySoA b;
    uint64_t *pairs;
    uint pairCount;
    uint _pad0;
    NpManifoldSoA cur;
    NpManifoldSoA prev;
    uint64_t *prevKeys;
    uint prevCount;
    uint _pad1;
    int *contactCounter;
    int *activeCounter;
    int contactCapacity;
    SolverParamsNp p;
    uint _pad2;
    uint *livePairs;
    int contactCap;
    uint _pad3;
    int *hullList;
    int *hullCounter;
    int *meshList;    // unused here; keeps the copy byte-identical to narrowphase.slang
    int *meshCounter;
};

// vk_args.h: NarrowphaseHullArgsGpu
struct NarrowphaseHullArgs
{
    NarrowphaseArgs np;
    uint8_t *geomHeap;
    uint *geomOffset;       // per shape, 0xFFFFFFFF none
    uint8_t *scratch;       // hullCapacity slots of scratchStride bytes
    float *satSepCur;       // per manifold slot
    uint *satBitsCur;
    float *satSepPrev;
    uint *satBitsPrev;
    uint64_t geomHeapBytes;
    uint hullCapacity;
    uint scratchStride;
};

// vk_args.h: NarrowphaseHullDispatchArgsGpu
struct NarrowphaseHullDispatchArgs
{
    int *counter;
    uint *indirect;
    uint capacity;
    uint groupSize;
};

static const uint HULL_GROUP = 32;

[shader("compute")]
[numthreads(1, 1, 1)]
void csNarrowphaseHullArgs(uint3 tid: SV_DispatchThreadID)
{
    NarrowphaseHullDispatchArgs *a = (NarrowphaseHullDispatchArgs *)pc.argsAddr;
    uint n = (uint)max(*a->counter, 0);
    n = min(n, a->capacity);
    a->indirect[0] = (n + a->groupSize - 1u) / a->groupSize;
    a->indirect[1] = 1u;
    a->indirect[2] = 1u;
}

void shapePoseNp(NpBodySoA b, int s, int body, out vec3 sp, out quat sq)
{
    sp = b.pos[body];
    sq = b.rot[body];
    if ((b.flags[s] & 1u) == 0u)
    {
        sp = add(sp, rotateQ(sq, b.localPos[s]));
        sq = mulQQ(sq, b.localRot[s]);
    }
}

// ---- per-lane scratch --------------------------------------------------------------------------
// One slot: [B3Scratch][B3HullAux][b3BoxHull blob], each 16-byte aligned. Locals have no address on
// device, so everything the b3 entry points take by pointer lives here.
struct B3HullAux
{
    b3LocalManifold manifold;
    b3SimplexCache simplexCache;
    b3SATCache sat;
    b3Sphere sphere;
    b3Capsule capsule;
};

static const uint BOX_BLOB_BYTES = 640u; // sizeof(b3BoxHull), asserted by avbd_test_vk_hulls

uint alignUp16(uint v) { return (v + 15u) & ~15u; }

// b3BoxHull layout (types.h), identity transform. The topology tables are hull.c's s_boxHull.
static const uint BOX_EDGES[24] = {
    0x00020102u, 0x05010011u, 0x00010304u, 0x03050214u, 0x00050506u, 0x04060417u, 0x00060700u, 0x02020612u,
    0x0100090au, 0x05030815u, 0x01030b0cu, 0x02070a10u, 0x01070d0eu, 0x04040c13u, 0x01040f08u, 0x03000e16u,
    0x02031107u, 0x05021009u, 0x0206130bu, 0x04071205u, 0x0301150fu, 0x05001401u, 0x03041703u, 0x0405160du };

void buildBoxHull(uint8_t *blob, vec3 half)
{
    b3HullData *hd = (b3HullData *)blob;
    float hx = max(half.x, 0.0002f), hy = max(half.y, 0.0002f), hz = max(half.z, 0.0002f);
    hd->version = (uint64_t(0x4A4C9587u) << 32) | uint64_t(0xDE57485Cu);
    hd->hash = 0;
    hd->aabb = b3AABB(b3Vec3(-hx, -hy, -hz), b3Vec3(hx, hy, hz));
    hd->surfaceArea = 8.0f * (hx * hy + hx * hz + hy * hz);
    hd->volume = 8.0f * hx * hy * hz;
    hd->innerRadius = min(hx, min(hy, hz));
    hd->center = b3Vec3(0.0f, 0.0f, 0.0f);
    hd->vertexCount = 8;
    hd->vertexOffset = 144;
    hd->pointOffset = 152;
    hd->edgeCount = 24;
    hd->edgeOffset = 248;
    hd->faceCount = 6;
    hd->planeOffset = 344;
    hd->faceOffset = 440;
    hd->soaVertexOffset = 448;
    hd->soaNormalOffset = 544;
    hd->byteCount = 640;

    uint *w = (uint *)blob;
    w[36] = 0x09010008u | 0u;      // vertex edges 8,1,0,9
    w[36] = 8u | (1u << 8) | (0u << 16) | (9u << 24);
    w[37] = 13u | (3u << 8) | (5u << 16) | (11u << 24);
    for (int i = 0; i < 24; ++i)
        w[62 + i] = BOX_EDGES[i];
    w[110] = 0u | (8u << 8) | (16u << 16) | (20u << 24); // faces 0,8,16,20,19,21 + 2 pad bytes
    w[111] = 19u | (21u << 8);

    // points: (h,h,h) (-h,h,h) (-h,-h,h) (h,-h,h) (h,h,-h) (-h,h,-h) (-h,-h,-h) (h,-h,-h)
    float *pt = (float *)(blob + 152);
    float *vx = (float *)(blob + 448);
    float *vy = (float *)(blob + 480);
    float *vz = (float *)(blob + 512);
    for (int i = 0; i < 8; ++i)
    {
        float sx = (i == 1 || i == 2 || i == 5 || i == 6) ? -1.0f : 1.0f;
        float sy = (i == 2 || i == 3 || i == 6 || i == 7) ? -1.0f : 1.0f;
        float sz = (i >= 4) ? -1.0f : 1.0f;
        pt[i * 3 + 0] = sx * hx;
        pt[i * 3 + 1] = sy * hy;
        pt[i * 3 + 2] = sz * hz;
        vx[i] = sx * hx;
        vy[i] = sy * hy;
        vz[i] = sz * hz;
    }
    // planes: -X +X -Y +Y -Z +Z, offset = half extent
    float *pl = (float *)(blob + 344);
    float *nx = (float *)(blob + 544);
    float *ny = (float *)(blob + 576);
    float *nz = (float *)(blob + 608);
    for (int i = 0; i < 6; ++i)
    {
        int axis = i >> 1;
        float sgn = (i & 1) == 0 ? -1.0f : 1.0f;
        float nxv = axis == 0 ? sgn : 0.0f;
        float nyv = axis == 1 ? sgn : 0.0f;
        float nzv = axis == 2 ? sgn : 0.0f;
        pl[i * 4 + 0] = nxv;
        pl[i * 4 + 1] = nyv;
        pl[i * 4 + 2] = nzv;
        pl[i * 4 + 3] = axis == 0 ? hx : (axis == 1 ? hy : hz);
        nx[i] = nxv;
        ny[i] = nyv;
        nz[i] = nzv;
    }
    nx[6] = 0.0f; nx[7] = 0.0f;
    ny[6] = 0.0f; ny[7] = 0.0f;
    nz[6] = 0.0f; nz[7] = 0.0f;
}

// A hull blob from the geometry heap, or null when the offset is missing, misaligned, or the
// header does not describe a blob that fits inside the heap and the scratch arrays.
b3HullData *loadHull(NarrowphaseHullArgs *H, int shape)
{
    uint off = H->geomOffset[shape];
    if (off == 0xFFFFFFFFu || (off & 15u) != 0u)
        return nullptr;
    if ((uint64_t)off + 144u > H->geomHeapBytes)
        return nullptr;
    b3HullData *h = (b3HullData *)(H->geomHeap + off);
    if (h->version != ((uint64_t(0x4A4C9587u) << 32) | uint64_t(0xDE57485Cu)))
        return nullptr;
    const int bc = h->byteCount, vc = h->vertexCount, fc = h->faceCount, ec = h->edgeCount;
    if (bc < 144)
        return nullptr;
    if ((uint64_t)off + (uint64_t)bc > H->geomHeapBytes)
        return nullptr;
    if (vc < 1)
        return nullptr;
    if (vc > B3_MAX_HULL_VERTICES)
        return nullptr;
    if (fc < 1)
        return nullptr;
    if (fc > B3_MAX_HULL_FACES)
        return nullptr;
    if (ec < 1)
        return nullptr;
    if (ec > B3_MAX_HULL_EDGES)
        return nullptr;
    return h;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void csNarrowphaseHull(uint3 tid: SV_DispatchThreadID)
{
    NarrowphaseHullArgs *H = (NarrowphaseHullArgs *)pc.argsAddr;
    NarrowphaseArgs *args = &H->np;
    uint l = tid.x;
    if (l >= (uint)max(*args->hullCounter, 0))
        return;
    int t = args->hullList[l];
    if (t < 0 || (uint)t >= args->pairCount)
        return;
    // The slot's SAT record is zeroed first, so every early return below leaves a valid (empty)
    // cache for the next step rather than whatever the buffer held.
    H->satSepCur[t] = 0.0f;
    H->satBitsCur[t] = 0u;
    // The host grows the scratch and reruns the pass when the counter exceeded hullCapacity
    // (the last attempt commits clamped: such a pair then has no contacts).
    if (l >= H->hullCapacity)
        return;

    uint64_t key = args->pairs[t];
    int sa = pairLoS(key), sb = pairHiS(key);
    int ia = args->b.shapeBody[sa], ib = args->b.shapeBody[sb];
    // csNarrowphase wrote key/bodies/shapes and contactCount = 0 for this slot already.

    vec3 posA = args->b.pos[ia], posB = args->b.pos[ib];
    quat rotA = args->b.rot[ia], rotB = args->b.rot[ib];
    vec3 spA, spB;
    quat sqA, sqB;
    shapePoseNp(args->b, sa, ia, spA, sqA);
    shapePoseNp(args->b, sb, ib, spB, sqB);
    int shapeA = args->b.shapeType != nullptr ? (int)args->b.shapeType[sa] : AVBD_SHAPE_BOX;
    int shapeB = args->b.shapeType != nullptr ? (int)args->b.shapeType[sb] : AVBD_SHAPE_BOX;

    // The hull is the primary (A) of the Box3D call.
    const bool swapped = shapeA != AVBD_SHAPE_HULL;
    const int sP = swapped ? sb : sa;
    const int sO = swapped ? sa : sb;
    const int typeO = swapped ? shapeA : shapeB;
    const vec3 spP = swapped ? spB : spA;
    const quat sqP = swapped ? sqB : sqA;
    const vec3 spO = swapped ? spA : spB;
    const quat sqO = swapped ? sqA : sqB;
    const vec3 halfO = args->b.halfExtents[sO];

    // This lane's scratch slot, then the guard that everything fits in it.
    const uint auxOff = alignUp16((uint)sizeof(B3Scratch));
    const uint blobOff = alignUp16(auxOff + (uint)sizeof(B3HullAux));
    if (blobOff + BOX_BLOB_BYTES > H->scratchStride)
        return;
    uint8_t *slot = H->scratch + (uint64_t)l * (uint64_t)H->scratchStride;
    g_b3scr = (B3Scratch *)slot;
    B3HullAux *aux = (B3HullAux *)(slot + auxOff);
    uint8_t *boxBlob = slot + blobOff;

    b3HullData *hullP = loadHull(H, sP);
    if (hullP == nullptr)
        return;
    b3HullData *hullO = nullptr;
    if (typeO == AVBD_SHAPE_HULL)
    {
        hullO = loadHull(H, sO);
        if (hullO == nullptr)
            return;
    }
    else if (typeO == AVBD_SHAPE_BOX)
    {
        buildBoxHull(boxBlob, halfO);
        hullO = (b3HullData *)boxBlob;
    }

    int prevIdx = (args->prevCount > 0) ? findPrevManifoldS(args->prevKeys, (int)args->prevCount, key) : -1;

    b3LocalManifold *m = &aux->manifold;
    m->normal = b3Vec3(0.0f, 0.0f, 0.0f);
    m->pointCount = 0;
    m->squaredDistance = 0.0f;
    m->feature = 0;
    m->triangleIndex = -1;
    m->i1 = -1;
    m->i2 = -1;
    m->i3 = -1;
    m->triangleFlags = 0;
    m->triangleNormal = b3Vec3(0.0f, 0.0f, 0.0f);

    // Box3D's transformBtoA: the other shape expressed in the hull's frame.
    const quat invP = conjugateQ(sqP);
    const vec3 pBA = rotateQ(invP, sub(spO, spP));
    const quat qBA = mulQQ(invP, sqO);
    b3Transform xf = b3Transform(b3Vec3(pBA.x, pBA.y, pBA.z), b3Quat(b3Vec3(qBA.x, qBA.y, qBA.z), qBA.w));

    b3SATCache *satp = &aux->sat;
    satp->separation = 0.0f;
    satp->type = (uint8_t)0;
    satp->indexA = (uint8_t)0;
    satp->indexB = (uint8_t)0;
    satp->hit = (uint8_t)0;
    const bool usesSat = typeO == AVBD_SHAPE_HULL || typeO == AVBD_SHAPE_BOX;
    if (usesSat && prevIdx >= 0 && args->prev.contactCount[prevIdx] >= 0)
    {
        uint bits = H->satBitsPrev[prevIdx];
        satp->separation = H->satSepPrev[prevIdx];
        satp->type = (uint8_t)(bits & 255u);
        satp->indexA = (uint8_t)((bits >> 8) & 255u);
        satp->indexB = (uint8_t)((bits >> 16) & 255u);
        satp->hit = (uint8_t)((bits >> 24) & 255u);
    }

    b3SimplexCache *sc = &aux->simplexCache;
    sc->metric = 0.0f;
    sc->count = 0;
    for (int k = 0; k < 4; ++k)
    {
        sc->indexA[k] = (uint8_t)0;
        sc->indexB[k] = (uint8_t)0;
    }

    const int capacity = 4;
    if (usesSat)
    {
        b3CollideHulls(m, capacity, hullP, hullO, xf, satp);
    }
    else if (typeO == AVBD_SHAPE_CAPSULE)
    {
        const float r = halfO.x;
        const float hs = max(halfO.z - halfO.x, 0.0f);
        b3Capsule *cb = &aux->capsule;
        *cb = b3Capsule(b3Vec3(0.0f, 0.0f, -hs), b3Vec3(0.0f, 0.0f, hs), r);
        b3CollideHullAndCapsule(m, capacity, hullP, cb, xf, sc);
    }
    else
    {
        b3Sphere *sbp = &aux->sphere;
        *sbp = b3Sphere(b3Vec3(0.0f, 0.0f, 0.0f), halfO.x);
        b3CollideHullAndSphere(m, capacity, hullP, sbp, xf, sc);
    }

    // The SAT cache is stored every step (a miss included) for the next one.
    {
        uint bits = (uint)satp->type | ((uint)satp->indexA << 8) | ((uint)satp->indexB << 16) | ((uint)satp->hit << 24);
        H->satSepCur[t] = usesSat ? satp->separation : 0.0f;
        H->satBitsCur[t] = usesSat ? bits : 0u;
    }

    // Manifold -> contact rows. Box3D's point is the midpoint of the two surface points and the
    // normal runs hull -> other shape; separation < 0 is penetration. Points past our contact margin
    // (Box3D keeps speculative points out to 2 cm) are not rows, as in the other pair classes.
    const int nIn = clamp(m->pointCount, 0, 4);
    vec3 xA[4];
    vec3 xB[4];
    int featKey[4];
    int n = 0;
    const vec3 nW = rotateQ(sqP, vec3(m->normal.x, m->normal.y, m->normal.z));
    const vec3 normalAB = swapped ? neg(nW) : nW;
    for (int i = 0; i < nIn; ++i)
    {
        const float sep = m->points[i].separation;
        if (!(sep <= COLLISION_MARGIN_NH))
            continue;
        const vec3 pl = vec3(m->points[i].point.x, m->points[i].point.y, m->points[i].point.z);
        const vec3 nl = vec3(m->normal.x, m->normal.y, m->normal.z);
        const vec3 onP = add(spP, rotateQ(sqP, sub(pl, scale(nl, 0.5f * sep))));
        const vec3 onO = add(spP, rotateQ(sqP, add(pl, scale(nl, 0.5f * sep))));
        xA[n] = swapped ? onO : onP;
        xB[n] = swapped ? onP : onO;
        const b3FeaturePair fp = m->points[i].pair;
        featKey[n] = ((int)fp.owner1 << 24) | ((int)fp.index1 << 16) | ((int)fp.owner2 << 8) | (int)fp.index2;
        ++n;
    }
    if (n <= 0)
        return;

    const float3x3 basis = orthonormalF(neg(normalAB));
    const quat invRotA = conjugateQ(rotA);
    const quat invRotB = conjugateQ(rotB);
    vec3 nRA[4];
    vec3 nRB[4];
    for (int k = 0; k < n; ++k)
    {
        nRA[k] = rotateQ(invRotA, sub(xA[k], posA));
        nRB[k] = rotateQ(invRotB, sub(xB[k], posB));
    }

    vec3 penalty[4];
    vec3 lambda[4];
    bool stick[4];
    for (int k = 0; k < n; ++k)
    {
        penalty[k] = vec3(0, 0, 0);
        lambda[k] = vec3(0, 0, 0);
        stick[k] = false;
    }

    const float friction = sqrt(args->b.friction[sa] * args->b.friction[sb]);

    if (prevIdx >= 0)
    {
        int oldOffset = args->prev.contactOffset[prevIdx];
        int oldCount = args->prev.contactCount[prevIdx];

        // A hull and a box are material (a stuck anchor is kept); a sphere or capsule is round.
        const bool roundA = shapeA == AVBD_SHAPE_SPHERE || shapeA == AVBD_SHAPE_CAPSULE;
        const bool roundB = shapeB == AVBD_SHAPE_SPHERE || shapeB == AVBD_SHAPE_CAPSULE;

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < oldCount; ++j)
            {
                int o = oldOffset + j;
                if (args->prev.feature[o] != featKey[i])
                    continue;

                vec3 newRA = nRA[i];
                vec3 newRB = nRB[i];

                penalty[i] = args->prev.penalty[o];
                lambda[i] = args->prev.lambda[o];
                stick[i] = (args->prev.stick[o] & 1) != 0;
                nRA[i] = args->prev.rA[o];
                nRB[i] = args->prev.rB[o];
                if (!stick[i] || roundA || roundB)
                    nRA[i] = newRA;
                if (!stick[i] || roundA || roundB)
                    nRB[i] = newRB;
                break;
            }
        }
    }

    int offset = subgroupReserve(args->contactCounter, n);
    if (offset + n > args->contactCapacity)
        return; // Overflow; reported to the host, which grows rather than limping on.

    subgroupReserveOne(args->activeCounter);

    args->cur.basis[t] = basis;
    args->cur.friction[t] = friction;
    args->cur.contactOffset[t] = offset;
    args->cur.contactCount[t] = n;

    for (int i = 0; i < n; ++i)
    {
        int o = offset + i;

        vec3 pA = transformQ(posA, rotA, nRA[i]);
        vec3 pB = transformQ(posB, rotB, nRB[i]);
        vec3 C0 = add(matVec(basis, sub(pA, pB)), vec3(COLLISION_MARGIN_NH, 0.0f, 0.0f));

        vec3 lam = scale(lambda[i], args->p.alpha * args->p.gamma);
        vec3 pen = clampV(scale(penalty[i], args->p.gamma), PENALTY_MIN_NH, PENALTY_MAX_NH);

        vec3 kAuto = vec3(0, 0, 0);
        if (args->p.autoKStart != 0 || args->p.dualDamping > 0.0f)
        {
            const float invMA = args->b.mass[ia] > 0.0f ? 1.0f / args->b.mass[ia] : 0.0f;
            const float invMB = args->b.mass[ib] > 0.0f ? 1.0f / args->b.mass[ib] : 0.0f;
            const vec3 rAW = rotateQ(rotA, nRA[i]);
            const vec3 rBW = rotateQ(rotB, nRB[i]);
            const vec3 momA = args->b.moment[ia];
            const vec3 momB = args->b.moment[ib];

            for (int axis = 0; axis < 3; ++axis)
            {
                const vec3 nrm = rowAtS(basis, axis);
                const vec3 cA = cross3(rAW, nrm);
                const vec3 cB = cross3(rBW, nrm);
                float w = invMA + invMB;
                if (momA.x > 0.0f)
                    w += cA.x * cA.x / momA.x + cA.y * cA.y / momA.y + cA.z * cA.z / momA.z;
                if (momB.x > 0.0f)
                    w += cB.x * cB.x / momB.x + cB.y * cB.y / momB.y + cB.z * cB.z / momB.z;
                float kv = w > 0.0f ? clampf(1.0f / (w * args->p.dt * args->p.dt), PENALTY_MIN_NH, PENALTY_MAX_NH)
                                    : PENALTY_MIN_NH;
                if (axis == 0) kAuto.x = kv; else if (axis == 1) kAuto.y = kv; else kAuto.z = kv;
            }

            if (args->p.autoKStart != 0 && penalty[i].x <= 0.0f)
                pen = kAuto;
        }

        if (args->p.dualDamping > 0.0f)
        {
            const vec3 vA = add(args->b.velLin[ia], cross3(args->b.velAng[ia], rotateQ(rotA, nRA[i])));
            const vec3 vB = add(args->b.velLin[ib], cross3(args->b.velAng[ib], rotateQ(rotB, nRB[i])));
            const vec3 dv = sub(vA, vB);
            const float vRel = sqrt(dv.x * dv.x + dv.y * dv.y + dv.z * dv.z);
            const float gate = args->p.dualDampVel > 0.0f
                                   ? max(0.0f, 1.0f - vRel / args->p.dualDampVel)
                                   : 1.0f;
            const float strength = args->p.dualDamping * gate;
            if (strength > 0.0f)
            {
                float penX = pen.x, penY = pen.y, penZ = pen.z;
                float kAX = kAuto.x, kAY = kAuto.y, kAZ = kAuto.z;
                if (penX > kAX && kAX > 0.0f) lam.x *= 1.0f - strength * (1.0f - kAX / penX);
                if (penY > kAY && kAY > 0.0f) lam.y *= 1.0f - strength * (1.0f - kAY / penY);
                if (penZ > kAZ && kAZ > 0.0f) lam.z *= 1.0f - strength * (1.0f - kAZ / penZ);
            }
        }

        args->cur.feature[o] = featKey[i];
        args->cur.rA[o] = nRA[i];
        args->cur.rB[o] = nRB[i];
        args->cur.C0[o] = C0;
        args->cur.penalty[o] = pen;
        args->cur.lambda[o] = lam;
        args->cur.stick[o] = (uint8_t)(stick[i] ? 1 : 0);
    }
}
