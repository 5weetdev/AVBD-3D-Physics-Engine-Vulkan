// static triangle mesh pairs. csNarrowphase (narrowphase.slang) does not collide a pair with
// a mesh shape: a mesh against a dynamic convex shape appends its pair slot to meshList. Then:
//   csMeshArgs        indirect dispatch for the midphase, from the list length
//   csMeshMidphase    one lane per listed pair walks the Box3D mesh BVH (b3MeshData in the geometry
//                     heap) and appends one child key per triangle whose AABB meets the convex AABB
//   csMeshChildCount  clamps the appended count to childCapacity (the sort and csMeshChild read it)
//   (radix sort of the child keys, VkNarrowphase::record)
//   csMeshChild       fixed dispatch of kMeshWorkers lanes; sorted child i is manifold slot
//                     childBase + i and runs the Box3D triangle-vs-convex collision (b3_gen.slang)
//   csMeshChildFilter same lanes; Box3D's ghost-collision filter across each pair's children
// A scene without a mesh shape never records any of these kernels.
//
// Child key layout (uint64):
//   bit 63       tag, always 1 (a pair key has bit 63 clear, so the two never sort together)
//   bits 42..62  lo shape index (21 bits), the smaller shape index of the pair
//   bits 21..41  hi shape index (21 bits), the larger shape index
//   bits  0..20  triangle index (21 bits), into the mesh's triangle array
// SceneBuilder refuses a scene whose shape count or a mesh's triangle count is >= 2^21 - 1.
//
// Manifold slots: pair slots are [0, pairCapacity); child slot childBase + i (childBase = pairCapacity).
// A child slot's key is its child key, so the next step finds the previous child by binary search over
// prev.key[prevChildBase, prevChildBase + prevChildCount).
//
// Frames: the triangle is shape A of the Box3D call (normal triangle -> convex) and the convex shape's
// own frame is the Box3D frame (convex at identity), as mesh_contact.c does. Points and the normal are
// mapped back to world space and to the pair's lo/hi order at the end.
//
// Not ported (phase notes): cross-triangle filtering (foundEdges/foundVertices), b3CullPoints,
// b3ReduceCluster, a per-triangle contact cache. Each child starts with an empty SAT and simplex cache.
//
// Memory safety (this shader is not checked by GPU-AV): every mesh header, node, triangle, vertex and flag
// read is range checked against the blob's byteCount and the heap size; a traversal stack that would
// overflow drops the push (recorded as a known gap, see the phase notes); a child lane writes only its
// own scratch slot; an append at or past childCapacity is only counted.

import maths;
import collide;
#include "b3/b3_gen.slang"

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

static const float COLLISION_MARGIN_MS = 0.01f;   // collision.h: COLLISION_MARGIN
static const float PENALTY_MIN_MS = 1.0f;
static const float PENALTY_MAX_MS = 10000000000.0f;
// The triangle AABB test widens by this much so a triangle about to touch is not culled before Box3D's
// speculative points (2 cm, COLLISION_MARGIN plus the speculative distance) can report it.
static const float MESH_AABB_MARGIN = 0.05f;
static const uint MESH_STACK = 128;
static const uint MESH_HEADER_BYTES = 96u;
static const uint MESH_NODE_BYTES = 32u;
static const uint MESH_TRI_BYTES = 12u;
static const uint MESH_VERSION_LO = 0xF1A8AAF7u; // B3_MESH_VERSION 0xAAAB9A00F1A8AAF7 (little endian words)
static const uint MESH_VERSION_HI = 0xAAAB9A00u;
static const uint64_t CHILD_TAG = uint64_t(1) << 63;
static const uint BOX_BLOB_BYTES_MS = 640u; // sizeof(b3BoxHull), asserted by avbd_test_vk_hulls

// ---- mirrors of narrowphase.slang (change together; layout_probe checks NarrowphaseArgs) ----------
int pairLoS(uint64_t k) { return (int)(uint)(k >> 32); }
int pairHiS(uint64_t k) { return (int)(uint)(k & uint64_t(0xFFFFFFFFu)); }

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

// Counted append for divergent lanes (the child kernel's lanes do not share control flow, so the
// subgroup forms above are not used there).
int atomicReserveMs(int *counter, int n)
{
    int base = 0;
    InterlockedAdd(*counter, n, base);
    return base;
}

uint64_t makeChildKey(int lo, int hi, uint tri)
{
    return CHILD_TAG | ((uint64_t)(uint)lo << 42) | ((uint64_t)(uint)hi << 21) | (uint64_t)tri;
}

int findPrevChildMs(uint64_t *keys, uint base, uint n, uint64_t want)
{
    int lo = 0, hi = (int)n - 1;
    while (lo <= hi)
    {
        int mid = (lo + hi) >> 1;
        uint64_t k = keys[base + (uint)mid];
        if (k == want)
            return mid;
        if (k < want)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return -1;
}

vec3 matVecMs(float3x3 m, vec3 v) { return vec3(dot3(m.row0, v), dot3(m.row1, v), dot3(m.row2, v)); }
vec3 rowAtMs(float3x3 m, int i) { return i == 0 ? m.row0 : (i == 1 ? m.row1 : m.row2); }
vec3 clampVMs(vec3 v, float lo, float hi) { return vec3(clampf(v.x, lo, hi), clampf(v.y, lo, hi), clampf(v.z, lo, hi)); }

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
    // Phase 2 fields, unused here. Must stay in step with narrowphase.slang's NpBodySoA (byte copy).
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
    int *meshList;
    int *meshCounter;
};

// vk_args.h: NarrowphaseMeshArgsGpu (csMeshMidphase)
struct NarrowphaseMeshArgs
{
    NarrowphaseArgs np;
    uint8_t *geomHeap;
    uint *geomOffset;       // per shape, 0xFFFFFFFF none
    uint64_t geomHeapBytes;
    uint64_t *childKeys;
    int *childCounter;
    uint childCapacity;
    uint meshCount;         // lanes that may hold a list entry (the pair capacity)
};

// vk_args.h: NarrowphaseMeshChildArgsGpu (csMeshChild)
struct NarrowphaseMeshChildArgs
{
    NarrowphaseArgs np;
    uint8_t *geomHeap;
    uint *geomOffset;
    uint8_t *scratch;       // kMeshWorkers slots of scratchStride bytes
    uint64_t *childSorted;
    int *childCounter;      // clamped count (csMeshChildCount)
    uint64_t *childInfo;    // per sorted child (the dead unsorted keys): see meshChildClassMs
    uint64_t geomHeapBytes;
    uint childCapacity;
    uint childBase;
    uint prevChildBase;
    uint prevChildCount;
    uint workers;
    uint scratchStride;
};

// vk_args.h: NarrowphaseMeshDispatchArgsGpu (csMeshArgs and csMeshChildCount)
struct NarrowphaseMeshDispatchArgs
{
    int *counter;
    uint *indirect;
    uint capacity;
    uint groupSize;
};

// ---- geometry heap: b3MeshData and b3MeshNode read by byte offset (mesh.c, types.h) -------------
// b3MeshData: version u64 @0, hash u64 @8, byteCount i32 @16, bounds @20..44, surfaceArea @44,
// treeHeight @48, degenerateCount @52, nodeOffset @56, nodeCount @60, vertexOffset @64,
// vertexCount @68, triangleOffset @72, triangleCount @76, materialOffset @80, materialCount @84,
// flagsOffset @88. b3MeshNode (32 bytes): lowerBound words 0-2, word 3 = (type or axis) in bits 0-1
// and (childOffset or triangleCount) in bits 2-31, upperBound words 4-6, triangleOffset word 7.
struct MeshView
{
    bool ok;
    uint8_t *base;
    uint nodeOff;
    uint nodeCount;
    uint vertOff;
    uint vertCount;
    uint triOff;
    uint triCount;
    uint8_t *flags;   // null when the blob has no flags array
};

uint rdU(uint8_t *b, uint o) { return *(uint *)(b + o); }
float rdF(uint8_t *b, uint o) { return *(float *)(b + o); }

MeshView loadMeshM(uint8_t *heap, uint64_t heapBytes, uint *geomOffset, int shape)
{
    MeshView v;
    v.ok = false;
    v.base = nullptr;
    v.nodeOff = 0u;
    v.nodeCount = 0u;
    v.vertOff = 0u;
    v.vertCount = 0u;
    v.triOff = 0u;
    v.triCount = 0u;
    v.flags = nullptr;
    uint off = geomOffset[shape];
    if (off == 0xFFFFFFFFu || (off & 15u) != 0u)
        return v;
    if ((uint64_t)off + (uint64_t)MESH_HEADER_BYTES > heapBytes)
        return v;
    uint8_t *b = heap + off;
    if (rdU(b, 0) != MESH_VERSION_LO || rdU(b, 4) != MESH_VERSION_HI)
        return v;
    const uint byteCount = rdU(b, 16);
    if (byteCount < MESH_HEADER_BYTES || (uint64_t)off + (uint64_t)byteCount > heapBytes)
        return v;
    const uint nOff = rdU(b, 56), nCnt = rdU(b, 60);
    const uint vOff = rdU(b, 64), vCnt = rdU(b, 68);
    const uint tOff = rdU(b, 72), tCnt = rdU(b, 76);
    const uint fOff = rdU(b, 88);
    if (nCnt < 1u || (uint64_t)nOff + (uint64_t)nCnt * MESH_NODE_BYTES > byteCount)
        return v;
    if (vCnt < 1u || (uint64_t)vOff + (uint64_t)vCnt * 12u > byteCount)
        return v;
    if (tCnt < 1u || (uint64_t)tOff + (uint64_t)tCnt * MESH_TRI_BYTES > byteCount)
        return v;
    v.ok = true;
    v.base = b;
    v.nodeOff = nOff;
    v.nodeCount = nCnt;
    v.vertOff = vOff;
    v.vertCount = vCnt;
    v.triOff = tOff;
    v.triCount = tCnt;
    if (fOff != 0u && (uint64_t)fOff + (uint64_t)tCnt <= (uint64_t)byteCount)
        v.flags = b + fOff;
    return v;
}

// Mesh-frame vertex i of the blob.
vec3 meshVertex(MeshView v, int i)
{
    float *p = (float *)(v.base + v.vertOff + (uint)i * 12u);
    return vec3(p[0], p[1], p[2]);
}

int meshTriIndex(MeshView v, uint tri, int k)
{
    return *(int *)(v.base + v.triOff + tri * MESH_TRI_BYTES + (uint)k * 4u);
}

// ---- convex shapes -------------------------------------------------------------------------------
// b3BoxHull layout (types.h), identity transform. The topology tables are hull.c's s_boxHull.
static const uint BOX_EDGES_MS[24] = {
    0x00020102u, 0x05010011u, 0x00010304u, 0x03050214u, 0x00050506u, 0x04060417u, 0x00060700u, 0x02020612u,
    0x0100090au, 0x05030815u, 0x01030b0cu, 0x02070a10u, 0x01070d0eu, 0x04040c13u, 0x01040f08u, 0x03000e16u,
    0x02031107u, 0x05021009u, 0x0206130bu, 0x04071205u, 0x0301150fu, 0x05001401u, 0x03041703u, 0x0405160du };

uint alignUp16m(uint v) { return (v + 15u) & ~15u; }

void buildBoxHullMs(uint8_t *blob, vec3 half)
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
    w[36] = 8u | (1u << 8) | (0u << 16) | (9u << 24);
    w[37] = 13u | (3u << 8) | (5u << 16) | (11u << 24);
    for (int i = 0; i < 24; ++i)
        w[62 + i] = BOX_EDGES_MS[i];
    w[110] = 0u | (8u << 8) | (16u << 16) | (20u << 24);
    w[111] = 19u | (21u << 8);

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

// A hull blob from the geometry heap, or null when it does not describe a blob that fits.
b3HullData *loadHullMs(uint8_t *heap, uint64_t heapBytes, uint *geomOffset, int shape)
{
    uint off = geomOffset[shape];
    if (off == 0xFFFFFFFFu || (off & 15u) != 0u)
        return nullptr;
    if ((uint64_t)off + 144u > heapBytes)
        return nullptr;
    b3HullData *h = (b3HullData *)(heap + off);
    if (h->version != ((uint64_t(0x4A4C9587u) << 32) | uint64_t(0xDE57485Cu)))
        return nullptr;
    const int bc = h->byteCount, vc = h->vertexCount, fc = h->faceCount, ec = h->edgeCount;
    if (bc < 144)
        return nullptr;
    if ((uint64_t)off + (uint64_t)bc > heapBytes)
        return nullptr;
    if (vc < 1 || vc > B3_MAX_HULL_VERTICES)
        return nullptr;
    if (fc < 1 || fc > B3_MAX_HULL_FACES)
        return nullptr;
    if (ec < 1 || ec > B3_MAX_HULL_EDGES)
        return nullptr;
    return h;
}

int shapeTypeOfMs(NarrowphaseArgs *args, int s)
{
    return args->b.shapeType != nullptr ? (int)args->b.shapeType[s] : AVBD_SHAPE_BOX;
}

// World pose of shape s on body `body`: the body pose for an identity shape, else composed with the
// local pose.
void shapePoseMs(NpBodySoA b, int s, int body, out vec3 sp, out quat sq)
{
    sp = b.pos[body];
    sq = b.rot[body];
    if ((b.flags[s] & 1u) == 0u)
    {
        sp = add(sp, rotateQ(sq, b.localPos[s]));
        sq = mulQQ(sq, b.localRot[s]);
    }
}

// Half extents of the AABB of a box with local half extents h, rotated by q (|R| h).
vec3 absRotHalfMs(quat q, vec3 h)
{
    vec3 cx = rotateQ(q, vec3(1.0f, 0.0f, 0.0f));
    vec3 cy = rotateQ(q, vec3(0.0f, 1.0f, 0.0f));
    vec3 cz = rotateQ(q, vec3(0.0f, 0.0f, 1.0f));
    return vec3(abs(cx.x) * h.x + abs(cy.x) * h.y + abs(cz.x) * h.z,
                abs(cx.y) * h.x + abs(cy.y) * h.y + abs(cz.y) * h.z,
                abs(cx.z) * h.x + abs(cy.z) * h.y + abs(cz.z) * h.z);
}

bool aabbOverlapMs(vec3 c0, vec3 h0, vec3 c1, vec3 h1)
{
    return abs(c0.x - c1.x) <= h0.x + h1.x && abs(c0.y - c1.y) <= h0.y + h1.y &&
           abs(c0.z - c1.z) <= h0.z + h1.z;
}

// ---- per-lane scratch ---------------------------------------------------------------------------
// One slot: [B3Scratch][B3MeshAux][b3BoxHull blob], each 16-byte aligned. Locals have no address on
// device, so everything the b3 entry points take by pointer lives here, the triangle included.
struct B3MeshAux
{
    b3LocalManifold manifold;
    b3SimplexCache simplexCache;
    b3SATCache sat;
    b3Sphere sphere;
    b3Capsule capsule;
    b3Vec3 tri[3];
};

// ---- kernels ------------------------------------------------------------------------------------

[shader("compute")]
[numthreads(1, 1, 1)]
void csMeshArgs(uint3 tid: SV_DispatchThreadID)
{
    NarrowphaseMeshDispatchArgs *a = (NarrowphaseMeshDispatchArgs *)pc.argsAddr;
    uint n = (uint)max(*a->counter, 0);
    n = min(n, a->capacity);
    a->indirect[0] = (n + a->groupSize - 1u) / a->groupSize;
    a->indirect[1] = 1u;
    a->indirect[2] = 1u;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void csMeshChildCount(uint3 tid: SV_DispatchThreadID)
{
    NarrowphaseMeshDispatchArgs *a = (NarrowphaseMeshDispatchArgs *)pc.argsAddr;
    int n = min(max(*a->counter, 0), (int)a->capacity);
    a->indirect[0] = (uint)n;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void csMeshMidphase(uint3 tid: SV_DispatchThreadID)
{
    NarrowphaseMeshArgs *M = (NarrowphaseMeshArgs *)pc.argsAddr;
    NarrowphaseArgs *args = &M->np;
    uint l = tid.x;
    if (l >= (uint)max(*args->meshCounter, 0) || l >= M->meshCount)
        return;
    int t = args->meshList[l];
    if (t < 0 || (uint)t >= args->pairCount)
        return;

    uint64_t key = args->pairs[t];
    int sa = pairLoS(key), sb = pairHiS(key);
    int meshS = shapeTypeOfMs(args, sa) == AVBD_SHAPE_MESH ? sa : sb;
    int convS = meshS == sa ? sb : sa;

    MeshView mv = loadMeshM(M->geomHeap, M->geomHeapBytes, M->geomOffset, meshS);
    if (!mv.ok)
        return;

    // The convex shape's AABB in the mesh's local frame: centre rotated into it, half extents |R| h.
    vec3 spM, spC;
    quat sqM, sqC;
    shapePoseMs(args->b, meshS, args->b.shapeBody[meshS], spM, sqM);
    shapePoseMs(args->b, convS, args->b.shapeBody[convS], spC, sqC);
    const vec3 cLocal = rotateQ(conjugateQ(sqM), sub(spC, spM));
    const vec3 hLocal = absRotHalfMs(mulQQ(conjugateQ(sqM), sqC), args->b.halfExtents[convS]);
    const vec3 hPad = add(hLocal, vec3(MESH_AABB_MARGIN, MESH_AABB_MARGIN, MESH_AABB_MARGIN));

    int stack[MESH_STACK];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0)
    {
        const int n = stack[--sp];
        if ((uint)n >= mv.nodeCount)
            continue;
        uint *nw = (uint *)(mv.base + mv.nodeOff + (uint)n * MESH_NODE_BYTES);
        float *nf = (float *)nw;
        const vec3 nlo = vec3(nf[0], nf[1], nf[2]);
        const vec3 nhi = vec3(nf[4], nf[5], nf[6]);
        const vec3 nc = scale(add(nlo, nhi), 0.5f);
        const vec3 nh = scale(sub(nhi, nlo), 0.5f);
        if (!aabbOverlapMs(nc, nh, cLocal, hPad))
            continue;

        const uint w3 = nw[3];
        if ((w3 & 3u) == 3u)
        {
            const uint cnt = w3 >> 2;
            const uint first = nw[7];
            for (uint i = 0; i < cnt; ++i)
            {
                const uint tri = first + i;
                if (tri >= mv.triCount)
                    continue;
                // Triangle AABB, mesh-local.
                vec3 tlo = vec3(0.0f, 0.0f, 0.0f), thi = vec3(0.0f, 0.0f, 0.0f);
                bool valid = true;
                for (int k = 0; k < 3; ++k)
                {
                    const int vi = meshTriIndex(mv, tri, k);
                    if (vi < 0 || (uint)vi >= mv.vertCount)
                    {
                        valid = false;
                        break;
                    }
                    const vec3 p = meshVertex(mv, vi);
                    if (k == 0)
                    {
                        tlo = p;
                        thi = p;
                    }
                    else
                    {
                        tlo = vec3(min(tlo.x, p.x), min(tlo.y, p.y), min(tlo.z, p.z));
                        thi = vec3(max(thi.x, p.x), max(thi.y, p.y), max(thi.z, p.z));
                    }
                }
                if (!valid)
                    continue;
                const vec3 tc = scale(add(tlo, thi), 0.5f);
                const vec3 th = scale(sub(thi, tlo), 0.5f);
                if (!aabbOverlapMs(tc, th, cLocal, hPad))
                    continue;
                const int slot = atomicReserveMs(M->childCounter, 1);
                if ((uint)slot < M->childCapacity)
                    M->childKeys[slot] = makeChildKey(min(sa, sb), max(sa, sb), tri);
            }
        }
        else
        {
            // Internal node: left child is n + 1, right child is n + childOffset. A push that would
            // overflow the stack is dropped (the traversal stays in bounds; see the header).
            const int right = n + (int)(w3 >> 2);
            if (sp + 2 <= (int)MESH_STACK)
            {
                stack[sp++] = right;
                stack[sp++] = n + 1;
            }
        }
    }
}

// childInfo[i], written by csMeshChild and read by csMeshChildFilter: bits 32-33 the class below (0 =
// no manifold), bits 34-37 the b3TriangleFeature, bits 0-31 the manifold's squared distance (float).
// Box3D's triangle classification (mesh_contact.c): 1 = accepted (a face feature, or a hull face whose
// normal is within 60 degrees of the triangle's, or a deep overlap), 2 = tentative (an edge or vertex
// feature, which may be a ghost collision with an internal edge).
uint meshChildClassMs(b3LocalManifold *m, b3Vec3 v1, b3Vec3 v2, b3Vec3 v3)
{
    if (m->feature == 1) // b3_featureTriangleFace
        return 1u;
    if (m->feature == 2) // b3_featureHullFace
    {
        const vec3 e1 = vec3(v2.x - v1.x, v2.y - v1.y, v2.z - v1.z);
        const vec3 e2 = vec3(v3.x - v1.x, v3.y - v1.y, v3.z - v1.z);
        const vec3 c = cross3(e1, e2);
        const float len = sqrt(c.x * c.x + c.y * c.y + c.z * c.z);
        const float cosA = len > 0.0f ? (c.x * m->normal.x + c.y * m->normal.y + c.z * m->normal.z) / len : 0.0f;
        if (cosA > 0.5f)
            return 1u;
        float minSep = m->points[0].separation;
        for (int i = 1; i < clamp(m->pointCount, 1, 4); ++i)
            minSep = min(minSep, m->points[i].separation);
        if (minSep < -2.0f * B3_LINEAR_SLOP)
            return 1u;
    }
    return 2u;
}

// One sorted child: triangle `key` of the mesh pair against its convex shape. Writes manifold slot
// childBase + i. Early returns leave contactCount 0 (set first).
void childLaneMs(NarrowphaseMeshChildArgs *C, NarrowphaseArgs *args, uint i, uint lane)
{
    const uint slot = C->childBase + i;
    const uint64_t key = C->childSorted[i];
    const int lo = (int)((key >> 42) & 0x1FFFFFull);
    const int hi = (int)((key >> 21) & 0x1FFFFFull);
    const uint tri = (uint)(key & 0x1FFFFFull);

    args->cur.key[slot] = key;
    args->cur.shapeA[slot] = lo;
    args->cur.shapeB[slot] = hi;
    const int ia = args->b.shapeBody[lo], ib = args->b.shapeBody[hi];
    args->cur.bodyA[slot] = ia;
    args->cur.bodyB[slot] = ib;
    args->cur.contactCount[slot] = 0;
    C->childInfo[i] = 0;

    const int shapeA = shapeTypeOfMs(args, lo);
    const int shapeB = shapeTypeOfMs(args, hi);
    const bool meshIsA = shapeA == AVBD_SHAPE_MESH;
    const int meshS = meshIsA ? lo : hi;
    const int convS = meshIsA ? hi : lo;
    const int typeC = meshIsA ? shapeB : shapeA;
    // Only mesh-vs-convex children exist; anything else is a corrupt key and writes nothing.
    if (meshIsA == (shapeB == AVBD_SHAPE_MESH))
        return;

    const uint auxOff = alignUp16m((uint)sizeof(B3Scratch));
    const uint blobOff = alignUp16m(auxOff + (uint)sizeof(B3MeshAux));
    if (blobOff + BOX_BLOB_BYTES_MS > C->scratchStride)
        return;
    uint8_t *slotMem = C->scratch + (uint64_t)lane * (uint64_t)C->scratchStride;
    g_b3scr = (B3Scratch *)slotMem;
    B3MeshAux *aux = (B3MeshAux *)(slotMem + auxOff);
    uint8_t *boxBlob = slotMem + blobOff;

    MeshView mv = loadMeshM(C->geomHeap, C->geomHeapBytes, C->geomOffset, meshS);
    if (!mv.ok)
        return;
    if (tri >= mv.triCount)
        return;

    vec3 spM, spC;
    quat sqM, sqC;
    shapePoseMs(args->b, meshS, args->b.shapeBody[meshS], spM, sqM);
    shapePoseMs(args->b, convS, args->b.shapeBody[convS], spC, sqC);

    const vec3 halfC = args->b.halfExtents[convS];
    const quat invC = conjugateQ(sqC);

    // The triangle in the convex shape's frame (Box3D frame, convex at identity).
    for (int k = 0; k < 3; ++k)
    {
        const int vi = meshTriIndex(mv, tri, k);
        if (vi < 0 || (uint)vi >= mv.vertCount)
            return;
        const vec3 vw = add(spM, rotateQ(sqM, meshVertex(mv, vi)));
        const vec3 vc = rotateQ(invC, sub(vw, spC));
        aux->tri[k] = b3Vec3(vc.x, vc.y, vc.z);
    }
    const int triFlags = mv.flags != nullptr ? (int)mv.flags[tri] : 0;

    // Box3D's manifold scratch (the output) and its caches for this child.
    b3LocalManifold *m = &aux->manifold;
    m->normal = b3Vec3(0.0f, 0.0f, 0.0f);
    m->pointCount = 0;
    m->squaredDistance = 0.0f;
    m->feature = 0;
    m->triangleIndex = (int)tri;
    m->i1 = -1;
    m->i2 = -1;
    m->i3 = -1;
    m->triangleFlags = triFlags;
    m->triangleNormal = b3Vec3(0.0f, 0.0f, 0.0f);

    b3SATCache *satp = &aux->sat;
    satp->separation = 0.0f;
    satp->type = (uint8_t)0;
    satp->indexA = (uint8_t)0;
    satp->indexB = (uint8_t)0;
    satp->hit = (uint8_t)0;
    b3SimplexCache *sc = &aux->simplexCache;
    sc->metric = 0.0f;
    sc->count = 0;
    for (int k = 0; k < 4; ++k)
    {
        sc->indexA[k] = (uint8_t)0;
        sc->indexB[k] = (uint8_t)0;
    }

    const int capacity = 4;
    if (typeC == AVBD_SHAPE_HULL)
    {
        b3HullData *hullC = loadHullMs(C->geomHeap, C->geomHeapBytes, C->geomOffset, convS);
        if (hullC == nullptr)
            return;
        b3CollideTriangleAndHull(m, capacity, aux->tri[0], aux->tri[1], aux->tri[2], triFlags, hullC, satp, true);
    }
    else if (typeC == AVBD_SHAPE_BOX)
    {
        buildBoxHullMs(boxBlob, halfC);
        b3CollideTriangleAndHull(m, capacity, aux->tri[0], aux->tri[1], aux->tri[2], triFlags,
                                 (b3HullData *)boxBlob, satp, true);
    }
    else if (typeC == AVBD_SHAPE_CAPSULE)
    {
        const float r = halfC.x;
        const float hs = max(halfC.z - halfC.x, 0.0f);
        b3Capsule *cb = &aux->capsule;
        *cb = b3Capsule(b3Vec3(0.0f, 0.0f, -hs), b3Vec3(0.0f, 0.0f, hs), r);
        b3CollideTriangleAndCapsule(m, capacity, &aux->tri[0], cb, sc);
    }
    else if (typeC == AVBD_SHAPE_SPHERE)
    {
        b3Sphere *sp = &aux->sphere;
        *sp = b3Sphere(b3Vec3(0.0f, 0.0f, 0.0f), halfC.x);
        b3CollideTriangleAndSphere(m, capacity, &aux->tri[0], sp);
    }
    else
    {
        return;
    }

    if (m->pointCount > 0)
    {
        const uint cls = meshChildClassMs(m, aux->tri[0], aux->tri[1], aux->tri[2]);
        C->childInfo[i] = ((uint64_t)(cls | ((uint)m->feature << 2)) << 32) | (uint64_t)asuint(m->squaredDistance);
    }

    // Manifold -> contact rows. The manifold lives in the convex frame, its normal runs triangle ->
    // convex (A -> B). Points past the contact margin are not rows.
    const int nIn = clamp(m->pointCount, 0, 4);
    const vec3 nL = vec3(m->normal.x, m->normal.y, m->normal.z);
    const vec3 nW = rotateQ(sqC, nL);
    const vec3 normalAB = meshIsA ? nW : neg(nW);
    vec3 xA[4];
    vec3 xB[4];
    int featKey[4];
    int n = 0;
    for (int p = 0; p < nIn; ++p)
    {
        const float sep = m->points[p].separation;
        if (!(sep <= COLLISION_MARGIN_MS))
            continue;
        const vec3 pl = vec3(m->points[p].point.x, m->points[p].point.y, m->points[p].point.z);
        const vec3 onMeshL = sub(pl, scale(nL, 0.5f * sep));
        const vec3 onConvL = add(pl, scale(nL, 0.5f * sep));
        const vec3 onMesh = add(spC, rotateQ(sqC, onMeshL));
        const vec3 onConv = add(spC, rotateQ(sqC, onConvL));
        xA[n] = meshIsA ? onMesh : onConv;
        xB[n] = meshIsA ? onConv : onMesh;
        const b3FeaturePair fp = m->points[p].pair;
        featKey[n] = ((int)fp.owner1 << 24) | ((int)fp.index1 << 16) | ((int)fp.owner2 << 8) | (int)fp.index2;
        ++n;
    }
    if (n <= 0)
        return;

    // Warm start: the previous step's child with the same key, if it existed.
    int prevIdx = -1;
    if (C->prevChildCount > 0)
    {
        const int r = findPrevChildMs(args->prev.key, C->prevChildBase, C->prevChildCount, key);
        if (r >= 0)
            prevIdx = (int)C->prevChildBase + r;
    }

    const float3x3 basis = orthonormalF(neg(normalAB));
    const vec3 posA = args->b.pos[ia], posB = args->b.pos[ib];
    const quat rotA = args->b.rot[ia], rotB = args->b.rot[ib];
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

    const float friction = sqrt(args->b.friction[lo] * args->b.friction[hi]);

    if (prevIdx >= 0)
    {
        const int oldOffset = args->prev.contactOffset[prevIdx];
        const int oldCount = args->prev.contactCount[prevIdx];
        // A mesh and a box are material (a stuck anchor is kept); a sphere or capsule is round.
        const bool roundA = shapeA == AVBD_SHAPE_SPHERE || shapeA == AVBD_SHAPE_CAPSULE;
        const bool roundB = shapeB == AVBD_SHAPE_SPHERE || shapeB == AVBD_SHAPE_CAPSULE;
        for (int k = 0; k < n; ++k)
        {
            for (int j = 0; j < oldCount; ++j)
            {
                const int o = oldOffset + j;
                if (args->prev.feature[o] != featKey[k])
                    continue;
                const vec3 newRA = nRA[k];
                const vec3 newRB = nRB[k];
                penalty[k] = args->prev.penalty[o];
                lambda[k] = args->prev.lambda[o];
                stick[k] = (args->prev.stick[o] & 1) != 0;
                nRA[k] = args->prev.rA[o];
                nRB[k] = args->prev.rB[o];
                if (!stick[k] || roundA || roundB)
                    nRA[k] = newRA;
                if (!stick[k] || roundA || roundB)
                    nRB[k] = newRB;
                break;
            }
        }
    }

    const int offset = atomicReserveMs(args->contactCounter, n);
    if (offset + n > args->contactCapacity)
        return; // Overflow; reported to the host, which grows and reruns.
    atomicReserveMs(args->activeCounter, 1);

    args->cur.basis[slot] = basis;
    args->cur.friction[slot] = friction;
    args->cur.contactOffset[slot] = offset;
    args->cur.contactCount[slot] = n;

    for (int k = 0; k < n; ++k)
    {
        const int o = offset + k;
        const vec3 pA = transformQ(posA, rotA, nRA[k]);
        const vec3 pB = transformQ(posB, rotB, nRB[k]);
        const vec3 C0 = add(matVecMs(basis, sub(pA, pB)), vec3(COLLISION_MARGIN_MS, 0.0f, 0.0f));

        vec3 lam = scale(lambda[k], args->p.alpha * args->p.gamma);
        vec3 pen = clampVMs(scale(penalty[k], args->p.gamma), PENALTY_MIN_MS, PENALTY_MAX_MS);

        vec3 kAuto = vec3(0, 0, 0);
        if (args->p.autoKStart != 0 || args->p.dualDamping > 0.0f)
        {
            const float invMA = args->b.mass[ia] > 0.0f ? 1.0f / args->b.mass[ia] : 0.0f;
            const float invMB = args->b.mass[ib] > 0.0f ? 1.0f / args->b.mass[ib] : 0.0f;
            const vec3 rAW = rotateQ(rotA, nRA[k]);
            const vec3 rBW = rotateQ(rotB, nRB[k]);
            const vec3 momA = args->b.moment[ia];
            const vec3 momB = args->b.moment[ib];
            for (int axis = 0; axis < 3; ++axis)
            {
                const vec3 nrm = rowAtMs(basis, axis);
                const vec3 cA = cross3(rAW, nrm);
                const vec3 cB = cross3(rBW, nrm);
                float w = invMA + invMB;
                if (momA.x > 0.0f)
                    w += cA.x * cA.x / momA.x + cA.y * cA.y / momA.y + cA.z * cA.z / momA.z;
                if (momB.x > 0.0f)
                    w += cB.x * cB.x / momB.x + cB.y * cB.y / momB.y + cB.z * cB.z / momB.z;
                const float kv = w > 0.0f ? clampf(1.0f / (w * args->p.dt * args->p.dt), PENALTY_MIN_MS, PENALTY_MAX_MS)
                                          : PENALTY_MIN_MS;
                if (axis == 0) kAuto.x = kv; else if (axis == 1) kAuto.y = kv; else kAuto.z = kv;
            }
            if (args->p.autoKStart != 0 && penalty[k].x <= 0.0f)
                pen = kAuto;
        }

        if (args->p.dualDamping > 0.0f)
        {
            const vec3 vA = add(args->b.velLin[ia], cross3(args->b.velAng[ia], rotateQ(rotA, nRA[k])));
            const vec3 vB = add(args->b.velLin[ib], cross3(args->b.velAng[ib], rotateQ(rotB, nRB[k])));
            const vec3 dv = sub(vA, vB);
            const float vRel = sqrt(dv.x * dv.x + dv.y * dv.y + dv.z * dv.z);
            const float gate = args->p.dualDampVel > 0.0f
                                   ? max(0.0f, 1.0f - vRel / args->p.dualDampVel)
                                   : 1.0f;
            const float strength = args->p.dualDamping * gate;
            if (strength > 0.0f)
            {
                const float penX = pen.x, penY = pen.y, penZ = pen.z;
                const float kAX = kAuto.x, kAY = kAuto.y, kAZ = kAuto.z;
                if (penX > kAX && kAX > 0.0f) lam.x *= 1.0f - strength * (1.0f - kAX / penX);
                if (penY > kAY && kAY > 0.0f) lam.y *= 1.0f - strength * (1.0f - kAY / penY);
                if (penZ > kAZ && kAZ > 0.0f) lam.z *= 1.0f - strength * (1.0f - kAZ / penZ);
            }
        }

        args->cur.feature[o] = featKey[k];
        args->cur.rA[o] = nRA[k];
        args->cur.rB[o] = nRB[k];
        args->cur.C0[o] = C0;
        args->cur.penalty[o] = pen;
        args->cur.lambda[o] = lam;
        args->cur.stick[o] = (uint8_t)(stick[k] ? 1 : 0);
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void csMeshChild(uint3 tid: SV_DispatchThreadID)
{
    NarrowphaseMeshChildArgs *C = (NarrowphaseMeshChildArgs *)pc.argsAddr;
    NarrowphaseArgs *args = &C->np;
    const uint g = tid.x;
    if (g >= C->workers)
        return;
    const uint n = (uint)max(*C->childCounter, 0);
    for (uint i = g; i < C->childCapacity; i += C->workers)
    {
        if (i >= n)
        {
            // Dead slot: no contacts, so every consumer sized by capacity skips it.
            const uint slot = C->childBase + i;
            args->cur.contactCount[slot] = 0;
            args->cur.key[slot] = 0;
            continue;
        }
        childLaneMs(C, args, i, g);
    }
}

// ---- ghost-collision filter (mesh_contact.c, after the per-triangle collide) ----------------------
// Box3D accepts face manifolds outright and treats edge and vertex manifolds as tentative: a convex
// shape's tentative manifold is dropped when its triangle's edges are all flat, or when one of its flat
// edges belongs to an accepted triangle; a sphere's is dropped when its feature edge or vertex belongs
// to an accepted triangle or to a closer tentative one. Without this a box sliding on a flat grid
// catches the internal edges. Every sibling's class is read, never written, so lanes do not race.

bool triHasVertexMs(int t0, int t1, int t2, int v)
{
    return t0 == v || t1 == v || t2 == v;
}

bool triHasEdgeMs(int t0, int t1, int t2, int a, int b)
{
    return triHasVertexMs(t0, t1, t2, a) && triHasVertexMs(t0, t1, t2, b);
}

void filterChildMs(NarrowphaseMeshChildArgs *C, NarrowphaseArgs *args, uint i, uint n)
{
    const uint64_t info = C->childInfo[i];
    if (((uint)(info >> 32) & 3u) != 2u)
        return;
    const uint slot = C->childBase + i;
    if (args->cur.contactCount[slot] <= 0)
        return;
    const uint64_t key = C->childSorted[i];
    const uint64_t pairKey = key >> 21;
    const int lo = (int)((key >> 42) & 0x1FFFFFull);
    const int hi = (int)((key >> 21) & 0x1FFFFFull);
    const uint tri = (uint)(key & 0x1FFFFFull);
    const bool meshIsA = shapeTypeOfMs(args, lo) == AVBD_SHAPE_MESH;
    const int meshS = meshIsA ? lo : hi;
    const int typeC = shapeTypeOfMs(args, meshIsA ? hi : lo);

    MeshView mv = loadMeshM(C->geomHeap, C->geomHeapBytes, C->geomOffset, meshS);
    if (!mv.ok || tri >= mv.triCount)
        return;
    int t[3];
    for (int k = 0; k < 3; ++k)
        t[k] = meshTriIndex(mv, tri, k);

    // The pair's children are contiguous in the sorted list.
    uint first = i, last = i;
    while (first > 0u && (C->childSorted[first - 1u] >> 21) == pairKey)
        --first;
    while (last + 1u < n && (C->childSorted[last + 1u] >> 21) == pairKey)
        ++last;

    bool drop = false;
    if (typeC == AVBD_SHAPE_SPHERE)
    {
        // b3_featureEdge1..3 = 3..5 (v1-v2, v2-v3, v3-v1), b3_featureVertex1..3 = 6..8.
        const uint feature = (uint)(info >> 34) & 15u;
        if (feature < 3u || feature > 8u)
            return;
        const bool isEdge = feature <= 5u;
        const int e = (int)(isEdge ? feature - 3u : feature - 6u);
        const int a = t[e];
        const int b = t[(e + 1) % 3];
        const float mySq = asfloat((uint)info);
        for (uint j = first; j <= last && !drop; ++j)
        {
            if (j == i)
                continue;
            const uint64_t infoJ = C->childInfo[j];
            const uint clsJ = (uint)(infoJ >> 32) & 3u;
            if (clsJ == 0u)
                continue;
            if (clsJ == 2u)
            {
                // Box3D visits tentative triangles closest first (ties here by sorted index).
                const float sqJ = asfloat((uint)infoJ);
                if (!(sqJ < mySq || (sqJ == mySq && j < i)))
                    continue;
            }
            const uint triJ = (uint)(C->childSorted[j] & 0x1FFFFFull);
            if (triJ >= mv.triCount)
                continue;
            const int u0 = meshTriIndex(mv, triJ, 0), u1 = meshTriIndex(mv, triJ, 1), u2 = meshTriIndex(mv, triJ, 2);
            drop = isEdge ? triHasEdgeMs(u0, u1, u2, a, b) : triHasVertexMs(u0, u1, u2, a);
        }
    }
    else
    {
        // b3_flatEdge1..3 = 0x11, 0x22, 0x44 (v1-v2, v2-v3, v3-v1).
        const uint flags = mv.flags != nullptr ? (uint)mv.flags[tri] : 0u;
        if ((flags & 0x77u) == 0x77u)
            drop = true;
        for (int e = 0; e < 3 && !drop; ++e)
        {
            const uint flat = 0x11u << (uint)e;
            if ((flags & flat) != flat)
                continue;
            const int a = t[e];
            const int b = t[(e + 1) % 3];
            for (uint j = first; j <= last && !drop; ++j)
            {
                if (j == i || ((uint)(C->childInfo[j] >> 32) & 3u) != 1u)
                    continue;
                const uint triJ = (uint)(C->childSorted[j] & 0x1FFFFFull);
                if (triJ >= mv.triCount)
                    continue;
                drop = triHasEdgeMs(meshTriIndex(mv, triJ, 0), meshTriIndex(mv, triJ, 1), meshTriIndex(mv, triJ, 2), a, b);
            }
        }
    }
    if (drop)
    {
        args->cur.contactCount[slot] = 0; // its reserved contact rows are left unused
        atomicReserveMs(args->activeCounter, -1);
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void csMeshChildFilter(uint3 tid: SV_DispatchThreadID)
{
    NarrowphaseMeshChildArgs *C = (NarrowphaseMeshChildArgs *)pc.argsAddr;
    NarrowphaseArgs *args = &C->np;
    const uint g = tid.x;
    if (g >= C->workers)
        return;
    const uint n = min((uint)max(*C->childCounter, 0), C->childCapacity);
    for (uint i = g; i < n; i += C->workers)
        filterChildMs(C, args, i, n);
}
