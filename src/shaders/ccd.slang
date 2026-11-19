// Continuous collision. After the solve, a body that moved more than a fraction of its
// smallest half extent is swept against the geometry it may have tunnelled through, and its pose
// is clamped to the earliest time of impact. Same scheme as Box3D's b3SolveContinuous and the 2D
// solver's ccd2d.slang:
//   * a post-solve pose clamp (like csExplosionGuard), not a change to the solve; x^t (initialLin/Ang) is left alone and
//     the velocity loses only its component into the obstacle, so the next step integrates from
//     the clamped pose;
//   * a hit counts only when the solver's final pose is really past the contact: the shape's
//     centre must end beyond the contact point along the TOI normal, within the core distance.
//     An ordinary landing or stack ends on the near side and is left to the speculative contacts
//     and the penalty solve;
//   * a shape that already overlaps at the start of the sweep (TOI fraction 0) is retried as a
//     small core sphere at its centre, so only deep penetration counts;
//   * pass 0 sweeps non-bullet shapes against static shapes; pass 1 sweeps bullet shapes against
//     static shapes and against non-bullet dynamic shapes at their final pose.
//
// Kernels, run per pass: csCcdFlag (append the fast shapes), csCcdToi (a fixed set of worker
// lanes, each strides over the list: one TOI per candidate, the earliest accepted fraction per
// body is an atomic min on a packed fraction + normal word), csCcdApply (move the body to the TOI pose and reset
// its hit word to all-ones, so the buffer is all-ones between passes).
//
// Shapes swept: box, sphere, capsule and hull (proxy from the hull points in the geometry heap).
// Targets: those plus static triangle meshes (and height fields, cooked as meshes): the mesh BVH
// is walked with the swept bounding-sphere segment and each overlapping triangle is a 3-point
// proxy, at most CCD_MAX_TRIS per (fast shape, mesh); past the cap the rest are dropped (a
// documented cap, no flag, no rerun). The list is bounded by the dynamic
// shape count, so it cannot overflow and needs no rerun rule.

import maths;
import collide;
import types;
#include "b3/b3_gen.slang"

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

static const uint BODY_BULLET_C = 256u;
static const float CCD_CORE_FRACTION = 0.25f;
static const float CCD_PAD = 0.02f;
static const uint CCD_SPLIT = 16u; // chunks each fast shape's target scan is split into
static const uint CCD_MAX_PTS = 128u; // = B3_MAX_HULL_VERTICES: hull points / box corners per proxy
static const uint CCD_NONE = 0xFFFFFFFFu; // empty slot of the packed (fraction, normal) word

// Packed hit word: high 16 bits the fraction (floored, so the pose stays on the near side), low 16
// bits an octahedral unit normal (8 + 8). Ordering by fraction makes the earliest hit an atomic min.
uint ccdPack(float f, vec3 n)
{
    uint fq = clamp((uint)(f * 65535.0f), 1u, 65534u);
    float l1 = abs(n.x) + abs(n.y) + abs(n.z);
    float inv = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    float ox = n.x * inv, oy = n.y * inv;
    if (n.z < 0.0f)
    {
        float tx = (1.0f - abs(oy)) * (ox >= 0.0f ? 1.0f : -1.0f);
        float ty = (1.0f - abs(ox)) * (oy >= 0.0f ? 1.0f : -1.0f);
        ox = tx;
        oy = ty;
    }
    uint ux = (uint)clamp((ox * 0.5f + 0.5f) * 255.0f + 0.5f, 0.0f, 255.0f);
    uint uy = (uint)clamp((oy * 0.5f + 0.5f) * 255.0f + 0.5f, 0.0f, 255.0f);
    return (fq << 16) | (ux << 8) | uy;
}

vec3 ccdUnpackNormal(uint w)
{
    float ox = (float)((w >> 8) & 255u) / 255.0f * 2.0f - 1.0f;
    float oy = (float)(w & 255u) / 255.0f * 2.0f - 1.0f;
    float oz = 1.0f - abs(ox) - abs(oy);
    if (oz < 0.0f)
    {
        float tx = (1.0f - abs(oy)) * (ox >= 0.0f ? 1.0f : -1.0f);
        float ty = (1.0f - abs(ox)) * (oy >= 0.0f ? 1.0f : -1.0f);
        ox = tx;
        oy = ty;
    }
    vec3 v = vec3(ox, oy, oz);
    float l = length3(v);
    return l > 0.0f ? scale(v, 1.0f / l) : vec3(0.0f, 0.0f, 0.0f);
}

struct CcdArgs
{
    vec3 *pos;
    quat *rot;
    vec3 *velLin;
    vec3 *initLin;
    quat *initAng;
    float *mass;
    uint8_t *awake;
    int *sleepTimer;
    BodyProps *props;       // may be null (no body has a non-default property)
    int *shpBody;
    vec3 *shpLocalPos;
    quat *shpLocalRot;
    vec3 *shpHalf;
    uint8_t *shpType;
    uint *shpFlags;
    uint64_t *shpCategory;
    uint64_t *shpMask;
    int *shpGroup;
    int *dynShape;
    int *staticShape;
    uint64_t *ignore;       // may be null when ignoreCount == 0
    uint8_t *geomHeap;      // may be null (no hull or mesh in the scene)
    uint *geomOffset;       // may be null
    uint64_t geomHeapBytes;
    int *list;
    int *counter;
    uint *frac;
    uint8_t *scratch;
    uint dynCount;
    uint staticCount;
    uint ignoreCount;
    uint scratchStride;
    uint lanes;
    uint pass;              // 0: non-bullets vs statics, 1: bullets vs statics + non-bullet dynamics
    float safety;           // flag when the body moved more than safety * its smallest half extent
    uint _pad;
};

b3Vec3 toB3(vec3 v) { return b3Vec3(v.x, v.y, v.z); }

bool ccdFiniteV(vec3 v)
{
    return (v.x == v.x) && (v.y == v.y) && (v.z == v.z) && !isinf(v.x) && !isinf(v.y) && !isinf(v.z);
}

bool ccdIsBullet(CcdArgs *a, int body)
{
    return a->props != nullptr && (a->props[body].flags & BODY_BULLET_C) != 0u;
}

vec3 ccdShapeCenter(CcdArgs *a, int s)
{
    return (a->shpFlags[s] & 1u) != 0u ? vec3(0.0f, 0.0f, 0.0f) : a->shpLocalPos[s];
}

quat ccdShapeRot(CcdArgs *a, int s)
{
    return (a->shpFlags[s] & 1u) != 0u ? quat(0.0f, 0.0f, 0.0f, 1.0f) : a->shpLocalRot[s];
}

static const uint CCD_HULL_LO = 0x4A4C9587u;   // b3HullData version, high word
static const uint CCD_HULL_HI = 0xDE57485Cu;   // low word

// A hull blob from the geometry heap, or null when the shape has none that fits (the same range
// checks as narrowphase_mesh.slang loadHullMs).
b3HullData *ccdHull(CcdArgs *a, int s)
{
    if (a->geomHeap == nullptr || a->geomOffset == nullptr)
        return nullptr;
    uint off = a->geomOffset[s];
    if (off == 0xFFFFFFFFu || (off & 15u) != 0u)
        return nullptr;
    if ((uint64_t)off + 144u > a->geomHeapBytes)
        return nullptr;
    b3HullData *h = (b3HullData *)(a->geomHeap + off);
    if (h->version != ((uint64_t(CCD_HULL_LO) << 32) | uint64_t(CCD_HULL_HI)))
        return nullptr;
    const int bc = h->byteCount, vc = h->vertexCount;
    if (bc < 144 || (uint64_t)off + (uint64_t)bc > a->geomHeapBytes)
        return nullptr;
    if (vc < 1 || vc > B3_MAX_HULL_VERTICES)
        return nullptr;
    if ((uint64_t)h->pointOffset + (uint64_t)vc * 12u > (uint64_t)bc)
        return nullptr;
    return h;
}

// Convex shapes the sweep handles: box, sphere, capsule, and a hull that loads.
bool ccdSweepable(CcdArgs *a, int s, int type)
{
    if (type == AVBD_SHAPE_HULL)
        return ccdHull(a, s) != nullptr;
    return type == AVBD_SHAPE_BOX || type == AVBD_SHAPE_SPHERE || type == AVBD_SHAPE_CAPSULE;
}

// Smallest half extent of a shape (the "thickness" the flag test compares the travel with).
float ccdMinHalf(CcdArgs *a, int s, int type)
{
    vec3 h = a->shpHalf[s];
    if (type == AVBD_SHAPE_HULL)
    {
        b3HullData *hd = ccdHull(a, s);
        b3Vec3 e = b3Sub(hd->aabb.upperBound, hd->aabb.lowerBound);
        return 0.5f * min(e.x, min(e.y, e.z));
    }
    if (type == AVBD_SHAPE_BOX)
        return min(h.x, min(h.y, h.z));
    return h.x;
}

// Radius of a sphere around ccdBoundCenter that holds the whole shape.
float ccdBoundRadius(CcdArgs *a, int s, int type)
{
    vec3 h = a->shpHalf[s];
    if (type == AVBD_SHAPE_HULL)
    {
        b3HullData *hd = ccdHull(a, s);
        b3Vec3 e = b3Sub(hd->aabb.upperBound, hd->aabb.lowerBound);
        return 0.5f * sqrt(e.x * e.x + e.y * e.y + e.z * e.z);
    }
    if (type == AVBD_SHAPE_BOX)
        return length3(h);
    if (type == AVBD_SHAPE_CAPSULE)
        return max(h.z, h.x);
    return h.x;
}

// Body-frame centre of the bounding sphere (the shape's centre; for a hull its AABB centre).
vec3 ccdBoundCenter(CcdArgs *a, int s, int type)
{
    vec3 off = ccdShapeCenter(a, s);
    if (type == AVBD_SHAPE_HULL)
    {
        b3HullData *hd = ccdHull(a, s);
        b3Vec3 c = b3Lerp(hd->aabb.lowerBound, hd->aabb.upperBound, 0.5f);
        off = add(off, rotateQ(ccdShapeRot(a, s), vec3(c.x, c.y, c.z)));
    }
    return off;
}

// Shape s as a convex proxy in its body's frame. `pts` is CCD_MAX_PTS b3Vec3 of scratch (the box
// corners or the hull vertices).
void ccdBuildProxy(CcdArgs *a, int s, b3Vec3 *pts, out b3ShapeProxy pr)
{
    pr.points = nullptr;
    pr.count = 1;
    pr.radius = 0.0f;
    pr.p0 = b3Vec3(0.0f, 0.0f, 0.0f);
    pr.p1 = b3Vec3(0.0f, 0.0f, 0.0f);
    int type = (int)a->shpType[s];
    vec3 h = a->shpHalf[s];
    vec3 off = ccdShapeCenter(a, s);
    quat q = ccdShapeRot(a, s);
    if (type == AVBD_SHAPE_HULL)
    {
        b3HullData *hd = ccdHull(a, s);
        int n = hd->vertexCount;
        b3Vec3 *hp = b3GetHullPoints(hd);
        for (int i = 0; i < n; i++)
            pts[i] = toB3(add(off, rotateQ(q, vec3(hp[i].x, hp[i].y, hp[i].z))));
        pr.points = pts;
        pr.count = n;
        return;
    }
    if (type == AVBD_SHAPE_BOX)
    {
        for (int i = 0; i < 8; i++)
        {
            vec3 c = vec3((i & 1) != 0 ? h.x : -h.x, (i & 2) != 0 ? h.y : -h.y, (i & 4) != 0 ? h.z : -h.z);
            pts[i] = toB3(add(off, rotateQ(q, c)));
        }
        pr.points = pts;
        pr.count = 8;
        return;
    }
    if (type == AVBD_SHAPE_CAPSULE)
    {
        vec3 ax = scale(rotateQ(q, vec3(0.0f, 0.0f, 1.0f)), max(h.z - h.x, 0.0f));
        pr.p0 = toB3(sub(off, ax));
        pr.p1 = toB3(add(off, ax));
        pr.count = 2;
        pr.radius = h.x;
        return;
    }
    pr.p0 = toB3(off);
    pr.count = 1;
    pr.radius = h.x;
}

bool ccdIgnored(CcdArgs *a, int bi, int bj)
{
    if (a->ignoreCount == 0u)
        return false;
    int lo = bi < bj ? bi : bj;
    int hi = bi < bj ? bj : bi;
    uint64_t key = ((uint64_t)(uint)lo << 32) | (uint)hi;
    int l = 0;
    int r = (int)a->ignoreCount - 1;
    while (l <= r)
    {
        int mid = (l + r) >> 1;
        uint64_t v = a->ignore[mid];
        if (v == key)
            return true;
        if (v < key)
            l = mid + 1;
        else
            r = mid - 1;
    }
    return false;
}

// The narrowphase's filter (narrowphase.slang): category/mask, shared group override.
bool ccdFiltered(CcdArgs *a, int sa, int sb)
{
    int gA = a->shpGroup[sa];
    int gB = a->shpGroup[sb];
    if (gA != 0 && gA == gB)
        return gA < 0;
    return (a->shpCategory[sa] & a->shpMask[sb]) == 0ull || (a->shpCategory[sb] & a->shpMask[sa]) == 0ull;
}

bool ccdFastBody(CcdArgs *a, int i)
{
    return a->mass[i] > 0.0f && a->awake[i] != 0 && a->sleepTimer[i] >= 0 &&
           (a->props == nullptr || (a->props[i].flags & BODY_KINEMATIC) == 0u);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csCcdFlag(uint3 tid: SV_DispatchThreadID)
{
    CcdArgs *a = (CcdArgs *)pc.argsAddr;
    uint t = tid.x;
    if (t >= a->dynCount)
        return;
    int s = a->dynShape[t];
    int body = a->shpBody[s];
    if (!ccdFastBody(a, body))
        return;
    if (ccdIsBullet(a, body) != (a->pass == 1u))
        return;
    int type = (int)a->shpType[s];
    if (!ccdSweepable(a, s, type))
        return;
    vec3 p0 = a->initLin[body];
    vec3 p1 = a->pos[body];
    quat dq = mulQQ(a->rot[body], conjugateQ(a->initAng[body]));
    float ang = 2.0f * length3(vec3(dq.x, dq.y, dq.z));
    float move = length3(sub(p1, p0)) + ang * ccdBoundRadius(a, s, type);
    if (!(move == move) || isinf(move) || !(move > a->safety * ccdMinHalf(a, s, type)))
        return;
    int slot = 0;
    InterlockedAdd(*a->counter, 1, slot);
    a->list[slot] = s;
}

// One TOI of the fast proxy pA against a stationary proxy pB; lowers `best` on an accepted hit.
// bcA is A's bounding-sphere centre in its body frame (the core sphere sits there) and w2 its
// world position at the end of the sweep.
void ccdSweepProxy(b3ShapeProxy pA, b3ShapeProxy pB, b3Sweep swA, b3Sweep swB, vec3 bcA, vec3 w2, float coreR,
                   inout float best, inout vec3 bestN)
{
    b3TOIInput inp;
    inp.proxyA = pA;
    inp.proxyB = pB;
    inp.sweepA = swA;
    inp.sweepB = swB;
    inp.maxFraction = best;
    b3TOIOutput o = b3TimeOfImpact(inp);
    if (o.state == b3_toiStateSeparated)
        return;
    if (!(o.state == b3_toiStateHit && o.fraction > 0.0f))
    {
        // Overlapping at the start of the sweep: retry with a small core sphere at the shape's
        // centre, so only deep penetration counts.
        b3ShapeProxy core;
        core.points = nullptr;
        core.count = 1;
        core.radius = coreR;
        core.p0 = toB3(bcA);
        core.p1 = core.p0;
        inp.proxyA = core;
        o = b3TimeOfImpact(inp);
        if (o.state != b3_toiStateHit)
            return;
    }
    if (!(o.fraction > 0.0f && o.fraction < best))
        return;
    // Only a final pose past the contact (within the core distance of it, or beyond) counts.
    vec3 n = vec3(o.normal.x, o.normal.y, o.normal.z);
    vec3 pt = vec3(o.point.x, o.point.y, o.point.z);
    if (dot3(sub(w2, pt), n) > -coreR)
    {
        best = o.fraction;
        bestN = n;
    }
}

b3Sweep ccdStaticSweep(vec3 posB, quat rotB)
{
    b3Sweep swB;
    swB.localCenter = b3Vec3(0.0f, 0.0f, 0.0f);
    swB.c1 = toB3(posB);
    swB.c2 = swB.c1;
    swB.q1 = b3Quat(b3Vec3(rotB.x, rotB.y, rotB.z), rotB.w);
    swB.q2 = swB.q1;
    return swB;
}

// ---- static triangle meshes: the b3MeshData BVH in the geometry heap, read as narrowphase_mesh.slang
// does (mirror of its MeshView / loadMeshM; change together) ----
static const uint CCD_MESH_HEADER_BYTES = 96u;
static const uint CCD_MESH_NODE_BYTES = 32u;
static const uint CCD_MESH_TRI_BYTES = 12u;
static const uint CCD_MESH_VERSION_LO = 0xF1A8AAF7u;
static const uint CCD_MESH_VERSION_HI = 0xAAAB9A00u;
static const uint CCD_MESH_STACK = 64u;
// Triangles swept per (fast shape, mesh). Documented cap: past it the rest of the traversal is
// dropped (no flag, no rerun).
static const int CCD_MAX_TRIS = 256;

struct CcdMesh
{
    bool ok;
    uint8_t *base;
    uint nodeOff;
    uint nodeCount;
    uint vertOff;
    uint vertCount;
    uint triOff;
    uint triCount;
};

uint ccdRdU(uint8_t *b, uint o) { return *(uint *)(b + o); }

CcdMesh ccdLoadMesh(CcdArgs *a, int s)
{
    CcdMesh v;
    v.ok = false;
    v.base = nullptr;
    v.nodeOff = 0u; v.nodeCount = 0u; v.vertOff = 0u; v.vertCount = 0u; v.triOff = 0u; v.triCount = 0u;
    if (a->geomHeap == nullptr || a->geomOffset == nullptr)
        return v;
    uint off = a->geomOffset[s];
    if (off == 0xFFFFFFFFu || (off & 15u) != 0u)
        return v;
    if ((uint64_t)off + (uint64_t)CCD_MESH_HEADER_BYTES > a->geomHeapBytes)
        return v;
    uint8_t *b = a->geomHeap + off;
    if (ccdRdU(b, 0) != CCD_MESH_VERSION_LO || ccdRdU(b, 4) != CCD_MESH_VERSION_HI)
        return v;
    const uint byteCount = ccdRdU(b, 16);
    if (byteCount < CCD_MESH_HEADER_BYTES || (uint64_t)off + (uint64_t)byteCount > a->geomHeapBytes)
        return v;
    const uint nOff = ccdRdU(b, 56), nCnt = ccdRdU(b, 60);
    const uint vOff = ccdRdU(b, 64), vCnt = ccdRdU(b, 68);
    const uint tOff = ccdRdU(b, 72), tCnt = ccdRdU(b, 76);
    if (nCnt < 1u || (uint64_t)nOff + (uint64_t)nCnt * CCD_MESH_NODE_BYTES > byteCount)
        return v;
    if (vCnt < 1u || (uint64_t)vOff + (uint64_t)vCnt * 12u > byteCount)
        return v;
    if (tCnt < 1u || (uint64_t)tOff + (uint64_t)tCnt * CCD_MESH_TRI_BYTES > byteCount)
        return v;
    v.ok = true;
    v.base = b;
    v.nodeOff = nOff; v.nodeCount = nCnt; v.vertOff = vOff; v.vertCount = vCnt; v.triOff = tOff; v.triCount = tCnt;
    return v;
}

// Slab test of one axis: narrows [t0, t1] to where p + t d lies in [lo, hi].
bool ccdSlab(float p, float d, float lo, float hi, inout float t0, inout float t1)
{
    if (abs(d) < 1e-9f)
        return p >= lo && p <= hi;
    float inv = 1.0f / d;
    float ta = (lo - p) * inv;
    float tb = (hi - p) * inv;
    if (ta > tb)
    {
        float t = ta;
        ta = tb;
        tb = t;
    }
    t0 = max(t0, ta);
    t1 = min(t1, tb);
    return t0 <= t1;
}

// Does the segment p -> q meet the box [lo, hi]?
bool ccdSegBox(vec3 p, vec3 q, vec3 lo, vec3 hi)
{
    float t0 = 0.0f, t1 = 1.0f;
    vec3 d = sub(q, p);
    return ccdSlab(p.x, d.x, lo.x, hi.x, t0, t1) && ccdSlab(p.y, d.y, lo.y, hi.y, t0, t1) &&
           ccdSlab(p.z, d.z, lo.z, hi.z, t0, t1);
}

// Sweep A against the triangles of static mesh shape sB whose boxes meet A's bounding-sphere path
// (the segment w1 -> w2, inflated by rA).
[noinline]
void ccdTestMesh(CcdArgs *a, int sB, b3Vec3 *ptsB, vec3 posB, quat rotB, b3ShapeProxy pA, b3Sweep swA,
                 vec3 bcA, vec3 w1, vec3 w2, float rA, float coreR, inout float best, inout vec3 bestN)
{
    CcdMesh mv = ccdLoadMesh(a, sB);
    if (!mv.ok)
        return;
    vec3 mOff = ccdShapeCenter(a, sB);
    quat mQ = ccdShapeRot(a, sB);
    quat invBody = conjugateQ(rotB);
    quat invShape = conjugateQ(mQ);
    vec3 m1 = rotateQ(invShape, sub(rotateQ(invBody, sub(w1, posB)), mOff));
    vec3 m2 = rotateQ(invShape, sub(rotateQ(invBody, sub(w2, posB)), mOff));
    float infl = rA + CCD_PAD;
    vec3 iv = vec3(infl, infl, infl);
    b3Sweep swB = ccdStaticSweep(posB, rotB);

    int stack[CCD_MESH_STACK];
    int sp = 0;
    stack[sp++] = 0;
    int tested = 0;
    while (sp > 0)
    {
        const int n = stack[--sp];
        if ((uint)n >= mv.nodeCount)
            continue;
        uint *nw = (uint *)(mv.base + mv.nodeOff + (uint)n * CCD_MESH_NODE_BYTES);
        float *nf = (float *)nw;
        if (!ccdSegBox(m1, m2, sub(vec3(nf[0], nf[1], nf[2]), iv), add(vec3(nf[4], nf[5], nf[6]), iv)))
            continue;
        const uint w3 = nw[3];
        if ((w3 & 3u) == 3u)
        {
            const uint cnt = w3 >> 2;
            const uint first = nw[7];
            for (uint i = 0u; i < cnt; ++i)
            {
                const uint tri = first + i;
                if (tri >= mv.triCount)
                    continue;
                vec3 v[3];
                bool valid = true;
                for (int k = 0; k < 3; ++k)
                {
                    const int vi = *(int *)(mv.base + mv.triOff + tri * CCD_MESH_TRI_BYTES + (uint)k * 4u);
                    if (vi < 0 || (uint)vi >= mv.vertCount)
                    {
                        valid = false;
                        break;
                    }
                    float *fp = (float *)(mv.base + mv.vertOff + (uint)vi * 12u);
                    v[k] = vec3(fp[0], fp[1], fp[2]);
                }
                if (!valid)
                    continue;
                vec3 tlo = vec3(min(v[0].x, min(v[1].x, v[2].x)), min(v[0].y, min(v[1].y, v[2].y)),
                                min(v[0].z, min(v[1].z, v[2].z)));
                vec3 thi = vec3(max(v[0].x, max(v[1].x, v[2].x)), max(v[0].y, max(v[1].y, v[2].y)),
                                max(v[0].z, max(v[1].z, v[2].z)));
                if (!ccdSegBox(m1, m2, sub(tlo, iv), add(thi, iv)))
                    continue;
                if (++tested > CCD_MAX_TRIS)
                    return;
                for (int k = 0; k < 3; ++k)
                    ptsB[k] = toB3(add(mOff, rotateQ(mQ, v[k])));
                b3ShapeProxy pB;
                pB.points = ptsB;
                pB.count = 3;
                pB.radius = 0.0f;
                pB.p0 = b3Vec3(0.0f, 0.0f, 0.0f);
                pB.p1 = pB.p0;
                ccdSweepProxy(pA, pB, swA, swB, bcA, w2, coreR, best, bestN);
            }
        }
        else
        {
            const int right = n + (int)(w3 >> 2);
            if (sp + 2 <= (int)CCD_MESH_STACK)
            {
                stack[sp++] = right;
                stack[sp++] = n + 1;
            }
        }
    }
}

// Sweep fast shape sA (body bA) against target shape sB; lowers `best` on an accepted hit.
void ccdTestPair(CcdArgs *a, int sA, int bA, int sB, b3Vec3 *ptsB, b3ShapeProxy pA, b3Sweep swA, float coreR,
                 inout float best, inout vec3 bestN)
{
    int bB = a->shpBody[sB];
    if (bB == bA)
        return;
    int typeB = (int)a->shpType[sB];
    if (typeB != AVBD_SHAPE_MESH && !ccdSweepable(a, sB, typeB))
        return;
    if (ccdFiltered(a, sA, sB) || ccdIgnored(a, bA, bB))
        return;

    vec3 posB = a->pos[bB];
    quat rotB = a->rot[bB];
    if (!ccdFiniteV(posB))
        return;

    // Bounding-sphere test: the segment travelled by A's bounding centre against B's sphere.
    int typeA = (int)a->shpType[sA];
    vec3 bcA = ccdBoundCenter(a, sA, typeA);
    vec3 w1 = add(a->initLin[bA], rotateQ(a->initAng[bA], bcA));
    vec3 w2 = add(a->pos[bA], rotateQ(a->rot[bA], bcA));
    float rA = ccdBoundRadius(a, sA, typeA);

    if (typeB == AVBD_SHAPE_MESH)
    {
        ccdTestMesh(a, sB, ptsB, posB, rotB, pA, swA, bcA, w1, w2, rA, coreR, best, bestN);
        return;
    }

    vec3 cB = add(posB, rotateQ(rotB, ccdBoundCenter(a, sB, typeB)));
    float rB = ccdBoundRadius(a, sB, typeB) + CCD_PAD;
    vec3 d = sub(w2, w1);
    float dd = lengthSq3(d);
    float u = dd > 0.0f ? clampf(dot3(sub(cB, w1), d) / dd, 0.0f, 1.0f) : 0.0f;
    vec3 closest = add(w1, scale(d, u));
    float reach = rA + rB;
    if (lengthSq3(sub(cB, closest)) > reach * reach)
        return;

    b3ShapeProxy pB;
    ccdBuildProxy(a, sB, ptsB, pB);
    ccdSweepProxy(pA, pB, swA, ccdStaticSweep(posB, rotB), bcA, w2, coreR, best, bestN);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csCcdToi(uint3 tid: SV_DispatchThreadID)
{
    CcdArgs *a = (CcdArgs *)pc.argsAddr;
    uint lane = tid.x;
    if (lane >= a->lanes)
        return;
    int n = a->counter[0];
    if (n <= 0)
        return;

    // This lane's scratch slot: [B3Scratch][CCD_MAX_PTS points of A][CCD_MAX_PTS points of B].
    const uint ptsOff = (((uint)sizeof(B3Scratch)) + 15u) & ~15u;
    if (ptsOff + 2u * CCD_MAX_PTS * 12u + 32u > a->scratchStride)
        return;
    uint8_t *slot = a->scratch + (uint64_t)lane * (uint64_t)a->scratchStride;
    g_b3scr = (B3Scratch *)slot;
    b3Vec3 *ptsA = (b3Vec3 *)(slot + ptsOff);
    b3Vec3 *ptsB = ptsA + CCD_MAX_PTS;

    // Work item = (listed shape, chunk of the target lists): a few fast bodies still fill the lanes.
    for (int item = (int)lane; item < n * CCD_SPLIT; item += (int)a->lanes)
    {
        const int li = item / CCD_SPLIT;
        const uint chunk = (uint)(item % CCD_SPLIT);
        int sA = a->list[li];
        int bA = a->shpBody[sA];
        vec3 p1 = a->pos[bA];
        quat q1 = a->initAng[bA];
        quat q2 = a->rot[bA];
        if (!ccdFiniteV(p1) || !ccdFiniteV(a->initLin[bA]))
            continue;
        if (q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w < 0.0f)
            q2 = quat(-q2.x, -q2.y, -q2.z, -q2.w);

        b3Sweep swA;
        swA.localCenter = b3Vec3(0.0f, 0.0f, 0.0f);
        swA.c1 = toB3(a->initLin[bA]);
        swA.c2 = toB3(p1);
        swA.q1 = b3Quat(b3Vec3(q1.x, q1.y, q1.z), q1.w);
        swA.q2 = b3Quat(b3Vec3(q2.x, q2.y, q2.z), q2.w);

        b3ShapeProxy pA;
        ccdBuildProxy(a, sA, ptsA, pA);
        float coreR = CCD_CORE_FRACTION * ccdMinHalf(a, sA, (int)a->shpType[sA]);

        float best = 1.0f;
        vec3 bestN = vec3(0.0f, 0.0f, 0.0f);
        for (uint k = chunk; k < a->staticCount; k += CCD_SPLIT)
            ccdTestPair(a, sA, bA, a->staticShape[k], ptsB, pA, swA, coreR, best, bestN);

        if (a->pass == 1u)
        {
            for (uint k = chunk; k < a->dynCount; k += CCD_SPLIT)
            {
                int sB = a->dynShape[k];
                int bB = a->shpBody[sB];
                if (ccdIsBullet(a, bB) || a->mass[bB] <= 0.0f)
                    continue;
                ccdTestPair(a, sA, bA, sB, ptsB, pA, swA, coreR, best, bestN);
            }
        }

        if (best < 1.0f)
        {
            InterlockedMin(a->frac[bA], ccdPack(best, bestN));
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csCcdApply(uint3 tid: SV_DispatchThreadID)
{
    CcdArgs *a = (CcdArgs *)pc.argsAddr;
    int t = (int)tid.x;
    if (t >= a->counter[0])
        return;
    int body = a->shpBody[a->list[t]];
    uint old = 0u;
    InterlockedExchange(a->frac[body], CCD_NONE, old);
    if ((old >> 16) >= 65535u)
        return;
    float f = (float)(old >> 16) / 65535.0f;

    vec3 p0 = a->initLin[body];
    vec3 p1 = a->pos[body];
    quat q0 = a->initAng[body];
    quat q1 = a->rot[body];
    if (q0.x * q1.x + q0.y * q1.y + q0.z * q1.z + q0.w * q1.w < 0.0f)
        q1 = quat(-q1.x, -q1.y, -q1.z, -q1.w);
    a->pos[body] = add(scale(p0, 1.0f - f), scale(p1, f));
    a->rot[body] = normalizeQ(addQQ(scaleQ(q0, 1.0f - f), scaleQ(q1, f)));

    // Inelastic stop: drop the velocity component into the obstacle (the contact solver cannot
    // absorb a whole step of travel in a few iterations).
    vec3 n = ccdUnpackNormal(old);
    vec3 v = a->velLin[body];
    float vn = dot3(v, n);
    if (vn > 0.0f)
        a->velLin[body] = sub(v, scale(n, vn));
}
