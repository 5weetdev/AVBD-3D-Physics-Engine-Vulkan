// Phase 3b: manifold builder
// (Narrowphase::build's k_narrowphase). Same structure: run collideShapes (imported
// from collide.slang, not duplicated -- see that file's module header), merge the
// previous step's dual state by feature key (persistent contact cache / warm start),
// compute C0 at q-, apply the alpha/gamma warm-start decay (Eq. 19), the auto-k
// effective-mass floor and the dual-damping rule, exactly as the reference source, same
// order. Host orchestration (double-buffered cur/prev, growth) is
// engine/vk/vk_narrowphase.h/.cpp.

import maths;
import collide;
import types;

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

static const float COLLISION_MARGIN_NP = 0.01f;   // collision.h: COLLISION_MARGIN
static const float PENALTY_MIN_NP = 1.0f;         // collision.h: PENALTY_MIN
static const float PENALTY_MAX_NP = 10000000000.0f; // collision.h: PENALTY_MAX

int pairLoS(uint64_t k) { return (int)(uint)(k >> 32); }
int pairHiS(uint64_t k) { return (int)(uint)(k & uint64_t(0xFFFFFFFFu)); }

// Subgroup-aggregated contiguous-slot reservation: subgroup equivalent of
// gpu_warp's warpReserve<MaxN>. WavePrefixSum/WaveActiveSum operate on the actual
// int contribution per lane, so this needs no per-bit ballot loop the way the reference
// bitmask version does -- same result (each lane gets a contiguous run of `n` slots
// starting at a warp/subgroup-shared base), simpler because HLSL's subgroup ops take
// the value directly.
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

// ---------------------------------------------------------------------------
// Minimal body SoA -- the fields k_narrowphase actually reads (pos/rot/halfExtents/
// shape/friction/mass/moment/velLin/velAng), mirroring collide.slang's approach of a
// slimmed-down BphBodySoA rather than pulling in every field of the real BodySoA that
// this kernel never touches (sleep/rest-pose/gyration are dead here).
// ---------------------------------------------------------------------------
struct NpBodySoA
{
    vec3 *pos;           // per BODY
    quat *rot;           // per BODY
    int *shapeBody;      // per SHAPE: owning body
    vec3 *localPos;      // per SHAPE: pose in the body frame
    quat *localRot;
    vec3 *halfExtents;   // per SHAPE
    uint8_t *shapeType;  // per SHAPE, may be null -> AVBD_SHAPE_BOX. One byte per shape, as
                         // VkWorld uploads it: an int* here read four tags as one, which an
                         // all-box scene hides.
    float *friction;     // per SHAPE
    uint *flags;         // per SHAPE: AVBD_SHAPE_FLAG_IDENTITY (bit 0)
    float *mass;         // per BODY from here down
    vec3 *moment;
    vec3 *velLin;
    vec3 *velAng;
    BodyProps *props;    // per BODY, may be null: no body has a non-default BodyProps
    uint64_t *category;      // per SHAPE: collision filter, may be null (no filtering)
    uint64_t *mask;
    int *group;
    uint count;
    uint _pad0;
};

// One manifold set (cur or prev), mirroring types.slang's ManifoldSoA field for field
// (this kernel writes/reads exactly the subset the narrowphase's ManifoldSoA does).
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
    int *shapeA;   // the shapes that produced the manifold (key = shape pair)
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
    // when set, pairCount is a capacity and *livePairs the broadphase's emit counter
    // (it may exceed the capacity; a thread below pairCount is then live). Dead slots get
    // contactCount 0 so every consumer sized by capacity skips them.
    uint *livePairs;
    // Fidelity trade (solver_params.h contactCap).
    int contactCap;      // 0 = no cap
    uint _pad3;
    // Hull pairs (either shape AVBD_SHAPE_HULL) are not collided here: their pair slot is appended
    // to hullList, which csNarrowphaseHull (narrowphase_hull.slang) consumes. Both may be null
    // when the scene has no hull shape; a hull pair is then dropped as before.
    int *hullList;       // may be null. Pair slots of hull pairs; capacity = pairCount
    int *hullCounter;    // may be null. Counts appends (zeroed by the host before the kernel)
    // Mesh pairs: a static triangle mesh against a convex shape is not collided here.
    // Its pair slot is appended to meshList for csMeshMidphase (narrowphase_mesh.slang). Both may
    // be null when the scene has no mesh; a mesh pair is then dropped, as before.
    int *meshList;       // may be null. Pair slots of mesh-vs-convex pairs; capacity = pairCount
    int *meshCounter;    // may be null. Counts appends (zeroed by the host before the kernel)
};

// Contact reduction to `cap` points (1 <= cap < n). Keeps the deepest point, then
// repeatedly the point furthest from every point kept so far (farthest-point
// sampling), which for a face contact keeps the corners and so the support area.
// Survivors keep their original order and feature keys, so warm-start matching
// below is unchanged for them. Ties go to the lower index.
// Pose of shape `s` of body `body`: the body pose itself for an identity shape (so the
// arithmetic of a single-shape body is unchanged), else the body pose composed with the
// shape's local transform.
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

int reduceContacts(inout ContactOut c[MAX_CONTACTS], int n, int cap, float3x3 basis,
                   vec3 posA, quat rotA, vec3 posB, quat rotB)
{
    vec3 at[MAX_CONTACTS];
    int first = 0;
    float best = 3.402823e38f;
    for (int i = 0; i < n; ++i)
    {
        vec3 xA = transformQ(posA, rotA, c[i].rA);
        vec3 xB = transformQ(posB, rotB, c[i].rB);
        at[i] = scale(add(xA, xB), 0.5f);
        float d = dot3(basis.row0, sub(xA, xB));
        if (d < best)
        {
            best = d;
            first = i;
        }
    }

    float minDist[MAX_CONTACTS];
    uint keep = 1u << (uint)first;
    for (int i = 0; i < n; ++i)
    {
        vec3 e = sub(at[i], at[first]);
        minDist[i] = dot3(e, e);
    }
    for (int k = 1; k < cap; ++k)
    {
        int pick = -1;
        float farD = -1.0f;
        for (int i = 0; i < n; ++i)
        {
            if ((keep >> (uint)i) & 1u)
                continue;
            if (minDist[i] > farD)
            {
                farD = minDist[i];
                pick = i;
            }
        }
        if (pick < 0)
            break;
        keep |= 1u << (uint)pick;
        for (int i = 0; i < n; ++i)
        {
            vec3 e = sub(at[i], at[pick]);
            minDist[i] = min(minDist[i], dot3(e, e));
        }
    }

    int m = 0;
    for (int i = 0; i < n; ++i)
    {
        if ((keep >> (uint)i) & 1u)
        {
            c[m] = c[i];
            ++m;
        }
    }
    return m;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void csNarrowphase(uint3 tid: SV_DispatchThreadID)
{
    NarrowphaseArgs *args = (NarrowphaseArgs *)pc.argsAddr;
    uint t = tid.x;
    if (t >= args->pairCount)
        return;
    if (args->livePairs != nullptr && t >= args->livePairs[0])
    {
        args->cur.contactCount[t] = 0;
        return;
    }

    uint64_t key = args->pairs[t];
    // The key is a SHAPE pair; the manifold still belongs to the two bodies.
    int sa = pairLoS(key), sb = pairHiS(key);
    int ia = args->b.shapeBody[sa], ib = args->b.shapeBody[sb];

    args->cur.key[t] = key;
    args->cur.bodyA[t] = ia;
    args->cur.bodyB[t] = ib;
    args->cur.shapeA[t] = sa;
    args->cur.shapeB[t] = sb;
    args->cur.contactCount[t] = 0;

    // Collision filter. Each shape's category must be in the other's mask; a shared
    // nonzero group overrides that (positive always collides, negative never). Null arrays
    // (no filter, e.g. a bare narrowphase test) and the defaults (category 1, mask all,
    // group 0) collide as before. Two bodies of mass 0 (a kinematic and a static) never do.
    if (args->b.category != nullptr)
    {
        const int gA = args->b.group[sa], gB = args->b.group[sb];
        if (gA != 0 && gA == gB)
        {
            if (gA < 0)
                return;
        }
        else if ((args->b.category[sa] & args->b.mask[sb]) == 0ull ||
                 (args->b.category[sb] & args->b.mask[sa]) == 0ull)
            return;
    }
    if (args->b.props != nullptr && args->b.mass[ia] <= 0.0f && args->b.mass[ib] <= 0.0f)
        return;

    // Body poses (the frame contact anchors live in) and shape poses (what collides).
    vec3 posA = args->b.pos[ia], posB = args->b.pos[ib];
    quat rotA = args->b.rot[ia], rotB = args->b.rot[ib];
    vec3 spA, spB;
    quat sqA, sqB;
    shapePoseNp(args->b, sa, ia, spA, sqA);
    shapePoseNp(args->b, sb, ib, spB, sqB);
    vec3 halfA = args->b.halfExtents[sa], halfB = args->b.halfExtents[sb];
    int shapeA = args->b.shapeType != nullptr ? (int)args->b.shapeType[sa] : AVBD_SHAPE_BOX;
    int shapeB = args->b.shapeType != nullptr ? (int)args->b.shapeType[sb] : AVBD_SHAPE_BOX;
    // Mesh pairs go to the mesh midphase. Mesh against mesh, and a mesh against a static body, never
    // collide. A slot below pairCount, so the list cannot overflow.
    if (shapeA == AVBD_SHAPE_MESH || shapeB == AVBD_SHAPE_MESH)
    {
        if (shapeA != AVBD_SHAPE_MESH || shapeB != AVBD_SHAPE_MESH)
        {
            if (args->b.mass[ia] > 0.0f || args->b.mass[ib] > 0.0f)
            {
                if (args->meshList != nullptr)
                {
                    int slot = subgroupReserveOne(args->meshCounter);
                    args->meshList[slot] = (int)t;
                }
            }
        }
        return;
    }
    // Hull pairs go to csNarrowphaseHull. A slot below pairCount, so the list cannot overflow.
    if (shapeA == AVBD_SHAPE_HULL || shapeB == AVBD_SHAPE_HULL)
    {
        if (args->hullList != nullptr)
        {
            int slot = subgroupReserveOne(args->hullCounter);
            args->hullList[slot] = (int)t;
        }
        return;
    }

    int cap = args->contactCap;

    ContactOut newContacts[MAX_CONTACTS];
    float3x3 basis;
    int n = collideShapesImpl(spA, sqA, halfA, shapeA, spB, sqB, halfB, shapeB, newContacts, basis);
    if (n <= 0)
        return;
    // Anchors are relative to the BODY frame (primal and dual transform them by the body
    // pose), so lift a non-identity shape's shape-frame anchors into it. An identity shape
    // is skipped: its anchors are already body-frame, bit for bit.
    if ((args->b.flags[sa] & 1u) == 0u)
        for (int k = 0; k < n; ++k)
            newContacts[k].rA = add(args->b.localPos[sa], rotateQ(args->b.localRot[sa], newContacts[k].rA));
    if ((args->b.flags[sb] & 1u) == 0u)
        for (int k = 0; k < n; ++k)
            newContacts[k].rB = add(args->b.localPos[sb], rotateQ(args->b.localRot[sb], newContacts[k].rB));
    if (cap > 0 && n > cap)
        n = reduceContacts(newContacts, n, cap, basis, posA, rotA, posB, rotB);

    // Dual state is NOT part of ContactOut (collideShapesImpl fills only feature/rA/rB,
    // same split as the reference Contact struct) -- parallel arrays, explicitly zeroed so a
    // brand-new contact starts exactly where the CPU reference starts it: no
    // accumulated impulse, not sticking. Matched contacts below overwrite all three.
    vec3 penalty[MAX_CONTACTS];
    vec3 lambda[MAX_CONTACTS];
    bool stick[MAX_CONTACTS];
    for (int k = 0; k < n; ++k)
    {
        penalty[k] = vec3(0, 0, 0);
        lambda[k] = vec3(0, 0, 0);
        stick[k] = false;
    }

    float friction = sqrt(args->b.friction[sa] * args->b.friction[sb]);
    // Per-body material: friction mixing rule (the larger enum of the two bodies wins) and
    // conveyor surface speed (body-local +X, in the tangent plane).
    vec3 beltVel = vec3(0, 0, 0);
    bool conveyor = false;
    if (args->b.props != nullptr)
    {
        const BodyProps pa = args->b.props[ia], pb = args->b.props[ib];
        const uint mix = max(pa.frictionMix & 0xFFu, pb.frictionMix & 0xFFu); // bits 8.. = restitution rule
        const float fa = args->b.friction[sa], fb = args->b.friction[sb];
        if (mix == 1u)
            friction = min(fa, fb);
        else if (mix == 2u)
            friction = max(fa, fb);
        else if (mix == 3u)
            friction = 0.5f * (fa + fb);
        else if (mix == 4u)
            friction = fa * fb;
        // Conveyor surface velocity, a body-frame vector (the shape's tangentVelocity).
        if (pa.tangentSpeed != 0.0f || pa.tangentY != 0.0f || pa.tangentZ != 0.0f)
        {
            beltVel = add(beltVel, rotateQ(rotA, vec3(pa.tangentSpeed, pa.tangentY, pa.tangentZ)));
            conveyor = true;
        }
        if (pb.tangentSpeed != 0.0f || pb.tangentY != 0.0f || pb.tangentZ != 0.0f)
        {
            beltVel = sub(beltVel, rotateQ(rotB, vec3(pb.tangentSpeed, pb.tangentY, pb.tangentZ)));
            conveyor = true;
        }
    }

    int prevIdx = (args->prevCount > 0) ? findPrevManifoldS(args->prevKeys, (int)args->prevCount, key) : -1;
    if (prevIdx >= 0)
    {
        int oldOffset = args->prev.contactOffset[prevIdx];
        int oldCount = args->prev.contactCount[prevIdx];

        const bool roundA = shapeA != AVBD_SHAPE_BOX;
        const bool roundB = shapeB != AVBD_SHAPE_BOX;

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < oldCount; ++j)
            {
                int o = oldOffset + j;
                if (args->prev.feature[o] != newContacts[i].featureKey)
                    continue;

                vec3 newRA = newContacts[i].rA;
                vec3 newRB = newContacts[i].rB;

                penalty[i] = args->prev.penalty[o];
                lambda[i] = args->prev.lambda[o];
                stick[i] = !conveyor && (args->prev.stick[o] & 1) != 0; // a belt never keeps anchors
                newContacts[i].rA = args->prev.rA[o];
                newContacts[i].rB = args->prev.rB[o];
                if (!stick[i] || roundA || roundB)
                    newContacts[i].rA = newRA;
                if (!stick[i] || roundA || roundB)
                    newContacts[i].rB = newRB;
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

        vec3 xA = transformQ(posA, rotA, newContacts[i].rA);
        vec3 xB = transformQ(posB, rotB, newContacts[i].rB);
        vec3 C0 = add(matVec(basis, sub(xA, xB)), vec3(COLLISION_MARGIN_NP, 0.0f, 0.0f));
        if (conveyor)
        {
            // The belt carries the other body along: shift the tangent error by the distance
            // the surface moves this step, so friction drags rather than resists.
            // evalContact scales C0 by (1 - alpha), so the shift is divided by it to land whole.
            const float beltScale = args->p.dt / max(1.0f - args->p.alpha, 0.01f);
            C0.y += dot3(rowAtS(basis, 1), beltVel) * beltScale;
            C0.z += dot3(rowAtS(basis, 2), beltVel) * beltScale;
        }

        vec3 lam = scale(lambda[i], args->p.alpha * args->p.gamma);
        vec3 pen = clampV(scale(penalty[i], args->p.gamma), PENALTY_MIN_NP, PENALTY_MAX_NP);

        vec3 kAuto = vec3(0, 0, 0);
        if (args->p.autoKStart != 0 || args->p.dualDamping > 0.0f)
        {
            const float invMA = args->b.mass[ia] > 0.0f ? 1.0f / args->b.mass[ia] : 0.0f;
            const float invMB = args->b.mass[ib] > 0.0f ? 1.0f / args->b.mass[ib] : 0.0f;
            const vec3 rAW = rotateQ(rotA, newContacts[i].rA);
            const vec3 rBW = rotateQ(rotB, newContacts[i].rB);
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
                float kv = w > 0.0f ? clampf(1.0f / (w * args->p.dt * args->p.dt), PENALTY_MIN_NP, PENALTY_MAX_NP)
                                    : PENALTY_MIN_NP;
                if (axis == 0) kAuto.x = kv; else if (axis == 1) kAuto.y = kv; else kAuto.z = kv;
            }

            if (args->p.autoKStart != 0 && penalty[i].x <= 0.0f)
                pen = kAuto;
        }

        if (args->p.dualDamping > 0.0f)
        {
            const vec3 vA = add(args->b.velLin[ia], cross3(args->b.velAng[ia], rotateQ(rotA, newContacts[i].rA)));
            const vec3 vB = add(args->b.velLin[ib], cross3(args->b.velAng[ib], rotateQ(rotB, newContacts[i].rB)));
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

        args->cur.feature[o] = newContacts[i].featureKey;
        args->cur.rA[o] = newContacts[i].rA;
        args->cur.rB[o] = newContacts[i].rB;
        args->cur.C0[o] = C0;
        args->cur.penalty[o] = pen;
        args->cur.lambda[o] = lam;
        args->cur.stick[o] = (uint8_t)(stick[i] ? 1 : 0);
    }
}
