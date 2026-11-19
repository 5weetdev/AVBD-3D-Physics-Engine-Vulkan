// The primal stage -- the primal position solve, ~82% of GPU time
// Two dispatches of the same update: csPrimalThread (one thread per body) and
// csPrimalWarp4 (a LANES=4 cooperative-lane variant, generic over WaveGetLaneCount() so
// lane groups never straddle a subgroup). Each is split into a *_Forces and *_NoForces
// entry point rather than a Slang generic over a bool, matching integrate.slang's
// precedent of concrete entry points per instantiation (one pipeline per instance).
// WITH_FORCES here means "joints/springs present".
//
// LANES beyond 4 (2/8/16/32, the reference launchPrimalLanes switch) are not instantiated here;
// only the plain thread-per-body and 4-lane cooperative variants are, matching what
// tools/test_vk_solve_iter.cpp exercises.

import maths;
import types;
import contact;
import forces;
import dual_body;

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// Mirrors the constraint builder. Not in types.slang (that
// file is owned by the concurrent constraint-graph port); declared locally here the same
// way integrate.slang declares its own private BodySoA subset.
struct ConstraintGraph
{
    uint64_t *key;
    int *isA;
    int *start;
    int *end;
};

struct PrimalArgs
{
    BodySoA b;
    ConstraintGraph g;
    ManifoldSoA m;
    JointSoA joints;
    SpringSoA springs;
    int *colorBodies;
    int listStart;
    int listCount;
    SolverParamsGpu p;
    // M4: when colorStart is set, the list is colour `color`'s device range,
    // [colorStart[color], colorEnd[color]), and listStart/listCount are ignored.
    int *colorStart;
    int *colorEnd;
    int color;
    int heavyDegree; // > 0: bodies with more graph edges are left to csPrimalWide*
    // Per shape rolling resistance (0 for a shape that is not a sphere or capsule). May be null:
    // no shape has any, and the contact path costs one pointer test.
    float *shapeRolling;
};

bool heavyBody(PrimalArgs a, int i)
{
    return a.heavyDegree > 0 && a.g.end[i] - a.g.start[i] > a.heavyDegree;
}

PrimalArgs resolveList(PrimalArgs a)
{
    if (a.colorStart != nullptr)
    {
        int s = a.colorStart[a.color];
        a.listStart = s;
        a.listCount = max(a.colorEnd[a.color] - s, 0);
    }
    return a;
}

// One body's contribution from a manifold edge, shared by the thread and lane variants.
void stampManifoldEdge(ManifoldSoA m, float *shapeRolling, int mi, bool isA, BodySoA b, vec3 dqLin_self, vec3 dqAng_self,
                        int i, quat rot_self, float alpha, float dt,
                        inout float3x3 lhsLin, inout float3x3 lhsAng, inout float3x3 lhsCross,
                        inout vec3 rhsLin, inout vec3 rhsAng)
{
    int ia = m.bodyA[mi];
    int ib = m.bodyB[mi];
    int n = m.contactCount[mi];
    if (n <= 0)
        return;

    float3x3 basis = m.basis[mi];
    float friction = m.friction[mi];
    int off = m.contactOffset[mi];
    float sLin = isA ? 1.0f : -1.0f;
    float3x3 basisT = transposeM(basis);
    float3x3 sBasis = scaleM(basis, sLin);
    float3x3 sBasisT = scaleM(basisT, sLin);

    quat rotA, rotB;
    vec3 dqALin, dqAAng, dqBLin, dqBAng;
    if (isA)
    {
        rotA = rot_self; dqALin = dqLin_self; dqAAng = dqAng_self;
        rotB = b.rot[ib]; dqBLin = sub(b.pos[ib], b.initialLin[ib]); dqBAng = subQQ(rotB, b.initialAng[ib]);
    }
    else
    {
        rotB = rot_self; dqBLin = dqLin_self; dqBAng = dqAng_self;
        rotA = b.rot[ia]; dqALin = sub(b.pos[ia], b.initialLin[ia]); dqAAng = subQQ(rotA, b.initialAng[ia]);
    }

    float fnSum = 0.0f; // total normal force, for rolling resistance (only read when b.props != null)
    for (int c = 0; c < n; c++)
    {
        int o = off + c;
        ContactEval ce = evalContact(basis, friction, rotA, rotB, m.rA[o], m.rB[o],
                                      dqALin, dqAAng, dqBLin, dqBAng, m.C0[o], m.penalty[o], m.lambda[o], alpha);
        fnSum += max(-ce.F.x, 0.0f);

        vec3 pen = m.penalty[o];
        float3x3 jAng = isA ? ce.jAAng : ce.jBAng;

        float3x3 jAngT = transposeM(jAng);
        float3x3 jAngTk = mulDiag(jAngT, pen);

        lhsLin = addM(lhsLin, mulMM(mulDiag(basisT, pen), basis));
        lhsAng = addM(lhsAng, mulMM(jAngTk, jAng));
        lhsCross = addM(lhsCross, mulMM(jAngTk, sBasis));

        rhsLin = add(rhsLin, mulMV(sBasisT, ce.F));
        rhsAng = add(rhsAng, mulMV(jAngT, ce.F));
    }

    // Rolling resistance: a stateless angular row on the part of the relative rotation that is
    // not about the contact normal, a spring against this step's rotation whose force is capped
    // at rr * total normal force. rr is the larger of the two shapes' values (shapeRolling,
    // per shape, 0 unless a sphere or capsule) and, for SceneBuilder scenes, the two bodies'
    // BodyProps.rollingResistance.
    if ((b.props != nullptr || shapeRolling != nullptr) && fnSum > 0.0f)
    {
        float rr = 0.0f;
        if (b.props != nullptr && b.shapeType != nullptr)
        {
            const uint ta = (uint)b.shapeType[ia], tb = (uint)b.shapeType[ib];
            const bool rollA = ta == 1u || ta == 2u, rollB = tb == 1u || tb == 2u;
            rr = max(rollA ? b.props[ia].rollingResistance : 0.0f,
                     rollB ? b.props[ib].rollingResistance : 0.0f);
        }
        if (shapeRolling != nullptr)
            rr = max(rr, max(shapeRolling[m.shapeA[mi]], shapeRolling[m.shapeB[mi]]));
        if (rr > 0.0f)
        {
            float mMax = 0.0f;
            if (b.mass[ia] > 0.0f)
                mMax = max(max(b.moment[ia].x, b.moment[ia].y), b.moment[ia].z);
            if (b.mass[ib] > 0.0f)
            {
                float mb = max(max(b.moment[ib].x, b.moment[ib].y), b.moment[ib].z);
                mMax = mMax > 0.0f ? min(mMax, mb) : mb;
            }
            const float kk = mMax / (dt * dt);
            vec3 nrm = basis.row0;
            float3x3 P = subM(diagonalM(1.0f, 1.0f, 1.0f), outerM(nrm, nrm));
            vec3 Cr = mulMV(P, sub(dqAAng, dqBAng));
            vec3 Fr = scale(Cr, kk);
            float cap = rr * fnSum;
            float fl = length3(Fr);
            if (fl > cap)
                Fr = scale(Fr, cap / fl);
            lhsAng = addM(lhsAng, scaleM(P, kk));
            rhsAng = add(rhsAng, scale(Fr, sLin));
        }
    }
}

// Motion locks (BodyProps flags): a locked degree of freedom takes no step (angular axes are
// world axes). b.props is null in a world with no non-default BodyProps.
vec3 zeroComp(vec3 v, int k)
{
    return vec3(k == 0 ? 0.0f : v.x, k == 1 ? 0.0f : v.y, k == 2 ? 0.0f : v.z);
}
// A locked degree of freedom takes no step. The block is solved as if it were free and the
// locked components of the step are dropped afterwards: that leaves the plain solve6 call sites
// (and so every unlocked body) compiling exactly as before, which the trajhash oracle pins.
void dropLockedSteps(BodySoA b, int i, inout vec3 dxLin, inout vec3 dxAng)
{
    if (b.props == nullptr)
        return;
    const uint f = b.props[i].flags;
    if ((f & BODY_LOCK_ANY) == 0u)
        return;
    for (int k = 0; k < 3; k++)
    {
        if ((f & (4u << (uint)k)) != 0u)
            dxLin = zeroComp(dxLin, k);
        if ((f & (32u << (uint)k)) != 0u)
            dxAng = zeroComp(dxAng, k);
    }
}

// World-frame inertia (SolverParams.worldFrameInertia, p.worldFrameInertia != 0): replaces the
// body-frame diag(moment) of the angular block by R diag(moment) R^T, R the rotation at the start
// of the step, and adds the explicit gyroscopic torque w x (I w) from the step-start angular
// velocity, so a torque-free body conserves the world angular momentum. Applied as a delta on
// top of the unchanged diagonal terms, so a call is skipped entirely with the flag off.
void worldInertiaDelta(BodySoA b, int i, vec3 moment, quat rot, float dt, inout float3x3 lhsAng, inout vec3 rhsAng)
{
    const quat q = b.initialAng[i];
    const vec3 cx = rotateQ(q, vec3(1, 0, 0));
    const vec3 cy = rotateQ(q, vec3(0, 1, 0));
    const vec3 cz = rotateQ(q, vec3(0, 0, 1));
    const float3x3 Mw = addM(addM(scaleM(outerM(cx, cx), moment.x), scaleM(outerM(cy, cy), moment.y)),
                             scaleM(outerM(cz, cz), moment.z));
    const float3x3 dM = divM(subM(Mw, diagonalM(moment.x, moment.y, moment.z)), dt * dt);
    const vec3 w = b.velAng[i];
    lhsAng = addM(lhsAng, dM);
    rhsAng = add(rhsAng, add(mulMV(dM, subQQ(rot, b.inertialAng[i])), cross3(w, mulMV(Mw, w))));
}

// ---------------------------------------------------------------------------
// Thread-per-body variant
// ---------------------------------------------------------------------------

void primalThreadBody(PrimalArgs args, int t, bool withForces)
{
    if (t >= args.listCount)
        return;
    int i = args.colorBodies[args.listStart + t];

    BodySoA b = args.b;
    float mass = b.mass[i];
    if (mass <= 0.0f || heavyBody(args, i))
        return;

    float dt = args.p.dt;
    vec3 moment = b.moment[i];
    vec3 pos = b.pos[i];
    quat rot = b.rot[i];
    vec3 dqLin_self = sub(pos, b.initialLin[i]);
    vec3 dqAng_self = subQQ(rot, b.initialAng[i]);

    float3x3 MLin = diagonalM(mass, mass, mass);
    float3x3 MAng = diagonalM(moment.x, moment.y, moment.z);

    float3x3 lhsLin = divM(MLin, dt * dt);
    float3x3 lhsAng = divM(MAng, dt * dt);
    float3x3 lhsCross = zeroM();

    vec3 rhsLin = mulMV(divM(MLin, dt * dt), sub(pos, b.inertialLin[i]));
    vec3 rhsAng = mulMV(divM(MAng, dt * dt), subQQ(rot, b.inertialAng[i]));
    if (args.p.worldFrameInertia != 0.0f)
        worldInertiaDelta(b, i, moment, rot, dt, lhsAng, rhsAng);

    ConstraintGraph g = args.g;
    for (int e = g.start[i]; e < g.end[i]; e++)
    {
        uint64_t key = g.key[e];
        int mi = graphIndex(key);
        bool isA = g.isA[e] != 0;

        if (withForces)
        {
            int type = graphType(key);
            if (type == FORCE_JOINT)
            {
                stampJoint(args.joints, mi, isA, b, args.p.alpha, lhsLin, lhsAng, lhsCross, rhsLin, rhsAng);
                continue;
            }
            if (type == FORCE_SPRING)
            {
                stampSpringA(args.springs, mi, isA, b, args.p.alpha, lhsLin, lhsAng, lhsCross, rhsLin, rhsAng);
                continue;
            }
        }

        stampManifoldEdge(args.m, args.shapeRolling, mi, isA, b, dqLin_self, dqAng_self, i, rot, args.p.alpha, args.p.dt,
                           lhsLin, lhsAng, lhsCross, rhsLin, rhsAng);
    }

    vec3 dxLin, dxAng;
    solve6(lhsLin, lhsAng, lhsCross, neg(rhsLin), neg(rhsAng), dxLin, dxAng);
    dropLockedSteps(b, i, dxLin, dxAng);
    b.pos[i] = add(pos, dxLin);
    b.rot[i] = addQV(rot, dxAng);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csPrimalThreadNoForces(uint3 tid: SV_DispatchThreadID)
{
    PrimalArgs *args = (PrimalArgs *)pc.argsAddr;
    primalThreadBody(resolveList(*args), (int)tid.x, false);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csPrimalThreadForces(uint3 tid: SV_DispatchThreadID)
{
    PrimalArgs *args = (PrimalArgs *)pc.argsAddr;
    primalThreadBody(resolveList(*args), (int)tid.x, true);
}

// ---------------------------------------------------------------------------
// LANES=4 cooperative variant. Reduction is a butterfly over WaveReadLaneAt keyed off
// WaveGetLaneIndex(), generic over the physical subgroup width: LANES consecutive
// lanes (aligned to a LANES boundary) cooperate on one body, same as the primal stage's
// warpReduceSum<LANES>, but built from wave intrinsics rather than a fixed __shfl mask
// so it is correct whether the physical subgroup is 32 (NVIDIA) or 64 (AMD) wide, as
// long as the dispatch's local group size is itself a multiple of the subgroup size --
// the host (vk_solver.cpp) is responsible for that; see its comment.
// ---------------------------------------------------------------------------

static const int LANES4 = 4;

// `selfLane` is the CALLING thread's own absolute lane (threadIdxX, per the identity-
// mapping note below), not a shared cluster base. Bug fixed here (found by
// tools/test_vk_solve_iter.cpp: csPrimalWarp4 diverged from the reference k_primal_color_warp
// by ~1e-2 after 8 iterations on a body with 3+ nonzero cooperating lanes, 0 diff with
// <=2): the previous version had every one of the LANES4 lanes in a cluster call this
// with the SAME clusterBase value, and added `offset` without wrapping modulo the
// cluster width. That means (a) all lanes read from the same fixed pair of source
// lanes every round instead of each lane reducing against its own partner, so a lane's
// contribution can be summed in twice while another's is dropped, and (b) for a lane
// near the top of its cluster, `lane + offset` could read past the cluster boundary
// into an unrelated body's data entirely. The fix below computes each lane's own
// position within its LANES4-wide cluster and XOR-butterflies within it (mirroring
// what the reference __shfl_down_sync(mask, val, offset, width=LANES) does implicitly via its
// `width` parameter -- Slang's WaveReadLaneAt has no such wrapping, so it is done here
// explicitly).
float waveReduceSumClustered(float val, int selfLane)
{
    int clusterBase = selfLane & ~(LANES4 - 1);
    int pos = selfLane - clusterBase;
    [unroll]
    for (int offset = LANES4 / 2; offset > 0; offset /= 2)
        val += WaveReadLaneAt(val, clusterBase + ((pos ^ offset) & (LANES4 - 1)));
    return val;
}

vec3 waveReduceSumClustered(vec3 v, int lane)
{
    return vec3(waveReduceSumClustered(v.x, lane), waveReduceSumClustered(v.y, lane), waveReduceSumClustered(v.z, lane));
}

float3x3 waveReduceSumClustered(float3x3 m, int lane)
{
    float3x3 r;
    r.row0 = waveReduceSumClustered(m.row0, lane);
    r.row1 = waveReduceSumClustered(m.row1, lane);
    r.row2 = waveReduceSumClustered(m.row2, lane);
    return r;
}

void primalWarp4Body(PrimalArgs args, uint threadIdxX, uint blockIdxX, uint blockDimX, bool withForces)
{
    int lane = (int)(threadIdxX & (LANES4 - 1));
    int bodyIdxInList = (int)(blockIdxX * (blockDimX / LANES4) + (threadIdxX / LANES4));
    if (bodyIdxInList >= args.listCount)
        return;

    BodySoA b = args.b;
    int i = args.colorBodies[args.listStart + bodyIdxInList];
    float mass = b.mass[i];
    if (mass <= 0.0f || heavyBody(args, i))
        return;

    quat rot_self = b.rot[i];
    vec3 dqLin_self = sub(b.pos[i], b.initialLin[i]);
    vec3 dqAng_self = subQQ(rot_self, b.initialAng[i]);

    float3x3 lhsLin_lane = zeroM();
    float3x3 lhsAng_lane = zeroM();
    float3x3 lhsCross_lane = zeroM();
    vec3 rhsLin_lane = vec3(0.0f, 0.0f, 0.0f);
    vec3 rhsAng_lane = vec3(0.0f, 0.0f, 0.0f);

    ConstraintGraph g = args.g;
    int start = g.start[i];
    int end = g.end[i];

    for (int e = start + lane; e < end; e += LANES4)
    {
        uint64_t key = g.key[e];
        int mi = graphIndex(key);
        bool isA = g.isA[e] != 0;

        if (withForces)
        {
            int type = graphType(key);
            if (type == FORCE_JOINT)
            {
                stampJoint(args.joints, mi, isA, b, args.p.alpha, lhsLin_lane, lhsAng_lane, lhsCross_lane, rhsLin_lane, rhsAng_lane);
                continue;
            }
            if (type == FORCE_SPRING)
            {
                stampSpringA(args.springs, mi, isA, b, args.p.alpha, lhsLin_lane, lhsAng_lane, lhsCross_lane, rhsLin_lane, rhsAng_lane);
                continue;
            }
        }

        stampManifoldEdge(args.m, args.shapeRolling, mi, isA, b, dqLin_self, dqAng_self, i, rot_self, args.p.alpha, args.p.dt,
                           lhsLin_lane, lhsAng_lane, lhsCross_lane, rhsLin_lane, rhsAng_lane);
    }

    int selfLane = (int)threadIdxX;
    lhsLin_lane = waveReduceSumClustered(lhsLin_lane, selfLane);
    lhsAng_lane = waveReduceSumClustered(lhsAng_lane, selfLane);
    lhsCross_lane = waveReduceSumClustered(lhsCross_lane, selfLane);
    rhsLin_lane = waveReduceSumClustered(rhsLin_lane, selfLane);
    rhsAng_lane = waveReduceSumClustered(rhsAng_lane, selfLane);

    if (lane == 0)
    {
        float dt = args.p.dt;
        vec3 moment = b.moment[i];
        vec3 pos = b.pos[i];
        quat rot = b.rot[i];

        float3x3 MLin = diagonalM(mass, mass, mass);
        float3x3 MAng = diagonalM(moment.x, moment.y, moment.z);

        lhsLin_lane = addM(lhsLin_lane, divM(MLin, dt * dt));
        lhsAng_lane = addM(lhsAng_lane, divM(MAng, dt * dt));

        rhsLin_lane = add(rhsLin_lane, mulMV(divM(MLin, dt * dt), sub(pos, b.inertialLin[i])));
        rhsAng_lane = add(rhsAng_lane, mulMV(divM(MAng, dt * dt), subQQ(rot, b.inertialAng[i])));
        if (args.p.worldFrameInertia != 0.0f)
            worldInertiaDelta(b, i, moment, rot, dt, lhsAng_lane, rhsAng_lane);

        vec3 dxLin, dxAng;
        solve6(lhsLin_lane, lhsAng_lane, lhsCross_lane, neg(rhsLin_lane), neg(rhsAng_lane), dxLin, dxAng);
        dropLockedSteps(b, i, dxLin, dxAng);
        b.pos[i] = add(pos, dxLin);
        b.rot[i] = addQV(rot, dxAng);
    }
}

// NOTE on the waveReduceSumClustered helper above: it keys the butterfly off a lane
// index computed from threadIdxX, not WaveGetLaneIndex(). That is correct only when
// the local group's thread-to-lane mapping is the identity (true for every Vulkan
// implementation actually exercised here: a 1D numthreads group is laid out lane 0..N-1
// == thread 0..N-1 on both NVIDIA and AMD's compute compilers), so `threadIdxX & mask`
// and `WaveGetLaneIndex()` agree; WaveReadLaneAt still dispatches through the real
// subgroup intrinsic, so this is genuinely wave-op reduction, not a shared-memory one.

[shader("compute")]
[numthreads(256, 1, 1)]
void csPrimalWarp4NoForces(uint3 tid: SV_DispatchThreadID, uint3 gtid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    PrimalArgs *args = (PrimalArgs *)pc.argsAddr;
    primalWarp4Body(resolveList(*args), gtid.x, gid.x, 256, false);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csPrimalWarp4Forces(uint3 tid: SV_DispatchThreadID, uint3 gtid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    PrimalArgs *args = (PrimalArgs *)pc.argsAddr;
    primalWarp4Body(resolveList(*args), gtid.x, gid.x, 256, true);
}

// ---------------------------------------------------------------------------
// Small-world fused iteration loop. The whole of VkWorld's phase 6 (every iteration: each
// colour's primal sweep in colour order, then csDual's pass, then csJointDual's pass) as ONE
// workgroup. It calls the same per-body / per-slot functions as the multi-dispatch kernels:
// primalWarp4Body with blockIdxX = the 64-body chunk of the colour's list (so a body keeps its
// 4-lane cluster and its accumulation order), dualSlot, jointDualSlot. A pipeline barrier
// between dispatches becomes AllMemoryBarrierWithGroupSync (one workgroup shares an SM's
// cache, so workgroup-scope ordering of the storage writes is all that is needed).
// Not used with the coarse correction; the host falls back then. The colour
// ranges are read on device here, as csPrimalWarp4* does through resolveList.
// ---------------------------------------------------------------------------
struct IterateArgs
{
    PrimalArgs *pa;   // colour fields ignored: colorStart/colorEnd select each colour's range
    DualArgs *da;
    JointDualArgs *jd;
    int iterations;
    int colorBound;
    int withForces;
    int jointSlots;   // jd->j.count
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csIterateSmall(uint3 gtid: SV_GroupThreadID)
{
    IterateArgs *ia = (IterateArgs *)pc.argsAddr;
    const PrimalArgs base = *ia->pa;
    const bool withForces = ia->withForces != 0;
    const int slots = ia->da->slots;
    const int jointSlots = ia->jointSlots;
    const int tx = (int)gtid.x;
    for (int it = 0; it < ia->iterations; it++)
    {
        for (int c = 0; c < ia->colorBound; c++)
        {
            PrimalArgs a = base;
            a.color = c;
            a = resolveList(a);
            const int chunks = (a.listCount + 63) / 64;
            for (int k = 0; k < chunks; k++)
                primalWarp4Body(a, (uint)tx, (uint)k, 256, withForces);
            AllMemoryBarrierWithGroupSync();
        }
        for (int t = tx; t < slots; t += 256)
            dualSlot(ia->da, t);
        AllMemoryBarrierWithGroupSync();
        for (int i = tx; i < jointSlots; i += 256)
            jointDualSlot(ia->jd, i);
        AllMemoryBarrierWithGroupSync();
    }
}

// ---------------------------------------------------------------------------
// Wide variant for high-degree bodies (a body with thousands of manifold/joint edges,
// which 4 lanes would walk serially). Launched over the same
// colour list at 256 entries per group; a group compacts its heavy entries and solves
// each with all 256 threads: a subgroup sum, then a shared-memory sum over subgroups.
// ---------------------------------------------------------------------------
static const int WIDE = 256;
static const int WIDE_ACC = 33; // lhsLin, lhsAng, lhsCross (27) + rhsLin, rhsAng (6)
groupshared int sWideBody[WIDE];
groupshared int sWideCount;
groupshared float sWideAcc[WIDE / 8][WIDE_ACC]; // subgroups of at least 8 lanes

void wideStore(int w, float3x3 a, float3x3 b, float3x3 c, vec3 r, vec3 s)
{
    float v[WIDE_ACC] = { a.row0.x, a.row0.y, a.row0.z, a.row1.x, a.row1.y, a.row1.z, a.row2.x, a.row2.y, a.row2.z,
                          b.row0.x, b.row0.y, b.row0.z, b.row1.x, b.row1.y, b.row1.z, b.row2.x, b.row2.y, b.row2.z,
                          c.row0.x, c.row0.y, c.row0.z, c.row1.x, c.row1.y, c.row1.z, c.row2.x, c.row2.y, c.row2.z,
                          r.x, r.y, r.z, s.x, s.y, s.z };
    [unroll]
    for (int k = 0; k < WIDE_ACC; k++)
    {
        float t = WaveActiveSum(v[k]);
        if (WaveIsFirstLane())
            sWideAcc[w][k] = t;
    }
}

float3x3 wideMat(float v[WIDE_ACC], int o)
{
    float3x3 m;
    m.row0 = vec3(v[o + 0], v[o + 1], v[o + 2]);
    m.row1 = vec3(v[o + 3], v[o + 4], v[o + 5]);
    m.row2 = vec3(v[o + 6], v[o + 7], v[o + 8]);
    return m;
}

void primalWideBody(PrimalArgs args, int tx, int gx, bool withForces)
{
    if (tx == 0)
        sWideCount = 0;
    GroupMemoryBarrierWithGroupSync();
    const int t = gx * WIDE + tx;
    BodySoA b = args.b;
    if (args.heavyDegree > 0 && t < args.listCount)
    {
        const int i = args.colorBodies[args.listStart + t];
        if (b.mass[i] > 0.0f && heavyBody(args, i))
        {
            int slot;
            InterlockedAdd(sWideCount, 1, slot);
            sWideBody[slot] = i;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    const int n = sWideCount;
    const int lanes = (int)WaveGetLaneCount();
    const int w = tx / lanes;
    for (int h = 0; h < n; h++)
    {
        const int i = sWideBody[h];
        const quat rot_self = b.rot[i];
        const vec3 dqLin_self = sub(b.pos[i], b.initialLin[i]);
        const vec3 dqAng_self = subQQ(rot_self, b.initialAng[i]);
        float3x3 lhsLin = zeroM(), lhsAng = zeroM(), lhsCross = zeroM();
        vec3 rhsLin = vec3(0, 0, 0), rhsAng = vec3(0, 0, 0);
        ConstraintGraph g = args.g;
        for (int e = g.start[i] + tx; e < g.end[i]; e += WIDE)
        {
            uint64_t key = g.key[e];
            int mi = graphIndex(key);
            bool isA = g.isA[e] != 0;
            if (withForces)
            {
                int type = graphType(key);
                if (type == FORCE_JOINT)
                {
                    stampJoint(args.joints, mi, isA, b, args.p.alpha, lhsLin, lhsAng, lhsCross, rhsLin, rhsAng);
                    continue;
                }
                if (type == FORCE_SPRING)
                {
                    stampSpringA(args.springs, mi, isA, b, args.p.alpha, lhsLin, lhsAng, lhsCross, rhsLin, rhsAng);
                    continue;
                }
            }
            stampManifoldEdge(args.m, args.shapeRolling, mi, isA, b, dqLin_self, dqAng_self, i, rot_self, args.p.alpha, args.p.dt,
                               lhsLin, lhsAng, lhsCross, rhsLin, rhsAng);
        }
        wideStore(w, lhsLin, lhsAng, lhsCross, rhsLin, rhsAng);
        GroupMemoryBarrierWithGroupSync();
        if (tx == 0)
        {
            float v[WIDE_ACC];
            for (int k = 0; k < WIDE_ACC; k++)
                v[k] = 0.0f;
            for (int s = 0; s < WIDE / lanes; s++)
                for (int k = 0; k < WIDE_ACC; k++)
                    v[k] += sWideAcc[s][k];
            lhsLin = wideMat(v, 0);
            lhsAng = wideMat(v, 9);
            lhsCross = wideMat(v, 18);
            rhsLin = vec3(v[27], v[28], v[29]);
            rhsAng = vec3(v[30], v[31], v[32]);
            const float dt = args.p.dt;
            const float mass = b.mass[i];
            const vec3 moment = b.moment[i];
            const vec3 pos = b.pos[i];
            float3x3 MLin = diagonalM(mass, mass, mass);
            float3x3 MAng = diagonalM(moment.x, moment.y, moment.z);
            lhsLin = addM(lhsLin, divM(MLin, dt * dt));
            lhsAng = addM(lhsAng, divM(MAng, dt * dt));
            rhsLin = add(rhsLin, mulMV(divM(MLin, dt * dt), sub(pos, b.inertialLin[i])));
            rhsAng = add(rhsAng, mulMV(divM(MAng, dt * dt), subQQ(rot_self, b.inertialAng[i])));
            if (args.p.worldFrameInertia != 0.0f)
                worldInertiaDelta(b, i, moment, rot_self, dt, lhsAng, rhsAng);
            vec3 dxLin, dxAng;
            solve6(lhsLin, lhsAng, lhsCross, neg(rhsLin), neg(rhsAng), dxLin, dxAng);
            dropLockedSteps(b, i, dxLin, dxAng);
            b.pos[i] = add(pos, dxLin);
            b.rot[i] = addQV(rot_self, dxAng);
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csPrimalWideNoForces(uint3 gtid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    PrimalArgs *args = (PrimalArgs *)pc.argsAddr;
    primalWideBody(resolveList(*args), (int)gtid.x, (int)gid.x, false);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csPrimalWideForces(uint3 gtid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    PrimalArgs *args = (PrimalArgs *)pc.argsAddr;
    primalWideBody(resolveList(*args), (int)gtid.x, (int)gid.x, true);
}

// ---------------------------------------------------------------------------
// csPrimalDispatchArgs (M4): one thread per colour turns the device colour ranges into
// VkDispatchIndirectCommand {groups, 1, 1}, so the per-colour primal sweep needs no
// host copy of the ranges. An empty colour gets zero groups, which is a legal no-op.
// ---------------------------------------------------------------------------
struct PrimalDispatchArgs
{
    int *colorStart;
    int *colorEnd;
    uint *out;
    uint colorCount;
    uint bodiesPerGroup;
};

[shader("compute")]
[numthreads(128, 1, 1)]
void csPrimalDispatchArgs(uint3 tid: SV_DispatchThreadID)
{
    PrimalDispatchArgs *args = (PrimalDispatchArgs *)pc.argsAddr;
    uint c = tid.x;
    if (c >= args->colorCount)
        return;
    int n = max(args->colorEnd[c] - args->colorStart[c], 0);
    args->out[3 * c + 0] = ((uint)n + args->bodiesPerGroup - 1) / args->bodiesPerGroup;
    args->out[3 * c + 1] = 1;
    args->out[3 * c + 2] = 1;
}

// ---------------------------------------------------------------------------
// csGateDispatchArgs: copies `count` VkDispatchIndirectCommands src -> dst, or zeroes
// them when the step's overflow flag is set. Lets a whole phase be skipped on device.
// ---------------------------------------------------------------------------
struct GateDispatchArgs
{
    uint *src;
    uint *dst;
    int *flag;
    int count;
    int _pad0;
};

[shader("compute")]
[numthreads(128, 1, 1)]
void csGateDispatchArgs(uint3 tid: SV_DispatchThreadID)
{
    GateDispatchArgs *args = (GateDispatchArgs *)pc.argsAddr;
    int t = (int)tid.x;
    if (t >= args->count)
        return;
    bool closed = (args->flag != nullptr) && (args->flag[0] != 0);
    for (int k = 0; k < 3; k++)
        args->dst[3 * t + k] = closed ? 0u : args->src[3 * t + k];
}

// ---------------------------------------------------------------------------
// Step-loop flags: any of these set stepState[0], which closes the gate on phases 6-8
// and makes the host rerun the step's recorded tail with larger bounds.
// csStepFlagIfSet: flag if a device counter/slot is non-zero.
// csStepColorBound: flag if a colour at or past the recorded bound is non-empty.
// ---------------------------------------------------------------------------
struct StepFlagIfSetArgs
{
    int *src;
    int *flag;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void csStepFlagIfSet(uint3 tid: SV_DispatchThreadID)
{
    StepFlagIfSetArgs *args = (StepFlagIfSetArgs *)pc.argsAddr;
    if (tid.x != 0)
        return;
    if (args->src[0] != 0)
        args->flag[0] = 1;
}

// csStepFlagIfGreater: capacity overflow. A step's broadphase pair counter or narrowphase
// contact counter past its capacity sets word 1 (the capacity flag, which closes the gate on
// phases 4-5) and word 0 (which closes the gate on phases 6-8).
struct StepFlagIfGreaterArgs
{
    int *count;
    int *flag;
    int cap;
    int _pad;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void csStepFlagIfGreater(uint3 tid: SV_DispatchThreadID)
{
    StepFlagIfGreaterArgs *args = (StepFlagIfGreaterArgs *)pc.argsAddr;
    if (tid.x != 0)
        return;
    if (args->count[0] > args->cap)
    {
        args->flag[0] = 1;
        args->flag[1] = 1;
    }
}

struct StepColorBoundArgs
{
    int *start;
    int *end;
    int *flag;
    int bound;
    int maxColors;
};

[shader("compute")]
[numthreads(128, 1, 1)]
void csStepColorBound(uint3 tid: SV_DispatchThreadID)
{
    StepColorBoundArgs *args = (StepColorBoundArgs *)pc.argsAddr;
    int c = args->bound + (int)tid.x;
    if (c >= args->maxColors)
        return;
    if (args->end[c] > args->start[c])
        args->flag[0] = 1;
}

// ---------------------------------------------------------------------------
// Two-level AVBD: after each colour sweep, a coarse
// correction restricted to rigid motions of spatial aggregates (grid cells). For member i
// of aggregate c, with r = p_i - centre(c), the prolongation is dx_i = [dp + dth x r, dth].
// The Galerkin restriction P^T H P is summed per aggregate; edges between two members of
// the same aggregate drop out exactly (rigid-invariant constraints), so they are skipped.
// ---------------------------------------------------------------------------
static const uint COARSE_EMPTY = 0xffffffffu;
static const int COARSE_ACC = 36; // LL 9, AL 9, AA 9, gl 3, ga 3, pad 3
static const int COARSE_PROBES = 32;

struct CoarseArgs
{
    PrimalArgs pa;          // b, g, m, joints, springs, p (list fields unused)
    int *active;            // active dynamic bodies
    int *activeCount;       // live count on device
    int *aggOf;             // per body: aggregate slot or -1
    uint *keys;             // per slot: packed cell or COARSE_EMPTY
    float *acc;             // per slot: COARSE_ACC floats
    float *delta;           // per slot: dp, dth
    int *stats;             // may be null: [0] probe failures, [1] aggregates
    int maxActive;
    int bodyCount;
    int tableSize;          // power of two
    float cell;
    float damping;
    int withForces;
    int color;              // cell parity 0..7 swept this dispatch (Gauss-Seidel over cells), -1 = all
    int _pad0;
};

// Same-parity cells are two cells apart, so with cell >= the largest body no two of them
// share an edge and a parity class can be solved in parallel without overshoot.
bool coarseColorOk(CoarseArgs *a, uint key)
{
    int want = a->color;
    int parity = (int)((key & 1u) | ((key >> 9) & 2u) | ((key >> 18) & 4u));
    return (want < 0) || (parity == want);
}

uint coarsePack(int3 c)
{
    return ((uint)(c.x + 512) & 1023u) | (((uint)(c.y + 512) & 1023u) << 10) | (((uint)(c.z + 512) & 1023u) << 20);
}

vec3 coarseCentre(uint key, float cell)
{
    int x = (int)(key & 1023u) - 512;
    int y = (int)((key >> 10) & 1023u) - 512;
    int z = (int)((key >> 20) & 1023u) - 512;
    return vec3(((float)x + 0.5f) * cell, ((float)y + 0.5f) * cell, ((float)z + 0.5f) * cell);
}

int coarseBody(CoarseArgs *a, int t)
{
    if (t >= min(a->activeCount[0], a->maxActive))
        return -1;
    int i = a->active[t];
    if (a->pa.b.props != nullptr && (a->pa.b.props[i].flags & BODY_LOCK_ANY) != 0u)
        return -1; // a locked body is not aggregated
    return a->pa.b.mass[i] > 0.0f ? i : -1;
}

// Once per step: empty the table and its accumulators, unassign every body.
[shader("compute")]
[numthreads(256, 1, 1)]
void csCoarseReset(uint3 tid: SV_DispatchThreadID)
{
    CoarseArgs *a = (CoarseArgs *)pc.argsAddr;
    int t = (int)tid.x;
    if (t < a->tableSize)
    {
        a->keys[t] = COARSE_EMPTY;
        for (int k = 0; k < COARSE_ACC; k++)
            a->acc[t * COARSE_ACC + k] = 0.0f;
    }
    if (t < a->bodyCount)
        a->aggOf[t] = -1;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csCoarseAssign(uint3 tid: SV_DispatchThreadID)
{
    CoarseArgs *a = (CoarseArgs *)pc.argsAddr;
    int i = coarseBody(a, (int)tid.x);
    if (i < 0)
        return;
    vec3 p = a->pa.b.pos[i];
    int3 c = int3((int)floor(p.x / a->cell), (int)floor(p.y / a->cell), (int)floor(p.z / a->cell));
    uint key = coarsePack(c);
    uint mask = (uint)a->tableSize - 1u;
    uint s = (key * 2654435761u) & mask;
    for (int k = 0; k < COARSE_PROBES; k++)
    {
        uint prev;
        InterlockedCompareExchange(a->keys[s], COARSE_EMPTY, key, prev);
        if (prev == COARSE_EMPTY || prev == key)
        {
            a->aggOf[i] = (int)s;
            if (prev == COARSE_EMPTY && a->stats != nullptr)
                InterlockedAdd(a->stats[1], 1);
            return;
        }
        s = (s + 1u) & mask;
    }
    if (a->stats != nullptr)
        InterlockedAdd(a->stats[0], 1); // reported: this body simply gets no coarse correction
}

void coarseAdd(float *dst, int o, float3x3 M)
{
    InterlockedAdd(dst[o + 0], M.row0.x); InterlockedAdd(dst[o + 1], M.row0.y); InterlockedAdd(dst[o + 2], M.row0.z);
    InterlockedAdd(dst[o + 3], M.row1.x); InterlockedAdd(dst[o + 4], M.row1.y); InterlockedAdd(dst[o + 5], M.row1.z);
    InterlockedAdd(dst[o + 6], M.row2.x); InterlockedAdd(dst[o + 7], M.row2.y); InterlockedAdd(dst[o + 8], M.row2.z);
}

float3x3 coarseLoad(float *src, int o)
{
    float3x3 M;
    M.row0 = vec3(src[o + 0], src[o + 1], src[o + 2]);
    M.row1 = vec3(src[o + 3], src[o + 4], src[o + 5]);
    M.row2 = vec3(src[o + 6], src[o + 7], src[o + 8]);
    return M;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void csCoarseAssemble(uint3 tid: SV_DispatchThreadID)
{
    CoarseArgs *a = (CoarseArgs *)pc.argsAddr;
    int i = coarseBody(a, (int)tid.x);
    if (i < 0)
        return;
    int agg = a->aggOf[i];
    if (agg < 0 || !coarseColorOk(a, a->keys[agg]))
        return;
    PrimalArgs args = a->pa;
    BodySoA b = args.b;
    float dt = args.p.dt;
    float mass = b.mass[i];
    vec3 moment = b.moment[i];
    vec3 pos = b.pos[i];
    quat rot = b.rot[i];
    vec3 dqLin_self = sub(pos, b.initialLin[i]);
    vec3 dqAng_self = subQQ(rot, b.initialAng[i]);

    float3x3 MLin = diagonalM(mass, mass, mass);
    float3x3 MAng = diagonalM(moment.x, moment.y, moment.z);
    float3x3 lhsLin = divM(MLin, dt * dt);
    float3x3 lhsAng = divM(MAng, dt * dt);
    float3x3 lhsCross = zeroM();
    vec3 rhsLin = mulMV(divM(MLin, dt * dt), sub(pos, b.inertialLin[i]));
    vec3 rhsAng = mulMV(divM(MAng, dt * dt), subQQ(rot, b.inertialAng[i]));
    if (args.p.worldFrameInertia != 0.0f)
        worldInertiaDelta(b, i, moment, rot, dt, lhsAng, rhsAng);

    ConstraintGraph g = args.g;
    for (int e = g.start[i]; e < g.end[i]; e++)
    {
        uint64_t key = g.key[e];
        int mi = graphIndex(key);
        bool isA = g.isA[e] != 0;
        int type = graphType(key);
        int other;
        if (type == FORCE_JOINT)
            other = isA ? args.joints.bodyB[mi] : args.joints.bodyA[mi];
        else if (type == FORCE_SPRING)
            other = isA ? args.springs.bodyB[mi] : args.springs.bodyA[mi];
        else
            other = isA ? args.m.bodyB[mi] : args.m.bodyA[mi];
        if (other >= 0 && a->aggOf[other] == agg)
            continue; // internal edge: no rigid-mode contribution

        if (a->withForces != 0)
        {
            if (type == FORCE_JOINT)
            {
                stampJoint(args.joints, mi, isA, b, args.p.alpha, lhsLin, lhsAng, lhsCross, rhsLin, rhsAng);
                continue;
            }
            if (type == FORCE_SPRING)
            {
                stampSpringA(args.springs, mi, isA, b, args.p.alpha, lhsLin, lhsAng, lhsCross, rhsLin, rhsAng);
                continue;
            }
        }
        stampManifoldEdge(args.m, args.shapeRolling, mi, isA, b, dqLin_self, dqAng_self, i, rot, args.p.alpha, args.p.dt,
                           lhsLin, lhsAng, lhsCross, rhsLin, rhsAng);
    }

    // P^T H P and P^T f with P = [[I, -R], [0, I]], R = [r]x (see the section comment).
    vec3 r = sub(pos, coarseCentre(a->keys[agg], a->cell));
    float3x3 R = skewM(r);
    float3x3 RL = mulMM(R, lhsLin);
    float3x3 AL = addM(lhsCross, RL);
    float3x3 XR = mulMM(lhsCross, R);
    // A - RLR - XR + R X^T, and R X^T = -(XR)^T since R is skew.
    float3x3 AA = subM(lhsAng, addM(addM(mulMM(RL, R), XR), transposeM(XR)));
    vec3 ga = add(rhsAng, cross3(r, rhsLin));

    float *dst = a->acc;
    int o = agg * COARSE_ACC;
    coarseAdd(dst, o, lhsLin);
    coarseAdd(dst, o + 9, AL);
    coarseAdd(dst, o + 18, AA);
    InterlockedAdd(dst[o + 27], rhsLin.x); InterlockedAdd(dst[o + 28], rhsLin.y); InterlockedAdd(dst[o + 29], rhsLin.z);
    InterlockedAdd(dst[o + 30], ga.x); InterlockedAdd(dst[o + 31], ga.y); InterlockedAdd(dst[o + 32], ga.z);
}

// Per slot: solve the aggregate's 6x6 system, then zero the accumulators for the next
// iteration (so no separate clear pass).
[shader("compute")]
[numthreads(128, 1, 1)]
void csCoarseSolve(uint3 tid: SV_DispatchThreadID)
{
    CoarseArgs *a = (CoarseArgs *)pc.argsAddr;
    int s = (int)tid.x;
    if (s >= a->tableSize || a->keys[s] == COARSE_EMPTY || !coarseColorOk(a, a->keys[s]))
        return;
    float *acc = a->acc;
    int o = s * COARSE_ACC;
    float3x3 LL = coarseLoad(acc, o);
    float3x3 AL = coarseLoad(acc, o + 9);
    float3x3 AA = coarseLoad(acc, o + 18);
    vec3 gl = vec3(acc[o + 27], acc[o + 28], acc[o + 29]);
    vec3 ga = vec3(acc[o + 30], acc[o + 31], acc[o + 32]);
    vec3 dp = vec3(0.0f, 0.0f, 0.0f);
    vec3 dth = vec3(0.0f, 0.0f, 0.0f);
    float w = a->damping;
    if (LL.row0.x > 0.0f)
        solve6(LL, AA, AL, neg(gl), neg(ga), dp, dth);
    a->delta[s * 6 + 0] = dp.x * w; a->delta[s * 6 + 1] = dp.y * w; a->delta[s * 6 + 2] = dp.z * w;
    a->delta[s * 6 + 3] = dth.x * w; a->delta[s * 6 + 4] = dth.y * w; a->delta[s * 6 + 5] = dth.z * w;
    for (int k = 0; k < COARSE_ACC; k++)
        acc[o + k] = 0.0f;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csCoarseApply(uint3 tid: SV_DispatchThreadID)
{
    CoarseArgs *a = (CoarseArgs *)pc.argsAddr;
    int i = coarseBody(a, (int)tid.x);
    if (i < 0)
        return;
    int s = a->aggOf[i];
    if (s < 0 || !coarseColorOk(a, a->keys[s]))
        return;
    vec3 dp = vec3(a->delta[s * 6 + 0], a->delta[s * 6 + 1], a->delta[s * 6 + 2]);
    vec3 dth = vec3(a->delta[s * 6 + 3], a->delta[s * 6 + 4], a->delta[s * 6 + 5]);
    BodySoA b = a->pa.b;
    vec3 pos = b.pos[i];
    vec3 r = sub(pos, coarseCentre(a->keys[s], a->cell));
    b.pos[i] = add(pos, add(dp, cross3(dth, r)));
    b.rot[i] = addQV(b.rot[i], dth);
}
