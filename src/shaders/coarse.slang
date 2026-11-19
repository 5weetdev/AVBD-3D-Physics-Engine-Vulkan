// The two-level AVBD coarse correction (csCoarse*). Split from primal.slang, whose library functions
// it uses.

import maths;
import types;
import contact;
import forces;
import dual_body;
import primal;

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
