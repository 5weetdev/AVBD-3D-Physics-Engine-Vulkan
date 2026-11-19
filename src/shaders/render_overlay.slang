// Overlay statistics and load accumulation kernels (split from render.slang).

import maths;
import types;
import render;

// ---------------------------------------------------------------------------
// Overlay structural aggregates (the overlay reductions)
// ---------------------------------------------------------------------------

struct OverlayStats
{
    int *push;
    int *pull;
    int *stress;
    int *cone;
    int *penetration;
    int *jointDegree;
};

void statMax(int *dst, int i, float v)
{
    if (dst != nullptr)
    {
        int old;
        InterlockedMax(dst[i], asint(v), old);
    }
}

struct OverlayStatsClearArgs
{
    OverlayStats s;
    int n;
    int _pad0;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csOverlayStatsClear(uint3 tid: SV_DispatchThreadID)
{
    OverlayStatsClearArgs *args = (OverlayStatsClearArgs *)pc.argsAddr;
    int i = (int)tid.x;
    if (i >= args->n)
        return;
    const int zero = asint(0.0f);
    const int none = asint(-1.0f);
    if (args->s.push != nullptr) args->s.push[i] = zero;
    if (args->s.pull != nullptr) args->s.pull[i] = zero;
    if (args->s.penetration != nullptr) args->s.penetration[i] = zero;
    if (args->s.stress != nullptr) args->s.stress[i] = none;
    if (args->s.cone != nullptr) args->s.cone[i] = none;
    if (args->s.jointDegree != nullptr) args->s.jointDegree[i] = 0;
}

struct OverlayStatsContactsArgs
{
    ManifoldSoA m;
    int slots;
    int _pad0;
    OverlayStats s;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csOverlayStatsContacts(uint3 tid: SV_DispatchThreadID)
{
    OverlayStatsContactsArgs *args = (OverlayStatsContactsArgs *)pc.argsAddr;
    ManifoldSoA m = args->m;
    int t = (int)tid.x;
    if (t >= args->slots || m.contactCount[t] <= 0)
        return;

    int off = m.contactOffset[t];
    float friction = m.friction[t];

    float peakForce = 0.0f;
    float peakCone = 0.0f;
    float peakPen = 0.0f;

    for (int c = 0; c < m.contactCount[t]; c++)
    {
        vec3 lam = m.lambda[off + c];
        float fn = abs(lam.x);
        peakForce = max(peakForce, fn);

        if (args->s.cone != nullptr)
        {
            float bounds = fn * friction;
            float ft = sqrt(lam.y * lam.y + lam.z * lam.z);
            if (bounds > 1e-6f)
                peakCone = max(peakCone, min(ft / bounds, 1.0f));
        }

        if (args->s.penetration != nullptr)
            peakPen = max(peakPen, -m.C0[off + c].x);
    }

    int ia = m.bodyA[t], ib = m.bodyB[t];
    statMax(args->s.push, ia, peakForce);
    statMax(args->s.push, ib, peakForce);
    statMax(args->s.cone, ia, peakCone);
    statMax(args->s.cone, ib, peakCone);
    statMax(args->s.penetration, ia, peakPen);
    statMax(args->s.penetration, ib, peakPen);
}

struct OverlayStatsForcesArgs
{
    BodySoA b;
    JointSoA j;
    SpringSoA sp;
    int jointsAuthored;
    int _pad0;
    OverlayStats s;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csOverlayStatsForces(uint3 tid: SV_DispatchThreadID)
{
    OverlayStatsForcesArgs *args = (OverlayStatsForcesArgs *)pc.argsAddr;
    BodySoA b = args->b;
    JointSoA j = args->j;
    SpringSoA sp = args->sp;

    int t = (int)tid.x;
    int total = args->jointsAuthored + (int)sp.count;
    if (t >= total)
        return;

    bool isJoint = t < args->jointsAuthored;
    int i = isJoint ? t : t - args->jointsAuthored;

    int ia, ib;
    vec3 rA, rB;
    if (isJoint)
    {
        if (j.broken[i])
            return;
        ia = j.bodyA[i];
        ib = j.bodyB[i];
        rA = j.rA[i];
        rB = j.rB[i];
    }
    else
    {
        ia = sp.bodyA[i];
        ib = sp.bodyB[i];
        rA = sp.rA[i];
        rB = sp.rB[i];
    }
    if (ib < 0 || (uint)ib >= b.count)
        return;

    if (args->s.jointDegree != nullptr)
    {
        if (ia >= 0)
        {
            int old;
            InterlockedAdd(args->s.jointDegree[ia], 1, old);
        }
        int old2;
        InterlockedAdd(args->s.jointDegree[ib], 1, old2);
    }

    vec3 pA = ia >= 0 ? transformQ(b.pos[ia], b.rot[ia], rA) : rA;
    vec3 pB = transformQ(b.pos[ib], b.rot[ib], rB);

    float axial = 0.0f;
    if (isJoint)
        axial = jointAxialLoadR(j.lambdaLin[i], pA, pB);
    else
    {
        float len = length3(sub(pB, pA));
        axial = sp.stiffness[i] * (len - sp.rest[i]);
    }

    float tension = max(axial, 0.0f);
    float compression = max(-axial, 0.0f);
    if (ia >= 0)
    {
        statMax(args->s.pull, ia, tension);
        statMax(args->s.push, ia, compression);
    }
    statMax(args->s.pull, ib, tension);
    statMax(args->s.push, ib, compression);

    if (args->s.stress != nullptr && isJoint)
    {
        float frac = j.fracture[i];
        if (!isinf(frac) && frac > 0.0f && !isnan(frac))
        {
            float f = min(max(length3(j.lambdaAng[i]), length3(j.lambdaLin[i])) / frac, 1.0f);
            if (ia >= 0)
                statMax(args->s.stress, ia, f);
            statMax(args->s.stress, ib, f);
        }
    }
}

struct OverlayLoadClearArgs
{
    uint *load;
    int n;
    int _pad0;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csOverlayLoadClear(uint3 tid: SV_DispatchThreadID)
{
    OverlayLoadClearArgs *args = (OverlayLoadClearArgs *)pc.argsAddr;
    int i = (int)tid.x;
    if (i < args->n)
        args->load[i] = 0u;
}

struct OverlayLoadAccumArgs
{
    BodySoA b;
    ManifoldSoA m;
    int slots;
    int _pad0;
    uint *load;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csOverlayLoadAccum(uint3 tid: SV_DispatchThreadID)
{
    OverlayLoadAccumArgs *args = (OverlayLoadAccumArgs *)pc.argsAddr;
    ManifoldSoA m = args->m;
    int t = (int)tid.x;
    if (t >= args->slots || m.contactCount[t] <= 0)
        return;

    int off = m.contactOffset[t];
    float peak = 0.0f;
    for (int c = 0; c < m.contactCount[t]; c++)
        peak = max(peak, abs(m.lambda[off + c].x));

    uint bits = asuint(peak);
    uint old;
    InterlockedMax(args->load[m.bodyA[t]], bits, old);
    InterlockedMax(args->load[m.bodyB[t]], bits, old);
}
