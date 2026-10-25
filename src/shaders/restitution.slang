// Restitution (bounce). Three kernels, recorded only while some body has restitution > 0
// (VkWorld::m_anyRestitution), so a scene without bouncy bodies runs none of this.
//
// AVBD is an implicit Euler position solve: a contact ends a step at rest along its normal,
// so bounce is a velocity correction after the solve (as Box3D's relax/restitution pass):
//
//   csRestitutionPre    (per manifold slot, after the narrowphase, before the iterations)
//                       vnPre = n . (vA + wA x rA - vB - wB x rB) at the manifold centroid,
//                       with the velocities still holding v^t.
//   csRestitutionImpulse(per manifold slot, after the velocity reconstruction)
//                       e = mix(eA, eB) of the two manifold shapes' restitution. A manifold that approached faster than the threshold
//                       (vnPre < -threshold) and pushed (some lambda.x < 0) gets one normal
//                       impulse at the centroid of its pushing points that sets the relative
//                       normal velocity there to -e * vnPre.
//   csRestitutionApply  (per dynamic body) sums the impulses of its manifolds through the
//                       constraint graph and writes only its own velocity, so there is no race.
//
// Conventions as contact.slang: n = basis row 0, C.x = n . (xA - xB) and C.x < 0 penetrates,
// so an approaching pair has vn < 0. The arms are rotateQ(rot, rA/rB): from the centre of
// mass, world orientation.

import maths;

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// Per manifold slot (32 bytes): the world arms of the impulse point, the impulse, vnPre.
struct RestitutionSlot
{
    float ax;
    float ay;
    float az;
    float impulse; // >= 0, along n; 0 = no bounce this step
    float bx;
    float by;
    float bz;
    float vnPre;
};

// Mirrors BodyProps (types.slang / scene_types.h); only restitution is read here.
struct RestitutionProps
{
    float linearDamping;
    float angularDamping;
    float gravityScale;
    float maxLinearSpeed;
    float sleepThreshold;
    uint flags;
    float tangentSpeed;
    float rollingResistance;
    uint userMaterialId;
    uint frictionMix;
    float restitution;
    float tangentY;
    float tangentZ;
    uint _pad;
};

struct RestitutionArgs
{
    // body state
    vec3 *pos;
    quat *rot;
    vec3 *velLin;
    vec3 *velAng;
    float *mass;
    vec3 *moment;
    RestitutionProps *props;
    // manifold slots (read only)
    int *bodyA;
    int *bodyB;
    float3x3 *basis;
    int *contactOffset;
    int *contactCount;
    vec3 *rA;
    vec3 *rB;
    vec3 *lambda;
    // constraint graph (csRestitutionApply)
    uint64_t *gKey;
    int *gIsA;
    int *gStart;
    int *gEnd;
    int *dyn;
    RestitutionSlot *out; // slots entries
    int *shapeA;          // per manifold slot: the shapes whose pair made it
    int *shapeB;
    float *shpRestitution; // per shape
    uint slots;
    uint dynCount;
    float threshold;      // m/s: slower approaches do not bounce
    uint _pad;
};

static const int FORCE_MANIFOLD_R = 0;
int graphTypeR(uint64_t k) { return (int)((uint)(k & 0xFFFFFFFFu) >> 29); }
int graphIndexR(uint64_t k) { return (int)((uint)(k & 0xFFFFFFFFu) & ((1u << 29) - 1u)); }

// World-space inverse inertia applied to v: R diag(1/moment) R^T v. Zero for mass 0.
vec3 invInertiaMul(RestitutionArgs *a, int i, vec3 v)
{
    if (a->mass[i] <= 0.0f)
        return vec3(0, 0, 0);
    quat q = a->rot[i];
    vec3 m = a->moment[i];
    vec3 l = rotateQ(conjugateQ(q), v);
    l = vec3(m.x > 0.0f ? l.x / m.x : 0.0f, m.y > 0.0f ? l.y / m.y : 0.0f, m.z > 0.0f ? l.z / m.z : 0.0f);
    return rotateQ(q, l);
}

float pointVn(RestitutionArgs *a, int ia, int ib, vec3 n, vec3 wa, vec3 wb)
{
    vec3 va = add(a->velLin[ia], cross3(a->velAng[ia], wa));
    vec3 vb = add(a->velLin[ib], cross3(a->velAng[ib], wb));
    return dot3(n, sub(va, vb));
}

[shader("compute")]
[numthreads(128, 1, 1)]
void csRestitutionPre(uint3 tid: SV_DispatchThreadID)
{
    RestitutionArgs *a = (RestitutionArgs *)pc.argsAddr;
    uint t = tid.x;
    if (t >= a->slots)
        return;
    RestitutionSlot s;
    s.ax = 0.0f; s.ay = 0.0f; s.az = 0.0f; s.impulse = 0.0f;
    s.bx = 0.0f; s.by = 0.0f; s.bz = 0.0f; s.vnPre = 0.0f;
    int n = a->contactCount[t];
    if (n > 0)
    {
        int ia = a->bodyA[t], ib = a->bodyB[t];
        quat qa = a->rot[ia], qb = a->rot[ib];
        vec3 nrm = a->basis[t].row0;
        int off = a->contactOffset[t];
        float vn = 0.0f;
        for (int k = 0; k < n; ++k)
            vn += pointVn(a, ia, ib, nrm, rotateQ(qa, a->rA[off + k]), rotateQ(qb, a->rB[off + k]));
        s.vnPre = vn / (float)n;
    }
    a->out[t] = s;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void csRestitutionImpulse(uint3 tid: SV_DispatchThreadID)
{
    RestitutionArgs *a = (RestitutionArgs *)pc.argsAddr;
    uint t = tid.x;
    if (t >= a->slots)
        return;
    RestitutionSlot s = a->out[t];
    s.impulse = 0.0f;
    int n = a->contactCount[t];
    int ia = n > 0 ? a->bodyA[t] : 0, ib = n > 0 ? a->bodyB[t] : 0;
    float e = 0.0f;
    if (n > 0)
    {
        // Per shape (the two shapes of the manifold); a body-level value (SceneBuilder scenes) is
        // a floor under it.
        const float ea = max(a->props[ia].restitution, a->shpRestitution[a->shapeA[t]]);
        const float eb = max(a->props[ib].restitution, a->shpRestitution[a->shapeB[t]]);
        // Mixing rule: bits 8..15 of frictionMix hold the rule + 1 (FrictionMix codes), 0 = the
        // default, the larger of the two.
        const uint rule = (max(a->props[ia].frictionMix, a->props[ib].frictionMix) >> 8u) & 0xFFu;
        e = max(ea, eb);
        if (rule == 1u)
            e = sqrt(ea * eb);
        else if (rule == 2u)
            e = min(ea, eb);
        else if (rule == 4u)
            e = 0.5f * (ea + eb);
        else if (rule == 5u)
            e = ea * eb;
    }
    if (n <= 0 || e <= 0.0f || s.vnPre >= -a->threshold)
    {
        a->out[t] = s;
        return;
    }

    quat qa = a->rot[ia], qb = a->rot[ib];
    vec3 nrm = a->basis[t].row0;
    int off = a->contactOffset[t];
    vec3 ca = vec3(0, 0, 0), cb = vec3(0, 0, 0);
    int pushing = 0;
    for (int k = 0; k < n; ++k)
    {
        if (a->lambda[off + k].x >= 0.0f)
            continue;
        ca = add(ca, rotateQ(qa, a->rA[off + k]));
        cb = add(cb, rotateQ(qb, a->rB[off + k]));
        pushing++;
    }
    if (pushing == 0)
    {
        a->out[t] = s;
        return;
    }
    ca = scale(ca, 1.0f / (float)pushing);
    cb = scale(cb, 1.0f / (float)pushing);

    float ma = a->mass[ia], mb = a->mass[ib];
    vec3 raN = cross3(ca, nrm), rbN = cross3(cb, nrm);
    float k = (ma > 0.0f ? 1.0f / ma : 0.0f) + (mb > 0.0f ? 1.0f / mb : 0.0f) +
              dot3(raN, invInertiaMul(a, ia, raN)) + dot3(rbN, invInertiaMul(a, ib, rbN));
    float vnNow = pointVn(a, ia, ib, nrm, ca, cb);
    float impulse = k > 0.0f ? max((-e * s.vnPre - vnNow) / k, 0.0f) : 0.0f;
    s.ax = ca.x; s.ay = ca.y; s.az = ca.z;
    s.bx = cb.x; s.by = cb.y; s.bz = cb.z;
    s.impulse = impulse;
    a->out[t] = s;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void csRestitutionApply(uint3 tid: SV_DispatchThreadID)
{
    RestitutionArgs *a = (RestitutionArgs *)pc.argsAddr;
    if (tid.x >= a->dynCount)
        return;
    int i = a->dyn[tid.x];
    float m = a->mass[i];
    if (m <= 0.0f)
        return;
    vec3 dv = vec3(0, 0, 0), dw = vec3(0, 0, 0);
    bool any = false;
    for (int e = a->gStart[i]; e < a->gEnd[i]; e++)
    {
        uint64_t key = a->gKey[e];
        if (graphTypeR(key) != FORCE_MANIFOLD_R)
            continue;
        int mi = graphIndexR(key);
        RestitutionSlot s = a->out[mi];
        if (s.impulse <= 0.0f)
            continue;
        bool isA = a->gIsA[e] != 0;
        vec3 nrm = a->basis[mi].row0;
        vec3 p = scale(nrm, isA ? s.impulse : -s.impulse);
        vec3 r = isA ? vec3(s.ax, s.ay, s.az) : vec3(s.bx, s.by, s.bz);
        dv = add(dv, scale(p, 1.0f / m));
        dw = add(dw, cross3(r, p));
        any = true;
    }
    if (!any)
        return;
    a->velLin[i] = add(a->velLin[i], dv);
    a->velAng[i] = add(a->velAng[i], invInertiaMul(a, i, dw));
}
