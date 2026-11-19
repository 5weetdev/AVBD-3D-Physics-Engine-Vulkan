// Phase-1 spike: k_integrate and k_velocity.
// Same operation order as the reference source so the two are easy to diff by eye; see
// tools/test_vk_integrate.cpp for the numerical comparison against the CPU reference.
//
// Kill/explosion-guard kernels are NOT ported here -- out of scope for the toolchain
// spike, which only needs to prove the inertial predictor and BDF1 velocity
// reconstruction round-trip through Slang cleanly.

import maths;

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// Mirrors the subset of BodySoA (the device structs) that k_integrate and
// k_velocity touch. Field order does not need to match the reference struct -- each field
// is its own pointer, so there is no shared layout to keep in sync -- but is kept in
// the same order anyway for easy comparison.
struct BodySoA
{
    vec3 *pos;
    quat *rot;
    vec3 *initialLin;
    quat *initialAng;
    vec3 *inertialLin;
    quat *inertialAng;
    vec3 *velLin;
    vec3 *velAng;
    vec3 *prevVelLin;
    float *mass;
    uint8_t *awake;
    uint count;
    // Explicit tail padding to round the struct to 8-byte alignment (it holds
    // 8-byte-aligned pointers). A C++ host struct with the same fields is forced to
    // this size by the standard's array-stride guarantee; Slang's scalar layout does
    // not automatically add it for a struct used only as a field (see the pose-bundle
    // layout bug this mirrors in engine/shaders/smoke.slang), so it must be explicit
    // here for BodySoA to match tools/test_vk_integrate.cpp's BodySoAAddr byte-for-byte.
    uint _pad0;
};

struct SolverParams
{
    float dt;
    float gravityX;
    float gravityY;
    float gravityZ;
    uint spinExact; // SolverParams.exactSpinVelocity: 1 = spinStep's exact inverse, 0 = legacy 2 vec(d)
    uint _padP;
};

struct IntegrateArgs
{
    BodySoA b;
    SolverParams p;
};

// The rotation of one step, w dt, from d = rot * inverse(rot0). Exact inverse of addQV
// (q' = normalize(q + 0.5 [w dt, 0] q)): d is proportional to (w dt / 2, 1), so
// w dt = 2 vec(d) / d.w. The plain 2 vec(d) loses (w dt)^2 / 8 of the spin every step (13%/s at
// 8 rad/s). The inverse diverges as a step nears half a turn, and addQV maps a huge w dt back to
// almost half a turn, so the step is capped at kMaxStepRotation (Box3D caps it too); without the
// cap Spring Ratio's stiff off-centre springs spun bodies up to 1e7 rad/s.
static const float kMaxStepRotation = 0.5f * 3.14159265f;
vec3 spinStep(quat d, uint exact)
{
    if (exact == 0u)
        return scale(vec3(d.x, d.y, d.z), 2.0f); // legacy form, kept so the golden gates replay
    const float dw = d.w < 0.0f ? -max(-d.w, 1e-6f) : max(d.w, 1e-6f);
    vec3 dv = scale(vec3(d.x, d.y, d.z), 2.0f / dw);
    const float a = sqrt(dv.x * dv.x + dv.y * dv.y + dv.z * dv.z);
    if (a > kMaxStepRotation)
        dv = scale(dv, kMaxStepRotation / a);
    return dv;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csIntegrate(uint3 tid: SV_DispatchThreadID)
{
    IntegrateArgs *args = (IntegrateArgs *)pc.argsAddr;
    BodySoA b = args->b;
    SolverParams p = args->p;

    uint i = tid.x;
    if (i >= b.count)
        return;

    const float dt = p.dt;
    const vec3 g = vec3(p.gravityX, p.gravityY, p.gravityZ);
    const float gLen = sqrt(lengthSq3(g));
    const float mass = b.mass[i];

    vec3 pos = b.pos[i];
    quat rot = b.rot[i];
    vec3 velLin = b.velLin[i];
    vec3 velAng = b.velAng[i];

    // Inertial position (Eq 2)
    vec3 inertialLin = add(pos, scale(velLin, dt));
    if (mass > 0.0f)
        inertialLin = add(inertialLin, scale(g, dt * dt));
    b.inertialLin[i] = inertialLin;
    b.inertialAng[i] = addQV(rot, scale(velAng, dt));

    // Adaptive warmstart: the component of the measured acceleration along gravity, as a fraction of |g|.
    vec3 accel = divs(sub(velLin, b.prevVelLin[i]), dt);
    float accelWeight = clampf(dot3(accel, g) / (gLen * gLen), 0.0f, 1.0f);
    if (!(accelWeight == accelWeight) || isinf(accelWeight)) // !isfinite (zero gravity)
        accelWeight = 0.0f;

    b.initialLin[i] = pos;
    b.initialAng[i] = rot;
    if (mass > 0.0f && b.awake[i] != 0)
    {
        b.pos[i] = add(add(pos, scale(velLin, dt)), scale(g, accelWeight * dt * dt));
        b.rot[i] = addQV(rot, scale(velAng, dt));
    }
}

struct VelocityArgs
{
    BodySoA b;
    SolverParams p;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csVelocity(uint3 tid: SV_DispatchThreadID)
{
    VelocityArgs *args = (VelocityArgs *)pc.argsAddr;
    BodySoA b = args->b;
    SolverParams p = args->p;

    uint i = tid.x;
    if (i >= b.count)
        return;

    b.prevVelLin[i] = b.velLin[i];
    if (b.mass[i] > 0.0f)
    {
        // quat - quat (maths.h): (a * inverse(b)).vec() * 2
        quat rot = b.rot[i];
        quat rot0 = b.initialAng[i];
        quat conj = quat(-rot0.x, -rot0.y, -rot0.z, rot0.w);
        float lsq = rot0.x * rot0.x + rot0.y * rot0.y + rot0.z * rot0.z + rot0.w * rot0.w;
        quat invRot0 = quat(conj.x / lsq, conj.y / lsq, conj.z / lsq, conj.w / lsq);
        quat d = mulQQ(rot, invRot0);
        vec3 dv = spinStep(d, p.spinExact);

        b.velLin[i] = divs(sub(b.pos[i], b.initialLin[i]), p.dt);
        b.velAng[i] = divs(dv, p.dt);
    }
}

// The integrate stage -- k_kill_fallen and k_explosion_guard. Both
// run once per step over the dynamic list, after k_velocity, exactly where VkWorld::step()
// places them: kill_fallen culls anything that fell out of the world, explosion_guard
// is the last-resort circuit breaker for a non-finite or WORLD_BOUND-exceeding pose.
// Phase 5A note: VkWorld::step() must run these before broadphase sees next step's
// pose, same ordering constraint the reference comment documents.

static const float WORLD_BOUND = 100000.0f;

// Separate pointer fields rather than adding to BodySoA above: BodySoA's layout is
// pinned byte-for-byte to tools/test_vk_integrate.cpp's BodySoAAddr (see the struct's
// own padding comment), so the rest/sleep fields these two kernels need travel as their
// own args-struct pointers instead of growing that shared struct.
struct KillFallenArgs
{
    BodySoA b;
    int *dyn;
    int dynCount;
    float killHeight; // kill threshold along (upX, upY, upZ), the unit up axis
    float upX;
    float upY;
    float upZ;
    float _padK;
    int *sleepTimer;
    vec3 *restPos;
    quat *restRot;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csKillFallen(uint3 tid: SV_DispatchThreadID)
{
    KillFallenArgs *args = (KillFallenArgs *)pc.argsAddr;
    int t = (int)tid.x;
    if (t >= args->dynCount)
        return;
    int i = args->dyn[t];
    BodySoA b = args->b;
    if (b.awake[i] == 0 || dot3(b.pos[i], vec3(args->upX, args->upY, args->upZ)) >= args->killHeight)
        return;

    b.awake[i] = 0;
    b.velLin[i] = vec3(0, 0, 0);
    b.velAng[i] = vec3(0, 0, 0);
    b.prevVelLin[i] = vec3(0, 0, 0);
    args->sleepTimer[i] = 0;
    args->restPos[i] = b.pos[i];
    args->restRot[i] = b.rot[i];
}

float clampFinite(float x, float bound)
{
    if (!(x == x) || isinf(x)) // !isfinite
        return 0.0f;
    return max(-bound, min(bound, x));
}

struct ExplosionGuardArgs
{
    BodySoA b;
    int *dyn;
    int dynCount;
    int *sleepTimer;
    vec3 *restPos;
    quat *restRot;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csExplosionGuard(uint3 tid: SV_DispatchThreadID)
{
    ExplosionGuardArgs *args = (ExplosionGuardArgs *)pc.argsAddr;
    int t = (int)tid.x;
    if (t >= args->dynCount)
        return;
    int i = args->dyn[t];
    BodySoA b = args->b;
    if (b.awake[i] == 0)
        return;

    vec3 p = b.pos[i];
    quat r = b.rot[i];
    bool pFinite = (p.x == p.x) && (p.y == p.y) && (p.z == p.z) && !isinf(p.x) && !isinf(p.y) && !isinf(p.z);
    bool rFinite = (r.x == r.x) && (r.y == r.y) && (r.z == r.z) && (r.w == r.w) &&
                   !isinf(r.x) && !isinf(r.y) && !isinf(r.z) && !isinf(r.w);
    bool finite = pFinite && rFinite;
    bool inBounds = finite && abs(p.x) <= WORLD_BOUND && abs(p.y) <= WORLD_BOUND && abs(p.z) <= WORLD_BOUND;
    if (inBounds)
        return;

    vec3 clamped = vec3(clampFinite(p.x, WORLD_BOUND), clampFinite(p.y, WORLD_BOUND), clampFinite(p.z, WORLD_BOUND));
    bool rotOk = rFinite && (r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w) > 1e-12f;
    quat rot = rotOk ? normalizeQ(r) : quat(0, 0, 0, 1);

    b.pos[i] = clamped;
    b.rot[i] = rot;
    b.velLin[i] = vec3(0, 0, 0);
    b.velAng[i] = vec3(0, 0, 0);
    b.prevVelLin[i] = vec3(0, 0, 0);
    b.awake[i] = 0;
    args->sleepTimer[i] = 0;
    args->restPos[i] = clamped;
    args->restRot[i] = rot;
}

// ---------------------------------------------------------------------------
// Per-body settings. csIntegrateProps / csVelocityProps replace csIntegrate /
// csVelocity only in a world where some body has a non-default BodyProps; the host
// dispatches the originals otherwise, so a default scene runs the code above bit for bit.
// They carry the props pointer as a separate args field (BodySoA above is byte-pinned).
//   damping        v *= 1/(1+dt*c), applied to the velocity that builds the inertial target
//                  (the stored velocity is untouched, so the adaptive warmstart's
//                  acceleration estimate still sees the real one)
//   gravityScale   scales gravity in the target and in the warmstart
//   maxLinearSpeed clamps the predicted velocity here and the reconstructed one in
//                  csVelocityProps
//   kinematic      mass 0 and BODY_KINEMATIC: advanced by its own velocity, never solved.
//                  Its initial pose is the pose before the move, so a contact sees the
//                  motion exactly as it sees a warm-started dynamic neighbour.
// BodyPropsI is a private copy of types.slang's BodyProps (this module cannot import types).
// ---------------------------------------------------------------------------
struct BodyPropsI
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

static const uint PROP_KINEMATIC = 1u;

struct IntegratePropsArgs
{
    BodySoA b;
    SolverParams p;
    BodyPropsI *props;
};

vec3 clampSpeed(vec3 v, float maxSpeed)
{
    if (maxSpeed <= 0.0f)
        return v;
    float s2 = lengthSq3(v);
    if (s2 > maxSpeed * maxSpeed)
        return scale(v, maxSpeed / sqrt(s2));
    return v;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csIntegrateProps(uint3 tid: SV_DispatchThreadID)
{
    IntegratePropsArgs *args = (IntegratePropsArgs *)pc.argsAddr;
    BodySoA b = args->b;
    SolverParams p = args->p;

    uint i = tid.x;
    if (i >= b.count)
        return;

    BodyPropsI pr = args->props[i];
    const float dt = p.dt;
    const float mass = b.mass[i];

    vec3 pos = b.pos[i];
    quat rot = b.rot[i];
    vec3 velLin = b.velLin[i];
    vec3 velAng = b.velAng[i];

    if (mass <= 0.0f && (pr.flags & PROP_KINEMATIC) != 0u)
    {
        vec3 np = add(pos, scale(velLin, dt));
        quat nr = addQV(rot, scale(velAng, dt));
        b.inertialLin[i] = np;
        b.inertialAng[i] = nr;
        b.initialLin[i] = pos;
        b.initialAng[i] = rot;
        b.pos[i] = np;
        b.rot[i] = nr;
        return;
    }

    // Motion locks: a locked axis carries no velocity and feels no gravity (the primal holds
    // its coordinate). This path is not hash-pinned, so it uses the full gravity vector.
    vec3 g = scale(vec3(p.gravityX, p.gravityY, p.gravityZ), pr.gravityScale);
    if (mass > 0.0f && (pr.flags & 252u) != 0u)
    {
        velLin = vec3((pr.flags & 4u) != 0u ? 0.0f : velLin.x, (pr.flags & 8u) != 0u ? 0.0f : velLin.y,
                      (pr.flags & 16u) != 0u ? 0.0f : velLin.z);
        velAng = vec3((pr.flags & 32u) != 0u ? 0.0f : velAng.x, (pr.flags & 64u) != 0u ? 0.0f : velAng.y,
                      (pr.flags & 128u) != 0u ? 0.0f : velAng.z);
        b.velLin[i] = velLin;
        b.velAng[i] = velAng;
        g = vec3((pr.flags & 4u) != 0u ? 0.0f : g.x, (pr.flags & 8u) != 0u ? 0.0f : g.y,
                 (pr.flags & 16u) != 0u ? 0.0f : g.z);
    }

    vec3 vLin = velLin;
    vec3 vAng = velAng;
    if (mass > 0.0f)
    {
        if (pr.linearDamping > 0.0f)
            vLin = scale(vLin, 1.0f / (1.0f + dt * pr.linearDamping));
        if (pr.angularDamping > 0.0f)
            vAng = scale(vAng, 1.0f / (1.0f + dt * pr.angularDamping));
        vLin = clampSpeed(vLin, pr.maxLinearSpeed);
    }

    vec3 inertialLin = add(pos, scale(vLin, dt));
    if (mass > 0.0f)
        inertialLin = add(inertialLin, scale(g, dt * dt));
    b.inertialLin[i] = inertialLin;
    b.inertialAng[i] = addQV(rot, scale(vAng, dt));

    vec3 accel = divs(sub(velLin, b.prevVelLin[i]), dt);
    // Component of the measured acceleration along this body's gravity, as a fraction of |g|.
    const float gLen2 = lengthSq3(g);
    float accelWeight = clampf((accel.x * g.x + accel.y * g.y + accel.z * g.z) / gLen2, 0.0f, 1.0f);
    if (!(accelWeight == accelWeight) || isinf(accelWeight))
        accelWeight = 0.0f;

    b.initialLin[i] = pos;
    b.initialAng[i] = rot;
    if (mass > 0.0f && b.awake[i] != 0)
    {
        b.pos[i] = add(add(pos, scale(vLin, dt)), scale(g, accelWeight * dt * dt));
        b.rot[i] = addQV(rot, scale(vAng, dt));
    }
}

struct VelocityPropsArgs
{
    BodySoA b;
    SolverParams p;
    BodyPropsI *props;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csVelocityProps(uint3 tid: SV_DispatchThreadID)
{
    VelocityPropsArgs *args = (VelocityPropsArgs *)pc.argsAddr;
    BodySoA b = args->b;
    SolverParams p = args->p;

    uint i = tid.x;
    if (i >= b.count)
        return;

    b.prevVelLin[i] = b.velLin[i];
    if (b.mass[i] > 0.0f)
    {
        quat rot = b.rot[i];
        quat rot0 = b.initialAng[i];
        quat conj = quat(-rot0.x, -rot0.y, -rot0.z, rot0.w);
        float lsq = rot0.x * rot0.x + rot0.y * rot0.y + rot0.z * rot0.z + rot0.w * rot0.w;
        quat invRot0 = quat(conj.x / lsq, conj.y / lsq, conj.z / lsq, conj.w / lsq);
        quat d = mulQQ(rot, invRot0);
        vec3 dv = spinStep(d, p.spinExact);

        vec3 vLin = divs(sub(b.pos[i], b.initialLin[i]), p.dt);
        b.velLin[i] = clampSpeed(vLin, args->props[i].maxLinearSpeed);
        b.velAng[i] = divs(dv, p.dt);
    }
}
