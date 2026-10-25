// The wake stage -- the implicit, velocity-gated wake path (one
// contact ring per step) plus GpuSolver::wakeAll(). The explicit, ungated mutation
// flood lives in mutate.slang (mirroring the mutation path's split). Kernel-for-kernel:
// k_wake_request -> csWakeRequest, k_touch_clear -> csTouchClear,
// k_touch_accum -> csTouchAccum, k_wake_lost_support -> csWakeLostSupport,
// k_wake_apply -> csWakeApply, k_touch_accum_joints -> csTouchAccumJoints,
// k_wake_request_joints -> csWakeRequestJoints, k_wake_island_seed -> csWakeIslandSeed,
// k_wake_all -> csWakeAll. Same guards, same hysteresis, same commutative-atomic
// fingerprint construction as the reference source -- see that file for why the hysteresis
// and the joint-edge ungating are both load-bearing (a sleeping body is a fake anchor
// on a joint, never on a contact).

import maths;
import types;

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// ---------------------------------------------------------------------------
// csWakeRequest -- k_wake_request
// ---------------------------------------------------------------------------
struct WakeRequestArgs
{
    BodySoA b;
    ManifoldSoA m;
    int slots;
    float wakeLinVel;
    float wakeAngVel;
    uint8_t *request;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csWakeRequest(uint3 tid: SV_DispatchThreadID)
{
    WakeRequestArgs *args = (WakeRequestArgs *)pc.argsAddr;
    BodySoA b = args->b;
    ManifoldSoA m = args->m;
    int t = (int)tid.x;
    if (t >= args->slots || m.contactCount[t] <= 0)
        return;

    int ia = m.bodyA[t], ib = m.bodyB[t];
    bool dynA = b.mass[ia] > 0.0f, dynB = b.mass[ib] > 0.0f;

    bool movingA = dynA && b.awake[ia] != 0 &&
                   (length3(b.velLin[ia]) >= args->wakeLinVel || length3(b.velAng[ia]) >= args->wakeAngVel);
    bool movingB = dynB && b.awake[ib] != 0 &&
                   (length3(b.velLin[ib]) >= args->wakeLinVel || length3(b.velAng[ib]) >= args->wakeAngVel);

    // A kinematic body (mass 0, BODY_KINEMATIC) that is moving wakes a sleeping dynamic neighbour.
    if (b.props != nullptr)
    {
        if (!dynA && (b.props[ia].flags & BODY_KINEMATIC) != 0u &&
            (length3(b.velLin[ia]) > 1e-4f || length3(b.velAng[ia]) > 1e-4f))
            movingA = true;
        if (!dynB && (b.props[ib].flags & BODY_KINEMATIC) != 0u &&
            (length3(b.velLin[ib]) > 1e-4f || length3(b.velAng[ib]) > 1e-4f))
            movingB = true;
    }

    if (movingA && dynB && b.awake[ib] == 0)
        args->request[ib] = 1;
    if (movingB && dynA && b.awake[ia] == 0)
        args->request[ia] = 1;
}

// ---------------------------------------------------------------------------
// csTouchClear / csTouchAccum -- k_touch_clear / k_touch_accum
// ---------------------------------------------------------------------------
struct TouchClearArgs
{
    int *dyn;
    int dynCount;
    int *count;
    int *hash;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csTouchClear(uint3 tid: SV_DispatchThreadID)
{
    TouchClearArgs *args = (TouchClearArgs *)pc.argsAddr;
    int t = (int)tid.x;
    if (t >= args->dynCount)
        return;
    int i = args->dyn[t];
    args->count[i] = 0;
    args->hash[i] = 0;
}

struct TouchAccumArgs
{
    BodySoA b;
    ManifoldSoA m;
    int slots;
    int _pad0;
    int *count;
    int *hash;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csTouchAccum(uint3 tid: SV_DispatchThreadID)
{
    TouchAccumArgs *args = (TouchAccumArgs *)pc.argsAddr;
    BodySoA b = args->b;
    ManifoldSoA m = args->m;
    int t = (int)tid.x;
    if (t >= args->slots || m.contactCount[t] <= 0)
        return;

    int ia = m.bodyA[t], ib = m.bodyB[t];

    if (b.mass[ia] > 0.0f)
    {
        int dummy;
        InterlockedAdd(args->count[ia], 1, dummy);
        InterlockedXor(args->hash[ia], (int)((uint)ib * 0x9E3779B1u + 0x85EBCA6Bu));
    }
    if (b.mass[ib] > 0.0f)
    {
        int dummy;
        InterlockedAdd(args->count[ib], 1, dummy);
        InterlockedXor(args->hash[ib], (int)((uint)ia * 0x9E3779B1u + 0x85EBCA6Bu));
    }
}

// ---------------------------------------------------------------------------
// csWakeLostSupport -- k_wake_lost_support
// ---------------------------------------------------------------------------
struct WakeLostSupportArgs
{
    BodySoA b;
    int *dyn;
    int dynCount;
    int _pad0;
    int *count;
    int *hash;
    int *snapCount;
    int *snapHash;
    uint8_t *request;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csWakeLostSupport(uint3 tid: SV_DispatchThreadID)
{
    WakeLostSupportArgs *args = (WakeLostSupportArgs *)pc.argsAddr;
    BodySoA b = args->b;
    int t = (int)tid.x;
    if (t >= args->dynCount)
        return;
    int i = args->dyn[t];
    if (b.awake[i] != 0)
        return;

    bool lost = args->count[i] < args->snapCount[i] ||
                (args->count[i] == args->snapCount[i] && args->hash[i] != args->snapHash[i]);
    if (lost)
        args->request[i] = 1;
}

// ---------------------------------------------------------------------------
// csWakeApply -- k_wake_apply
// ---------------------------------------------------------------------------
struct WakeApplyArgs
{
    BodySoA b;
    int *dyn;
    int dynCount;
    int _pad0;
    uint8_t *request;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csWakeApply(uint3 tid: SV_DispatchThreadID)
{
    WakeApplyArgs *args = (WakeApplyArgs *)pc.argsAddr;
    BodySoA b = args->b;
    int t = (int)tid.x;
    if (t >= args->dynCount)
        return;
    int i = args->dyn[t];
    if (args->request[i] == 0)
        return;
    if (b.awake[i] != 0)
        return;
    b.awake[i] = 1;
    b.sleepTimer[i] = 0;
}

// ---------------------------------------------------------------------------
// csTouchAccumJoints -- k_touch_accum_joints
// ---------------------------------------------------------------------------
struct TouchAccumJointsArgs
{
    BodySoA b;
    JointSoA j;
    SpringSoA sp;
    int *count;
    int *hash;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csTouchAccumJoints(uint3 tid: SV_DispatchThreadID)
{
    TouchAccumJointsArgs *args = (TouchAccumJointsArgs *)pc.argsAddr;
    BodySoA b = args->b;
    JointSoA j = args->j;
    SpringSoA sp = args->sp;
    int t = (int)tid.x;
    int total = (int)j.count + (int)sp.count;
    if (t >= total)
        return;

    int ia, ib;
    if (t < (int)j.count)
    {
        if (j.broken[t] != 0)
            return;
        ia = j.bodyA[t];
        ib = j.bodyB[t];
    }
    else
    {
        int k = t - (int)j.count;
        ia = sp.bodyA[k];
        ib = sp.bodyB[k];
    }

    int ka = ia < 0 ? -1 : ia;
    int kb = ib < 0 ? -1 : ib;
    if (ia >= 0 && b.mass[ia] > 0.0f)
    {
        int dummy;
        InterlockedAdd(args->count[ia], 1, dummy);
        InterlockedXor(args->hash[ia], (int)((uint)kb * 0x27D4EB2Du + 0xC2B2AE35u));
    }
    if (ib >= 0 && b.mass[ib] > 0.0f)
    {
        int dummy;
        InterlockedAdd(args->count[ib], 1, dummy);
        InterlockedXor(args->hash[ib], (int)((uint)ka * 0x27D4EB2Du + 0xC2B2AE35u));
    }
}

// ---------------------------------------------------------------------------
// csWakeRequestJoints -- k_wake_request_joints
// ---------------------------------------------------------------------------
struct WakeRequestJointsArgs
{
    BodySoA b;
    JointSoA j;
    SpringSoA sp;
    uint8_t *request;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csWakeRequestJoints(uint3 tid: SV_DispatchThreadID)
{
    WakeRequestJointsArgs *args = (WakeRequestJointsArgs *)pc.argsAddr;
    BodySoA b = args->b;
    JointSoA j = args->j;
    SpringSoA sp = args->sp;
    int t = (int)tid.x;
    int total = (int)j.count + (int)sp.count;
    if (t >= total)
        return;

    int ia, ib;
    if (t < (int)j.count)
    {
        if (j.broken[t] != 0)
            return;
        ia = j.bodyA[t];
        ib = j.bodyB[t];
    }
    else
    {
        int k = t - (int)j.count;
        ia = sp.bodyA[k];
        ib = sp.bodyB[k];
    }

    bool dynA = ia >= 0 && b.mass[ia] > 0.0f;
    bool dynB = ib >= 0 && b.mass[ib] > 0.0f;

    if (dynA && b.awake[ia] != 0 && dynB && b.awake[ib] == 0)
        args->request[ib] = 1;
    if (dynB && b.awake[ib] != 0 && dynA && b.awake[ia] == 0)
        args->request[ia] = 1;
}

// ---------------------------------------------------------------------------
// csWakeIslandSeed -- k_wake_island_seed
// ---------------------------------------------------------------------------
struct WakeIslandSeedArgs
{
    BodySoA b;
    int *dyn;
    int dynCount;
    int _pad0;
    uint8_t *request;
    uint8_t *seed;
    int *any;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csWakeIslandSeed(uint3 tid: SV_DispatchThreadID)
{
    WakeIslandSeedArgs *args = (WakeIslandSeedArgs *)pc.argsAddr;
    BodySoA b = args->b;
    int t = (int)tid.x;
    if (t >= args->dynCount)
        return;
    int i = args->dyn[t];
    if (b.awake[i] == 0 && args->request[i] == 0)
        return;
    args->seed[i] = 1;
    *args->any = 1;
}

// ---------------------------------------------------------------------------
// csWakeAll -- k_wake_all (GpuSolver::wakeAll())
// ---------------------------------------------------------------------------
struct WakeAllArgs
{
    BodySoA b;
    int *dyn;
    int dynCount;
    int _pad0;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csWakeAll(uint3 tid: SV_DispatchThreadID)
{
    WakeAllArgs *args = (WakeAllArgs *)pc.argsAddr;
    BodySoA b = args->b;
    int t = (int)tid.x;
    if (t >= args->dynCount)
        return;
    int i = args->dyn[t];
    b.awake[i] = 1;
    b.sleepTimer[i] = 0;
}
