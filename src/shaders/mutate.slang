// The mutation path -- the batched external mutation apply
// (k_mutate_apply -> csMutateApply, k_body_gather -> csBodyGather) and the explicit,
// UNGATED wake flood every mutation seeds (k_wake_seed_region -> csWakeSeedRegion,
// k_wake_flood_contacts -> csWakeFloodContacts, k_wake_flood_joints -> csWakeFloodJoints,
// k_wake_seed_merge -> csWakeSeedMerge). See that file's header comment for why this
// path has no velocity gate: a teleported/impulsed/grabbed body has not moved yet when
// the mutation lands, so a gate would silently drop the first frames of every
// interaction.
//
// `type` mirrors GpuSolver::BodyWrite / BodyRead as a plain int (Transform=0,
// LinearVelocity=1, AngularVelocity=2, Impulse=3, Force=4, Torque=5, Awake=6 for writes;
// Pose=0, LinearVelocity=1, AngularVelocity=2, Awake=3 for reads) -- the host picks the
// same numeric values the real C++ enums have, so this file need not know their names.
//
// The `data`/`out` byte buffers are read/written through a typed pointer cast plus
// ordinary array indexing (`((BodyTransform *)args->data)[t]`), exactly mirroring the
// the reference source's `((const GpuSolver::BodyTransform *)data)[t]` -- Slang raw pointers
// compute the same byte stride from the cast-to type, so no explicit sizeof/byte-offset
// arithmetic is needed here.

import maths;
import types;

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// Mirrors GpuSolver::BodyTransform / BodyPointVector (engine/host/gpu_solver.h) byte for
// byte: vec3 (12B) + quat (16B) = 28B, vec3 + vec3 = 24B, both 4-byte aligned throughout
// (no 8-byte pointers inside), so no padding is needed on either side.
struct BodyTransform
{
    vec3 pos;
    quat rot;
};

struct BodyPointVector
{
    vec3 vector;
    vec3 worldPoint;
};

static const int BW_TRANSFORM = 0;
static const int BW_LINVEL = 1;
static const int BW_ANGVEL = 2;
static const int BW_IMPULSE = 3;
static const int BW_FORCE = 4;
static const int BW_TORQUE = 5;
static const int BW_AWAKE = 6;
static const int BW_PROPS = 7;
static const int BW_TARGET = 8;

// Mirrors scene_types.h BodyTarget: pos, rot, dt.
struct BodyTarget
{
    vec3 pos;
    quat rot;
    float dt;
};

static const int BR_POSE = 0;
static const int BR_LINVEL = 1;
static const int BR_ANGVEL = 2;
static const int BR_AWAKE = 3;

// ---------------------------------------------------------------------------
// csMutateApply -- k_mutate_apply
// ---------------------------------------------------------------------------
struct MutateApplyArgs
{
    BodySoA b;
    int type;
    int *idx;
    uint8_t *data;
    int count;
    float dt;
    uint8_t *seed;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csMutateApply(uint3 tid: SV_DispatchThreadID)
{
    MutateApplyArgs *args = (MutateApplyArgs *)pc.argsAddr;
    BodySoA b = args->b;
    int t = (int)tid.x;
    if (t >= args->count)
        return;
    int i = args->idx[t];
    if (i < 0 || (uint)i >= b.count)
        return;

    float mass = b.mass[i];
    bool sleepSeed = true; // set false only by the forced-sleep Awake(false) branch
    // A kinematic body (mass 0, BODY_KINEMATIC) takes velocity writes but is never a wake seed.
    bool kinematic = mass <= 0.0f && b.props != nullptr && (b.props[i].flags & BODY_KINEMATIC) != 0u;

    if (args->type == BW_PROPS)
    {
        if (b.props != nullptr)
        {
            BodyProps np = ((BodyProps *)args->data)[t];
            np.flags = (np.flags & ~BODY_KINEMATIC) | (b.props[i].flags & BODY_KINEMATIC);
            b.props[i] = np;
        }
        return;
    }
    else if (args->type == BW_TARGET)
    {
        if (kinematic)
        {
            BodyTarget tg = ((BodyTarget *)args->data)[t];
            float idt = tg.dt > 0.0f ? 1.0f / tg.dt : 0.0f;
            b.velLin[i] = scale(sub(tg.pos, b.pos[i]), idt);
            b.velAng[i] = scale(subQQ(tg.rot, b.rot[i]), idt);
        }
        return;
    }
    else if (args->type == BW_TRANSFORM)
    {
        BodyTransform x = ((BodyTransform *)args->data)[t];
        b.pos[i] = x.pos;
        b.rot[i] = x.rot;

        b.restPos[i] = x.pos;
        b.restRot[i] = x.rot;
        b.sleepTimer[i] = 0;
    }
    else if (args->type == BW_LINVEL)
    {
        if (mass > 0.0f || kinematic)
            b.velLin[i] = ((vec3 *)args->data)[t];
    }
    else if (args->type == BW_ANGVEL)
    {
        if (mass > 0.0f || kinematic)
            b.velAng[i] = ((vec3 *)args->data)[t];
    }
    else if (args->type == BW_IMPULSE || args->type == BW_FORCE)
    {
        if (mass > 0.0f)
        {
            BodyPointVector pv = ((BodyPointVector *)args->data)[t];
            vec3 J = (args->type == BW_FORCE) ? scale(pv.vector, args->dt) : pv.vector;
            b.velLin[i] = add(b.velLin[i], divs(J, mass));

            vec3 torque = cross3(sub(pv.worldPoint, b.pos[i]), J);
            quat rot = b.rot[i];
            vec3 tLocal = rotateQ(conjugateQ(rot), torque);
            vec3 mom = b.moment[i];
            vec3 dwLocal = vec3(mom.x > 0.0f ? tLocal.x / mom.x : 0.0f,
                                 mom.y > 0.0f ? tLocal.y / mom.y : 0.0f,
                                 mom.z > 0.0f ? tLocal.z / mom.z : 0.0f);
            b.velAng[i] = add(b.velAng[i], rotateQ(rot, dwLocal));
        }
    }
    else if (args->type == BW_TORQUE)
    {
        if (mass > 0.0f)
        {
            vec3 torque = scale(((vec3 *)args->data)[t], args->dt);
            quat rot = b.rot[i];
            vec3 tLocal = rotateQ(conjugateQ(rot), torque);
            vec3 mom = b.moment[i];
            vec3 dwLocal = vec3(mom.x > 0.0f ? tLocal.x / mom.x : 0.0f,
                                 mom.y > 0.0f ? tLocal.y / mom.y : 0.0f,
                                 mom.z > 0.0f ? tLocal.z / mom.z : 0.0f);
            b.velAng[i] = add(b.velAng[i], rotateQ(rot, dwLocal));
        }
    }
    else if (args->type == BW_AWAKE)
    {
        if (mass <= 0.0f)
            return;
        if (args->data[t] != 0)
        {
            // waking is the seed below; nothing else to do
        }
        else
        {
            b.awake[i] = 0;
            b.velLin[i] = vec3(0.0f, 0.0f, 0.0f);
            b.velAng[i] = vec3(0.0f, 0.0f, 0.0f);
            b.prevVelLin[i] = vec3(0.0f, 0.0f, 0.0f);
            b.sleepTimer[i] = 0;
            b.restPos[i] = b.pos[i];
            b.restRot[i] = b.rot[i];
            return; // explicitly NOT a wake seed
        }
    }

    if (mass > 0.0f)
        args->seed[i] = 1;
}

// ---------------------------------------------------------------------------
// csWakeSeedRegion -- k_wake_seed_region
// ---------------------------------------------------------------------------
struct WakeSeedRegionArgs
{
    BodySoA b;
    int *dyn;
    int dynCount;
    int _pad0;
    vec3 lo;
    vec3 hi;
    uint8_t *seed;
    int *changed;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csWakeSeedRegion(uint3 tid: SV_DispatchThreadID)
{
    WakeSeedRegionArgs *args = (WakeSeedRegionArgs *)pc.argsAddr;
    BodySoA b = args->b;
    int t = (int)tid.x;
    if (t >= args->dynCount)
        return;
    int i = args->dyn[t];
    vec3 p = b.pos[i];
    if (p.x < args->lo.x || p.x > args->hi.x || p.y < args->lo.y || p.y > args->hi.y ||
        p.z < args->lo.z || p.z > args->hi.z)
        return;
    args->seed[i] = 1;
    *args->changed = 1;
}

// ---------------------------------------------------------------------------
// csWakeFloodContacts -- k_wake_flood_contacts
// ---------------------------------------------------------------------------
struct WakeFloodContactsArgs
{
    BodySoA b;
    ManifoldSoA m;
    int slots;
    int _pad0;
    uint8_t *seed;
    int *changed;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csWakeFloodContacts(uint3 tid: SV_DispatchThreadID)
{
    WakeFloodContactsArgs *args = (WakeFloodContactsArgs *)pc.argsAddr;
    BodySoA b = args->b;
    ManifoldSoA m = args->m;
    int t = (int)tid.x;
    if (t >= args->slots || m.contactCount[t] <= 0)
        return;

    int ia = m.bodyA[t], ib = m.bodyB[t];
    bool dynA = b.mass[ia] > 0.0f, dynB = b.mass[ib] > 0.0f;
    if (!dynA || !dynB)
        return;

    uint8_t sa = args->seed[ia], sb = args->seed[ib];
    if (sa == sb)
        return;
    if (sa != 0)
    {
        args->seed[ib] = 1;
        *args->changed = 1;
    }
    else
    {
        args->seed[ia] = 1;
        *args->changed = 1;
    }
}

// ---------------------------------------------------------------------------
// csWakeFloodJoints -- k_wake_flood_joints
// ---------------------------------------------------------------------------
struct WakeFloodJointsArgs
{
    BodySoA b;
    JointSoA j;
    SpringSoA sp;
    uint8_t *seed;
    int *changed;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csWakeFloodJoints(uint3 tid: SV_DispatchThreadID)
{
    WakeFloodJointsArgs *args = (WakeFloodJointsArgs *)pc.argsAddr;
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
    if (!dynA || !dynB)
        return;

    uint8_t sa = args->seed[ia], sb = args->seed[ib];
    if (sa == sb)
        return;
    if (sa != 0)
    {
        args->seed[ib] = 1;
        *args->changed = 1;
    }
    else
    {
        args->seed[ia] = 1;
        *args->changed = 1;
    }
}

// ---------------------------------------------------------------------------
// csWakeSeedMerge -- k_wake_seed_merge
// ---------------------------------------------------------------------------
struct WakeSeedMergeArgs
{
    int *dyn;
    int dynCount;
    int _pad0;
    uint8_t *seed;
    uint8_t *request;
    // M4: when set, the merge is skipped while *gate is nonzero -- the flood's last
    // recorded round still changed something, so the seeds are not a fixed point yet
    // and must survive for the host to record more rounds. Null = merge unconditionally.
    int *gate;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csWakeSeedMerge(uint3 tid: SV_DispatchThreadID)
{
    WakeSeedMergeArgs *args = (WakeSeedMergeArgs *)pc.argsAddr;
    int t = (int)tid.x;
    if (t >= args->dynCount)
        return;
    if (args->gate != nullptr && args->gate[0] != 0)
        return;
    int i = args->dyn[t];
    if (args->seed[i] != 0)
    {
        args->request[i] = 1;
        args->seed[i] = 0;
    }
}

// ---------------------------------------------------------------------------
// csBodyGather -- k_body_gather
// ---------------------------------------------------------------------------
struct BodyGatherArgs
{
    BodySoA b;
    int type;
    int *idx;
    uint8_t *out;
    int count;
    int _pad0; // tail pad: round 152+4+8+8+4=176(+4)=180 up to 184
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csBodyGather(uint3 tid: SV_DispatchThreadID)
{
    BodyGatherArgs *args = (BodyGatherArgs *)pc.argsAddr;
    BodySoA b = args->b;
    int t = (int)tid.x;
    if (t >= args->count)
        return;
    int i = args->idx[t];
    bool valid = i >= 0 && (uint)i < b.count;

    if (args->type == BR_POSE)
    {
        BodyTransform x;
        x.pos = valid ? b.pos[i] : vec3(0.0f, 0.0f, 0.0f);
        x.rot = valid ? b.rot[i] : quat(0.0f, 0.0f, 0.0f, 1.0f);
        ((BodyTransform *)args->out)[t] = x;
    }
    else if (args->type == BR_LINVEL)
    {
        ((vec3 *)args->out)[t] = valid ? b.velLin[i] : vec3(0.0f, 0.0f, 0.0f);
    }
    else if (args->type == BR_ANGVEL)
    {
        ((vec3 *)args->out)[t] = valid ? b.velAng[i] : vec3(0.0f, 0.0f, 0.0f);
    }
    else if (args->type == BR_AWAKE)
    {
        args->out[t] = valid ? b.awake[i] : (uint8_t)0;
    }
}

// ---------------------------------------------------------------------------
// csJointWrite -- the joint row rewrite (VkWorld::setJointRows)
// One record per row whose authored fields changed. Mirrors
// engine/vk/vk_args.h:JointWriteRecGpu (176 bytes, 4-byte scalars only, so the
// host struct and this one need no padding reasoning). The authored fields are
// copied into the device SoA. Body indices, the broken flag and the dual state (C0 is rebuilt each step,
// penalty and lambda carry over) are left alone, so a changed motor speed does not
// restart the row's warm start.
// ---------------------------------------------------------------------------
struct JointRowWrite
{
    uint row;
    uint _pad0;
    float rA[3];
    float rB[3];
    float restAng[4];
    float frameA[4];
    float frameB[4];
    float kLin;
    float kAng;
    float frac;
    float fracAng;
    float arm;
    uint generic;
    uint axisMode[6];
    float axisLo[6];
    float axisHi[6];
};

struct JointWriteArgs
{
    JointSoA j;
    JointRowWrite *recs;
    int count;
    uint _pad0;
};

[shader("compute")]
[numthreads(64, 1, 1)]
void csJointWrite(uint3 tid: SV_DispatchThreadID)
{
    JointWriteArgs *args = (JointWriteArgs *)pc.argsAddr;
    int t = (int)tid.x;
    if (t >= args->count)
        return;
    JointSoA j = args->j;
    JointRowWrite r = args->recs[t];
    if (r.row >= j.count)
        return;
    uint i = r.row;

    j.rA[i] = vec3(r.rA[0], r.rA[1], r.rA[2]);
    j.rB[i] = vec3(r.rB[0], r.rB[1], r.rB[2]);
    quat ra;
    ra.x = r.restAng[0]; ra.y = r.restAng[1]; ra.z = r.restAng[2]; ra.w = r.restAng[3];
    j.restAng[i] = ra;
    quat fa;
    fa.x = r.frameA[0]; fa.y = r.frameA[1]; fa.z = r.frameA[2]; fa.w = r.frameA[3];
    j.frameA[i] = fa;
    quat fb;
    fb.x = r.frameB[0]; fb.y = r.frameB[1]; fb.z = r.frameB[2]; fb.w = r.frameB[3];
    j.frameB[i] = fb;
    j.stiffnessLin[i] = r.kLin;
    j.stiffnessAng[i] = r.kAng;
    j.fracture[i] = r.frac;
    if (j.fractureAng != nullptr)
        j.fractureAng[i] = r.fracAng;
    j.torqueArm[i] = r.arm;
    if (j.generic != nullptr)
        j.generic[i] = (uint8_t)r.generic;
    for (int k = 0; k < 6; k++)
    {
        if (j.axisMode != nullptr)
            j.axisMode[i * 6 + k] = (uint8_t)r.axisMode[k];
        if (j.axisLo != nullptr)
            j.axisLo[i * 6 + k] = r.axisLo[k];
        if (j.axisHi != nullptr)
            j.axisHi[i * 6 + k] = r.axisHi[k];
    }

}
