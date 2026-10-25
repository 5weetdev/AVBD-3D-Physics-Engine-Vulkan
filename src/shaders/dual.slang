// The dual stage -- k_dual, k_joint_init, k_joint_dual,
// k_contact_quality. Phase 4b scope note (see forces.slang): joints are the legacy
// weld path only.

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

// ---------------------------------------------------------------------------
// k_joint_init
// ---------------------------------------------------------------------------

struct JointInitArgs
{
    BodySoA b;
    JointSoA j;
    SolverParamsGpu p;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csJointInit(uint3 tid: SV_DispatchThreadID)
{
    JointInitArgs *args = (JointInitArgs *)pc.argsAddr;
    BodySoA b = args->b;
    JointSoA j = args->j;
    int i = (int)tid.x;
    if ((uint)i >= j.count || j.broken[i] != 0)
        return;
    if (jointDormant(b, j.bodyA[i], j.bodyB[i]))
        return;
    jointInitialize(j, i, b, args->p.alpha, args->p.gamma, args->p.dt,
                        args->p.dualDamping, args->p.dualDampVel);
}

// ---------------------------------------------------------------------------
// k_joint_dual
// ---------------------------------------------------------------------------

[shader("compute")]
[numthreads(256, 1, 1)]
void csJointDual(uint3 tid: SV_DispatchThreadID)
{
    JointDualArgs *args = (JointDualArgs *)pc.argsAddr;
    jointDualSlot(args, (int)tid.x);
}

// ---------------------------------------------------------------------------
// k_dual
// ---------------------------------------------------------------------------

[shader("compute")]
[numthreads(256, 1, 1)]
void csDual(uint3 tid: SV_DispatchThreadID)
{
    DualArgs *args = (DualArgs *)pc.argsAddr;
    dualSlot(args, (int)tid.x);
}

