// The drag path: the mouse grab expressed as a
// temporary joint, k_drag_update and k_drag_release. Field-for-field identical to the
// the reference source; the grabGroup/grabSurvivors host-side graph walks and the
// dragBody/dragBodies/releaseDrag orchestration stay on the CPU (they touch no device
// buffers this kernel doesn't already cover) and live in vk_interact.cpp instead.

import maths;
import types;

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

struct DragUpdateArgs
{
    BodySoA b;
    JointSoA j;
    int slot0;
    int count;
    int *bodies;
    vec3 *targets;
    vec3 *locals;
    float *stiffness;
    int grab;
    int _pad0;
};

[shader("compute")]
[numthreads(128, 1, 1)]
void csDragUpdate(uint3 tid: SV_DispatchThreadID)
{
    DragUpdateArgs *args = (DragUpdateArgs *)pc.argsAddr;
    BodySoA b = args->b;
    JointSoA j = args->j;

    int k = (int)tid.x;
    if (k >= args->count)
        return;
    int slot = args->slot0 + k;
    if (slot < 0 || (uint)slot >= j.count)
        return;

    int body = args->bodies[k];

    j.bodyA[slot] = -1;
    j.bodyB[slot] = body;
    j.rA[slot] = args->targets[k];
    j.rB[slot] = args->locals[k];
    j.stiffnessLin[slot] = args->stiffness[k];
    j.stiffnessAng[slot] = 0.0f;
    j.fracture[slot] = 1.0f / 0.0f; // INFINITY
    j.torqueArm[slot] = 0.0f;
    j.restAng[slot] = quat(0.0f, 0.0f, 0.0f, 1.0f);
    j.broken[slot] = 0;

    const float PENALTY_MIN = 1.0f; // engine/core/collision.h: PENALTY_MIN
    if (args->grab != 0)
    {
        j.penaltyLin[slot] = vec3(PENALTY_MIN, PENALTY_MIN, PENALTY_MIN);
        j.penaltyAng[slot] = vec3(0.0f, 0.0f, 0.0f);
        j.lambdaLin[slot] = vec3(0.0f, 0.0f, 0.0f);
        j.lambdaAng[slot] = vec3(0.0f, 0.0f, 0.0f);
    }

    if (body >= 0 && (uint)body < b.count)
    {
        b.awake[body] = 1;
        b.sleepTimer[body] = 0;
    }
}

struct DragReleaseArgs
{
    JointSoA j;
    int slot0;
    int count;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csDragRelease(uint3 tid: SV_DispatchThreadID)
{
    DragReleaseArgs *args = (DragReleaseArgs *)pc.argsAddr;
    JointSoA j = args->j;

    int k = (int)tid.x;
    if (k >= args->count)
        return;
    int slot = args->slot0 + k;
    if (slot < 0 || (uint)slot >= j.count)
        return;

    j.broken[slot] = 1;
    j.penaltyLin[slot] = vec3(0.0f, 0.0f, 0.0f);
    j.penaltyAng[slot] = vec3(0.0f, 0.0f, 0.0f);
    j.lambdaLin[slot] = vec3(0.0f, 0.0f, 0.0f);
    j.lambdaAng[slot] = vec3(0.0f, 0.0f, 0.0f);
}
