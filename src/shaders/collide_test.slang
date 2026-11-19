// Test-only entry points over collide.slang's contact generators (avbd_test_vk_collide and
// avbd_test_vk_shapes): the solver's kernels import collide.slang as a library and never dispatch these.

import maths;
import collide;

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// ---------------------------------------------------------------------------
// Shape-aware buffers/kernel: like BodyBoxSoA but with a per-body shape tag, so the
// same dispatch csCollideShapes drives box-box, round-round and mixed pairs -- unlike
// csCollideBoxes above, which is box-only (kept for the phase-3 spike's exact
// box-box comparison).
// ---------------------------------------------------------------------------
struct BodyShapeSoA
{
    vec3 *pos;
    quat *rot;
    vec3 *halfExtents;
    int *shape;
    uint count;
    uint _pad0;
};

struct CollideShapesArgs
{
    BodyShapeSoA a;
    BodyShapeSoA b;
    CollideOut out;
};

[shader("compute")]
[numthreads(64, 1, 1)]
void csCollideShapes(uint3 tid: SV_DispatchThreadID)
{
    CollideShapesArgs *args = (CollideShapesArgs *)pc.argsAddr;
    uint i = tid.x;
    if (i >= args->a.count)
        return;

    vec3 posA = args->a.pos[i];
    quat rotA = args->a.rot[i];
    vec3 halfA = args->a.halfExtents[i];
    int shapeA = args->a.shape[i];
    vec3 posB = args->b.pos[i];
    quat rotB = args->b.rot[i];
    vec3 halfB = args->b.halfExtents[i];
    int shapeB = args->b.shape[i];

    ContactOut contacts[MAX_CONTACTS];
    float3x3 basis;
    int count = collideShapesImpl(posA, rotA, halfA, shapeA, posB, rotB, halfB, shapeB, contacts, basis);

    args->out.count[i] = count;
    args->out.basis[i] = basis;
    for (int k = 0; k < MAX_CONTACTS; ++k)
        args->out.contacts[i * MAX_CONTACTS + k] = contacts[k];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csCollideBoxes(uint3 tid: SV_DispatchThreadID)
{
    CollideArgs *args = (CollideArgs *)pc.argsAddr;
    uint i = tid.x;
    if (i >= args->a.count)
        return;

    vec3 posA = args->a.pos[i];
    quat rotA = args->a.rot[i];
    vec3 halfA = args->a.halfExtents[i];
    vec3 posB = args->b.pos[i];
    quat rotB = args->b.rot[i];
    vec3 halfB = args->b.halfExtents[i];

    ContactOut contacts[MAX_CONTACTS];
    for (int k = 0; k < MAX_CONTACTS; ++k) { contacts[k].featureKey = -1; contacts[k].rA = vec3(0,0,0); contacts[k].rB = vec3(0,0,0); }
    float3x3 basis;
    int count = collideBoxesImpl(posA, rotA, halfA, posB, rotB, halfB, contacts, basis);

    args->out.count[i] = count;
    args->out.basis[i] = basis;
    for (int k = 0; k < MAX_CONTACTS; ++k)
        args->out.contacts[i * MAX_CONTACTS + k] = contacts[k];
}
