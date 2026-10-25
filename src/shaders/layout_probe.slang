// End-to-end layout gate. Reads a struct instance through the ACTUAL types.slang
// struct declarations (imported, not re-typed here) and writes every field back, in
// declaration order, as one uint64 slot per field (pointers as their raw address,
// scalars zero/bit-extended into the low 32 bits). The host fills the REAL C++ struct
// (gpu_types.cuh / gpu_forces.cuh / the packed SolverParamsGpu) with a distinct sentinel
// per field, uploads its raw bytes unchanged, dispatches, and compares field by field.
//
// This is deliberately different from tools/test_vk_layout.cpp's old approach (a
// hand-written C++ mirror compared to the real struct by offsetof): that only proves the
// mirror struct the test author wrote is self-consistent, and never touches the GPU or
// the Slang compiler's own idea of the layout. Here, the Slang COMPILER computes every
// field's offset when it lowers `p->field` to a load; if types.slang has the wrong
// padding, the compiler reads the wrong bytes and a sentinel lands in the wrong slot (or
// the wrong struct entirely), which is exactly the bug class this gate exists to catch.

import types;
import maths;

// The shared PipelineLayout gives every kernel exactly one 8-byte push constant (see
// vk_pipeline.h), so the two addresses this probe needs live in a small ring-buffer
// struct instead, exactly like CollideArgs in collide.slang -- pc.argsAddr points to a
// ProbeArgs, not directly to the source struct.
struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

struct ProbeArgs
{
    uint64_t srcAddr; // pointer to the real, sentinel-filled struct (uploaded raw bytes)
    uint64_t outAddr; // uint64_t out[fieldCount], one slot per field, declaration order
};

// ---------------------------------------------------------------------------
// BodySoA: 20 fields (18 pointers, count, _pad0).
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(1, 1, 1)]
void csProbeBody(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    BodySoA *p = (BodySoA *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)p->pos;
    out[k++] = (uint64_t)p->rot;
    out[k++] = (uint64_t)p->initialLin;
    out[k++] = (uint64_t)p->initialAng;
    out[k++] = (uint64_t)p->inertialLin;
    out[k++] = (uint64_t)p->inertialAng;
    out[k++] = (uint64_t)p->velLin;
    out[k++] = (uint64_t)p->velAng;
    out[k++] = (uint64_t)p->prevVelLin;
    out[k++] = (uint64_t)p->halfExtents;
    out[k++] = (uint64_t)p->shapeType;
    out[k++] = (uint64_t)p->moment;
    out[k++] = (uint64_t)p->mass;
    out[k++] = (uint64_t)p->friction;
    out[k++] = (uint64_t)p->awake;
    out[k++] = (uint64_t)p->sleepTimer;
    out[k++] = (uint64_t)p->restPos;
    out[k++] = (uint64_t)p->restRot;
    out[k++] = (uint64_t)p->gyration; out[k++] = (uint64_t)p->props;
    out[k++] = (uint64_t)p->count;
    out[k++] = (uint64_t)p->_pad0;
}

// ---------------------------------------------------------------------------
// ManifoldSoA: 18 fields, no explicit pad (ends on an 8-byte boundary already).
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(1, 1, 1)]
void csProbeManifold(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    ManifoldSoA *p = (ManifoldSoA *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)p->key;
    out[k++] = (uint64_t)p->bodyA;
    out[k++] = (uint64_t)p->bodyB;
    out[k++] = (uint64_t)p->basis;
    out[k++] = (uint64_t)p->friction;
    out[k++] = (uint64_t)p->contactOffset;
    out[k++] = (uint64_t)p->contactCount;
    out[k++] = (uint64_t)(uint)p->count;
    out[k++] = (uint64_t)(uint)p->capacity;
    out[k++] = (uint64_t)p->feature;
    out[k++] = (uint64_t)p->rA;
    out[k++] = (uint64_t)p->rB;
    out[k++] = (uint64_t)p->C0;
    out[k++] = (uint64_t)p->penalty;
    out[k++] = (uint64_t)p->lambda;
    out[k++] = (uint64_t)p->stick;
    out[k++] = (uint64_t)(uint)p->contactTotal;
    out[k++] = (uint64_t)(uint)p->contactCapacity;
    out[k++] = (uint64_t)p->shapeA;
    out[k++] = (uint64_t)p->shapeB;
}

// ---------------------------------------------------------------------------
// ShapeSoA: 11 pointers, count, _pad0.
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(1, 1, 1)]
void csProbeShape(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    ShapeSoA *p = (ShapeSoA *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)p->body;
    out[k++] = (uint64_t)p->localPos;
    out[k++] = (uint64_t)p->localRot;
    out[k++] = (uint64_t)p->halfExtents;
    out[k++] = (uint64_t)p->shapeType;
    out[k++] = (uint64_t)p->friction;
    out[k++] = (uint64_t)p->category;
    out[k++] = (uint64_t)p->mask;
    out[k++] = (uint64_t)p->group;
    out[k++] = (uint64_t)p->flags;
    out[k++] = (uint64_t)p->geomOffset;
    out[k++] = (uint64_t)p->count;
    out[k++] = (uint64_t)p->_pad0;
}

// ---------------------------------------------------------------------------
// JointSoA: 24 fields (22 pointers, count, _pad0).
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(1, 1, 1)]
void csProbeJoint(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    JointSoA *p = (JointSoA *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)p->bodyA;
    out[k++] = (uint64_t)p->bodyB;
    out[k++] = (uint64_t)p->rA;
    out[k++] = (uint64_t)p->rB;
    out[k++] = (uint64_t)p->C0Lin;
    out[k++] = (uint64_t)p->C0Ang;
    out[k++] = (uint64_t)p->penaltyLin;
    out[k++] = (uint64_t)p->penaltyAng;
    out[k++] = (uint64_t)p->lambdaLin;
    out[k++] = (uint64_t)p->lambdaAng;
    out[k++] = (uint64_t)p->stiffnessLin;
    out[k++] = (uint64_t)p->stiffnessAng;
    out[k++] = (uint64_t)p->fracture;
    out[k++] = (uint64_t)p->torqueArm;
    out[k++] = (uint64_t)p->restAng;
    out[k++] = (uint64_t)p->broken;
    out[k++] = (uint64_t)p->frameA;
    out[k++] = (uint64_t)p->frameB;
    out[k++] = (uint64_t)p->axisMode;
    out[k++] = (uint64_t)p->axisLo;
    out[k++] = (uint64_t)p->axisHi;
    out[k++] = (uint64_t)p->generic;
    out[k++] = (uint64_t)p->fractureAng;
    out[k++] = (uint64_t)p->count;
    out[k++] = (uint64_t)p->_pad0;
}

// ---------------------------------------------------------------------------
// SpringSoA: 8 fields (6 pointers, count, _pad0).
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(1, 1, 1)]
void csProbeSpring(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    SpringSoA *p = (SpringSoA *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)p->bodyA;
    out[k++] = (uint64_t)p->bodyB;
    out[k++] = (uint64_t)p->rA;
    out[k++] = (uint64_t)p->rB;
    out[k++] = (uint64_t)p->rest;
    out[k++] = (uint64_t)p->stiffness;
    out[k++] = (uint64_t)p->count;
    out[k++] = (uint64_t)p->_pad0;
}

// ---------------------------------------------------------------------------
// SolverParamsGpu: 20 scalar fields, no pointers, no padding needed (every field is
// exactly 4 bytes). Each slot's low 32 bits carry the field's raw bits (asuint for
// floats), matching engine/vk/solver_params_pack.h's packed layout.
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(1, 1, 1)]
void csProbeSolverParams(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    SolverParamsGpu *p = (SolverParamsGpu *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)asuint(p->dt);
    out[k++] = (uint64_t)asuint(p->worldFrameInertia);
    out[k++] = (uint64_t)(uint)p->iterations;
    out[k++] = (uint64_t)(uint)p->primalLanes;
    out[k++] = (uint64_t)asuint(p->alpha);
    out[k++] = (uint64_t)asuint(p->betaLin);
    out[k++] = (uint64_t)asuint(p->betaAng);
    out[k++] = (uint64_t)asuint(p->gamma);
    out[k++] = (uint64_t)p->sleepEnabled;
    out[k++] = (uint64_t)(uint)p->sleepFrames;
    out[k++] = (uint64_t)asuint(p->sleepLinVel);
    out[k++] = (uint64_t)asuint(p->sleepAngVel);
    out[k++] = (uint64_t)asuint(p->sleepDisp);
    out[k++] = (uint64_t)asuint(p->sleepFracture);
    out[k++] = (uint64_t)p->autoKStart;
    out[k++] = (uint64_t)asuint(p->dualDamping);
    out[k++] = (uint64_t)asuint(p->dualDampVel);
    out[k++] = (uint64_t)p->killFallenEnabled;
    out[k++] = (uint64_t)asuint(p->killHeight);
    out[k++] = (uint64_t)asuint(p->grabStrength);
}

// ---------------------------------------------------------------------------
// IntegrateArgs -- the arg struct integrate.slang actually receives (its own private
// BodySoA subset + its 2-field SolverParams), NOT types.slang's BodySoA. Declared here
// rather than imported from integrate.slang (which has no `import`able public surface
// for its private structs) as a byte-for-byte transcription kept in the same field
// order; see the header comment on why this still counts as testing "the Slang
// compiler's own offsets" -- it is compiled independently by slangc here, same as
// integrate.slang's copy, and any mismatch between the two would itself be a bug this
// probe is positioned to catch if their field lists ever drift apart.
// ---------------------------------------------------------------------------
struct IntegrateBodySoA
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
    uint _pad0;
};
struct IntegrateSolverParams
{
    float dt;
    float gravityX;
    float gravityY;
    float gravityZ;
    uint spinExact;
    uint _padP;
};
struct IntegrateArgsProbe
{
    IntegrateBodySoA b;
    IntegrateSolverParams p;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void csProbeIntegrateArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    IntegrateArgsProbe *a = (IntegrateArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->b.pos;
    out[k++] = (uint64_t)a->b.rot;
    out[k++] = (uint64_t)a->b.initialLin;
    out[k++] = (uint64_t)a->b.initialAng;
    out[k++] = (uint64_t)a->b.inertialLin;
    out[k++] = (uint64_t)a->b.inertialAng;
    out[k++] = (uint64_t)a->b.velLin;
    out[k++] = (uint64_t)a->b.velAng;
    out[k++] = (uint64_t)a->b.prevVelLin;
    out[k++] = (uint64_t)a->b.mass;
    out[k++] = (uint64_t)a->b.awake;
    out[k++] = (uint64_t)a->b.count;
    out[k++] = (uint64_t)a->b._pad0;
    out[k++] = (uint64_t)asuint(a->p.dt);
    out[k++] = (uint64_t)asuint(a->p.gravityX);
    out[k++] = (uint64_t)asuint(a->p.gravityY);
    out[k++] = (uint64_t)asuint(a->p.gravityZ);
    out[k++] = (uint64_t)a->p.spinExact;
    out[k++] = (uint64_t)a->p._padP;
}

// ---------------------------------------------------------------------------
// VelocityArgs / KillFallenArgs / ExplosionGuardArgs -- same situation and same
// precedent as IntegrateArgsProbe above: integrate.slang's structs are private, so each
// is re-declared here byte-for-byte in the same field order and probed independently.
// ---------------------------------------------------------------------------
struct VelocityArgsProbe
{
    IntegrateBodySoA b;
    IntegrateSolverParams p;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void csProbeVelocityArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    VelocityArgsProbe *a = (VelocityArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->b.pos;
    out[k++] = (uint64_t)a->b.rot;
    out[k++] = (uint64_t)a->b.initialLin;
    out[k++] = (uint64_t)a->b.initialAng;
    out[k++] = (uint64_t)a->b.inertialLin;
    out[k++] = (uint64_t)a->b.inertialAng;
    out[k++] = (uint64_t)a->b.velLin;
    out[k++] = (uint64_t)a->b.velAng;
    out[k++] = (uint64_t)a->b.prevVelLin;
    out[k++] = (uint64_t)a->b.mass;
    out[k++] = (uint64_t)a->b.awake;
    out[k++] = (uint64_t)a->b.count;
    out[k++] = (uint64_t)a->b._pad0;
    out[k++] = (uint64_t)asuint(a->p.dt);
    out[k++] = (uint64_t)asuint(a->p.gravityX);
    out[k++] = (uint64_t)asuint(a->p.gravityY);
    out[k++] = (uint64_t)asuint(a->p.gravityZ);
    out[k++] = (uint64_t)a->p.spinExact;
    out[k++] = (uint64_t)a->p._padP;
}

struct KillFallenArgsProbe
{
    IntegrateBodySoA b;
    int *dyn;
    int dynCount;
    float killHeight;
    float upX;
    float upY;
    float upZ;
    float _padK;
    int *sleepTimer;
    vec3 *restPos;
    quat *restRot;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void csProbeKillFallenArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    KillFallenArgsProbe *a = (KillFallenArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->b.pos;
    out[k++] = (uint64_t)a->b.rot;
    out[k++] = (uint64_t)a->b.initialLin;
    out[k++] = (uint64_t)a->b.initialAng;
    out[k++] = (uint64_t)a->b.inertialLin;
    out[k++] = (uint64_t)a->b.inertialAng;
    out[k++] = (uint64_t)a->b.velLin;
    out[k++] = (uint64_t)a->b.velAng;
    out[k++] = (uint64_t)a->b.prevVelLin;
    out[k++] = (uint64_t)a->b.mass;
    out[k++] = (uint64_t)a->b.awake;
    out[k++] = (uint64_t)a->b.count;
    out[k++] = (uint64_t)a->b._pad0;
    out[k++] = (uint64_t)a->dyn;
    out[k++] = (uint64_t)a->dynCount;
    out[k++] = (uint64_t)asuint(a->killHeight);
    out[k++] = (uint64_t)asuint(a->upX);
    out[k++] = (uint64_t)asuint(a->upY);
    out[k++] = (uint64_t)asuint(a->upZ);
    out[k++] = (uint64_t)asuint(a->_padK);
    out[k++] = (uint64_t)a->sleepTimer;
    out[k++] = (uint64_t)a->restPos;
    out[k++] = (uint64_t)a->restRot;
}

struct ExplosionGuardArgsProbe
{
    IntegrateBodySoA b;
    int *dyn;
    int dynCount;
    int *sleepTimer;
    vec3 *restPos;
    quat *restRot;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void csProbeExplosionGuardArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    ExplosionGuardArgsProbe *a = (ExplosionGuardArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->b.pos;
    out[k++] = (uint64_t)a->b.rot;
    out[k++] = (uint64_t)a->b.initialLin;
    out[k++] = (uint64_t)a->b.initialAng;
    out[k++] = (uint64_t)a->b.inertialLin;
    out[k++] = (uint64_t)a->b.inertialAng;
    out[k++] = (uint64_t)a->b.velLin;
    out[k++] = (uint64_t)a->b.velAng;
    out[k++] = (uint64_t)a->b.prevVelLin;
    out[k++] = (uint64_t)a->b.mass;
    out[k++] = (uint64_t)a->b.awake;
    out[k++] = (uint64_t)a->b.count;
    out[k++] = (uint64_t)a->b._pad0;
    out[k++] = (uint64_t)a->dyn;
    out[k++] = (uint64_t)a->dynCount;
    out[k++] = (uint64_t)a->sleepTimer;
    out[k++] = (uint64_t)a->restPos;
    out[k++] = (uint64_t)a->restRot;
}

// ===========================================================================
// broadphase.slang / narrowphase.slang / {scan,compact,radix}.slang args structs --
// transcribed byte-for-byte from those files (same precedent as IntegrateArgsProbe
// above: these structs are not `public`/importable, so each is re-declared here with
// the identical field order its owning .slang file uses, and compiled independently by
// slangc -- any drift between the two field lists is itself exactly the bug class this
// probe exists to catch). See engine/vk/vk_args.h for the C++ side.
// ===========================================================================

// ---- broadphase.slang ----

struct BphBodySoAProbe
{
    vec3 *pos;
    quat *rot;
    int *shapeBody;
    vec3 *localPos;
    quat *localRot;
    vec3 *halfExtents;
    uint *flags;
    uint count;
    uint _pad0;
};

struct AabbBinArgsProbe
{
    BphBodySoAProbe b;
    int *idx;
    uint n;
    float invCellSize;
    vec3 *aabbMin;
    vec3 *aabbMax;
    uint64_t *cellId;
    uint *hash;
    int *binBody;
    uint mask;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void csProbeAabbBinArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    AabbBinArgsProbe *a = (AabbBinArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->b.pos;
    out[k++] = (uint64_t)a->b.rot;
    out[k++] = (uint64_t)a->b.shapeBody;
    out[k++] = (uint64_t)a->b.localPos;
    out[k++] = (uint64_t)a->b.localRot;
    out[k++] = (uint64_t)a->b.halfExtents;
    out[k++] = (uint64_t)a->b.flags;
    out[k++] = (uint64_t)a->b.count;
    out[k++] = (uint64_t)a->b._pad0;
    out[k++] = (uint64_t)a->idx;
    out[k++] = (uint64_t)a->n;
    out[k++] = (uint64_t)asuint(a->invCellSize);
    out[k++] = (uint64_t)a->aabbMin;
    out[k++] = (uint64_t)a->aabbMax;
    out[k++] = (uint64_t)a->cellId;
    out[k++] = (uint64_t)a->hash;
    out[k++] = (uint64_t)a->binBody;
    out[k++] = (uint64_t)a->mask;
}

struct StaticBinArgsProbe
{
    BphBodySoAProbe b;
    int *idx;
    uint n;
    float invCellSize;
    vec3 *aabbMin;
    vec3 *aabbMax;
    uint64_t *cellId;
    uint *hash;
    int *binBody;
    uint mask;
    float cellSize;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void csProbeStaticBinArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    StaticBinArgsProbe *a = (StaticBinArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->b.pos;
    out[k++] = (uint64_t)a->b.rot;
    out[k++] = (uint64_t)a->b.shapeBody;
    out[k++] = (uint64_t)a->b.localPos;
    out[k++] = (uint64_t)a->b.localRot;
    out[k++] = (uint64_t)a->b.halfExtents;
    out[k++] = (uint64_t)a->b.flags;
    out[k++] = (uint64_t)a->b.count;
    out[k++] = (uint64_t)a->b._pad0;
    out[k++] = (uint64_t)a->idx;
    out[k++] = (uint64_t)a->n;
    out[k++] = (uint64_t)asuint(a->invCellSize);
    out[k++] = (uint64_t)a->aabbMin;
    out[k++] = (uint64_t)a->aabbMax;
    out[k++] = (uint64_t)a->cellId;
    out[k++] = (uint64_t)a->hash;
    out[k++] = (uint64_t)a->binBody;
    out[k++] = (uint64_t)a->mask;
    out[k++] = (uint64_t)asuint(a->cellSize);
}

struct AabbStaticArgsProbe
{
    BphBodySoAProbe b;
    int *idx;
    uint n;
    vec3 *aabbMin;
    vec3 *aabbMax;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void csProbeAabbStaticArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    AabbStaticArgsProbe *a = (AabbStaticArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->b.pos;
    out[k++] = (uint64_t)a->b.rot;
    out[k++] = (uint64_t)a->b.shapeBody;
    out[k++] = (uint64_t)a->b.localPos;
    out[k++] = (uint64_t)a->b.localRot;
    out[k++] = (uint64_t)a->b.halfExtents;
    out[k++] = (uint64_t)a->b.flags;
    out[k++] = (uint64_t)a->b.count;
    out[k++] = (uint64_t)a->b._pad0;
    out[k++] = (uint64_t)a->idx;
    out[k++] = (uint64_t)a->n;
    out[k++] = (uint64_t)a->aabbMin;
    out[k++] = (uint64_t)a->aabbMax;
}

struct ResetRangesArgsProbe
{
    int *cellStart;
    int *cellEnd;
    uint tableSize;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void csProbeResetRangesArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    ResetRangesArgsProbe *a = (ResetRangesArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->cellStart;
    out[k++] = (uint64_t)a->cellEnd;
    out[k++] = (uint64_t)a->tableSize;
}

struct CellRangesArgsProbe
{
    uint *hashSorted;
    uint n;
    int *cellStart;
    int *cellEnd;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void csProbeCellRangesArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    CellRangesArgsProbe *a = (CellRangesArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->hashSorted;
    out[k++] = (uint64_t)a->n;
    out[k++] = (uint64_t)a->cellStart;
    out[k++] = (uint64_t)a->cellEnd;
}

struct EmitPairsArgsProbe
{
    int *binBodySorted;
    uint n;
    uint64_t *cellId;
    vec3 *aabbMin;
    vec3 *aabbMax;
    int *cellStart;
    int *cellEnd;
    float invCellSize;
    uint mask;
    uint64_t *pairs;
    int *counter;
    int capacity;
    uint64_t *ignore;
    int ignoreCount;
    int *shapeBody;
    int *sBinBodySorted;
    int *sCellStart;
    int *sCellEnd;
    uint sMask;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void csProbeEmitPairsArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    EmitPairsArgsProbe *a = (EmitPairsArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->binBodySorted;
    out[k++] = (uint64_t)a->n;
    out[k++] = (uint64_t)a->cellId;
    out[k++] = (uint64_t)a->aabbMin;
    out[k++] = (uint64_t)a->aabbMax;
    out[k++] = (uint64_t)a->cellStart;
    out[k++] = (uint64_t)a->cellEnd;
    out[k++] = (uint64_t)asuint(a->invCellSize);
    out[k++] = (uint64_t)a->mask;
    out[k++] = (uint64_t)a->pairs;
    out[k++] = (uint64_t)a->counter;
    out[k++] = (uint64_t)(uint)a->capacity;
    out[k++] = (uint64_t)a->ignore;
    out[k++] = (uint64_t)(uint)a->ignoreCount;
    out[k++] = (uint64_t)a->shapeBody;
    out[k++] = (uint64_t)a->sBinBodySorted;
    out[k++] = (uint64_t)a->sCellStart;
    out[k++] = (uint64_t)a->sCellEnd;
    out[k++] = (uint64_t)a->sMask;
}

struct EmitCrossPairsArgsProbe
{
    int *aIdx;
    uint nA;
    int *bIdx;
    uint nB;
    int *bStart;
    int *bEnd;
    vec3 *aabbMin;
    vec3 *aabbMax;
    uint64_t *pairs;
    int *counter;
    int capacity;
    uint64_t *ignore;
    int ignoreCount;
    int *shapeBody;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void csProbeEmitCrossPairsArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    EmitCrossPairsArgsProbe *a = (EmitCrossPairsArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->aIdx;
    out[k++] = (uint64_t)a->nA;
    out[k++] = (uint64_t)a->bIdx;
    out[k++] = (uint64_t)a->nB;
    out[k++] = (uint64_t)a->bStart;
    out[k++] = (uint64_t)a->bEnd;
    out[k++] = (uint64_t)a->aabbMin;
    out[k++] = (uint64_t)a->aabbMax;
    out[k++] = (uint64_t)a->pairs;
    out[k++] = (uint64_t)a->counter;
    out[k++] = (uint64_t)(uint)a->capacity;
    out[k++] = (uint64_t)a->ignore;
    out[k++] = (uint64_t)(uint)a->ignoreCount;
    out[k++] = (uint64_t)a->shapeBody;
}

// ---- narrowphase.slang ----

struct NpBodySoAProbe
{
    vec3 *pos;
    quat *rot;
    int *shapeBody;
    vec3 *localPos;
    quat *localRot;
    vec3 *halfExtents;
    uint8_t *shapeType;
    float *friction;
    uint *flags;
    float *mass;
    vec3 *moment;
    vec3 *velLin, *velAng;
    uint *props;
    uint *category, *mask, *group;
    uint count;
    uint _pad0;
};

struct NpManifoldSoAProbe
{
    uint64_t *key;
    int *bodyA, *bodyB;
    float3x3 *basis;
    float *friction;
    int *contactOffset, *contactCount;
    uint *feature;
    vec3 *rA, *rB;
    vec3 *C0;
    vec3 *penalty, *lambda;
    uint8_t *stick;
    int *shapeA, *shapeB;
};

struct SolverParamsNpProbe
{
    float dt;
    float alpha;
    float gamma;
    uint autoKStart;
    float dualDamping;
    float dualDampVel;
};

struct NarrowphaseArgsProbe
{
    NpBodySoAProbe b;
    uint64_t *pairs;
    uint pairCount;
    uint _pad0;
    NpManifoldSoAProbe cur;
    NpManifoldSoAProbe prev;
    uint64_t *prevKeys;
    uint prevCount;
    uint _pad1;
    int *contactCounter;
    int *activeCounter;
    int contactCapacity;
    SolverParamsNpProbe p;
    uint _pad2;
    uint *livePairs;
    int contactCap;
    uint _pad3;
    int *hullList;
    int *hullCounter;
    int *meshList;
    int *meshCounter;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void csProbeNarrowphaseArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    NarrowphaseArgsProbe *a = (NarrowphaseArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->b.pos;
    out[k++] = (uint64_t)a->b.rot;
    out[k++] = (uint64_t)a->b.shapeBody;
    out[k++] = (uint64_t)a->b.localPos;
    out[k++] = (uint64_t)a->b.localRot;
    out[k++] = (uint64_t)a->b.halfExtents;
    out[k++] = (uint64_t)a->b.shapeType;
    out[k++] = (uint64_t)a->b.friction;
    out[k++] = (uint64_t)a->b.flags;
    out[k++] = (uint64_t)a->b.mass;
    out[k++] = (uint64_t)a->b.moment;
    out[k++] = (uint64_t)a->b.velLin;
    out[k++] = (uint64_t)a->b.velAng;
    out[k++] = (uint64_t)a->b.props;
    out[k++] = (uint64_t)a->b.category; out[k++] = (uint64_t)a->b.mask; out[k++] = (uint64_t)a->b.group;
    out[k++] = (uint64_t)a->b.count;
    out[k++] = (uint64_t)a->pairs;
    out[k++] = (uint64_t)a->pairCount;
    // cur
    out[k++] = (uint64_t)a->cur.key;
    out[k++] = (uint64_t)a->cur.bodyA;
    out[k++] = (uint64_t)a->cur.bodyB;
    out[k++] = (uint64_t)a->cur.basis;
    out[k++] = (uint64_t)a->cur.friction;
    out[k++] = (uint64_t)a->cur.contactOffset;
    out[k++] = (uint64_t)a->cur.contactCount;
    out[k++] = (uint64_t)a->cur.feature;
    out[k++] = (uint64_t)a->cur.rA;
    out[k++] = (uint64_t)a->cur.rB;
    out[k++] = (uint64_t)a->cur.C0;
    out[k++] = (uint64_t)a->cur.penalty;
    out[k++] = (uint64_t)a->cur.lambda;
    out[k++] = (uint64_t)a->cur.stick;
    out[k++] = (uint64_t)a->cur.shapeA;
    out[k++] = (uint64_t)a->cur.shapeB;
    // prev
    out[k++] = (uint64_t)a->prev.key;
    out[k++] = (uint64_t)a->prev.bodyA;
    out[k++] = (uint64_t)a->prev.bodyB;
    out[k++] = (uint64_t)a->prev.basis;
    out[k++] = (uint64_t)a->prev.friction;
    out[k++] = (uint64_t)a->prev.contactOffset;
    out[k++] = (uint64_t)a->prev.contactCount;
    out[k++] = (uint64_t)a->prev.feature;
    out[k++] = (uint64_t)a->prev.rA;
    out[k++] = (uint64_t)a->prev.rB;
    out[k++] = (uint64_t)a->prev.C0;
    out[k++] = (uint64_t)a->prev.penalty;
    out[k++] = (uint64_t)a->prev.lambda;
    out[k++] = (uint64_t)a->prev.stick;
    out[k++] = (uint64_t)a->prev.shapeA;
    out[k++] = (uint64_t)a->prev.shapeB;
    out[k++] = (uint64_t)a->prevKeys;
    out[k++] = (uint64_t)a->prevCount;
    out[k++] = (uint64_t)a->contactCounter;
    out[k++] = (uint64_t)a->activeCounter;
    out[k++] = (uint64_t)(uint)a->contactCapacity;
    out[k++] = (uint64_t)asuint(a->p.dt);
    out[k++] = (uint64_t)asuint(a->p.alpha);
    out[k++] = (uint64_t)asuint(a->p.gamma);
    out[k++] = (uint64_t)a->p.autoKStart;
    out[k++] = (uint64_t)asuint(a->p.dualDamping);
    out[k++] = (uint64_t)asuint(a->p.dualDampVel);
    out[k++] = (uint64_t)a->livePairs;
    out[k++] = (uint64_t)(uint)a->contactCap;
    out[k++] = (uint64_t)a->hullList;
    out[k++] = (uint64_t)a->hullCounter;
    out[k++] = (uint64_t)a->meshList;
    out[k++] = (uint64_t)a->meshCounter;
}

// ---- constraints.slang ----

struct CsResetColorsArgsProbe { int *color; int *dyn; uint dynCount; uint *dynCountPtr; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeCsResetColorsArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    CsResetColorsArgsProbe *a = (CsResetColorsArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->color; out[k++] = (uint64_t)a->dyn; out[k++] = (uint64_t)a->dynCount;
    out[k++] = (uint64_t)a->dynCountPtr;
}

struct CsEmitManifoldArgsProbe
{
    int *mBodyA, *mBodyB, *mContactCount;
    uint slots;
    float *mass; uint8_t *awake;
    uint64_t *keys; int *isA; int *counter;
    int capacity;
};
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeCsEmitManifoldArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    CsEmitManifoldArgsProbe *a = (CsEmitManifoldArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->mBodyA; out[k++] = (uint64_t)a->mBodyB; out[k++] = (uint64_t)a->mContactCount;
    out[k++] = (uint64_t)a->slots;
    out[k++] = (uint64_t)a->mass; out[k++] = (uint64_t)a->awake;
    out[k++] = (uint64_t)a->keys; out[k++] = (uint64_t)a->isA; out[k++] = (uint64_t)a->counter;
    out[k++] = (uint64_t)(uint)a->capacity;
}

struct CsEmitForceArgsProbe
{
    int *fBodyA, *fBodyB;
    uint count;
    uint8_t *broken;
    float *mass; uint8_t *awake;
    uint64_t *keys; int *isA; int *counter;
    int capacity;
    int type;
};
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeCsEmitForceArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    CsEmitForceArgsProbe *a = (CsEmitForceArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->fBodyA; out[k++] = (uint64_t)a->fBodyB; out[k++] = (uint64_t)a->count;
    out[k++] = (uint64_t)a->broken;
    out[k++] = (uint64_t)a->mass; out[k++] = (uint64_t)a->awake;
    out[k++] = (uint64_t)a->keys; out[k++] = (uint64_t)a->isA; out[k++] = (uint64_t)a->counter;
    out[k++] = (uint64_t)(uint)a->capacity; out[k++] = (uint64_t)(uint)a->type;
}

struct CsRowsArgsProbe { uint64_t *keys; uint n; int *start; int *end; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeCsRowsArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    CsRowsArgsProbe *a = (CsRowsArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->keys; out[k++] = (uint64_t)a->n; out[k++] = (uint64_t)a->start; out[k++] = (uint64_t)a->end;
}

struct CsJpRoundArgsProbe
{
    uint64_t *gKey; int *gIsA; int *gStart; int *gEnd;
    int *mBodyA, *mBodyB;
    int *jBodyA, *jBodyB;
    int *sBodyA, *sBodyB;
    float *mass; uint8_t *awake;
    int *dyn;
    uint dynCount;
    int *color; uint *remaining;
    uint *dynCountPtr;
};
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeCsJpRoundArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    CsJpRoundArgsProbe *a = (CsJpRoundArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->gKey; out[k++] = (uint64_t)a->gIsA; out[k++] = (uint64_t)a->gStart; out[k++] = (uint64_t)a->gEnd;
    out[k++] = (uint64_t)a->mBodyA; out[k++] = (uint64_t)a->mBodyB;
    out[k++] = (uint64_t)a->jBodyA; out[k++] = (uint64_t)a->jBodyB;
    out[k++] = (uint64_t)a->sBodyA; out[k++] = (uint64_t)a->sBodyB;
    out[k++] = (uint64_t)a->mass; out[k++] = (uint64_t)a->awake;
    out[k++] = (uint64_t)a->dyn; out[k++] = (uint64_t)a->dynCount;
    out[k++] = (uint64_t)a->color; out[k++] = (uint64_t)a->remaining;
    out[k++] = (uint64_t)a->dynCountPtr;
}

struct CsColorKeysArgsProbe { int *color; int *dyn; uint dynCount; uint *keys; int *vals; uint *dynCountPtr; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeCsColorKeysArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    CsColorKeysArgsProbe *a = (CsColorKeysArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->color; out[k++] = (uint64_t)a->dyn; out[k++] = (uint64_t)a->dynCount;
    out[k++] = (uint64_t)a->keys; out[k++] = (uint64_t)a->vals; out[k++] = (uint64_t)a->dynCountPtr;
}

struct CsColorRowsArgsProbe { uint *keys; uint n; int *start; int *end; uint *dynCountPtr; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeCsColorRowsArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    CsColorRowsArgsProbe *a = (CsColorRowsArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->keys; out[k++] = (uint64_t)a->n; out[k++] = (uint64_t)a->start; out[k++] = (uint64_t)a->end;
    out[k++] = (uint64_t)a->dynCountPtr;
}

// ---- scan.slang / compact.slang / radix.slang ----

struct ScanBlockArgsProbe { uint *in, *out, *blockSums; uint n; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeScanBlockArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    ScanBlockArgsProbe *a = (ScanBlockArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->in; out[k++] = (uint64_t)a->out;
    out[k++] = (uint64_t)a->blockSums; out[k++] = (uint64_t)a->n;
}

struct AddBlockSumsArgsProbe { uint *out, *blockOffsets; uint n; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeAddBlockSumsArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    AddBlockSumsArgsProbe *a = (AddBlockSumsArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->out; out[k++] = (uint64_t)a->blockOffsets; out[k++] = (uint64_t)a->n;
}

struct FlagsToU32ArgsProbe { uint8_t *flags; uint *out; uint n; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeFlagsToU32Args(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    FlagsToU32ArgsProbe *a = (FlagsToU32ArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->flags; out[k++] = (uint64_t)a->out; out[k++] = (uint64_t)a->n;
}

struct ScatterCompactArgsProbe { int *items; uint8_t *flags; uint *scanned; int *out; uint n; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeScatterCompactArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    ScatterCompactArgsProbe *a = (ScatterCompactArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->items; out[k++] = (uint64_t)a->flags;
    out[k++] = (uint64_t)a->scanned; out[k++] = (uint64_t)a->out; out[k++] = (uint64_t)a->n;
}

struct NumSelectedArgsProbe { uint *scanned; uint8_t *flags; int *numSelected; uint n; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeNumSelectedArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    NumSelectedArgsProbe *a = (NumSelectedArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->scanned; out[k++] = (uint64_t)a->flags;
    out[k++] = (uint64_t)a->numSelected; out[k++] = (uint64_t)a->n;
}

struct HistArgsU32PairsProbe { uint *keysIn, *digitCount, *skipMask; uint n, numBlocks, pass; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeHistArgsU32Pairs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    HistArgsU32PairsProbe *a = (HistArgsU32PairsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->keysIn; out[k++] = (uint64_t)a->digitCount; out[k++] = (uint64_t)a->skipMask;
    out[k++] = (uint64_t)a->n; out[k++] = (uint64_t)a->numBlocks; out[k++] = (uint64_t)a->pass;
}

struct ScatterArgsU32PairsProbe { uint *keysIn, *valsIn, *digitCountScanned, *skipMask, *keysOut, *valsOut; uint n, numBlocks, pass; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeScatterArgsU32Pairs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    ScatterArgsU32PairsProbe *a = (ScatterArgsU32PairsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->keysIn; out[k++] = (uint64_t)a->valsIn;
    out[k++] = (uint64_t)a->digitCountScanned; out[k++] = (uint64_t)a->skipMask;
    out[k++] = (uint64_t)a->keysOut; out[k++] = (uint64_t)a->valsOut;
    out[k++] = (uint64_t)a->n; out[k++] = (uint64_t)a->numBlocks; out[k++] = (uint64_t)a->pass;
}

struct HistArgsU64PairsProbe { uint64_t *keysIn; uint *digitCount, *skipMask; uint n, numBlocks, pass; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeHistArgsU64Pairs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    HistArgsU64PairsProbe *a = (HistArgsU64PairsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->keysIn; out[k++] = (uint64_t)a->digitCount; out[k++] = (uint64_t)a->skipMask;
    out[k++] = (uint64_t)a->n; out[k++] = (uint64_t)a->numBlocks; out[k++] = (uint64_t)a->pass;
}

struct ScatterArgsU64PairsProbe { uint64_t *keysIn; uint *valsIn, *digitCountScanned, *skipMask; uint64_t *keysOut; uint *valsOut; uint n, numBlocks, pass; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeScatterArgsU64Pairs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    ScatterArgsU64PairsProbe *a = (ScatterArgsU64PairsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->keysIn; out[k++] = (uint64_t)a->valsIn;
    out[k++] = (uint64_t)a->digitCountScanned; out[k++] = (uint64_t)a->skipMask;
    out[k++] = (uint64_t)a->keysOut; out[k++] = (uint64_t)a->valsOut;
    out[k++] = (uint64_t)a->n; out[k++] = (uint64_t)a->numBlocks; out[k++] = (uint64_t)a->pass;
}

struct HistArgsU64KeysProbe { uint64_t *keysIn; uint *digitCount, *skipMask; uint n, numBlocks, pass; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeHistArgsU64Keys(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    HistArgsU64KeysProbe *a = (HistArgsU64KeysProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->keysIn; out[k++] = (uint64_t)a->digitCount; out[k++] = (uint64_t)a->skipMask;
    out[k++] = (uint64_t)a->n; out[k++] = (uint64_t)a->numBlocks; out[k++] = (uint64_t)a->pass;
}

struct ScatterArgsU64KeysProbe { uint64_t *keysIn; uint *digitCountScanned, *skipMask; uint64_t *keysOut; uint n, numBlocks, pass; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeScatterArgsU64Keys(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    ScatterArgsU64KeysProbe *a = (ScatterArgsU64KeysProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->keysIn; out[k++] = (uint64_t)a->digitCountScanned; out[k++] = (uint64_t)a->skipMask;
    out[k++] = (uint64_t)a->keysOut;
    out[k++] = (uint64_t)a->n; out[k++] = (uint64_t)a->numBlocks; out[k++] = (uint64_t)a->pass;
}

struct ReduceOrAndArgsU32Probe { uint *keysIn, *outOr, *outAnd; uint n; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeReduceOrAndArgsU32(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    ReduceOrAndArgsU32Probe *a = (ReduceOrAndArgsU32Probe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->keysIn; out[k++] = (uint64_t)a->outOr; out[k++] = (uint64_t)a->outAnd; out[k++] = (uint64_t)a->n;
}

struct ReduceOrAndArgsU64Probe { uint64_t *keysIn, *outOr, *outAnd; uint n; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeReduceOrAndArgsU64(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    ReduceOrAndArgsU64Probe *a = (ReduceOrAndArgsU64Probe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->keysIn; out[k++] = (uint64_t)a->outOr; out[k++] = (uint64_t)a->outAnd; out[k++] = (uint64_t)a->n;
}

struct FinalizeSkipMaskArgsU32Probe { uint *orAddr, *andAddr, *skipMask; uint numPasses; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeFinalizeSkipMaskArgsU32(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    FinalizeSkipMaskArgsU32Probe *a = (FinalizeSkipMaskArgsU32Probe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->orAddr; out[k++] = (uint64_t)a->andAddr;
    out[k++] = (uint64_t)a->skipMask; out[k++] = (uint64_t)a->numPasses;
}

struct FinalizeSkipMaskArgsU64Probe { uint64_t *orAddr, *andAddr; uint *skipMask; uint numPasses; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeFinalizeSkipMaskArgsU64(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    FinalizeSkipMaskArgsU64Probe *a = (FinalizeSkipMaskArgsU64Probe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->orAddr; out[k++] = (uint64_t)a->andAddr;
    out[k++] = (uint64_t)a->skipMask; out[k++] = (uint64_t)a->numPasses;
}

// ===========================================================================
// primal.slang / dual.slang args structs -- Phase 4b. Transcribed byte-for-byte from
// those files: PrimalArgs/JointInitArgs/JointDualArgs/DualArgs are not
// public, so each is re-declared here in the exact field order the owning .slang file
// uses -- including BodySoA/ManifoldSoA/JointSoA/SpringSoA/SolverParamsGpu, which ARE
// imported from types.slang so their own layout is already covered by csProbeBody/
// csProbeManifold/csProbeJoint/csProbeSpring/csProbeSolverParams above; what these probes
// additionally prove is the COMPOSITE struct's own field offsets -- i.e. that nothing
// pads between b/g/m/joints/springs/p the way DualArgsGpu's now-removed spurious _pad0
// briefly would have. See engine/vk/vk_args.h's Full*SoAGpu / PrimalArgsGpu /
// JointInitArgsGpu / JointDualArgsGpu / DualArgsGpu.
// ===========================================================================

struct PrimalConstraintGraphProbe { uint64_t *key; int *isA; int *start; int *end; };

struct PrimalArgsProbe
{
    BodySoA b;
    PrimalConstraintGraphProbe g;
    ManifoldSoA m;
    JointSoA joints;
    SpringSoA springs;
    int *colorBodies;
    int listStart;
    int listCount;
    SolverParamsGpu p;
};

[shader("compute")] [numthreads(1, 1, 1)]
void csProbePrimalArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    PrimalArgsProbe *a = (PrimalArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->b.pos;
    out[k++] = (uint64_t)a->b.rot;
    out[k++] = (uint64_t)a->b.initialLin;
    out[k++] = (uint64_t)a->b.initialAng;
    out[k++] = (uint64_t)a->b.inertialLin;
    out[k++] = (uint64_t)a->b.inertialAng;
    out[k++] = (uint64_t)a->b.velLin;
    out[k++] = (uint64_t)a->b.velAng;
    out[k++] = (uint64_t)a->b.prevVelLin;
    out[k++] = (uint64_t)a->b.halfExtents;
    out[k++] = (uint64_t)a->b.shapeType;
    out[k++] = (uint64_t)a->b.moment;
    out[k++] = (uint64_t)a->b.mass;
    out[k++] = (uint64_t)a->b.friction;
    out[k++] = (uint64_t)a->b.awake;
    out[k++] = (uint64_t)a->b.sleepTimer;
    out[k++] = (uint64_t)a->b.restPos;
    out[k++] = (uint64_t)a->b.restRot;
    out[k++] = (uint64_t)a->b.gyration; out[k++] = (uint64_t)a->b.props;
    out[k++] = (uint64_t)a->b.count;
    out[k++] = (uint64_t)a->g.key;
    out[k++] = (uint64_t)a->g.isA;
    out[k++] = (uint64_t)a->g.start;
    out[k++] = (uint64_t)a->g.end;
    out[k++] = (uint64_t)a->m.key;
    out[k++] = (uint64_t)a->m.bodyA;
    out[k++] = (uint64_t)a->m.bodyB;
    out[k++] = (uint64_t)a->m.basis;
    out[k++] = (uint64_t)a->m.friction;
    out[k++] = (uint64_t)a->m.contactOffset;
    out[k++] = (uint64_t)a->m.contactCount;
    out[k++] = (uint64_t)(uint)a->m.count;
    out[k++] = (uint64_t)(uint)a->m.capacity;
    out[k++] = (uint64_t)a->m.feature;
    out[k++] = (uint64_t)a->m.rA;
    out[k++] = (uint64_t)a->m.rB;
    out[k++] = (uint64_t)a->m.C0;
    out[k++] = (uint64_t)a->m.penalty;
    out[k++] = (uint64_t)a->m.lambda;
    out[k++] = (uint64_t)a->m.stick;
    out[k++] = (uint64_t)(uint)a->m.contactTotal;
    out[k++] = (uint64_t)(uint)a->m.contactCapacity; out[k++] = (uint64_t)a->m.shapeA; out[k++] = (uint64_t)a->m.shapeB;
    out[k++] = (uint64_t)a->joints.bodyA;
    out[k++] = (uint64_t)a->joints.bodyB;
    out[k++] = (uint64_t)a->joints.rA;
    out[k++] = (uint64_t)a->joints.rB;
    out[k++] = (uint64_t)a->joints.C0Lin;
    out[k++] = (uint64_t)a->joints.C0Ang;
    out[k++] = (uint64_t)a->joints.penaltyLin;
    out[k++] = (uint64_t)a->joints.penaltyAng;
    out[k++] = (uint64_t)a->joints.lambdaLin;
    out[k++] = (uint64_t)a->joints.lambdaAng;
    out[k++] = (uint64_t)a->joints.stiffnessLin;
    out[k++] = (uint64_t)a->joints.stiffnessAng;
    out[k++] = (uint64_t)a->joints.fracture;
    out[k++] = (uint64_t)a->joints.torqueArm;
    out[k++] = (uint64_t)a->joints.restAng;
    out[k++] = (uint64_t)a->joints.broken;
    out[k++] = (uint64_t)a->joints.frameA;
    out[k++] = (uint64_t)a->joints.frameB;
    out[k++] = (uint64_t)a->joints.axisMode;
    out[k++] = (uint64_t)a->joints.axisLo;
    out[k++] = (uint64_t)a->joints.axisHi;
    out[k++] = (uint64_t)a->joints.generic;
    out[k++] = (uint64_t)a->joints.count;
    out[k++] = (uint64_t)a->springs.bodyA;
    out[k++] = (uint64_t)a->springs.bodyB;
    out[k++] = (uint64_t)a->springs.rA;
    out[k++] = (uint64_t)a->springs.rB;
    out[k++] = (uint64_t)a->springs.rest;
    out[k++] = (uint64_t)a->springs.stiffness;
    out[k++] = (uint64_t)a->springs.count;
    out[k++] = (uint64_t)a->colorBodies;
    out[k++] = (uint64_t)(uint)a->listStart;
    out[k++] = (uint64_t)(uint)a->listCount;
    out[k++] = (uint64_t)asuint(a->p.dt);
    out[k++] = (uint64_t)asuint(a->p.worldFrameInertia);
    out[k++] = (uint64_t)(uint)a->p.iterations;
    out[k++] = (uint64_t)(uint)a->p.primalLanes;
    out[k++] = (uint64_t)asuint(a->p.alpha);
    out[k++] = (uint64_t)asuint(a->p.betaLin);
    out[k++] = (uint64_t)asuint(a->p.betaAng);
    out[k++] = (uint64_t)asuint(a->p.gamma);
}

struct JointInitArgsProbe { BodySoA b; JointSoA j; SolverParamsGpu p; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeJointInitArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    JointInitArgsProbe *a = (JointInitArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->b.pos;
    out[k++] = (uint64_t)a->b.mass;
    out[k++] = (uint64_t)a->b.awake;
    out[k++] = (uint64_t)a->b.count;
    out[k++] = (uint64_t)a->j.bodyA;
    out[k++] = (uint64_t)a->j.penaltyLin;
    out[k++] = (uint64_t)a->j.broken;
    out[k++] = (uint64_t)a->j.count;
    out[k++] = (uint64_t)asuint(a->p.dt);
    out[k++] = (uint64_t)asuint(a->p.alpha);
    out[k++] = (uint64_t)asuint(a->p.gamma);
}

struct JointDualArgsProbe { BodySoA b; JointSoA j; SolverParamsGpu p; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeJointDualArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    JointDualArgsProbe *a = (JointDualArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->b.pos;
    out[k++] = (uint64_t)a->b.mass;
    out[k++] = (uint64_t)a->b.awake;
    out[k++] = (uint64_t)a->b.count;
    out[k++] = (uint64_t)a->j.bodyA;
    out[k++] = (uint64_t)a->j.lambdaLin;
    out[k++] = (uint64_t)a->j.broken;
    out[k++] = (uint64_t)a->j.count;
    out[k++] = (uint64_t)asuint(a->p.alpha);
    out[k++] = (uint64_t)asuint(a->p.betaLin);
    out[k++] = (uint64_t)asuint(a->p.betaAng);
}

// DualArgs -- deliberately no padding between slots and p; see vk_args.h's DualArgsGpu
// comment for why (this is the field list that comment refers to).
struct DualArgsProbe { BodySoA b; ManifoldSoA m; int slots; SolverParamsGpu p; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeDualArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    DualArgsProbe *a = (DualArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->b.pos;
    out[k++] = (uint64_t)a->b.rot;
    out[k++] = (uint64_t)a->b.initialLin;
    out[k++] = (uint64_t)a->b.initialAng;
    out[k++] = (uint64_t)a->b.count;
    out[k++] = (uint64_t)a->m.bodyA;
    out[k++] = (uint64_t)a->m.bodyB;
    out[k++] = (uint64_t)a->m.basis;
    out[k++] = (uint64_t)a->m.friction;
    out[k++] = (uint64_t)a->m.contactOffset;
    out[k++] = (uint64_t)a->m.contactCount;
    out[k++] = (uint64_t)a->m.rA;
    out[k++] = (uint64_t)a->m.rB;
    out[k++] = (uint64_t)a->m.C0;
    out[k++] = (uint64_t)a->m.penalty;
    out[k++] = (uint64_t)a->m.lambda;
    out[k++] = (uint64_t)a->m.stick;
    out[k++] = (uint64_t)(uint)a->slots;
    out[k++] = (uint64_t)asuint(a->p.alpha);
    out[k++] = (uint64_t)asuint(a->p.betaLin);
}

// ===========================================================================
// render.slang / drag.slang args structs -- Phase 5C. Same convention as
// the primal/dual probes above: BodySoA/JointSoA are imported from types.slang (their
// own layout is already covered by csProbeBody/csProbeJoint), so what these probes
// additionally prove is the composite wrapper struct's own field offsets. See
// engine/vk/vk_args.h's WriteInstancesArgsGpu /
// DragUpdateArgsGpu / DragReleaseArgsGpu.
// ===========================================================================

struct OverlayViewProbe
{
    int mode;
    float speedMax, loadMax, degreeMax, invCellSize, sleepFrames;
    int colorCount, colorStatics;
    uint64_t graphColor, load, degree;
    float jointLoadMax, degreeJointMax, penetrationMax;
    int _pad0;
    uint64_t jointDegree, push, pull, stress, cone, penetration;
};

struct WriteInstancesArgsProbe
{
    BodySoA b;
    uint64_t out;
    int maxInstances;
    int _pad0;
    uint64_t frustum;
    uint64_t visible;
    OverlayViewProbe ov;
    uint64_t groupOf;
    uint64_t shapeBase;
    uint64_t shapeVisible;
    ShapeSoA s;
};
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeWriteInstancesArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    WriteInstancesArgsProbe *a = (WriteInstancesArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->b.pos;
    out[k++] = (uint64_t)a->b.rot;
    out[k++] = (uint64_t)a->b.initialLin;
    out[k++] = (uint64_t)a->b.initialAng;
    out[k++] = (uint64_t)a->b.inertialLin;
    out[k++] = (uint64_t)a->b.inertialAng;
    out[k++] = (uint64_t)a->b.velLin;
    out[k++] = (uint64_t)a->b.velAng;
    out[k++] = (uint64_t)a->b.prevVelLin;
    out[k++] = (uint64_t)a->b.halfExtents;
    out[k++] = (uint64_t)a->b.shapeType;
    out[k++] = (uint64_t)a->b.moment;
    out[k++] = (uint64_t)a->b.mass;
    out[k++] = (uint64_t)a->b.friction;
    out[k++] = (uint64_t)a->b.awake;
    out[k++] = (uint64_t)a->b.sleepTimer;
    out[k++] = (uint64_t)a->b.restPos;
    out[k++] = (uint64_t)a->b.restRot;
    out[k++] = (uint64_t)a->b.gyration; out[k++] = (uint64_t)a->b.props;
    out[k++] = (uint64_t)a->b.count;
    out[k++] = (uint64_t)a->out;
    out[k++] = (uint64_t)(uint)a->maxInstances;
    out[k++] = (uint64_t)a->frustum;
    out[k++] = (uint64_t)a->visible;
    out[k++] = (uint64_t)(uint)a->ov.mode;
    out[k++] = (uint64_t)asuint(a->ov.speedMax);
    out[k++] = (uint64_t)asuint(a->ov.loadMax);
    out[k++] = (uint64_t)asuint(a->ov.degreeMax);
    out[k++] = (uint64_t)asuint(a->ov.invCellSize);
    out[k++] = (uint64_t)asuint(a->ov.sleepFrames);
    out[k++] = (uint64_t)(uint)a->ov.colorCount;
    out[k++] = (uint64_t)(uint)a->ov.colorStatics;
    out[k++] = (uint64_t)a->ov.graphColor;
    out[k++] = (uint64_t)a->ov.load;
    out[k++] = (uint64_t)a->ov.degree;
    out[k++] = (uint64_t)asuint(a->ov.jointLoadMax);
    out[k++] = (uint64_t)asuint(a->ov.degreeJointMax);
    out[k++] = (uint64_t)asuint(a->ov.penetrationMax);
    out[k++] = (uint64_t)a->ov.jointDegree;
    out[k++] = (uint64_t)a->ov.push;
    out[k++] = (uint64_t)a->ov.pull;
    out[k++] = (uint64_t)a->ov.stress;
    out[k++] = (uint64_t)a->ov.cone;
    out[k++] = (uint64_t)a->ov.penetration;
    out[k++] = (uint64_t)a->groupOf;
    out[k++] = (uint64_t)a->shapeBase;
    out[k++] = (uint64_t)a->shapeVisible;
    out[k++] = (uint64_t)a->s.body;
    out[k++] = (uint64_t)a->s.localPos;
    out[k++] = (uint64_t)a->s.localRot;
    out[k++] = (uint64_t)a->s.halfExtents;
    out[k++] = (uint64_t)a->s.shapeType;
    out[k++] = (uint64_t)a->s.friction;
    out[k++] = (uint64_t)a->s.category;
    out[k++] = (uint64_t)a->s.mask;
    out[k++] = (uint64_t)a->s.group;
    out[k++] = (uint64_t)a->s.flags;
    out[k++] = (uint64_t)a->s.count;
}

struct DragUpdateArgsProbe
{
    BodySoA b;
    JointSoA j;
    int slot0, count;
    uint64_t bodies, targets, locals, stiffness;
    int grab, _pad0;
};
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeDragUpdateArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    DragUpdateArgsProbe *a = (DragUpdateArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->b.pos;
    out[k++] = (uint64_t)a->b.count;
    out[k++] = (uint64_t)a->j.bodyA;
    out[k++] = (uint64_t)a->j.bodyB;
    out[k++] = (uint64_t)a->j.rA;
    out[k++] = (uint64_t)a->j.count;
    out[k++] = (uint64_t)(uint)a->slot0;
    out[k++] = (uint64_t)(uint)a->count;
    out[k++] = (uint64_t)a->bodies;
    out[k++] = (uint64_t)a->targets;
    out[k++] = (uint64_t)a->locals;
    out[k++] = (uint64_t)a->stiffness;
    out[k++] = (uint64_t)(uint)a->grab;
}

struct DragReleaseArgsProbe { JointSoA j; int slot0, count; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeDragReleaseArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    DragReleaseArgsProbe *a = (DragReleaseArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->j.bodyA;
    out[k++] = (uint64_t)a->j.count;
    out[k++] = (uint64_t)(uint)a->slot0;
    out[k++] = (uint64_t)(uint)a->count;
}

struct OverlayStatsProbe { uint64_t push, pull, stress, cone, penetration, jointDegree; };

struct OverlayStatsClearArgsProbe { OverlayStatsProbe s; int n; int _pad0; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeOverlayStatsClearArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    OverlayStatsClearArgsProbe *a = (OverlayStatsClearArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->s.push; out[k++] = (uint64_t)a->s.pull; out[k++] = (uint64_t)a->s.stress;
    out[k++] = (uint64_t)a->s.cone; out[k++] = (uint64_t)a->s.penetration; out[k++] = (uint64_t)a->s.jointDegree;
    out[k++] = (uint64_t)(uint)a->n;
}

struct OverlayStatsContactsArgsProbe { ManifoldSoA m; int slots; int _pad0; OverlayStatsProbe s; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeOverlayStatsContactsArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    OverlayStatsContactsArgsProbe *a = (OverlayStatsContactsArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->m.key; out[k++] = (uint64_t)a->m.bodyA; out[k++] = (uint64_t)a->m.bodyB;
    out[k++] = (uint64_t)a->m.basis; out[k++] = (uint64_t)a->m.friction;
    out[k++] = (uint64_t)a->m.contactOffset; out[k++] = (uint64_t)a->m.contactCount;
    out[k++] = (uint64_t)(uint)a->m.count; out[k++] = (uint64_t)(uint)a->m.capacity;
    out[k++] = (uint64_t)a->m.feature; out[k++] = (uint64_t)a->m.rA; out[k++] = (uint64_t)a->m.rB;
    out[k++] = (uint64_t)a->m.C0; out[k++] = (uint64_t)a->m.penalty; out[k++] = (uint64_t)a->m.lambda;
    out[k++] = (uint64_t)a->m.stick;
    out[k++] = (uint64_t)(uint)a->m.contactTotal; out[k++] = (uint64_t)(uint)a->m.contactCapacity; out[k++] = (uint64_t)a->m.shapeA; out[k++] = (uint64_t)a->m.shapeB;
    out[k++] = (uint64_t)(uint)a->slots;
    out[k++] = (uint64_t)a->s.push; out[k++] = (uint64_t)a->s.pull; out[k++] = (uint64_t)a->s.stress;
    out[k++] = (uint64_t)a->s.cone; out[k++] = (uint64_t)a->s.penetration; out[k++] = (uint64_t)a->s.jointDegree;
}

struct OverlayStatsForcesArgsProbe { BodySoA b; JointSoA j; SpringSoA sp; int jointsAuthored; int _pad0; OverlayStatsProbe s; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeOverlayStatsForcesArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    OverlayStatsForcesArgsProbe *a = (OverlayStatsForcesArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->b.pos; out[k++] = (uint64_t)a->b.rot;
    out[k++] = (uint64_t)a->b.initialLin; out[k++] = (uint64_t)a->b.initialAng;
    out[k++] = (uint64_t)a->b.inertialLin; out[k++] = (uint64_t)a->b.inertialAng;
    out[k++] = (uint64_t)a->b.velLin; out[k++] = (uint64_t)a->b.velAng; out[k++] = (uint64_t)a->b.prevVelLin;
    out[k++] = (uint64_t)a->b.halfExtents; out[k++] = (uint64_t)a->b.shapeType;
    out[k++] = (uint64_t)a->b.moment; out[k++] = (uint64_t)a->b.mass; out[k++] = (uint64_t)a->b.friction;
    out[k++] = (uint64_t)a->b.awake; out[k++] = (uint64_t)a->b.sleepTimer;
    out[k++] = (uint64_t)a->b.restPos; out[k++] = (uint64_t)a->b.restRot;
    out[k++] = (uint64_t)a->b.gyration; out[k++] = (uint64_t)a->b.props; out[k++] = (uint64_t)a->b.count;
    out[k++] = (uint64_t)a->j.bodyA; out[k++] = (uint64_t)a->j.bodyB;
    out[k++] = (uint64_t)a->j.rA; out[k++] = (uint64_t)a->j.rB;
    out[k++] = (uint64_t)a->j.C0Lin; out[k++] = (uint64_t)a->j.C0Ang;
    out[k++] = (uint64_t)a->j.penaltyLin; out[k++] = (uint64_t)a->j.penaltyAng;
    out[k++] = (uint64_t)a->j.lambdaLin; out[k++] = (uint64_t)a->j.lambdaAng;
    out[k++] = (uint64_t)a->j.stiffnessLin; out[k++] = (uint64_t)a->j.stiffnessAng;
    out[k++] = (uint64_t)a->j.fracture; out[k++] = (uint64_t)a->j.torqueArm;
    out[k++] = (uint64_t)a->j.restAng; out[k++] = (uint64_t)a->j.broken;
    out[k++] = (uint64_t)a->j.frameA; out[k++] = (uint64_t)a->j.frameB;
    out[k++] = (uint64_t)a->j.axisMode; out[k++] = (uint64_t)a->j.axisLo; out[k++] = (uint64_t)a->j.axisHi;
    out[k++] = (uint64_t)a->j.generic; out[k++] = (uint64_t)a->j.count;
    out[k++] = (uint64_t)a->sp.bodyA; out[k++] = (uint64_t)a->sp.bodyB;
    out[k++] = (uint64_t)a->sp.rA; out[k++] = (uint64_t)a->sp.rB;
    out[k++] = (uint64_t)a->sp.rest; out[k++] = (uint64_t)a->sp.stiffness;
    out[k++] = (uint64_t)a->sp.count;
    out[k++] = (uint64_t)(uint)a->jointsAuthored;
    out[k++] = (uint64_t)a->s.push; out[k++] = (uint64_t)a->s.pull; out[k++] = (uint64_t)a->s.stress;
    out[k++] = (uint64_t)a->s.cone; out[k++] = (uint64_t)a->s.penetration; out[k++] = (uint64_t)a->s.jointDegree;
}

struct OverlayLoadClearArgsProbe { uint64_t load; int n; int _pad0; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeOverlayLoadClearArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    OverlayLoadClearArgsProbe *a = (OverlayLoadClearArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->load; out[k++] = (uint64_t)(uint)a->n;
}

struct OverlayLoadAccumArgsProbe { BodySoA b; ManifoldSoA m; int slots; int _pad0; uint64_t load; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeOverlayLoadAccumArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    OverlayLoadAccumArgsProbe *a = (OverlayLoadAccumArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->b.pos; out[k++] = (uint64_t)a->b.rot;
    out[k++] = (uint64_t)a->b.initialLin; out[k++] = (uint64_t)a->b.initialAng;
    out[k++] = (uint64_t)a->b.inertialLin; out[k++] = (uint64_t)a->b.inertialAng;
    out[k++] = (uint64_t)a->b.velLin; out[k++] = (uint64_t)a->b.velAng; out[k++] = (uint64_t)a->b.prevVelLin;
    out[k++] = (uint64_t)a->b.halfExtents; out[k++] = (uint64_t)a->b.shapeType;
    out[k++] = (uint64_t)a->b.moment; out[k++] = (uint64_t)a->b.mass; out[k++] = (uint64_t)a->b.friction;
    out[k++] = (uint64_t)a->b.awake; out[k++] = (uint64_t)a->b.sleepTimer;
    out[k++] = (uint64_t)a->b.restPos; out[k++] = (uint64_t)a->b.restRot;
    out[k++] = (uint64_t)a->b.gyration; out[k++] = (uint64_t)a->b.props; out[k++] = (uint64_t)a->b.count;
    out[k++] = (uint64_t)a->m.key; out[k++] = (uint64_t)a->m.bodyA; out[k++] = (uint64_t)a->m.bodyB;
    out[k++] = (uint64_t)a->m.basis; out[k++] = (uint64_t)a->m.friction;
    out[k++] = (uint64_t)a->m.contactOffset; out[k++] = (uint64_t)a->m.contactCount;
    out[k++] = (uint64_t)(uint)a->m.count; out[k++] = (uint64_t)(uint)a->m.capacity;
    out[k++] = (uint64_t)a->m.feature; out[k++] = (uint64_t)a->m.rA; out[k++] = (uint64_t)a->m.rB;
    out[k++] = (uint64_t)a->m.C0; out[k++] = (uint64_t)a->m.penalty; out[k++] = (uint64_t)a->m.lambda;
    out[k++] = (uint64_t)a->m.stick;
    out[k++] = (uint64_t)(uint)a->m.contactTotal; out[k++] = (uint64_t)(uint)a->m.contactCapacity; out[k++] = (uint64_t)a->m.shapeA; out[k++] = (uint64_t)a->m.shapeB;
    out[k++] = (uint64_t)(uint)a->slots;
    out[k++] = (uint64_t)a->load;
}

struct DebugDrawViewProbe { int contactPoints, contactNormals, manifoldLinks, tint; float pointSize, normalLength, forceMax; int _pad0; };
struct WriteDebugInstancesArgsProbe { BodySoA b; ManifoldSoA m; int slots; int _pad0; uint64_t out; int maxInstances; int _pad1; uint64_t counter; DebugDrawViewProbe d; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeWriteDebugInstancesArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    WriteDebugInstancesArgsProbe *a = (WriteDebugInstancesArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->b.pos; out[k++] = (uint64_t)a->b.rot;
    out[k++] = (uint64_t)a->b.initialLin; out[k++] = (uint64_t)a->b.initialAng;
    out[k++] = (uint64_t)a->b.inertialLin; out[k++] = (uint64_t)a->b.inertialAng;
    out[k++] = (uint64_t)a->b.velLin; out[k++] = (uint64_t)a->b.velAng; out[k++] = (uint64_t)a->b.prevVelLin;
    out[k++] = (uint64_t)a->b.halfExtents; out[k++] = (uint64_t)a->b.shapeType;
    out[k++] = (uint64_t)a->b.moment; out[k++] = (uint64_t)a->b.mass; out[k++] = (uint64_t)a->b.friction;
    out[k++] = (uint64_t)a->b.awake; out[k++] = (uint64_t)a->b.sleepTimer;
    out[k++] = (uint64_t)a->b.restPos; out[k++] = (uint64_t)a->b.restRot;
    out[k++] = (uint64_t)a->b.gyration; out[k++] = (uint64_t)a->b.props; out[k++] = (uint64_t)a->b.count;
    out[k++] = (uint64_t)a->m.key; out[k++] = (uint64_t)a->m.bodyA; out[k++] = (uint64_t)a->m.bodyB;
    out[k++] = (uint64_t)a->m.basis; out[k++] = (uint64_t)a->m.friction;
    out[k++] = (uint64_t)a->m.contactOffset; out[k++] = (uint64_t)a->m.contactCount;
    out[k++] = (uint64_t)(uint)a->m.count; out[k++] = (uint64_t)(uint)a->m.capacity;
    out[k++] = (uint64_t)a->m.feature; out[k++] = (uint64_t)a->m.rA; out[k++] = (uint64_t)a->m.rB;
    out[k++] = (uint64_t)a->m.C0; out[k++] = (uint64_t)a->m.penalty; out[k++] = (uint64_t)a->m.lambda;
    out[k++] = (uint64_t)a->m.stick;
    out[k++] = (uint64_t)(uint)a->m.contactTotal; out[k++] = (uint64_t)(uint)a->m.contactCapacity; out[k++] = (uint64_t)a->m.shapeA; out[k++] = (uint64_t)a->m.shapeB;
    out[k++] = (uint64_t)(uint)a->slots;
    out[k++] = (uint64_t)a->out;
    out[k++] = (uint64_t)(uint)a->maxInstances;
    out[k++] = (uint64_t)a->counter;
    out[k++] = (uint64_t)(uint)a->d.contactPoints;
    out[k++] = (uint64_t)(uint)a->d.contactNormals;
    out[k++] = (uint64_t)(uint)a->d.manifoldLinks;
    out[k++] = (uint64_t)(uint)a->d.tint;
    out[k++] = (uint64_t)asuint(a->d.pointSize);
    out[k++] = (uint64_t)asuint(a->d.normalLength);
    out[k++] = (uint64_t)asuint(a->d.forceMax);
}

// ===========================================================================
// Phase 5B: engine/shaders/{sleep,wake,mutate}.slang args structs
// (engine/vk/vk_args.h). Same reasoning as the Phase 4b/5C probes above --
// the two real bugs this gate exists to catch (an extra/missing pad silently shifting
// every later field) are exactly what a probe over only the fields a parity test
// happens to touch would miss, so every field of every struct is read back here.
// ===========================================================================

// Emits every BodySoA/ManifoldSoA/JointSoA/SpringSoA field, in the exact order
// fullBodyFields/fullManifoldFieldsAll/fullJointFieldsAll/fullSpringFieldsAll expect on
// the C++ side (test_vk_layout.cpp) -- these are plain statement blocks (Slang has no
// preprocessor macros usable across these signatures), copy-pasted once per struct
// exactly like csProbePrimalArgs above, so a probe here can never silently emit fewer
// fields than the host test expects (the earlier bug in this file).
void emitBody(BodySoA b, uint64_t *out, inout int k)
{
    out[k++] = (uint64_t)b.pos;
    out[k++] = (uint64_t)b.rot;
    out[k++] = (uint64_t)b.initialLin;
    out[k++] = (uint64_t)b.initialAng;
    out[k++] = (uint64_t)b.inertialLin;
    out[k++] = (uint64_t)b.inertialAng;
    out[k++] = (uint64_t)b.velLin;
    out[k++] = (uint64_t)b.velAng;
    out[k++] = (uint64_t)b.prevVelLin;
    out[k++] = (uint64_t)b.halfExtents;
    out[k++] = (uint64_t)b.shapeType;
    out[k++] = (uint64_t)b.moment;
    out[k++] = (uint64_t)b.mass;
    out[k++] = (uint64_t)b.friction;
    out[k++] = (uint64_t)b.awake;
    out[k++] = (uint64_t)b.sleepTimer;
    out[k++] = (uint64_t)b.restPos;
    out[k++] = (uint64_t)b.restRot;
    out[k++] = (uint64_t)b.gyration; out[k++] = (uint64_t)b.props;
    out[k++] = (uint64_t)(uint)b.count;
}

void emitManifold(ManifoldSoA m, uint64_t *out, inout int k)
{
    out[k++] = (uint64_t)m.key;
    out[k++] = (uint64_t)m.bodyA;
    out[k++] = (uint64_t)m.bodyB;
    out[k++] = (uint64_t)m.basis;
    out[k++] = (uint64_t)m.friction;
    out[k++] = (uint64_t)m.contactOffset;
    out[k++] = (uint64_t)m.contactCount;
    out[k++] = (uint64_t)(uint)m.count;
    out[k++] = (uint64_t)(uint)m.capacity;
    out[k++] = (uint64_t)m.feature;
    out[k++] = (uint64_t)m.rA;
    out[k++] = (uint64_t)m.rB;
    out[k++] = (uint64_t)m.C0;
    out[k++] = (uint64_t)m.penalty;
    out[k++] = (uint64_t)m.lambda;
    out[k++] = (uint64_t)m.stick;
    out[k++] = (uint64_t)(uint)m.contactTotal;
    out[k++] = (uint64_t)(uint)m.contactCapacity;
    out[k++] = (uint64_t)m.shapeA;
    out[k++] = (uint64_t)m.shapeB;
}

void emitJoint(JointSoA j, uint64_t *out, inout int k)
{
    out[k++] = (uint64_t)j.bodyA;
    out[k++] = (uint64_t)j.bodyB;
    out[k++] = (uint64_t)j.rA;
    out[k++] = (uint64_t)j.rB;
    out[k++] = (uint64_t)j.C0Lin;
    out[k++] = (uint64_t)j.C0Ang;
    out[k++] = (uint64_t)j.penaltyLin;
    out[k++] = (uint64_t)j.penaltyAng;
    out[k++] = (uint64_t)j.lambdaLin;
    out[k++] = (uint64_t)j.lambdaAng;
    out[k++] = (uint64_t)j.stiffnessLin;
    out[k++] = (uint64_t)j.stiffnessAng;
    out[k++] = (uint64_t)j.fracture;
    out[k++] = (uint64_t)j.torqueArm;
    out[k++] = (uint64_t)j.restAng;
    out[k++] = (uint64_t)j.broken;
    out[k++] = (uint64_t)j.frameA;
    out[k++] = (uint64_t)j.frameB;
    out[k++] = (uint64_t)j.axisMode;
    out[k++] = (uint64_t)j.axisLo;
    out[k++] = (uint64_t)j.axisHi;
    out[k++] = (uint64_t)j.generic;
    out[k++] = (uint64_t)(uint)j.count;
}

void emitSpring(SpringSoA sp, uint64_t *out, inout int k)
{
    out[k++] = (uint64_t)sp.bodyA;
    out[k++] = (uint64_t)sp.bodyB;
    out[k++] = (uint64_t)sp.rA;
    out[k++] = (uint64_t)sp.rB;
    out[k++] = (uint64_t)sp.rest;
    out[k++] = (uint64_t)sp.stiffness;
    out[k++] = (uint64_t)(uint)sp.count;
}

struct SleepBlockFractureArgsProbe { BodySoA b; JointSoA j; float ratio; uint64_t block; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeSleepBlockFractureArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    SleepBlockFractureArgsProbe *a = (SleepBlockFractureArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    emitBody(a->b, out, k);
    emitJoint(a->j, out, k);
    out[k++] = (uint64_t)asuint(a->ratio);
    out[k++] = (uint64_t)a->block;
}

struct SleepBlockDragArgsProbe { uint64_t dragIdx; int dragCount; uint64_t block; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeSleepBlockDragArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    SleepBlockDragArgsProbe *a = (SleepBlockDragArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->dragIdx;
    out[k++] = (uint64_t)(uint)a->dragCount;
    out[k++] = (uint64_t)a->block;
}

struct SleepMarkArgsProbe
{
    BodySoA b; uint64_t dyn; int dynCount; uint64_t candidate;
    int frames; float linVel; float angVel; float disp;
    uint64_t block; uint hasBlock; float leak;
};
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeSleepMarkArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    SleepMarkArgsProbe *a = (SleepMarkArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    emitBody(a->b, out, k);
    out[k++] = (uint64_t)a->dyn;
    out[k++] = (uint64_t)(uint)a->dynCount;
    out[k++] = (uint64_t)a->candidate;
    out[k++] = (uint64_t)(uint)a->frames;
    out[k++] = (uint64_t)asuint(a->linVel);
    out[k++] = (uint64_t)asuint(a->angVel);
    out[k++] = (uint64_t)asuint(a->disp);
    out[k++] = (uint64_t)a->block;
    out[k++] = (uint64_t)a->hasBlock;
    out[k++] = (uint64_t)asuint(a->leak);
}

struct SleepCommitConstraintGraphProbe { uint64_t *key; int *isA; int *start; int *end; };
struct SleepCommitArgsProbe
{
    BodySoA b; SleepCommitConstraintGraphProbe g; ManifoldSoA m; JointSoA j; SpringSoA sp;
    uint64_t active; int activeCount; uint64_t candidate;
    uint64_t touchCount; uint64_t touchHash; uint64_t snapCount; uint64_t snapHash;
    uint64_t activeCountPtr;
};
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeSleepCommitArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    SleepCommitArgsProbe *a = (SleepCommitArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    emitBody(a->b, out, k);
    out[k++] = (uint64_t)a->g.key; out[k++] = (uint64_t)a->g.isA; out[k++] = (uint64_t)a->g.start; out[k++] = (uint64_t)a->g.end;
    emitManifold(a->m, out, k);
    emitJoint(a->j, out, k);
    emitSpring(a->sp, out, k);
    out[k++] = (uint64_t)a->active;
    out[k++] = (uint64_t)(uint)a->activeCount;
    out[k++] = (uint64_t)a->candidate;
    out[k++] = (uint64_t)a->touchCount;
    out[k++] = (uint64_t)a->touchHash;
    out[k++] = (uint64_t)a->snapCount;
    out[k++] = (uint64_t)a->snapHash;
    out[k++] = (uint64_t)a->activeCountPtr;
}

struct ActiveFlagsArgsProbe { BodySoA b; uint64_t dyn; int dynCount; uint64_t flags; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeActiveFlagsArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    ActiveFlagsArgsProbe *a = (ActiveFlagsArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    emitBody(a->b, out, k);
    out[k++] = (uint64_t)a->dyn;
    out[k++] = (uint64_t)(uint)a->dynCount;
    out[k++] = (uint64_t)a->flags;
}

struct WakeRequestArgsProbe { BodySoA b; ManifoldSoA m; int slots; float wakeLinVel; float wakeAngVel; uint64_t request; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeWakeRequestArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    WakeRequestArgsProbe *a = (WakeRequestArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    emitBody(a->b, out, k);
    emitManifold(a->m, out, k);
    out[k++] = (uint64_t)(uint)a->slots;
    out[k++] = (uint64_t)asuint(a->wakeLinVel);
    out[k++] = (uint64_t)asuint(a->wakeAngVel);
    out[k++] = (uint64_t)a->request;
}

struct TouchClearArgsProbe { uint64_t dyn; int dynCount; uint64_t count; uint64_t hash; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeTouchClearArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    TouchClearArgsProbe *a = (TouchClearArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->dyn;
    out[k++] = (uint64_t)(uint)a->dynCount;
    out[k++] = (uint64_t)a->count;
    out[k++] = (uint64_t)a->hash;
}

struct TouchAccumArgsProbe { BodySoA b; ManifoldSoA m; int slots; uint64_t count; uint64_t hash; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeTouchAccumArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    TouchAccumArgsProbe *a = (TouchAccumArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    emitBody(a->b, out, k);
    emitManifold(a->m, out, k);
    out[k++] = (uint64_t)(uint)a->slots;
    out[k++] = (uint64_t)a->count;
    out[k++] = (uint64_t)a->hash;
}

struct WakeLostSupportArgsProbe
{
    BodySoA b; uint64_t dyn; int dynCount;
    uint64_t count; uint64_t hash; uint64_t snapCount; uint64_t snapHash; uint64_t request;
};
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeWakeLostSupportArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    WakeLostSupportArgsProbe *a = (WakeLostSupportArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    emitBody(a->b, out, k);
    out[k++] = (uint64_t)a->dyn;
    out[k++] = (uint64_t)(uint)a->dynCount;
    out[k++] = (uint64_t)a->count;
    out[k++] = (uint64_t)a->hash;
    out[k++] = (uint64_t)a->snapCount;
    out[k++] = (uint64_t)a->snapHash;
    out[k++] = (uint64_t)a->request;
}

struct WakeApplyArgsProbe { BodySoA b; uint64_t dyn; int dynCount; uint64_t request; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeWakeApplyArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    WakeApplyArgsProbe *a = (WakeApplyArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    emitBody(a->b, out, k);
    out[k++] = (uint64_t)a->dyn;
    out[k++] = (uint64_t)(uint)a->dynCount;
    out[k++] = (uint64_t)a->request;
}

struct TouchAccumJointsArgsProbe { BodySoA b; JointSoA j; SpringSoA sp; uint64_t count; uint64_t hash; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeTouchAccumJointsArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    TouchAccumJointsArgsProbe *a = (TouchAccumJointsArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    emitBody(a->b, out, k);
    emitJoint(a->j, out, k);
    emitSpring(a->sp, out, k);
    out[k++] = (uint64_t)a->count;
    out[k++] = (uint64_t)a->hash;
}

struct WakeRequestJointsArgsProbe { BodySoA b; JointSoA j; SpringSoA sp; uint64_t request; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeWakeRequestJointsArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    WakeRequestJointsArgsProbe *a = (WakeRequestJointsArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    emitBody(a->b, out, k);
    emitJoint(a->j, out, k);
    emitSpring(a->sp, out, k);
    out[k++] = (uint64_t)a->request;
}

struct WakeIslandSeedArgsProbe { BodySoA b; uint64_t dyn; int dynCount; uint64_t request; uint64_t seed; uint64_t any; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeWakeIslandSeedArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    WakeIslandSeedArgsProbe *a = (WakeIslandSeedArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    emitBody(a->b, out, k);
    out[k++] = (uint64_t)a->dyn;
    out[k++] = (uint64_t)(uint)a->dynCount;
    out[k++] = (uint64_t)a->request;
    out[k++] = (uint64_t)a->seed;
    out[k++] = (uint64_t)a->any;
}

struct WakeAllArgsProbe { BodySoA b; uint64_t dyn; int dynCount; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeWakeAllArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    WakeAllArgsProbe *a = (WakeAllArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    emitBody(a->b, out, k);
    out[k++] = (uint64_t)a->dyn;
    out[k++] = (uint64_t)(uint)a->dynCount;
}

struct MutateApplyArgsProbe { BodySoA b; int type; uint64_t idx; uint64_t data; int count; float dt; uint64_t seed; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeMutateApplyArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    MutateApplyArgsProbe *a = (MutateApplyArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    emitBody(a->b, out, k);
    out[k++] = (uint64_t)(uint)a->type;
    out[k++] = (uint64_t)a->idx;
    out[k++] = (uint64_t)a->data;
    out[k++] = (uint64_t)(uint)a->count;
    out[k++] = (uint64_t)asuint(a->dt);
    out[k++] = (uint64_t)a->seed;
}

struct WakeSeedRegionArgsProbe { BodySoA b; uint64_t dyn; int dynCount; int _pad0; vec3 lo; vec3 hi; uint64_t seed; uint64_t changed; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeWakeSeedRegionArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    WakeSeedRegionArgsProbe *a = (WakeSeedRegionArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    emitBody(a->b, out, k);
    out[k++] = (uint64_t)a->dyn;
    out[k++] = (uint64_t)(uint)a->dynCount;
    out[k++] = (uint64_t)asuint(a->lo.x); out[k++] = (uint64_t)asuint(a->lo.y); out[k++] = (uint64_t)asuint(a->lo.z);
    out[k++] = (uint64_t)asuint(a->hi.x); out[k++] = (uint64_t)asuint(a->hi.y); out[k++] = (uint64_t)asuint(a->hi.z);
    out[k++] = (uint64_t)a->seed;
    out[k++] = (uint64_t)a->changed;
}

struct WakeFloodContactsArgsProbe { BodySoA b; ManifoldSoA m; int slots; uint64_t seed; uint64_t changed; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeWakeFloodContactsArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    WakeFloodContactsArgsProbe *a = (WakeFloodContactsArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    emitBody(a->b, out, k);
    emitManifold(a->m, out, k);
    out[k++] = (uint64_t)(uint)a->slots;
    out[k++] = (uint64_t)a->seed;
    out[k++] = (uint64_t)a->changed;
}

struct WakeFloodJointsArgsProbe { BodySoA b; JointSoA j; SpringSoA sp; uint64_t seed; uint64_t changed; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeWakeFloodJointsArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    WakeFloodJointsArgsProbe *a = (WakeFloodJointsArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    emitBody(a->b, out, k);
    emitJoint(a->j, out, k);
    emitSpring(a->sp, out, k);
    out[k++] = (uint64_t)a->seed;
    out[k++] = (uint64_t)a->changed;
}

struct WakeSeedMergeArgsProbe { uint64_t dyn; int dynCount; uint64_t seed; uint64_t request; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeWakeSeedMergeArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    WakeSeedMergeArgsProbe *a = (WakeSeedMergeArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->dyn;
    out[k++] = (uint64_t)(uint)a->dynCount;
    out[k++] = (uint64_t)a->seed;
    out[k++] = (uint64_t)a->request;
}

struct BodyGatherArgsProbe { BodySoA b; int type; uint64_t idx; uint64_t out_; int count; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeBodyGatherArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    BodyGatherArgsProbe *a = (BodyGatherArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    emitBody(a->b, out, k);
    out[k++] = (uint64_t)(uint)a->type;
    out[k++] = (uint64_t)a->idx;
    out[k++] = (uint64_t)a->out_;
    out[k++] = (uint64_t)(uint)a->count;
}

struct GateDispatchArgsProbe { uint64_t src; uint64_t dst; uint64_t flag; int count; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeGateDispatchArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    GateDispatchArgsProbe *a = (GateDispatchArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->src;
    out[k++] = (uint64_t)a->dst;
    out[k++] = (uint64_t)a->flag;
    out[k++] = (uint64_t)(uint)a->count;
}

struct StepFlagIfSetProbe { uint64_t src; uint64_t flag; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeStepFlagIfSet(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    StepFlagIfSetProbe *a = (StepFlagIfSetProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->src;
    out[k++] = (uint64_t)a->flag;
}

struct StepFlagIfGreaterProbe { uint64_t count; uint64_t flag; int cap; int _pad; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeStepFlagIfGreater(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    StepFlagIfGreaterProbe *a = (StepFlagIfGreaterProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->count;
    out[k++] = (uint64_t)a->flag;
    out[k++] = (uint64_t)(uint)a->cap;
}

struct StepColorBoundProbe { uint64_t start; uint64_t end; uint64_t flag; int bound; int maxColors; };
[shader("compute")] [numthreads(1, 1, 1)]
void csProbeStepColorBound(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    StepColorBoundProbe *a = (StepColorBoundProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->start;
    out[k++] = (uint64_t)a->end;
    out[k++] = (uint64_t)a->flag;
    out[k++] = (uint64_t)(uint)a->bound;
    out[k++] = (uint64_t)(uint)a->maxColors;
}

// CoarseArgs (primal.slang): PrimalArgs (proven above) followed by the coarse fields.
struct CoarseArgsProbe
{
    PrimalArgsProbe pa;
    int *colorStart;
    int *colorEnd;
    int paColor;
    int _pad3;
    float *shapeRolling;
    int *active;
    int *activeCount;
    int *aggOf;
    uint *keys;
    float *acc;
    float *delta;
    int *stats;
    int maxActive;
    int bodyCount;
    int tableSize;
    float cell;
    float damping;
    int withForces;
    int color;
    int _pad0;
};

[shader("compute")] [numthreads(1, 1, 1)]
void csProbeCoarseArgs(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    ProbeArgs *args = (ProbeArgs *)pc.argsAddr;
    CoarseArgsProbe *a = (CoarseArgsProbe *)args->srcAddr;
    uint64_t *out = (uint64_t *)args->outAddr;
    int k = 0;
    out[k++] = (uint64_t)a->pa.colorBodies;
    out[k++] = (uint64_t)a->colorStart;
    out[k++] = (uint64_t)(uint)a->paColor;
    out[k++] = (uint64_t)a->shapeRolling;
    out[k++] = (uint64_t)a->active;
    out[k++] = (uint64_t)a->activeCount;
    out[k++] = (uint64_t)a->aggOf;
    out[k++] = (uint64_t)a->keys;
    out[k++] = (uint64_t)a->acc;
    out[k++] = (uint64_t)a->delta;
    out[k++] = (uint64_t)a->stats;
    out[k++] = (uint64_t)(uint)a->maxActive;
    out[k++] = (uint64_t)(uint)a->bodyCount;
    out[k++] = (uint64_t)(uint)a->tableSize;
    out[k++] = (uint64_t)asuint(a->cell);
    out[k++] = (uint64_t)asuint(a->damping);
    out[k++] = (uint64_t)(uint)a->withForces;
    out[k++] = (uint64_t)(uint)a->color;
}
