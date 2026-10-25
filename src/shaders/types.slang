// Device-side mirrors of device-side pointer-bundle structs (engine/core/gpu_types.cuh,
// engine/core/gpu_forces.cuh) and the solver's per-step params (engine/host/gpu_solver.h:
// SolverParams), for Slang kernels.
//
// Phase 1 found that Slang's scalar layout (-fvk-use-scalar-layout) does NOT tail-pad a
// nested struct the way the C++/device ABI does: a C++ struct that ends in a 4-byte field
// after a run of 8-byte pointers is padded by the compiler to a multiple of 8 (the
// pointers force 8-byte alignment on the whole struct), but Slang under scalar layout
// packs fields back-to-back with no such trailing pad unless one is written explicitly.
// See integrate.slang's BodySoA for the first instance of this; every struct below
// follows the same rule: after the last field, add `uint _padN` for every 4 bytes needed
// to bring the struct up to a multiple of 8 (the alignment its 8-byte pointers demand).
//
// tools/test_vk_layout.cpp is the mechanical check, and it is an end-to-end GPU one: it
// fills the REAL the reference structs (gpu_types.cuh, gpu_forces.cuh, gpu_solver.h via
// engine/vk/solver_params_pack.h) with a distinct sentinel per field, uploads
// the raw bytes unchanged, and dispatches engine/shaders/layout_probe.slang, which
// imports THIS file and reads the instance back through the actual struct declarations
// below -- the Slang compiler computes every offset itself. A dropped pad or reordered
// field makes a sentinel land in the wrong slot, caught by field name on readback. (An
// earlier version of this file's comment described an offsetof-only host-side check;
// that version could not see the Slang compiler's own layout and has been replaced.)
//
// IMPORTANT -- SolverParams is NOT mirrored as a raw byte-for-byte struct here. The real
// C++ SolverParams (engine/host/gpu_solver.h) interleaves `bool` fields (1 byte in the
// Itanium/MSVC C++ ABI) with `float`/`int` fields (4-byte aligned), so the compiler
// inserts 3 bytes of UNINITIALIZED padding after each bool. HLSL/Slang have no 1-byte
// bool -- `bool` there is a 4-byte type -- so there is no scalar-layout Slang struct that
// is simultaneously (a) byte-identical to the C++ layout and (b) safe to read on the GPU:
// reading the 3 padding bytes as part of a wider load risks treating uninitialized stack
// garbage as meaningful bits. SolverParamsGpu below is instead an explicitly *packed*
// struct the host must marshal into (bools converted to uint32_t 0/1, no reliance on
// struct padding), matching what integrate.slang's minimal SolverParams already does by
// accident (it only carries dt/gravity, which happen to have no bools between them).

import maths;

// ---------------------------------------------------------------------------
// BodySoA -- mirrors engine/core/gpu_types.cuh:BodySoA (152 bytes in the C++ ABI:
// 18 pointers + int count, padded to a multiple of 8).
// ---------------------------------------------------------------------------
// Per-body settings, 56 bytes (mirrors engine/core/scene_types.h BodyProps). All-zero except
// gravityScale = 1 is the default.
public static const uint BODY_KINEMATIC = 1u;
public static const uint BODY_NO_SLEEP = 2u;
public static const uint BODY_LOCK_LIN = 4u | 8u | 16u;
public static const uint BODY_LOCK_ANG = 32u | 64u | 128u;
public static const uint BODY_LOCK_ANY = 4u | 8u | 16u | 32u | 64u | 128u;
public static const uint BODY_BULLET = 256u;

public struct BodyProps
{
    public float linearDamping;
    public float angularDamping;
    public float gravityScale;
    public float maxLinearSpeed;
    public float sleepThreshold;
    public uint flags;
    public float tangentSpeed;
    public float rollingResistance;
    public uint userMaterialId;
    public uint frictionMix;
    public float restitution;
    public float tangentY;
    public float tangentZ;
    public uint _pad;
};

public struct BodySoA
{
    public vec3 *pos;
    public quat *rot;
    public vec3 *initialLin;
    public quat *initialAng;
    public vec3 *inertialLin;
    public quat *inertialAng;
    public vec3 *velLin;
    public vec3 *velAng;
    public vec3 *prevVelLin;
    public vec3 *halfExtents;
    public uint8_t *shapeType;
    public vec3 *moment;
    public float *mass;
    public float *friction;
    public uint8_t *awake;
    public int *sleepTimer;
    public vec3 *restPos;
    public quat *restRot;
    public float *gyration;
    // Per-body settings (BodyProps). NULL unless some body has a non-default setting, so
    // every kernel that reads it first checks `props != nullptr` and the default scene
    // runs the original code.
    public BodyProps *props;
    public uint count;
    public uint _pad0; // tail pad: round 19*8 + 4 = 156 up to 160
};

// ---------------------------------------------------------------------------
// ShapeSoA -- the collision shapes, one entry per shape (a body owns one or more). Shape
// poses are body pose o local pose. Mass/inertia stay on the body. category/mask/group
// are the collision filter, stored but not yet consulted. flags bit 0
// (AVBD_SHAPE_FLAG_IDENTITY) means the local pose is identity: kernels then use the body
// pose as is. geomOffset indexes the byte heap holding Box3D hull blobs (VkWorld::addGeometry).
// 11 pointers + count, padded to 8.
// ---------------------------------------------------------------------------
public struct ShapeSoA
{
    public int *body;
    public vec3 *localPos;
    public quat *localRot;
    public vec3 *halfExtents;
    public uint8_t *shapeType;
    public float *friction;
    public uint64_t *category;
    public uint64_t *mask;
    public int *group;
    public uint *flags;
    public uint *geomOffset; // byte offset of the shape's blob in the geometry heap; 0xFFFFFFFF: none
    public uint count;
    public uint _pad0;
};

// ---------------------------------------------------------------------------
// ManifoldSoA -- mirrors engine/core/gpu_types.cuh:ManifoldSoA (128 bytes; ends on an
// 8-byte boundary already, so no trailing pad is needed).
// ---------------------------------------------------------------------------
public struct ManifoldSoA
{
    public uint64_t *key;          // PairKey* (unsigned long long*)
    public int *bodyA;
    public int *bodyB;
    public float3x3 *basis;
    public float *friction;
    public int *contactOffset;
    public int *contactCount;
    public int count;
    public int capacity;
    public int *feature;           // FeaturePair* (union, 4 bytes) -- read as its int alias
    public vec3 *rA;
    public vec3 *rB;
    public vec3 *C0;
    public vec3 *penalty;
    public vec3 *lambda;
    public uint8_t *stick;
    public int contactTotal;
    public int contactCapacity;
    public int *shapeA;            // the shapes whose pair produced the manifold (key = shape pair)
    public int *shapeB;
};

// ---------------------------------------------------------------------------
// JointSoA -- mirrors engine/core/gpu_forces.cuh:JointSoA (184 bytes: 22 pointers +
// int count, padded to a multiple of 8).
// ---------------------------------------------------------------------------
public struct JointSoA
{
    public int *bodyA;
    public int *bodyB;
    public vec3 *rA;
    public vec3 *rB;
    public vec3 *C0Lin;
    public vec3 *C0Ang;
    public vec3 *penaltyLin;
    public vec3 *penaltyAng;
    public vec3 *lambdaLin;
    public vec3 *lambdaAng;
    public float *stiffnessLin;
    public float *stiffnessAng;
    public float *fracture;
    public float *torqueArm;
    public quat *restAng;
    public uint8_t *broken;
    public quat *frameA;
    public quat *frameB;
    public uint8_t *axisMode;
    public float *axisLo;
    public float *axisHi;
    public uint8_t *generic;
    public float *fractureAng; // N*m bending capacity; < 0: legacy |lambdaAng| test. may be null (legacy)
    public uint count;
    public uint _pad0; // tail pad: 23*8 + 4 = 188 up to 192
};

// ---------------------------------------------------------------------------
// SpringSoA -- mirrors engine/vk/vk_args.h:FullSpringSoAGpu (56 bytes: 6 pointers +
// int count, padded to a multiple of 8): the plain distance spring.
// ---------------------------------------------------------------------------
public struct SpringSoA
{
    public int *bodyA;
    public int *bodyB;
    public vec3 *rA;
    public vec3 *rB;
    public float *rest;
    public float *stiffness;
    public uint count;
    public uint _pad0; // tail pad: round 6*8 + 4 = 52 up to 56
};

// ---------------------------------------------------------------------------
// SolverParamsGpu -- an explicit, hand-packed subset of engine/host/gpu_solver.h's
// SolverParams. See the file-header note: this is deliberately NOT a byte-mirror of the
// C++ struct. Every bool becomes a uint (0/1) the host writes explicitly; there is no
// reliance on C++ struct padding being zero or meaningful. Field set here covers what
// the ported kernels (integrate, collide, dual, primal) need; extend as more of the
// solver is ported, keeping the same "no raw bool" rule.
// ---------------------------------------------------------------------------
public struct SolverParamsGpu
{
    public float dt;
    public float worldFrameInertia; // 0/1 (was a vestigial gravity copy): primal.slang worldInertia
    public int iterations;
    public int primalLanes;
    public float alpha;
    public float betaLin;
    public float betaAng;
    public float gamma;
    public uint sleepEnabled;
    public int sleepFrames;
    public float sleepLinVel;
    public float sleepAngVel;
    public float sleepDisp;
    public float sleepFracture;
    public uint autoKStart;
    public float dualDamping;
    public float dualDampVel;
    public uint killFallenEnabled;
    public float killHeight;
    public float grabStrength;
};
