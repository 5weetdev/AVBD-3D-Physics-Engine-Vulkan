/*
 * Copyright (c) 2026 Chris Giles
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies.
 * Chris Giles makes no representations about the suitability
 * of this software for any purpose.
 * It is provided "as is" without express or implied warranty.
 */

#pragma once

// Plain-data types shared by every backend and by the scene authoring layer: joint
// descriptions, overlay/debug-draw options, readback structs, the body tier. No device code, no
// Vulkan, no solver state -- a scene described in these terms can be built without either.

#include "collision.h"
#include "maths.h"

// One axis of a generic 6-DOF joint. `mode` is an AVBD-level mirror of JointAxisMode
// in the shaders, repeated here so a caller does not have to include them.
struct JointAxis
{
    int mode = 0;         // JointAxisFree
    float lo = 0.0f;      // limit low / motor target / spring target
    float hi = 0.0f;      // limit high / motor max force / spring stiffness
    float damping = 0.0f; // spring damping ratio zeta (SPRING / SPRING_F only), 1/16 steps up to 1.94
};

enum JointAxisKind
{
    JointAxisFree = 0,
    JointAxisLocked = 1,
    JointAxisLimited = 2,
    JointAxisMotor = 3,
    JointAxisSpring = 4,
    // Phase 3 (joint_presets.h). SpringF: lo target, hi stiffness, force bounded by
    // SceneBuilder::setJointForceBounds. VelocityMotor: lo speed, hi max force. Cone: angular
    // axis 3 of a swing-twist joint, hi = half angle.
    JointAxisSpringF = 5,
    JointAxisVelocityMotor = 6,
    JointAxisCone = 7
};

// A generic joint, authored in each body's own space. The frames are what the axis
// modes are measured in: axis 0..2 are linear x/y/z of frame A, 3..5 angular.
struct JointDesc
{
    vec3 rA{0, 0, 0}, rB{0, 0, 0};
    quat frameA{0, 0, 0, 1}, frameB{0, 0, 0, 1};
    JointAxis axis[6];
    float fracture = INFINITY;
};

// Debug overlays.
//
// Every one of these recolours the existing box instances in place, inside the same
// kernel that already packs them for the renderer -- there is no second draw call, no
// line geometry and no readback, so an overlay is affordable at 100k bodies where an
// immediate-mode debug draw is not. The solver state is never touched: an overlay only
// changes the byte of colour a body is written with, so what you see is the simulation
// running exactly as it runs with the overlay off.
//
// The modes are mutually exclusive because they all compete for the same channel.
enum OverlayMode
{
    OVERLAY_NONE = 0,
    OVERLAY_SLEEP,        // asleep vs awake, with the settling countdown as a gradient
    OVERLAY_GRID,         // broadphase hash grid: one hue per occupied cell
    OVERLAY_GRAPH_COLOR,  // graph coloring, i.e. the parallel batches of the primal solve
    OVERLAY_LOAD,         // peak contact normal force on the body: the force chains
    OVERLAY_SPEED,        // linear speed
    OVERLAY_DEGREE,       // number of bodies this one is touching
    OVERLAY_JOINT_DEGREE, // number of unbroken joints and springs attached to this body
    OVERLAY_JOINT_LOAD,   // worst connected joint load, signed: compression vs tension
    OVERLAY_JOINT_STRESS, // worst connected joint load as a fraction of its fracture limit
    OVERLAY_AXIAL_LOAD,   // structural load, signed: contacts pushing vs joints pulling
    OVERLAY_FRICTION,     // fraction of the friction cone in use: where the scene shears
    OVERLAY_PENETRATION,  // deepest contact penetration: where the solve is not holding
    OVERLAY_MODE_COUNT
};

struct OverlayOptions
{
    OverlayMode mode = OVERLAY_NONE;

    // Upper end of each heat ramp. Scenes differ by orders of magnitude -- a 100k pile
    // and a five-block tower have nothing in common in newtons -- so these are exposed
    // rather than guessed.
    float speedMax = 10.0f;
    float loadMax = 200.0f;
    int degreeMax = 12;

    // The structural overlays. `jointLoadMax` is the newton value at each end of the
    // diverging compression/tension ramp; `penetrationMax` is in metres. Stress is a
    // fraction of each joint's own fracture limit, so it needs no range.
    float jointLoadMax = 500.0f;
    int jointDegreeMax = 6;
    float penetrationMax = 0.02f;

    // Statics stay their neutral grey instead of joining the overlay. The ground plane
    // is in none of these data sets (not binned in the grid, not in the constraint
    // graph, never asleep), so colouring it in says something untrue about it.
    bool colorStatics = false;
};

// How a contact marker is tinted. The marker is drawn either way; this only picks
// which of the contact's numbers it reports.
enum ContactTint
{
    CONTACT_TINT_FORCE = 0, // normal force, on the same heat ramp as the load overlay
    CONTACT_TINT_FRICTION,  // fraction of the friction cone in use: blue held, red slipping
    CONTACT_TINT_STICK,     // green where static friction is holding, orange where it is slipping
    CONTACT_TINT_FLAT       // one colour, for when the geometry alone is the point
};

// Geometry drawn *in addition to* the bodies, rather than a recolouring of them.
//
// These cost what they draw, so unlike the per-body overlays they are budgeted: a
// 50k-body pile carries ~230k contacts, and a marker each is a quarter-million extra
// instances. Everything here is off by default and, while off, launches no kernel and
// grows no buffer -- the whole feature is one `if` on the render path.
struct DebugDrawOptions
{
    bool contactPoints = false;  // a small cube at each contact point
    bool contactNormals = false; // a needle along the contact normal, scaled by force
    bool manifoldLinks = false;  // a rod joining the centres of each touching pair
    bool jointLoadRods = false;  // recolour the joint rods by signed axial load

    ContactTint contactTint = CONTACT_TINT_FORCE;

    float contactPointSize = 0.04f; // half-extent of a contact marker, metres
    float normalLength = 0.30f;     // needle length at full force, metres
    float forceMax = 200.0f;        // force at which the marker ramp saturates

    // Hard ceiling on markers. Reached, the extra contacts are simply not drawn --
    // dropped rather than decimated, since a stable subset is easier to read across
    // frames than a set that reshuffles. Reported back so the UI can say so.
    int maxMarkers = 120000;

    bool any() const { return contactPoints || contactNormals || manifoldLinks; }
};

// Constraint force and torque of one joint, world frame, from the joint's multipliers
// (VkWorld::downloadJointForces). A soft row (spring) carries no multiplier, so it reads zero.
// `events` is a mask of JOINT_EVENT_*; it only flags, it never breaks the joint (fracture is
// the separate, destructive threshold).
enum JointEventBits
{
    JOINT_EVENT_FORCE = 1, // |force| > the joint's force threshold
    JOINT_EVENT_TORQUE = 2 // |torque| > the joint's torque threshold
};

struct JointForce
{
    vec3 force, torque;
    unsigned events;
};

// External mutation payloads (GpuSolver / VkWorld::setBodyData, getBodyData). The
// numeric values of the enums are the shaders' contract (mutate.slang BW_*/BR_*).
enum class BodyWrite
{
    Transform,       // BodyTransform: absolute pose. Re-anchors the rest pose.
    LinearVelocity,  // vec3, absolute
    AngularVelocity, // vec3, absolute
    Impulse,         // BodyPointVector: impulse at a world point
    Force,           // BodyPointVector: force at a world point, applied over one dt
    Torque,          // vec3
    Awake,           // unsigned char: 1 wakes, 0 forces to sleep
    Props,           // BodyProps: replaces the body's settings
    Target,          // BodyTarget: kinematic target pose, converted to a velocity
};

enum class BodyRead
{
    Pose,            // BodyTransform
    LinearVelocity,  // vec3
    AngularVelocity, // vec3
    Awake,           // unsigned char
};

struct BodyTransform
{
    vec3 pos;
    quat rot;
};

// Per-body settings. The device mirror is BodyProps in types.slang, 56 bytes;
// the all-default value is "no special behaviour" and a world whose bodies are all
// default never reads or dispatches anything for it.
enum BodyPropFlags : unsigned
{
    BODY_KINEMATIC = 1u, // infinite mass, moved by its own velocity, not solved
    BODY_NO_SLEEP = 2u,  // never fall asleep (sleep test only)
    BODY_LOCK_LIN_X = 4u,
    BODY_LOCK_LIN_Y = 8u,
    BODY_LOCK_LIN_Z = 16u,
    BODY_LOCK_ANG_X = 32u,
    BODY_LOCK_ANG_Y = 64u,
    BODY_LOCK_ANG_Z = 128u,
    BODY_LOCK_ANY = 4u | 8u | 16u | 32u | 64u | 128u,
    BODY_BULLET = 256u, // swept against non-bullet dynamics too (ccd.slang)
};

// How two bodies' friction coefficients combine; a pair uses the larger enum value of
// the two bodies (so any body asking for a rule overrides the default).
enum FrictionMix : unsigned
{
    FRICTION_MIX_GEOMETRIC = 0, // sqrt(a*b), the default
    FRICTION_MIX_MIN = 1,
    FRICTION_MIX_MAX = 2,
    FRICTION_MIX_AVERAGE = 3,
    FRICTION_MIX_MULTIPLY = 4,
};

struct BodyProps
{
    float linearDamping = 0.0f; // v *= 1/(1+dt*c) before the inertial target
    float angularDamping = 0.0f;
    float gravityScale = 1.0f;
    float maxLinearSpeed = 0.0f;    // m/s, 0 = no clamp
    float sleepThreshold = 0.0f;    // m/s, 0 = SolverParams.sleepLinVel/AngVel
    unsigned flags = 0;             // BodyPropFlags
    float tangentSpeed = 0.0f;      // conveyor surface velocity in the body frame, m/s: x
    float rollingResistance = 0.0f; // bound = rr * |Fn| on the contact rolling torque
    unsigned userMaterialId = 0;    // passed through to manifolds, never read by the solver
    unsigned frictionMix = 0;       // FrictionMix
    float restitution = 0.0f;       // bounce, max of the pair; > 0 turns on restitution.slang
    float tangentY = 0.0f;          // ... y of the body-frame belt velocity
    float tangentZ = 0.0f;          // ... z of the body-frame belt velocity
    unsigned _pad = 0;
};
static_assert(sizeof(BodyProps) == 56, "BodyProps mirrors types.slang BodyProps");

inline bool bodyPropsDefault(const BodyProps &p)
{
    return p.linearDamping == 0.0f && p.angularDamping == 0.0f && p.gravityScale == 1.0f && p.maxLinearSpeed == 0.0f &&
           p.sleepThreshold == 0.0f && p.flags == 0 && p.tangentSpeed == 0.0f && p.tangentY == 0.0f &&
           p.tangentZ == 0.0f && p.rollingResistance == 0.0f && p.userMaterialId == 0 && p.frictionMix == 0 &&
           p.restitution == 0.0f;
}

// BodyWrite::Target payload: move a kinematic body to this pose over dt.
struct BodyTarget
{
    vec3 pos;
    quat rot;
    float dt;
};

struct BodyPointVector
{
    vec3 vector;
    vec3 worldPoint;
};

// One rendered box. Matches the vertex attribute layout in vk_renderer.cpp, and is
// written directly into the instance buffer by the solver, so the field order here is part of
// the renderer's contract.
struct BodyInstance
{
    float pos[3];
    float rot[4];
    float half[3];
    unsigned int rgba;
};

// Which half of the tiered world a body belongs to (UNITY_DOTS_PLAN.md §1). This is a
// pure *readback* classification and nothing else: both tiers are simulated identically,
// collide with each other, sleep, and are rendered from the same device arrays. The only
// thing it changes is whether the body's pose is packed into the pinned host buffer that
// stepEnd() publishes -- which is exactly the transfer that does not scale, because it is
// the one thing per body that has to cross PCIe and then be touched by the CPU.
//
// Default is Interactive, so a caller that never mentions tiers gets phase 3's behaviour
// unchanged: every body downloaded, every step.
enum class BodyTier : unsigned char
{
    Interactive = 0, // pose downloaded every stepEnd(); the tier a script may read
    Bulk = 1,        // never downloaded; exists only as device rows + instance data
};

// One spawned body of a group, in world space.
struct SpawnBody
{
    vec3 size;
    float density;
    float friction;
    vec3 position;
    quat rotation = quat{0, 0, 0, 1};
    vec3 velocity = vec3{0, 0, 0};
    vec3 angularVelocity = vec3{0, 0, 0};
};

// One joint of a spawned group. The body indices are global, i.e. the ones
// spawnGroup() is about to hand out: base + i for the i'th body of the group, or an
// existing body, or -1 for a world anchor.
struct SpawnJoint
{
    int bodyA;
    int bodyB;
    vec3 rA, rB;
    float stiffnessLin = INFINITY;
    float stiffnessAng = 0.0f;
    float fracture = INFINITY;
};
