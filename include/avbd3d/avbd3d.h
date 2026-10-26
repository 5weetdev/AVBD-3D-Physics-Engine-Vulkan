/* SPDX-FileCopyrightText: 2026 avbd3d contributors
   SPDX-License-Identifier: MIT */

/* avbd3d: a 3D rigid body solver on Vulkan compute, using Augmented Vertex Block Descent
   (Giles, Diaz, Yuksel, SIGGRAPH 2025). This is the public C API. It follows the Box3D model: a world
   holds bodies, a body holds shapes, joints connect two bodies. The headers are:

     avbd3d.h          this file: worlds, bodies, shapes, joints, queries, events, grab
     avbd3d_geometry.h hulls, meshes, height fields, compounds and the character mover helpers
     avbd3d_vulkan.h   the shared Vulkan device, render data and contact markers

   Conventions

    - Every call returns an Avbd3dResult. AVBD3D_OK is 0; every error is negative. The only
      exceptions are the *_is_valid calls (they take only an id and return int32_t), the
      avbd3d_default_* calls (they return the struct), avbd3d_abi_version, avbd3d_result_string
      and the pure helpers (avbd3d_make_box_hull, avbd3d_box_hull_data, avbd3d_solve_planes,
      avbd3d_clip_vector).
    - Results come back through the last parameter: "out" for a value, "out_count" for a count,
      "out_id" for a created object, "out_world" for a created world. An output pointer must not
      be NULL.
    - Booleans are int32_t (0 or 1). Structs are plain data; call the matching avbd3d_default_*
      function first and edit the fields you need.
    - Ids are small value types: {index, world, generation}. An id whose object was destroyed is
      stale; every call that takes one returns AVBD3D_ERR_INVALID_ARG. A NULL world returns
      AVBD3D_ERR_NULL_HANDLE.
    - Nothing is thread safe. Call one world from one thread at a time.
    - A Vulkan failure never aborts the process. The world records it and is dead from then on:
      every call on it returns AVBD3D_ERR_DEVICE_LOST (or AVBD3D_ERR_DEVICE or
      AVBD3D_ERR_OUT_OF_MEMORY, whichever stopped it) and only avbd3d_destroy_world works. A dead
      world is not recovered: destroy it and create a new one. There is no sticky error to poll
      and no per-thread error: the Result of the call that failed says why.
    - A step that had to clamp results because a hard capacity ceiling was reached returns
      AVBD3D_ERR_CAPACITY. The world keeps working; avbd3d_get_stats says what overflowed.
    - Arrays are passed as a pointer and a capacity. A call that fills an array writes at most
      capacity elements and reports the number it wrote (or, for a count query with capacity 0,
      the number available) in out_count.

   Steps and queued work

    avbd3d_step is avbd3d_step_begin followed by avbd3d_step_end. step_begin records and submits
    the whole step to the GPU and returns. While it runs every call stays available and none
    waits for it: getters, queries and events read the newest finished step, and setters,
    creates and destroys are queued and applied at the top of the next step. The exceptions end
    the step first because they use the device directly: the render-data writes of
    avbd3d_vulkan.h, grab move and end, and avbd3d_get_stats. The intended loop is

        avbd3d_step_end(world);        finish step N (a no-op on the first frame)
        ... read poses and events of step N, write forces and velocities ...
        avbd3d_step_begin(world, dt);  step N + 1 runs while the frame goes on

    A body written since the last finished step keeps the written value until the step that
    applies it has finished. Forces and torques act for the next step only; impulses change the
    velocity before it.

   Body origin and centre of mass

    A body has a frame, its origin, and a centre of mass computed from its shapes. Positions,
    rotations and transforms set and read through the body calls are the origin frame.
    avbd3d_body_get_world_center and avbd3d_body_get_local_center give the centre of mass. Shapes
    are attached in the body's origin frame; a body whose centre of mass is not at its origin
    rotates about the centre of mass. avbd3d_body_set_mass_data replaces the computed mass,
    centre and inertia until avbd3d_body_apply_mass_from_shapes restores them.

   Units and gravity

    Lengths are metres, masses kilograms, times seconds, angles radians. The default gravity is
    (0, -10, 0): the solver is y-up by default, but any gravity direction works and the world has
    no other notion of "up".

   Linking

    Build the library with AVBD3D_BUILD_SHARED. A program that links the shared library on
    Windows defines AVBD3D_USE_SHARED before including this header. Everything is extern "C".
*/
#ifndef AVBD3D_H
#define AVBD3D_H

#include <stddef.h>
#include <stdint.h>

#define AVBD3D_ABI_VERSION_MAJOR 1
#define AVBD3D_ABI_VERSION_MINOR 0

#if defined(_WIN32)
#if defined(AVBD3D_BUILD_SHARED)
#define AVBD3D_API __declspec(dllexport)
#elif defined(AVBD3D_USE_SHARED)
#define AVBD3D_API __declspec(dllimport)
#else
#define AVBD3D_API
#endif
#else
#define AVBD3D_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* The result of every call. AVBD3D_OK is 0 and every error is negative. */
typedef enum Avbd3dResult
{
	/* The call succeeded. */
	AVBD3D_OK = 0,
	/* A world (or GPU) pointer was NULL or not a live one. */
	AVBD3D_ERR_NULL_HANDLE = -1,
	/* An argument was NULL, out of range, malformed or a stale id. */
	AVBD3D_ERR_INVALID_ARG = -2,
	/* The feature is not available on this build or device. */
	AVBD3D_ERR_UNSUPPORTED = -3,
	/* No usable Vulkan device, or a Vulkan call failed for another reason. The world is dead. */
	AVBD3D_ERR_DEVICE = -4,
	/* A hard capacity ceiling was reached: the step committed clamped results. */
	AVBD3D_ERR_CAPACITY = -5,
	/* VK_ERROR_DEVICE_LOST (a driver reset or a faulting kernel). The world is dead. */
	AVBD3D_ERR_DEVICE_LOST = -6,
	/* Host or device memory ran out. The world is dead. */
	AVBD3D_ERR_OUT_OF_MEMORY = -7
} Avbd3dResult;

/* A world: bodies, shapes, joints and their device state. Opaque. */
typedef struct Avbd3dWorld Avbd3dWorld;

/* Opaque geometry owned by the caller (see avbd3d_geometry.h). */
typedef struct Avbd3dHullData Avbd3dHullData;
typedef struct Avbd3dMeshData Avbd3dMeshData;
typedef struct Avbd3dHeightFieldData Avbd3dHeightFieldData;
typedef struct Avbd3dCompoundData Avbd3dCompoundData;

/* The ABI version of the library you linked: (major << 16) | minor. Compare it with
   AVBD3D_ABI_VERSION_MAJOR / MINOR from the header you compiled against. */
AVBD3D_API uint32_t avbd3d_abi_version(void);

/* A static, human readable name for a result such as "AVBD3D_ERR_INVALID_ARG". Never NULL. */
AVBD3D_API const char* avbd3d_result_string(Avbd3dResult result);

/* An sRGB colour as 0xRRGGBB. */
typedef uint32_t Avbd3dHexColor;

/* Limits that appear in struct layouts. */
#define AVBD3D_MAX_MANIFOLD_POINTS 4
#define AVBD3D_MAX_SHAPE_CAST_POINTS 128
/* Most bodies one grab holds: the size of the arrays avbd3d_grab_begin fills. */
#define AVBD3D_MAX_GRAB_BODIES 256

/* Body id references a body instance. This should be treated as an opaque handle. */
typedef struct Avbd3dBodyId
{
	int32_t index1;
	uint16_t world0;
	uint16_t generation;
} Avbd3dBodyId;

/* Shape id references a shape instance. This should be treated as an opaque handle. */
typedef struct Avbd3dShapeId
{
	int32_t index1;
	uint16_t world0;
	uint16_t generation;
} Avbd3dShapeId;

/* Joint id references a joint instance. This should be treated as an opaque handle. */
typedef struct Avbd3dJointId
{
	int32_t index1;
	uint16_t world0;
	uint16_t generation;
} Avbd3dJointId;

/* Contact id references a contact instance. This should be treated as an opaque handle. */
typedef struct Avbd3dContactId
{
	int32_t index1;
	uint16_t world0;
	int16_t padding;
	uint32_t generation;
} Avbd3dContactId;

/* A 3D vector. */
typedef struct Avbd3dVec3
{
	float x;
	float y;
	float z;
} Avbd3dVec3;

/* A quaternion. */
typedef struct Avbd3dQuat
{
	Avbd3dVec3 v;
	float s;
} Avbd3dQuat;

/* A rigid transform. */
typedef struct Avbd3dTransform
{
	Avbd3dVec3 p;
	Avbd3dQuat q;
} Avbd3dTransform;

/* A 3x3 matrix. */
typedef struct Avbd3dMatrix3
{
	Avbd3dVec3 cx, cy, cz;
} Avbd3dMatrix3;

/* Axis aligned bounding box. */
typedef struct Avbd3dAABB
{
	Avbd3dVec3 lowerBound;
	Avbd3dVec3 upperBound;
} Avbd3dAABB;

/* A plane.
   separation = dot(normal, point) - offset */
typedef struct Avbd3dPlane
{
	Avbd3dVec3 normal;
	float offset;
} Avbd3dPlane;

/* Prototype callback for overlap queries.
   Called for each shape found in the query.
   @see avbd3d_world_overlap_aabb
   @return false to terminate the query. */
typedef int32_t Avbd3dOverlapResultFcn( Avbd3dShapeId shapeId, void* context );

/* Prototype callback for ray casts.
   Called for each shape found in the query. You control how the ray cast
   proceeds by returning a float:
   return -1: ignore this shape and continue
   return 0: terminate the ray cast
   return fraction: clip the ray to this point
   return 1: don't clip the ray and continue
   @param shapeId the shape hit by the ray
   @param point the point of initial intersection
   @param normal the normal vector at the point of intersection
   @param fraction the fraction along the ray at the point of intersection
   @param userMaterialId the shape or triangle surface type
   @param triangleIndex the triangle index for mesh or height field shapes or -1 for other shape types
   @param childIndex the child shape index for compound shapes
   @param context the user context
   @return -1 to filter, 0 to terminate, fraction to clip the ray for closest hit, 1 to continue
   @see avbd3d_world_cast_ray */
typedef float Avbd3dCastResultFcn( Avbd3dShapeId shapeId, Avbd3dVec3 point, Avbd3dVec3 normal, float fraction, uint64_t userMaterialId,
							   int32_t triangleIndex, int32_t childIndex, void* context );

/* World definition used to create a simulation world. Must be initialized using avbd3d_default_world_def. */
typedef struct Avbd3dWorldDef
{
	/* Gravity vector. There is no up vector: gravity defines down. */
	Avbd3dVec3 gravity;

	/* Restitution speed threshold, usually in m/s. Collisions above this
	   speed have restitution applied (will bounce). */
	float restitutionThreshold;

	/* Hit event speed threshold, usually in m/s. Collisions above this
	   speed can generate hit events if the shape also enables hit events. */
	float hitEventThreshold;

	/* Can bodies go to sleep to improve performance */
	int32_t enableSleep;

	/* Enable continuous collision */
	int32_t enableContinuous;
} Avbd3dWorldDef;

/* The body simulation type.
   Each body is one of these three types. The type determines how the body behaves in the simulation. */
typedef enum Avbd3dBodyType
{
	/* zero mass, zero velocity, may be manually moved */
	AVBD3D_STATIC_BODY = 0,
	/* zero mass, velocity set by user, moved by solver */
	AVBD3D_KINEMATIC_BODY = 1,
	/* positive mass, velocity determined by forces, moved by solver */
	AVBD3D_DYNAMIC_BODY = 2,
} Avbd3dBodyType;

/* Motion locks to restrict the body movement */
typedef struct Avbd3dMotionLocks
{
	/* Prevent translation along the x-axis */
	int32_t linearX;

	/* Prevent translation along the y-axis */
	int32_t linearY;

	/* Prevent translation along the z-axis */
	int32_t linearZ;

	/* Prevent rotation around the x-axis */
	int32_t angularX;

	/* Prevent rotation around the y-axis */
	int32_t angularY;

	/* Prevent rotation around the z-axis */
	int32_t angularZ;
} Avbd3dMotionLocks;

/* A body definition holds all the data needed to construct a rigid body.
   You can safely re-use body definitions. Shapes are added to a body after construction.
   Body definitions are temporary objects used to bundle creation parameters.
   Must be initialized using avbd3d_default_body_def(). */
typedef struct Avbd3dBodyDef
{
	/* The body type: static, kinematic, or dynamic. */
	int32_t /* Avbd3dBodyType */ type;

	/* The initial world position of the body. Bodies should be created with the desired position.
	   Note: Creating bodies at the origin and then moving them nearly doubles the cost of body creation, especially
	   if the body is moved after shapes have been added. */
	Avbd3dVec3 position;

	/* The initial world rotation of the body. */
	Avbd3dQuat rotation;

	/* The initial linear velocity of the body's origin. Usually in meters per second. */
	Avbd3dVec3 linearVelocity;

	/* The initial angular velocity of the body. Radians per second. */
	Avbd3dVec3 angularVelocity;

	/* Linear damping is used to reduce the linear velocity. The damping parameter
	   can be larger than 1 but the damping effect becomes sensitive to the
	   time step when the damping parameter is large.
	   Generally linear damping is undesirable because it makes objects move slowly
	   as if they are floating. */
	float linearDamping;

	/* Angular damping is used to reduce the angular velocity. The damping parameter
	   can be larger than 1.0f but the damping effect becomes sensitive to the
	   time step when the damping parameter is large.
	   Angular damping can be used to slow down rotating bodies. */
	float angularDamping;

	/* Scale the gravity applied to this body. Non-dimensional. */
	float gravityScale;

	/* Use this to store application specific body data. */
	void* userData;

	/* Motions locks to restrict linear and angular movement */
	Avbd3dMotionLocks motionLocks;

	/* Is this body initially awake or sleeping? */
	int32_t isAwake;

	/* Treat this body as a high speed object that performs continuous collision detection
	   against dynamic and kinematic bodies, but not other bullet bodies.
	   Warning: Bullets should be used sparingly. They are not a solution for general dynamic-versus-dynamic
	   continuous collision. They do not guarantee accurate collision if both bodies are fast moving because
	   the bullet does a continuous check after all non-bullet bodies have moved. You could get unlucky and have
	   the bullet body end a time step very close to a non-bullet body and the non-bullet body then moves over
	   the bullet body. In continuous collision, initial overlap is ignored to avoid freezing bodies in place.
	   I do not recommend using them for game projectiles if precise collision timing is needed. Instead consider
	   using a ray or shape cast. You can use a marching ray or shape cast for projectile that moves over time.
	   If you want a fast moving projectile to collide with a fast moving target, you need to consider the relative
	   movement in your ray or shape cast. This is out of scope.
	   So what are good use cases for bullets? Pinball games or games with dynamic containers that hold other objects.
	   It should be a use case where it doesn't break the game if there is a collision missed, but having them
	   captured improves the quality of the game. */
	int32_t isBullet;

	/* Used to disable a body. A disabled body does not move or collide. */
	int32_t isEnabled;
} Avbd3dBodyDef;

/* This is used to filter collision on shapes. It affects shape-vs-shape collision
   and shape-versus-query collision (such as avbd3d_world_cast_ray). */
typedef struct Avbd3dFilter
{
	/* The collision category bits. Normally you would just set one bit. The category bits should
	   represent your application object types. For example:
	   @code{.cpp}
	   enum MyCategories
	   {
	   Static  = 0x00000001,
	   Dynamic = 0x00000002,
	   Debris  = 0x00000004,
	   Player  = 0x00000008,
	   // etc
	   };
	   @endcode */
	uint64_t categoryBits;

	/* The collision mask bits. This states the categories that this
	   shape would accept for collision.
	   For example, you may want your player to only collide with static objects
	   and other players.
	   @code{.c}
	   maskBits = Static | Player;
	   @endcode */
	uint64_t maskBits;

	/* Collision groups allow a certain group of objects to never collide (negative)
	   or always collide (positive). A group index of zero has no effect. Non-zero group filtering
	   always wins against the mask bits.
	   For example, you may want ragdolls to collide with other ragdolls but you don't want
	   ragdoll self-collision. In this case you would give each ragdoll a unique negative group index
	   and apply that group index to all shapes on the ragdoll. */
	int32_t groupIndex;
} Avbd3dFilter;

/* Material properties supported per triangle on meshes and height fields */
typedef struct Avbd3dSurfaceMaterial
{
	/* The Coulomb (dry) friction coefficient, usually in the range [0,1]. */
	float friction;

	/* The coefficient of restitution (bounce) usually in the range [0,1].
	   https://en.wikipedia.org/wiki/Coefficient_of_restitution */
	float restitution;

	/* The rolling resistance usually in the range [0,1]. This is only used for spheres and capsules. */
	float rollingResistance;

	/* The tangent velocity for conveyor belts. This is local to the shape and will be projected
	   onto the contact surface. */
	Avbd3dVec3 tangentVelocity;

	/* User material identifier. This is passed with query results and to friction and restitution
	   combining functions. It is not used internally. */
	uint64_t userMaterialId;
} Avbd3dSurfaceMaterial;

/* Shape type */
typedef enum Avbd3dShapeType
{
	/* A capsule is an extruded sphere */
	AVBD3D_CAPSULE_SHAPE,
	/* A baked compound shape composed of spheres, capsules, hulls, and meshes */
	AVBD3D_COMPOUND_SHAPE,
	/* A height field useful for terrain */
	AVBD3D_HEIGHT_FIELD_SHAPE,
	/* A convex hull */
	AVBD3D_HULL_SHAPE,
	/* A triangle soup */
	AVBD3D_MESH_SHAPE,
	/* A sphere with an offset */
	AVBD3D_SPHERE_SHAPE,
	/* The number of shape types */
} Avbd3dShapeType;

/* Used to create a shape */
typedef struct Avbd3dShapeDef
{
	/* Use this to store application specific shape data. */
	void* userData;

	/* The base surface material. Ignored for compound shapes. */
	Avbd3dSurfaceMaterial baseMaterial;

	/* The density, usually in kg/m^3. */
	float density;

	/* Explosion scale for avbd3d_world_explode. non-dimensional */
	float explosionScale;

	/* Contact filtering data. */
	Avbd3dFilter filter;

	/* A sensor shape generates overlap events but never generates a collision response.
	   Sensors do not have continuous collision. Instead, use a ray or shape cast for those scenarios.
	   Sensors still contribute to the body mass if they have non-zero density.
	   Note: Sensor events are disabled by default.
	   @see enableSensorEvents */
	int32_t isSensor;

	/* Enable sensor events for this shape. This applies to sensors and non-sensors. False by default, even for sensors.
	   Only convex shapes may act as sensor visitors. */
	int32_t enableSensorEvents;

	/* Enable contact events for this shape. Only applies to kinematic and dynamic bodies. Ignored for sensors. False by default. */
	int32_t enableContactEvents;

	/* Enable hit events for this shape. Only applies to kinematic and dynamic bodies. Ignored for sensors. False by default. */
	int32_t enableHitEvents;

	/* Should the body update the mass properties when this shape is created. Default is true.
	   Warning: if this is false, you MUST call avbd3d_body_apply_mass_from_shapes or avbd3d_body_set_mass_data before simulating the world. */
	int32_t updateBodyMass;
} Avbd3dShapeDef;

/* Joint type enumeration. This is useful because all joint types use Avbd3dJointId and sometimes you
   want to get the type of a joint. */
typedef enum Avbd3dJointType
{
	AVBD3D_PARALLEL_JOINT,
	AVBD3D_DISTANCE_JOINT,
	AVBD3D_FILTER_JOINT,
	AVBD3D_MOTOR_JOINT,
	AVBD3D_PRISMATIC_JOINT,
	AVBD3D_REVOLUTE_JOINT,
	AVBD3D_SPHERICAL_JOINT,
	AVBD3D_WELD_JOINT,
	AVBD3D_WHEEL_JOINT,
} Avbd3dJointType;

/* Base joint definition used by all joint types. The local frames are measured from the
   body's origin rather than the center of mass because:
   1. You might not know where the center of mass will be.
   2. If you add/remove shapes from a body and recompute the mass, the joints will be broken. */
typedef struct Avbd3dJointDef
{
	/* User data pointer */
	void* userData;

	/* The first attached body */
	Avbd3dBodyId bodyIdA;

	/* The second attached body */
	Avbd3dBodyId bodyIdB;

	/* The first local joint frame */
	Avbd3dTransform localFrameA;

	/* The second local joint frame */
	Avbd3dTransform localFrameB;

	/* Force threshold for joint events */
	float forceThreshold;

	/* Torque threshold for joint events */
	float torqueThreshold;

	/* Set this flag to true if the attached bodies should collide */
	int32_t collideConnected;
} Avbd3dJointDef;

/* Distance joint definition.
   Connects a point on body A with a point on body B by a segment.
   Useful for ropes and springs. */
typedef struct Avbd3dDistanceJointDef
{
	/* Base joint definition */
	Avbd3dJointDef base;

	/* The rest length of this joint. Clamped to a stable minimum value. */
	float length;

	/* Enable the distance constraint to behave like a spring. If false
	   then the distance joint will be rigid, overriding the limit and motor. */
	int32_t enableSpring;

	/* The lower spring force controls how much tension it can sustain */
	float lowerSpringForce;

	/* The upper spring force controls how much compression it can sustain */
	float upperSpringForce;

	/* The spring linear stiffness Hertz, cycles per second */
	float hertz;

	/* The spring linear damping ratio, non-dimensional */
	float dampingRatio;

	/* Enable/disable the joint limit */
	int32_t enableLimit;

	/* Minimum length. Clamped to a stable minimum value. */
	float minLength;

	/* Maximum length. Must be greater than or equal to the minimum length. */
	float maxLength;

	/* Enable/disable the joint motor */
	int32_t enableMotor;

	/* The maximum motor force, usually in newtons */
	float maxMotorForce;

	/* The desired motor speed, usually in meters per second */
	float motorSpeed;
} Avbd3dDistanceJointDef;

/* A motor joint is used to control the relative position and velocity between two bodies. */
typedef struct Avbd3dMotorJointDef
{
	/* Base joint definition */
	Avbd3dJointDef base;

	/* The desired linear velocity */
	Avbd3dVec3 linearVelocity;

	/* The maximum motor force in newtons */
	float maxVelocityForce;

	/* The desired angular velocity */
	Avbd3dVec3 angularVelocity;

	/* The maximum motor torque in newton-meters */
	float maxVelocityTorque;

	/* Linear spring hertz for position control */
	float linearHertz;

	/* Linear spring damping ratio */
	float linearDampingRatio;

	/* Maximum spring force in newtons */
	float maxSpringForce;

	/* Angular spring hertz for position control */
	float angularHertz;

	/* Angular spring damping ratio */
	float angularDampingRatio;

	/* Maximum spring torque in newton-meters */
	float maxSpringTorque;
} Avbd3dMotorJointDef;

/* A filter joint is used to disable collision between two specific bodies. */
typedef struct Avbd3dFilterJointDef
{
	/* Base joint definition */
	Avbd3dJointDef base;
} Avbd3dFilterJointDef;

/* Parallel joint definition. Constrains the angle between axis z in body A and axis z in body B
   using a spring. Useful to keep a body upright. */
typedef struct Avbd3dParallelJointDef
{
	/* Base joint definition */
	Avbd3dJointDef base;

	/* The spring stiffness Hertz, cycles per second */
	float hertz;

	/* The spring damping ratio, non-dimensional */
	float dampingRatio;

	/* The maximum spring torque, typically in newton-meters. */
	float maxTorque;
} Avbd3dParallelJointDef;

/* Prismatic joint definition. Body B may slide along the x-axis in local frame A.
   Body B cannot rotate relative to body A. The joint translation is zero when the
   local frame origins coincide in world space. */
typedef struct Avbd3dPrismaticJointDef
{
	/* Base joint definition */
	Avbd3dJointDef base;

	/* Enable a linear spring along the prismatic joint axis */
	int32_t enableSpring;

	/* The spring stiffness Hertz, cycles per second */
	float hertz;

	/* The spring damping ratio, non-dimensional */
	float dampingRatio;

	/* The target translation for the joint in meters. The spring-damper will drive
	   to this translation. */
	float targetTranslation;

	/* Enable/disable the joint limit */
	int32_t enableLimit;

	/* The lower translation limit */
	float lowerTranslation;

	/* The upper translation limit */
	float upperTranslation;

	/* Enable/disable the joint motor */
	int32_t enableMotor;

	/* The maximum motor force, typically in newtons */
	float maxMotorForce;

	/* The desired motor speed, typically in meters per second */
	float motorSpeed;
} Avbd3dPrismaticJointDef;

/* Revolute joint definition. A point on body B is fixed to a point on body A.
   Allows relative rotation about the z-axis. */
typedef struct Avbd3dRevoluteJointDef
{
	/* Base joint definition. */
	Avbd3dJointDef base;

	/* The bodyB angle minus bodyA angle in the reference state (radians).
	   This defines the zero angle for the joint limit. */
	float targetAngle;

	/* Enable a rotational spring on the revolute hinge axis. */
	int32_t enableSpring;

	/* The spring stiffness Hertz, cycles per second. */
	float hertz;

	/* The spring damping ratio, non-dimensional. */
	float dampingRatio;

	/* A flag to enable joint limits. */
	int32_t enableLimit;

	/* The lower angle for the joint limit in radians. Minimum of -0.99*pi radians. */
	float lowerAngle;

	/* The upper angle for the joint limit in radians. Maximum of 0.99*pi radians. */
	float upperAngle;

	/* A flag to enable the joint motor. */
	int32_t enableMotor;

	/* The maximum motor torque, typically in newton-meters. */
	float maxMotorTorque;

	/* The desired motor speed in radians per second. */
	float motorSpeed;
} Avbd3dRevoluteJointDef;

/* Spherical joint definition. A point on body B is fixed to a point on body A.
   Allows rotation about the shared point. */
typedef struct Avbd3dSphericalJointDef
{
	/* Base joint definition */
	Avbd3dJointDef base;

	/* Enable a rotational spring that attempts to align the two joint frames. */
	int32_t enableSpring;

	/* The spring stiffness Hertz, cycles per second. This may be clamped internally
	   according to the time step to maintain stability. Non-negative number. */
	float hertz;

	/* The spring damping ratio, non-dimensional. Non-negative number. */
	float dampingRatio;

	/* Target spring rotation, joint frame B relative to joint frame A. */
	Avbd3dQuat targetRotation;

	/* A flag to enable the cone limit. The cone is centered on the frameA z-axis. */
	int32_t enableConeLimit;

	/* The angle for the cone limit in radians. Valid range is [0, pi] */
	float coneAngle;

	/* A flag to enable the twist limit. The twist is centered on the frameB z-axis. */
	int32_t enableTwistLimit;

	/* The angle for the lower twist limit in radians. Minimum of -0.99*pi radians. */
	float lowerTwistAngle;

	/* The angle for the upper twist limit in radians. Maximum of 0.99*pi radians. */
	float upperTwistAngle;

	/* A flag to enable the joint motor */
	int32_t enableMotor;

	/* The maximum motor torque, typically in newton-meters. Non-negative number. */
	float maxMotorTorque;

	/* The desired motor angular velocity in radians per second. This is the relative angular
	   velocity between the two bodies in world space.
	   motorVelocity = angularVelocityB - angularVelocityA */
	Avbd3dVec3 motorVelocity;
} Avbd3dSphericalJointDef;

/* Weld joint definition
   Connects two bodies together rigidly. This constraint provides springs to mimic
   soft-body simulation.
   Note: the approximate solver cannot hold many bodies together rigidly. */
typedef struct Avbd3dWeldJointDef
{
	/* Base joint definition */
	Avbd3dJointDef base;

	/* Linear stiffness expressed as Hertz (cycles per second). Use zero for maximum stiffness. */
	float linearHertz;

	/* Angular stiffness as Hertz (cycles per second). Use zero for maximum stiffness. */
	float angularHertz;

	/* Linear damping ratio, non-dimensional. Use 1 for critical damping. */
	float linearDampingRatio;

	/* Linear damping ratio, non-dimensional. Use 1 for critical damping. */
	float angularDampingRatio;
} Avbd3dWeldJointDef;

/* Wheel joint definition
   Body A is the chassis and body B is the wheel.
   The wheel rotates around the local z-axis in frame B.
   The wheel translates along the local x-axis in frame A.
   The wheel can optionally steer along the x-axis in frame A. */
typedef struct Avbd3dWheelJointDef
{
	/* Base joint definition */
	Avbd3dJointDef base;

	/* Enable a linear spring along the local axis */
	int32_t enableSuspensionSpring;

	/* Spring stiffness in Hertz */
	float suspensionHertz;

	/* Spring damping ratio, non-dimensional */
	float suspensionDampingRatio;

	/* Enable/disable the joint linear limit */
	int32_t enableSuspensionLimit;

	/* The lower suspension translation limit */
	float lowerSuspensionLimit;

	/* The upper translation limit */
	float upperSuspensionLimit;

	/* Enable/disable the joint rotational motor */
	int32_t enableSpinMotor;

	/* The maximum motor torque, typically in newton-meters */
	float maxSpinTorque;

	/* The desired motor speed in radians per second */
	float spinSpeed;

	/* Enable steering, otherwise the steering is fixed forward */
	int32_t enableSteering;

	/* Steering stiffness in Hertz */
	float steeringHertz;

	/* Spring damping ratio, non-dimensional */
	float steeringDampingRatio;

	/* The target steering angle in radians */
	float targetSteeringAngle;

	/* The maximum steering torque in N*m */
	float maxSteeringTorque;

	/* Enable/disable the steering angular limit */
	int32_t enableSteeringLimit;

	/* The lower steering angle in radians */
	float lowerSteeringLimit;

	/* The upper steering angle in radians */
	float upperSteeringLimit;
} Avbd3dWheelJointDef;

/* The explosion definition is used to configure options for explosions. Explosions
   consider shape geometry when computing the impulse. */
typedef struct Avbd3dExplosionDef
{
	/* Mask bits to filter shapes */
	uint64_t maskBits;

	/* The center of the explosion in world space */
	Avbd3dVec3 position;

	/* The radius of the explosion */
	float radius;

	/* The falloff distance beyond the radius. Impulse is reduced to zero at this distance. */
	float falloff;

	/* Impulse per unit area. This applies an impulse according to the shape area that
	   is facing the explosion. Explosions only apply to spheres, capsules, and hulls. This
	   may be negative for implosions. */
	float impulsePerArea;
} Avbd3dExplosionDef;

/* A begin-touch event is generated when a shape starts to overlap a sensor shape. */
typedef struct Avbd3dSensorBeginTouchEvent
{
	/* The id of the sensor shape */
	Avbd3dShapeId sensorShapeId;

	/* The id of the shape that began touching the sensor shape */
	Avbd3dShapeId visitorShapeId;
} Avbd3dSensorBeginTouchEvent;

/* An end touch event is generated when a shape stops overlapping a sensor shape.
   These include things like setting the transform, destroying a body or shape, or changing
   a filter. You will also get an end event if the sensor or visitor are destroyed.
   Therefore you should always confirm the shape id is valid using avbd3d_shape_is_valid. */
typedef struct Avbd3dSensorEndTouchEvent
{
	/* The id of the sensor shape
	   Warning: this shape may have been destroyed
	   @see avbd3d_shape_is_valid */
	Avbd3dShapeId sensorShapeId;

	/* The id of the shape that stopped touching the sensor shape
	   Warning: this shape may have been destroyed
	   @see avbd3d_shape_is_valid */
	Avbd3dShapeId visitorShapeId;
} Avbd3dSensorEndTouchEvent;

/* Sensor events are buffered in the world and are available
   as begin/end overlap event arrays after the time step is complete.
   Note: these may become invalid if bodies and/or shapes are destroyed */
typedef struct Avbd3dSensorEvents
{
	/* Array of sensor begin touch events */
	Avbd3dSensorBeginTouchEvent* beginEvents;

	/* Array of sensor end touch events */
	Avbd3dSensorEndTouchEvent* endEvents;

	/* The number of begin touch events */
	int32_t beginCount;

	/* The number of end touch events */
	int32_t endCount;
} Avbd3dSensorEvents;

/* A begin-touch event is generated when two shapes begin touching. */
typedef struct Avbd3dContactBeginTouchEvent
{
	/* Id of the first shape */
	Avbd3dShapeId shapeIdA;

	/* Id of the second shape */
	Avbd3dShapeId shapeIdB;

	/* The transient contact id. This contact may be destroyed automatically when the world is modified or simulated.
	   Use avbd3d_contact_is_valid before using this id. */
	Avbd3dContactId contactId;
} Avbd3dContactBeginTouchEvent;

/* An end touch event is generated when two shapes stop touching.
   You will get an end event if you do anything that destroys contacts previous to the last
   world step. These include things like setting the transform, destroying a body
   or shape, or changing a filter or body type. */
typedef struct Avbd3dContactEndTouchEvent
{
	/* Id of the first shape
	   Warning: this shape may have been destroyed
	   @see avbd3d_shape_is_valid */
	Avbd3dShapeId shapeIdA;

	/* Id of the first shape
	   Warning: this shape may have been destroyed
	   @see avbd3d_shape_is_valid */
	Avbd3dShapeId shapeIdB;

	/* Id of the contact.
	   Warning: this contact may have been destroyed
	   @see avbd3d_contact_is_valid */
	Avbd3dContactId contactId;
} Avbd3dContactEndTouchEvent;

/* A hit touch event is generated when two shapes collide with a speed faster than the hit speed threshold.
   This may be reported for speculative contacts that have a confirmed impulse. */
typedef struct Avbd3dContactHitEvent
{
	/* Id of the first shape */
	Avbd3dShapeId shapeIdA;

	/* Id of the second shape */
	Avbd3dShapeId shapeIdB;

	/* Id of the contact.
	   Warning: this contact may have been destroyed
	   @see avbd3d_contact_is_valid */
	Avbd3dContactId contactId;

	/* Point where the shapes hit at the beginning of the time step.
	   This is a mid-point between the two surfaces. It could be at speculative
	   point where the two shapes were not touching at the beginning of the time step. */
	Avbd3dVec3 point;

	/* Normal vector pointing from shape A to shape B */
	Avbd3dVec3 normal;

	/* The speed the shapes are approaching. Always positive. Typically in meters per second. */
	float approachSpeed;

	/* User material on shape A */
	uint64_t userMaterialIdA;

	/* User material on shape B */
	uint64_t userMaterialIdB;
} Avbd3dContactHitEvent;

/* Contact events are buffered in the world and are available
   as event arrays after the time step is complete.
   Note: these may become invalid if bodies and/or shapes are destroyed */
typedef struct Avbd3dContactEvents
{
	/* Array of begin touch events */
	Avbd3dContactBeginTouchEvent* beginEvents;

	/* Array of end touch events */
	Avbd3dContactEndTouchEvent* endEvents;

	/* Array of hit events */
	Avbd3dContactHitEvent* hitEvents;

	/* Number of begin touch events */
	int32_t beginCount;

	/* Number of end touch events */
	int32_t endCount;

	/* Number of hit events */
	int32_t hitCount;
} Avbd3dContactEvents;

/* Body move events triggered when a body moves.
   Triggered when a body moves due to simulation. Not reported for bodies moved by the user.
   This also has a flag to indicate that the body went to sleep so the application can also
   sleep that actor/entity/object associated with the body.
   On the other hand if the flag does not indicate the body went to sleep then the application
   can treat the actor/entity/object associated with the body as awake.
   This is an efficient way for an application to update game object transforms rather than
   calling functions such as avbd3d_body_get_transform() because this data is delivered as a contiguous array
   and it is only populated with bodies that have moved.
   Note: If sleeping is disabled all dynamic and kinematic bodies will trigger move events. */
typedef struct Avbd3dBodyMoveEvent
{
	/* The body user data. */
	void* userData;

	/* The body transform. */
	Avbd3dTransform transform;

	/* The body id. */
	Avbd3dBodyId bodyId;

	/* Did the body fall asleep this time step? */
	int32_t fellAsleep;
} Avbd3dBodyMoveEvent;

/* Body events are buffered in the world and are available
   as event arrays after the time step is complete.
   Note: this data becomes invalid if bodies are destroyed */
typedef struct Avbd3dBodyEvents
{
	/* Array of move events */
	Avbd3dBodyMoveEvent* moveEvents;

	/* Number of move events */
	int32_t moveCount;
} Avbd3dBodyEvents;

/* Joint events report joints that are awake and have a force and/or torque exceeding the threshold
   The observed forces and torques are not returned for efficiency reasons. */
typedef struct Avbd3dJointEvent
{
	/* The joint id */
	Avbd3dJointId jointId;

	/* The user data from the joint for convenience */
	void* userData;
} Avbd3dJointEvent;

/* Joint events are buffered in the world and are available
   as event arrays after the time step is complete.
   Note: this data becomes invalid if joints are destroyed */
typedef struct Avbd3dJointEvents
{
	/* Array of events */
	Avbd3dJointEvent* jointEvents;

	/* Number of events */
	int32_t count;
} Avbd3dJointEvents;

/* A manifold point is a contact point belonging to a contact manifold.
   It holds details related to the geometry and dynamics of the contact points.
   Contacts are speculative, so some contact points may be separated.
   You may use the maxNormalImpulse to determine if there was an interaction during
   the time step. */
typedef struct Avbd3dManifoldPoint
{
	/* Location of the contact point relative to the bodyA center of mass in world space. */
	Avbd3dVec3 anchorA;

	/* Location of the contact point relative to the bodyB center of mass in world space. */
	Avbd3dVec3 anchorB;

	/* The separation of the contact point, negative if penetrating */
	float separation;

	/* Cached separation used for contact recycling */
	float baseSeparation;

	/* The impulse along the manifold normal vector. The solver iterates within one step, so this is
	   the result of the last iteration. */
	float normalImpulse;

	/* The total normal impulse applied during the step. This is important
	   to identify speculative contact points that had an interaction in the time step. */
	float totalNormalImpulse;

	/* Relative normal velocity pre-solve. Used for hit events. If the normal impulse is
	   zero then there was no hit. Negative means shapes are approaching. */
	float normalVelocity;

	/* Local point for matching
	   Uniquely identifies a contact point between two shapes */
	uint32_t featureId;

	/* Triangle index if one of the shapes is a mesh or height field */
	int32_t triangleIndex;

	/* Did this contact point exist in the previous step? */
	int32_t persisted;
} Avbd3dManifoldPoint;

/* A contact manifold describes the contact points between colliding shapes.
   Note: Contacts are speculative, so some contact points may be separated. */
typedef struct Avbd3dManifold
{
	/* The manifold points. There may be 1 to 4 valid points. */
	Avbd3dManifoldPoint points[AVBD3D_MAX_MANIFOLD_POINTS];

	/* The unit normal vector in world space, points from shape A to shape B */
	Avbd3dVec3 normal;

	/* Central friction angular impulse (applied about the normal) */
	float twistImpulse;

	/* Central friction linear impulse */
	Avbd3dVec3 frictionImpulse;

	/* Rolling resistance angular impulse */
	Avbd3dVec3 rollingImpulse;

	/* The number of contact points, will be 0 to 4 */
	int32_t pointCount;
} Avbd3dManifold;

/* The contact data for two shapes. By convention the manifold normal points
   from shape A to shape B.
   @see avbd3d_shape_get_contact_data() and avbd3d_body_get_contact_data() */
typedef struct Avbd3dContactData
{
	/* The contact id. You may hold onto this to track a contact across time steps.
	   This id may become orphaned. Use avbd3d_contact_is_valid before using it for other functions. */
	Avbd3dContactId contactId;

	/* The first shape id. */
	Avbd3dShapeId shapeIdA;

	/* The second shape id. */
	Avbd3dShapeId shapeIdB;

	/* The contact manifold. This points to internal data and may become invalid. Do not store
	   this pointer. */
	const struct Avbd3dManifold* manifolds;

	/* The number of contact manifolds. For mesh and height-field collision there can be multiple manifolds. */
	int32_t manifoldCount;
} Avbd3dContactData;

/* The query filter is used to filter collisions between queries and shapes. For example,
   you may want a ray-cast representing a projectile to hit players and the static environment
   but not debris. */
typedef struct Avbd3dQueryFilter
{
	/* The collision category bits of this query. Normally you would just set one bit. */
	uint64_t categoryBits;

	/* The collision mask bits. This states the shape categories that this
	   query would accept for collision. */
	uint64_t maskBits;
} Avbd3dQueryFilter;

/* Result from avbd3d_world_cast_ray_closest. */
typedef struct Avbd3dRayResult
{
	/* The shape hit. */
	Avbd3dShapeId shapeId;

	/* The world point of the hit. */
	Avbd3dVec3 point;

	/* The world normal of the shape surface at the hit point. */
	Avbd3dVec3 normal;

	/* The user material id at the hit point. This can be per triangle
	   if the shape is a mesh, height-field, or compound with child mesh. */
	uint64_t userMaterialId;

	/* The fraction of the input ray. */
	float fraction;

	/* The triangle index if the shape is a mesh, height-field, or compound with
	   child mesh. */
	int32_t triangleIndex;

	/* The child index if the shape is a compound. */
	int32_t childIndex;

	/* Did the ray hit? If false, all other data is invalid. */
	int32_t hit;
} Avbd3dRayResult;

/* A shape proxy is used by the GJK algorithm. It can represent a convex shape. */
typedef struct Avbd3dShapeProxy
{
	/* The point cloud. */
	const Avbd3dVec3* points;

	/* The number of points. Do not exceed AVBD3D_MAX_SHAPE_CAST_POINTS. */
	int32_t count;

	/* The external radius of the point cloud. */
	float radius;
} Avbd3dShapeProxy;

/* Low level ray cast or shape-cast output data. */
typedef struct Avbd3dCastOutput
{
	/* The surface normal at the hit point. */
	Avbd3dVec3 normal;

	/* The surface hit point. */
	Avbd3dVec3 point;

	/* The fraction of the input translation at collision. */
	float fraction;

	/* The number of iterations used. */
	int32_t iterations;

	/* The index of the mesh or height field triangle hit. */
	int32_t triangleIndex;

	/* The index of the compound child shape. */
	int32_t childIndex;

	/* The material index. May be -1 for null. */
	int32_t materialIndex;

	/* Did the cast hit? */
	int32_t hit;
} Avbd3dCastOutput;

/* Body cast result for ray and shape casts. */
typedef struct Avbd3dBodyCastResult
{
	/* The shape hit. */
	Avbd3dShapeId shapeId;

	/* The world point on the shape surface. */
	Avbd3dVec3 point;

	/* The world normal vector on the shape surface. */
	Avbd3dVec3 normal;

	/* The fraction along the ray hit.
	   hit point = origin + fraction * translation */
	float fraction;

	/* The triangle index if the shape is a mesh or height-field. */
	int32_t triangleIndex;

	/* The user material id at the hit point. This can be per triangle
	   if the shape is a mesh, height-field, or compound with child mesh. */
	uint64_t userMaterialId;

	/* The number of iterations used. Diagnostic. */
	int32_t iterations;

	/* Did the cast hit? If false, all other fields are invalid. */
	int32_t hit;
} Avbd3dBodyCastResult;

/* The plane between a character mover and a shape */
typedef struct Avbd3dPlaneResult
{
	/* Outward pointing plane. */
	Avbd3dPlane plane;

	/* Closest point on the shape. May not be unique. */
	Avbd3dVec3 point;

	/* The index of the mesh or height field triangle hit. */
	int32_t triangleIndex;

	/* The index of the compound child shape. */
	int32_t childIndex;

	/* The material index. */
	int32_t materialIndex;
} Avbd3dPlaneResult;

/* Body plane result for movers. */
typedef struct Avbd3dBodyPlaneResult
{
	/* The shape id on the body. */
	Avbd3dShapeId shapeId;

	/* The plane result. */
	Avbd3dPlaneResult result;
} Avbd3dBodyPlaneResult;

/* Used to collect collision planes for character movers.
   Return true to continue gathering planes. */
typedef int32_t Avbd3dPlaneResultFcn( Avbd3dShapeId shapeId, const Avbd3dPlaneResult* plane, int32_t planeCount, void* context );

/* Used to filter shapes for shape casting character movers.
   Return true to accept the collision */
typedef int32_t Avbd3dMoverFilterFcn( Avbd3dShapeId shapeId, void* context );

/* This holds the mass data computed for a shape. */
typedef struct Avbd3dMassData
{
	/* The shape mass */
	float mass;

	/* The local center of mass position. */
	Avbd3dVec3 center;

	/* The inertia tensor about the shape center of mass. */
	Avbd3dMatrix3 inertia;
} Avbd3dMassData;

/* A solid sphere */
typedef struct Avbd3dSphere
{
	/* The local center */
	Avbd3dVec3 center;

	/* The radius */
	float radius;
} Avbd3dSphere;

/* A solid capsule can be viewed as two hemispheres connected
   by a rectangle. */
typedef struct Avbd3dCapsule
{
	/* Local center of the first hemisphere */
	Avbd3dVec3 center1;

	/* Local center of the second hemisphere */
	Avbd3dVec3 center2;

	/* The radius of the hemispheres */
	float radius;
} Avbd3dCapsule;

/* Solver parameters, set with avbd3d_world_set_solver_def. The time
   step is an argument of the step call and gravity is part of Avbd3dWorldDef, so neither is
   repeated here. Set between steps. */
typedef struct Avbd3dSolverDef
{
	/* Primal/dual iterations per step. Default 12. */
	int32_t iterations;

	/* Cooperating GPU lanes per body in the primal sweep. 1 = one thread per body. Any value
	   above 1 changes accumulation order. Default 4. */
	int32_t primalLanes;

	/* Constraint error regularisation in [0, 1] (paper Eq. 18). Default 0.97. */
	float alpha;

	/* Penalty ramp rate for linear (contact) constraints. Default 10000. */
	float betaLin;

	/* Penalty ramp rate for angular constraints. Default 100. */
	float betaAng;

	/* Dual and penalty warm start decay in [0, 1) (paper Eq. 19). Default 0.99. */
	float gamma;

	/* Initialise a new contact's penalty from effective mass instead of a constant floor. */
	int32_t autoKStart;

	/* Damps the dual warm start of highly ramped quasi-static contacts, in [0, 1]. Default 0.1. */
	float dualDamping;

	/* Relative contact speed (m/s) at which dual damping fades to zero. Default 0.1. */
	float dualDampVel;

	/* Quiet steps before a body may sleep. Used when sleeping is enabled in the world def.
	   Default 64. */
	int32_t sleepFrames;

	/* Fast-reject speeds for the rest test; also set the wake hysteresis. Default 0.4 each. */
	float sleepLinearVelocity;
	float sleepAngularVelocity;

	/* Net displacement (m) allowed over the rest window. Default 0.005. */
	float sleepDisplacement;

	/* Joint load fraction of the fracture force that keeps its endpoints awake. Default 0.8. */
	float sleepFracture;

	/* Leaky rest streak. 0 = off. */
	float sleepLeak;

	/* Force-sleep bodies that fall below `killHeight` (height along the up axis, -normalize(gravity)). Default on, -50. */
	int32_t enableKillFallen;
	float killHeight;

	/* Multiplier on the mouse-grab spring stiffness. Default 1. */
	float grabStrength;

	/* Fidelity trade: maximum points per manifold, 0 = off. 1-3 run as 4. */
	int32_t contactCap;

	/* Two-level AVBD coarse correction: iterations that get it (0 = off), aggregate edge in
	   metres (0 = automatic) and step damping. Defaults 0, 0, 0.7. */
	int32_t coarseIterations;
	float coarseCell;
	float coarseDamping;

	/* World-frame angular inertia: R diag( I ) R^T in the primal plus the gyroscopic torque, so a
	   torque-free spinning body conserves angular momentum. Default on. */
	int32_t worldFrameInertia;
} Avbd3dSolverDef;

/* How the two sides of a contact combine a material coefficient. Selects the combination by data, not by callback, so it runs on the device. */
typedef enum Avbd3dMixingRule
{
	/* sqrt( a * b ): the friction default. */
	AVBD3D_MIX_GEOMETRIC = 0,
	/* ( a + b ) / 2 */
	AVBD3D_MIX_AVERAGE,
	/* min( a, b ) */
	AVBD3D_MIX_MIN,
	/* max( a, b ): the restitution default. */
	AVBD3D_MIX_MAX,
	/* a * b */
	AVBD3D_MIX_MULTIPLY,
} Avbd3dMixingRule;

/* Which property a batched call moves. `data` is an array of `count` elements of the type
   in the comment, tightly packed, in the order of the ids. */
typedef enum Avbd3dBodyDataType
{
	/* Avbd3dVec3, the body origin in world space. */
	AVBD3D_BODY_DATA_POSITION,
	/* Avbd3dQuat, the body orientation. */
	AVBD3D_BODY_DATA_ROTATION,
	/* Avbd3dTransform (origin + rotation) of the body. */
	AVBD3D_BODY_DATA_TRANSFORM,
	/* Avbd3dVec3, linear velocity of the center of mass. */
	AVBD3D_BODY_DATA_LINEAR_VELOCITY,
	/* Avbd3dVec3, angular velocity. */
	AVBD3D_BODY_DATA_ANGULAR_VELOCITY,
	/* uint8_t, 1 if the body is awake. Read only. */
	AVBD3D_BODY_DATA_AWAKE,
} Avbd3dBodyDataType;

/* How an AVBD joint's rows act. */
typedef enum Avbd3dGenericJointKind
{
	/* The anchors coincide under linearStiffness, and the relative rotation is held at its */
	/* creation value under angularStiffness (0: free, a ball joint). The axes are unused. */
	AVBD3D_GENERIC_POINT = 0,
	/* Six axes in the local frames (linear x, y, z, then angular x, y, z), angles as a rotation */
	/* vector. linearStiffness and angularStiffness bound a springForce axis. */
	AVBD3D_GENERIC_SIX_DOF = 1,
	/* The three linear rows act along the anchor-to-anchor direction; axis 0 sets the length. */
	/* The force is bounded to [-angularStiffness, +linearStiffness]. */
	AVBD3D_GENERIC_DISTANCE = 2,
	/* Six axes as for AVBD3D_GENERIC_SIX_DOF, angles as swing-twist (a cone on angular axis 3). */
	AVBD3D_GENERIC_SWING_TWIST = 3,
} Avbd3dGenericJointKind;

/* What one axis of a six-axis AVBD joint does. */
typedef enum Avbd3dGenericAxisMode
{
	AVBD3D_AXIS_FREE = 0,
	AVBD3D_AXIS_LOCKED,
	/* Held inside [lo, hi]. */
	AVBD3D_AXIS_LIMITED,
	/* Driven to position lo with at most force hi. */
	AVBD3D_AXIS_MOTOR,
	/* Pulled to lo with stiffness hi. */
	AVBD3D_AXIS_SPRING,
	/* Pulled to lo with stiffness hi, the force bounded by the joint's stiffness fields. */
	AVBD3D_AXIS_SPRING_FORCE,
	/* Driven at speed lo with at most force hi. */
	AVBD3D_AXIS_VELOCITY_MOTOR,
	/* Angular axis 3 of a swing-twist joint: swing held inside half angle hi. */
	AVBD3D_AXIS_CONE,
} Avbd3dGenericAxisMode;

typedef struct Avbd3dGenericAxis
{
	int32_t /* Avbd3dGenericAxisMode */ mode;
	float lo;
	float hi;
} Avbd3dGenericAxis;

/* Use avbd3d_default_generic_joint_def() to initialize this structure. The base's local frames give
   the anchors (p) and the axis frames (q, six-axis kinds only); its thresholds, userData and
   collideConnected apply as for every joint. */
typedef struct Avbd3dGenericJointDef
{
	Avbd3dJointDef base;
	int32_t /* Avbd3dGenericJointKind */ kind;
	float linearStiffness;
	float angularStiffness;
	Avbd3dGenericAxis axes[6];
	/* Constraint force at which the joint breaks, N. INFINITY: unbreakable. */
	float fractureForce;
	/* Bending capacity, N*m, when fractureForce is finite. < 0: the force test covers bending. */
	float fractureTorque;
} Avbd3dGenericJointDef;

/* Use avbd3d_default_spring_def() to initialize this structure. The base's local frame
   positions are the anchors. */
typedef struct Avbd3dSpringDef
{
	Avbd3dJointDef base;
	/* N/m. */
	float stiffness;
	/* < 0: the anchors' distance when the spring is created. */
	float restLength;
} Avbd3dSpringDef;

/* One handle: pull `body` so that its `localAnchor` reaches `target`. */
typedef struct Avbd3dGrabHandle
{
	Avbd3dBodyId body;

	/* Anchor in body space, from the pick hit. */
	Avbd3dVec3 localAnchor;

	/* Where the anchor should be, world space. */
	Avbd3dVec3 target;

	/* Stiffness scale in [0, 1]: 1 full strength, 0 no hold. Shapes the falloff away from the
	   clicked body. */
	float weight;
} Avbd3dGrabHandle;

/* What the device and the build offer. Set struct_size to sizeof(Avbd3dCapabilities) before
   the call. */
typedef struct Avbd3dCapabilities
{
	uint32_t struct_size;
	/* Vulkan physical device name, NUL terminated. */
	char device_name[256];
	/* PCI vendor id of the device. */
	uint32_t vendor_id;
	/* Device subgroup size. The cooperative primal kernels assume 32. */
	uint32_t subgroup_size;
	/* Device-local memory in bytes. */
	uint64_t device_memory_bytes;
	/* Limits of the shape model. */
	int32_t max_hull_vertices;
	int32_t max_hull_faces;
	int32_t max_hull_edges;
	int32_t max_manifold_points;
	int32_t max_child_shapes;
	/* Hard limit on constraint-graph colours. A scene that needs more reports a failed step. */
	int32_t max_colors;
} Avbd3dCapabilities;

/* Counts, capacities, the result of the last step and its timings. Timings are in
   milliseconds; the GPU ones come from timestamps and are 0 if the device has none. */
typedef struct Avbd3dStats
{
	/* Bodies, joints and contacts the last step saw. */
	int32_t bodies;
	int32_t awake_bodies;
	int32_t joint_rows;
	int32_t springs;
	int32_t pairs;
	int32_t manifolds;
	int32_t contacts;
	/* Constraint graph colours and the Jones-Plassmann rounds that found them. */
	int32_t colors;
	int32_t coloring_rounds;
	/* Current capacities of the growing pools. */
	int32_t pair_capacity;
	int32_t contact_capacity;
	int32_t manifold_capacity;
	/* Broadphase cell size in metres. */
	float cell_size;
	/* The result of the last step. ok is 0 only when a hard capacity ceiling was reached and
	   the step committed clamped results; truncated is 1 when results were clamped. */
	int32_t ok;
	int32_t truncated;
	/* Command buffer submissions of the last step: 1, or more when a pass was rerun. */
	int32_t submits;
	/* Reruns by cause. */
	int32_t capacity_reruns;
	int32_t flood_reruns;
	int32_t island_reruns;
	int32_t jp_reruns;
	int32_t color_reruns;
	/* 1 if the contact event list was cut short by its capacity. */
	int32_t contact_events_truncated;
	/* Device memory the world holds, in bytes. */
	uint64_t device_bytes;
	/* Timings of the last step, in milliseconds. */
	float step_ms;
	float gpu_ms;
	float front_ms;
	float broadphase_ms;
	float narrowphase_ms;
	float build_ms;
	float gpu_build_ms;
	float solve_ms;
	float iterations_ms;
	float primal_ms;
	float dual_ms;
	float tail_ms;
	float host_pre_ms;
	float host_record_ms;
	float host_submit_ms;
	float host_wait_ms;
} Avbd3dStats;

/* Debug draw callbacks for avbd3d_world_draw. A NULL callback is skipped. */
typedef struct Avbd3dDebugDraw
{
	/* Draws a shape. userShape is the userData of the shape's def. */
	void ( *DrawShapeFcn )( void* userShape, Avbd3dTransform transform, Avbd3dHexColor color, void* context );
	/* Draws a bounding box. */
	void ( *DrawBoundsFcn )( Avbd3dAABB aabb, Avbd3dHexColor color, void* context );
	/* Only shapes whose bounds overlap this box are drawn. */
	Avbd3dAABB drawingBounds;
	int32_t drawShapes;
	int32_t drawBounds;
	void* context;
} Avbd3dDebugDraw;

/* ---------------------------------------------------------------------------
   Defaults
   --------------------------------------------------------------------------- */

/* All callbacks NULL, drawing flags off, bounds +-100 m. */
AVBD3D_API Avbd3dDebugDraw avbd3d_default_debug_draw(void);

/* Use this to initialize your world definition */
AVBD3D_API Avbd3dWorldDef avbd3d_default_world_def(void);

/* Use this to initialize your body definition */
AVBD3D_API Avbd3dBodyDef avbd3d_default_body_def(void);

/* Use this to initialize your filter */
AVBD3D_API Avbd3dFilter avbd3d_default_filter(void);

/* Use this to initialize your surface material */
AVBD3D_API Avbd3dSurfaceMaterial avbd3d_default_surface_material(void);

/* Use this to initialize your shape definition */
AVBD3D_API Avbd3dShapeDef avbd3d_default_shape_def(void);

/* Use this to initialize your joint definition */
AVBD3D_API Avbd3dDistanceJointDef avbd3d_default_distance_joint_def(void);

/* Use this to initialize your joint definition */
AVBD3D_API Avbd3dMotorJointDef avbd3d_default_motor_joint_def(void);

/* Use this to initialize your joint definition */
AVBD3D_API Avbd3dFilterJointDef avbd3d_default_filter_joint_def(void);

/* Use this to initialize your joint definition */
AVBD3D_API Avbd3dParallelJointDef avbd3d_default_parallel_joint_def(void);

/* Use this to initialize your joint definition */
AVBD3D_API Avbd3dPrismaticJointDef avbd3d_default_prismatic_joint_def(void);

/* Use this to initialize your joint definition. */
AVBD3D_API Avbd3dRevoluteJointDef avbd3d_default_revolute_joint_def(void);

/* Use this to initialize your joint definition. */
AVBD3D_API Avbd3dSphericalJointDef avbd3d_default_spherical_joint_def(void);

/* Use this to initialize your joint definition */
AVBD3D_API Avbd3dWeldJointDef avbd3d_default_weld_joint_def(void);

/* Use this to initialize your joint definition */
AVBD3D_API Avbd3dWheelJointDef avbd3d_default_wheel_joint_def(void);

/* Use this to initialize your explosion definition */
AVBD3D_API Avbd3dExplosionDef avbd3d_default_explosion_def(void);

/* Use this to initialize your query filter */
AVBD3D_API Avbd3dQueryFilter avbd3d_default_query_filter(void);

/* Use this to initialize your solver definition. */
AVBD3D_API Avbd3dSolverDef avbd3d_default_solver_def(void);

/* A point joint: hard anchors, free rotation, unbreakable. */
AVBD3D_API Avbd3dGenericJointDef avbd3d_default_generic_joint_def(void);

/* A spring of stiffness 100 N/m at its creation length. */
AVBD3D_API Avbd3dSpringDef avbd3d_default_spring_def(void);

/* ---------------------------------------------------------------------------
   World
   --------------------------------------------------------------------------- */

/* Creates a world on its own Vulkan device. Returns AVBD3D_ERR_DEVICE when no usable device
   exists and AVBD3D_ERR_OUT_OF_MEMORY when memory ran out; *out_world is then NULL. */
AVBD3D_API Avbd3dResult avbd3d_create_world(const Avbd3dWorldDef *def, Avbd3dWorld **out_world);

/* Destroys a world and everything in it. Works on a dead world. A NULL world is a no-op that
   returns AVBD3D_ERR_NULL_HANDLE. */
AVBD3D_API Avbd3dResult avbd3d_destroy_world(Avbd3dWorld *world);

/* Advances the simulation by dt seconds: avbd3d_step_begin followed by avbd3d_step_end.
   Returns AVBD3D_ERR_CAPACITY if the step had to clamp results (see avbd3d_get_stats). */
AVBD3D_API Avbd3dResult avbd3d_step(Avbd3dWorld *world, float dt);

/* Applies queued work, records the whole step and submits it once. Returns without waiting. */
AVBD3D_API Avbd3dResult avbd3d_step_begin(Avbd3dWorld *world, float dt);

/* Waits for the step started by avbd3d_step_begin and reads back its status. A no-op when no
   step is in flight. Returns AVBD3D_ERR_CAPACITY if the step had to clamp results. */
AVBD3D_API Avbd3dResult avbd3d_step_end(Avbd3dWorld *world);

/* Counts, capacities, the result of the last step and its timings. Ends a step in flight. */
AVBD3D_API Avbd3dResult avbd3d_get_stats(Avbd3dWorld *world, Avbd3dStats *out);

/* The device and limits of this world. Set out->struct_size first. */
AVBD3D_API Avbd3dResult avbd3d_get_capabilities(Avbd3dWorld *world, Avbd3dCapabilities *out);

/* Calls the callbacks of draw for every shape whose category is in maskBits and whose bounds
   overlap draw->drawingBounds. */
AVBD3D_API Avbd3dResult avbd3d_world_draw(Avbd3dWorld *world, const Avbd3dDebugDraw *draw, uint64_t maskBits);

/* Get the world's bounds. This is the bounding box that covers the current simulation. May have a small
   amount of padding. */
AVBD3D_API Avbd3dResult avbd3d_world_get_bounds(Avbd3dWorld *world, Avbd3dAABB *out);

/* Enable/disable sleep. If your application does not need sleeping, you can gain some performance
   by disabling sleep completely at the world level.
   @see Avbd3dWorldDef */
AVBD3D_API Avbd3dResult avbd3d_world_enable_sleeping(Avbd3dWorld *world, int32_t flag);

/* Enable/disable continuous collision between dynamic and static bodies. Generally you should keep continuous
   collision enabled to prevent fast moving objects from going through static objects. The performance gain from
   disabling continuous collision is minor.
   @see Avbd3dWorldDef */
AVBD3D_API Avbd3dResult avbd3d_world_enable_continuous(Avbd3dWorld *world, int32_t flag);

/* Adjust the restitution threshold. It is recommended not to make this value very small
   because it will prevent bodies from sleeping. Usually in meters per second.
   @see Avbd3dWorldDef */
AVBD3D_API Avbd3dResult avbd3d_world_set_restitution_threshold(Avbd3dWorld *world, float value);

/* Adjust the hit event threshold. This controls the collision speed needed to generate a Avbd3dContactHitEvent.
   Usually in meters per second.
   @see Avbd3dWorldDef::hitEventThreshold */
AVBD3D_API Avbd3dResult avbd3d_world_set_hit_event_threshold(Avbd3dWorld *world, float value);

/* Set the gravity vector for the entire world. There is no concept of an up direction and this
   is left as a decision for the application. Usually in m/s^2.
   @see Avbd3dWorldDef */
AVBD3D_API Avbd3dResult avbd3d_world_set_gravity(Avbd3dWorld *world, Avbd3dVec3 gravity);

/* Get the gravity vector */
AVBD3D_API Avbd3dResult avbd3d_world_get_gravity(Avbd3dWorld *world, Avbd3dVec3 *out);

/* Apply a radial explosion
   @param world The world id
   @param explosionDef The explosion definition */
AVBD3D_API Avbd3dResult avbd3d_world_explode(Avbd3dWorld *world, const Avbd3dExplosionDef *explosionDef);

/* @return true between StepBegin and StepEnd. */
AVBD3D_API Avbd3dResult avbd3d_is_step_in_flight(Avbd3dWorld *world, int32_t *out);

/* Replace the solver parameters. Takes effect at the next step. */
AVBD3D_API Avbd3dResult avbd3d_world_set_solver_def(Avbd3dWorld *world, const Avbd3dSolverDef *def);

/* Rule that combines the friction coefficients of the two shapes of a contact. Default
   AVBD3D_MIX_GEOMETRIC. Applies to every body, existing and future. Takes effect at the next step. */
AVBD3D_API Avbd3dResult avbd3d_world_set_friction_mixing(Avbd3dWorld *world, Avbd3dMixingRule rule);

/* Rule that combines the restitution of the two bodies of a contact (each body carries its
   primary shape's restitution). Default AVBD3D_MIX_MAX. Takes effect at the next step. */
AVBD3D_API Avbd3dResult avbd3d_world_set_restitution_mixing(Avbd3dWorld *world, Avbd3dMixingRule rule);

/* Copy one property of `count` bodies into `data`. Invalid ids write zeros.
   @return the number of valid ids. */
AVBD3D_API Avbd3dResult avbd3d_world_get_body_data(Avbd3dWorld *world, const Avbd3dBodyId *ids,
                                                   int32_t count, Avbd3dBodyDataType type, void *data,
                                                   int32_t *out);

/* Write one property of `count` bodies from `data` (not AVBD3D_BODY_DATA_AWAKE).
   @return the number of valid ids. */
AVBD3D_API Avbd3dResult avbd3d_world_set_body_data(Avbd3dWorld *world, const Avbd3dBodyId *ids,
                                                   int32_t count, Avbd3dBodyDataType type, const void *data,
                                                   int32_t *out);

/* ---------------------------------------------------------------------------
   Events
   --------------------------------------------------------------------------- */

/* Get the body events for the current time step. The event data is transient. Do not store a reference to this data. */
AVBD3D_API Avbd3dResult avbd3d_world_get_body_events(Avbd3dWorld *world, Avbd3dBodyEvents *out);

/* Get sensor events for the current time step. The event data is transient. Do not store a reference to this data. */
AVBD3D_API Avbd3dResult avbd3d_world_get_sensor_events(Avbd3dWorld *world, Avbd3dSensorEvents *out);

/* Get contact events for this current time step. The event data is transient. Do not store a reference to this data. */
AVBD3D_API Avbd3dResult avbd3d_world_get_contact_events(Avbd3dWorld *world, Avbd3dContactEvents *out);

/* Get the joint events for the current time step. The event data is transient. Do not store a reference to this data. */
AVBD3D_API Avbd3dResult avbd3d_world_get_joint_events(Avbd3dWorld *world, Avbd3dJointEvents *out);

/* ---------------------------------------------------------------------------
   Queries
   --------------------------------------------------------------------------- */

/* Cast a ray into the world to collect the closest hit. This is a convenience function. Ignores initial overlap.
   This is less general than avbd3d_world_cast_ray() and does not allow for custom filtering. Ignores back-side collision
   on meshes and height-fields. */
AVBD3D_API Avbd3dResult avbd3d_world_cast_ray_closest(Avbd3dWorld *world, Avbd3dVec3 origin,
                                                      Avbd3dVec3 translation, Avbd3dQueryFilter filter,
                                                      Avbd3dRayResult *out);

/* Calls fcn for each shape whose bounds overlap aabb. Return 0 from fcn to stop. */
AVBD3D_API Avbd3dResult avbd3d_world_overlap_aabb(Avbd3dWorld *world, Avbd3dAABB aabb,
                                                  Avbd3dQueryFilter filter,
                                                  Avbd3dOverlapResultFcn *fcn, void *context);

/* Calls fcn for each shape that overlaps the convex proxy placed at origin. */
AVBD3D_API Avbd3dResult avbd3d_world_overlap_shape(Avbd3dWorld *world, Avbd3dVec3 origin,
                                                   const Avbd3dShapeProxy *proxy,
                                                   Avbd3dQueryFilter filter, Avbd3dOverlapResultFcn *fcn, void *context);

/* Casts a ray and calls fcn for each hit. fcn returns the new max fraction: -1 to ignore the
   hit, 0 to stop, the hit fraction to clip, 1 to continue. */
AVBD3D_API Avbd3dResult avbd3d_world_cast_ray(Avbd3dWorld *world, Avbd3dVec3 origin, Avbd3dVec3 translation,
                                              Avbd3dQueryFilter filter, Avbd3dCastResultFcn *fcn, void *context);

/* Sweeps a convex proxy along translation and calls fcn for each hit. */
AVBD3D_API Avbd3dResult avbd3d_world_cast_shape(Avbd3dWorld *world, Avbd3dVec3 origin,
                                                const Avbd3dShapeProxy *proxy,
                                                Avbd3dVec3 translation, Avbd3dQueryFilter filter,
                                                Avbd3dCastResultFcn *fcn, void *context);

/* Sweeps a character capsule along translation. fcn may be NULL; it returns 0 to skip a shape.
   out_fraction is the fraction of translation the capsule can travel (1 if unobstructed). */
AVBD3D_API Avbd3dResult avbd3d_world_cast_mover(Avbd3dWorld *world, Avbd3dVec3 origin,
                                                const Avbd3dCapsule *mover,
                                                Avbd3dVec3 translation, Avbd3dQueryFilter filter,
                                                Avbd3dMoverFilterFcn *fcn, void *context, float *out_fraction);

/* Calls fcn with the contact planes of every shape the character capsule touches at origin. */
AVBD3D_API Avbd3dResult avbd3d_world_collide_mover(Avbd3dWorld *world, Avbd3dVec3 origin,
                                                   const Avbd3dCapsule *mover,
                                                   Avbd3dQueryFilter filter, Avbd3dPlaneResultFcn *fcn, void *context);

/* ---------------------------------------------------------------------------
   Bodies
   --------------------------------------------------------------------------- */

/* Create a rigid body given a definition. No reference to the definition is retained. So you can create the definition
   on the stack and pass it as a pointer.
   @code{.c}
   Avbd3dBodyDef bodyDef = avbd3d_default_body_def();
   Avbd3dBodyId myBodyId = avbd3d_create_body(myWorldId, &bodyDef);
   @endcode
   Warning: This function is locked during callbacks. */
AVBD3D_API Avbd3dResult avbd3d_create_body(Avbd3dWorld *world, const Avbd3dBodyDef *def,
                                           Avbd3dBodyId *out_id);

/* Destroy a rigid body given an id. This destroys all shapes and joints attached to the body.
   Do not keep references to the associated shapes and joints. */
AVBD3D_API Avbd3dResult avbd3d_destroy_body(Avbd3dBodyId bodyId);

/* Get the body type: static, kinematic, or dynamic */
AVBD3D_API Avbd3dResult avbd3d_body_get_type(Avbd3dBodyId id, Avbd3dBodyType *out);

/* Change the body type. This is an expensive operation. This automatically updates the mass
   properties regardless of the automatic mass setting. */
AVBD3D_API Avbd3dResult avbd3d_body_set_type(Avbd3dBodyId id, Avbd3dBodyType type);

/* Set the user data for a body */
AVBD3D_API Avbd3dResult avbd3d_body_set_user_data(Avbd3dBodyId id, void *userData);

/* Get the user data stored in a body */
AVBD3D_API Avbd3dResult avbd3d_body_get_user_data(Avbd3dBodyId id, void **out);

/* Get the world position of a body. This is the location of the body origin. */
AVBD3D_API Avbd3dResult avbd3d_body_get_position(Avbd3dBodyId id, Avbd3dVec3 *out);

/* Get the world rotation of a body as a quaternion */
AVBD3D_API Avbd3dResult avbd3d_body_get_rotation(Avbd3dBodyId id, Avbd3dQuat *out);

/* Get the world transform of a body. */
AVBD3D_API Avbd3dResult avbd3d_body_get_transform(Avbd3dBodyId id, Avbd3dTransform *out);

/* Set the world transform of a body. This acts as a teleport and is fairly expensive.
   Note: Generally you should create a body with the intended transform.
   @see Avbd3dBodyDef::position and Avbd3dBodyDef::rotation. */
AVBD3D_API Avbd3dResult avbd3d_body_set_transform(Avbd3dBodyId id, Avbd3dVec3 position, Avbd3dQuat rotation);

/* Get a local point on a body given a world point. */
AVBD3D_API Avbd3dResult avbd3d_body_get_local_point(Avbd3dBodyId id, Avbd3dVec3 worldPoint, Avbd3dVec3 *out);

/* Get a world point on a body given a local point. */
AVBD3D_API Avbd3dResult avbd3d_body_get_world_point(Avbd3dBodyId id, Avbd3dVec3 localPoint, Avbd3dVec3 *out);

/* Get a local vector on a body given a world vector. */
AVBD3D_API Avbd3dResult avbd3d_body_get_local_vector(Avbd3dBodyId id, Avbd3dVec3 worldVector,
                                                     Avbd3dVec3 *out);

/* Get a world vector on a body given a local vector. */
AVBD3D_API Avbd3dResult avbd3d_body_get_world_vector(Avbd3dBodyId id, Avbd3dVec3 localVector,
                                                     Avbd3dVec3 *out);

/* Get the linear velocity of a body's center of mass. Usually in meters per second. */
AVBD3D_API Avbd3dResult avbd3d_body_get_linear_velocity(Avbd3dBodyId id, Avbd3dVec3 *out);

/* Get the angular velocity of a body in radians per second. */
AVBD3D_API Avbd3dResult avbd3d_body_get_angular_velocity(Avbd3dBodyId id, Avbd3dVec3 *out);

/* Set the linear velocity of a body at the center of mass. Usually in meters per second. */
AVBD3D_API Avbd3dResult avbd3d_body_set_linear_velocity(Avbd3dBodyId id, Avbd3dVec3 linearVelocity);

/* Set the angular velocity of a body in radians per second. */
AVBD3D_API Avbd3dResult avbd3d_body_set_angular_velocity(Avbd3dBodyId id, Avbd3dVec3 angularVelocity);

/* Set the velocity to reach the given body origin transform after a given time step. Meant for kinematic bodies;
   ignored for static or disabled bodies and when timeStep <= 0. A sleeping body is only moved (and woken) if wake is
   true and the movement is above its sleep threshold. */
AVBD3D_API Avbd3dResult avbd3d_body_set_target_transform(Avbd3dBodyId id, Avbd3dTransform target,
                                                         float timeStep, int32_t wake);

/* Get the linear velocity of a local point attached to a body. Usually in meters per second. */
AVBD3D_API Avbd3dResult avbd3d_body_get_local_point_velocity(Avbd3dBodyId id, Avbd3dVec3 localPoint,
                                                             Avbd3dVec3 *out);

/* Get the linear velocity of a world point attached to a body. Usually in meters per second. */
AVBD3D_API Avbd3dResult avbd3d_body_get_world_point_velocity(Avbd3dBodyId id, Avbd3dVec3 worldPoint,
                                                             Avbd3dVec3 *out);

/* Apply a force at a world point. If the force is not applied at the center of mass,
   it will generate a torque and affect the angular velocity. This optionally wakes up the body.
   The force is ignored if the body is not awake.
   @param id The body id
   @param force The world force vector, usually in newtons (N)
   @param point The world position of the point of application
   @param wake Option to wake up the body */
AVBD3D_API Avbd3dResult avbd3d_body_apply_force(Avbd3dBodyId id, Avbd3dVec3 force, Avbd3dVec3 point,
                                                int32_t wake);

/* Apply a force to the center of mass. This optionally wakes up the body.
   The force is ignored if the body is not awake.
   @param id The body id
   @param force the world force vector, usually in newtons (N).
   @param wake also wake up the body */
AVBD3D_API Avbd3dResult avbd3d_body_apply_force_to_center(Avbd3dBodyId id, Avbd3dVec3 force, int32_t wake);

/* Apply a torque. This affects the angular velocity without affecting the linear velocity.
   This optionally wakes the body. The torque is ignored if the body is not awake.
   @param id The body id
   @param torque the world torque vector, usually in N*m.
   @param wake also wake up the body */
AVBD3D_API Avbd3dResult avbd3d_body_apply_torque(Avbd3dBodyId id, Avbd3dVec3 torque, int32_t wake);

/* Apply an impulse at a point. This immediately modifies the velocity.
   It also modifies the angular velocity if the point of application
   is not at the center of mass. This optionally wakes the body.
   The impulse is ignored if the body is not awake.
   @param id The body id
   @param impulse the world impulse vector, usually in N*s or kg*m/s.
   @param point the world position of the point of application.
   @param wake also wake up the body
   Warning: This should be used for one-shot impulses. If you need a steady force,
   use a force instead, which works better with the solver. */
AVBD3D_API Avbd3dResult avbd3d_body_apply_linear_impulse(Avbd3dBodyId id, Avbd3dVec3 impulse,
                                                         Avbd3dVec3 point, int32_t wake);

/* Apply an impulse to the center of mass. This immediately modifies the velocity.
   The impulse is ignored if the body is not awake. This optionally wakes the body.
   @param id The body id
   @param impulse the world impulse vector, usually in N*s or kg*m/s.
   @param wake also wake up the body
   Warning: This should be used for one-shot impulses. If you need a steady force,
   use a force instead, which works better with the solver. */
AVBD3D_API Avbd3dResult avbd3d_body_apply_linear_impulse_to_center(Avbd3dBodyId id, Avbd3dVec3 impulse,
                                                                   int32_t wake);

/* Apply an angular impulse in world space. The impulse is ignored if the body is not awake.
   This optionally wakes the body.
   @param id The body id
   @param impulse the world angular impulse vector, usually in units of kg*m*m/s
   @param wake also wake up the body
   Warning: This should be used for one-shot impulses. If you need a steady torque,
   use a torque instead, which works better with the solver. */
AVBD3D_API Avbd3dResult avbd3d_body_apply_angular_impulse(Avbd3dBodyId id, Avbd3dVec3 impulse, int32_t wake);

/* Get the mass of the body, usually in kilograms */
AVBD3D_API Avbd3dResult avbd3d_body_get_mass(Avbd3dBodyId id, float *out);

/* Get the rotational inertia of the body in local space, usually in kg*m^2 */
AVBD3D_API Avbd3dResult avbd3d_body_get_local_rotational_inertia(Avbd3dBodyId id, Avbd3dMatrix3 *out);

/* Get the inverse mass of the body, usually in 1/kilograms */
AVBD3D_API Avbd3dResult avbd3d_body_get_inverse_mass(Avbd3dBodyId id, float *out);

/* Get the inverse rotational inertia of the body in world space, usually in 1/kg*m^2 */
AVBD3D_API Avbd3dResult avbd3d_body_get_world_inverse_rotational_inertia(Avbd3dBodyId id, Avbd3dMatrix3 *out);

/* Get the center of mass position of the body in local space */
AVBD3D_API Avbd3dResult avbd3d_body_get_local_center(Avbd3dBodyId id, Avbd3dVec3 *out);

/* Get the center of mass position of the body in world space */
AVBD3D_API Avbd3dResult avbd3d_body_get_world_center(Avbd3dBodyId id, Avbd3dVec3 *out);

/* Override the body's mass properties. Normally this is computed automatically using the
   shape geometry and density. This information is lost if a shape is added or removed or if the
   body type changes. */
AVBD3D_API Avbd3dResult avbd3d_body_set_mass_data(Avbd3dBodyId id, Avbd3dMassData massData);

/* Get the mass data for a body */
AVBD3D_API Avbd3dResult avbd3d_body_get_mass_data(Avbd3dBodyId id, Avbd3dMassData *out);

/* This updates the mass properties to the sum of the mass properties of the shapes.
   This normally does not need to be called unless you called SetMassData to override
   the mass and you later want to reset the mass.
   You may also use this when automatic mass computation has been disabled.
   You should call this regardless of body type. */
AVBD3D_API Avbd3dResult avbd3d_body_apply_mass_from_shapes(Avbd3dBodyId id);

/* Adjust the linear damping. Normally this is set in Avbd3dBodyDef before creation. */
AVBD3D_API Avbd3dResult avbd3d_body_set_linear_damping(Avbd3dBodyId id, float linearDamping);

/* Adjust the angular damping. Normally this is set in Avbd3dBodyDef before creation. */
AVBD3D_API Avbd3dResult avbd3d_body_set_angular_damping(Avbd3dBodyId id, float angularDamping);

/* Adjust the gravity scale. Normally this is set in Avbd3dBodyDef before creation.
   @see Avbd3dBodyDef::gravityScale */
AVBD3D_API Avbd3dResult avbd3d_body_set_gravity_scale(Avbd3dBodyId id, float gravityScale);

/* @return true if this body is awake */
AVBD3D_API Avbd3dResult avbd3d_body_is_awake(Avbd3dBodyId id, int32_t *out);

/* Wake a body from sleep. This wakes the entire island the body is touching.
   Warning: Putting a body to sleep will put the entire island of bodies touching this body to sleep,
   which can be expensive and possibly unintuitive. */
AVBD3D_API Avbd3dResult avbd3d_body_set_awake(Avbd3dBodyId id, int32_t awake);

/* Returns true if this body is enabled */
AVBD3D_API Avbd3dResult avbd3d_body_is_enabled(Avbd3dBodyId id, int32_t *out);

/* Disable a body by removing it completely from the simulation. This is expensive. */
AVBD3D_API Avbd3dResult avbd3d_body_disable(Avbd3dBodyId id);

/* Enable a body by adding it to the simulation. This is expensive. */
AVBD3D_API Avbd3dResult avbd3d_body_enable(Avbd3dBodyId id);

/* Set the motion locks on this body. */
AVBD3D_API Avbd3dResult avbd3d_body_set_motion_locks(Avbd3dBodyId id, Avbd3dMotionLocks locks);

/* Set this body to be a bullet. A bullet does continuous collision detection
   against dynamic bodies (but not other bullets). */
AVBD3D_API Avbd3dResult avbd3d_body_set_bullet(Avbd3dBodyId id, int32_t flag);

/* Enable/disable hit events on all shapes
   @see Avbd3dShapeDef::enableHitEvents */
AVBD3D_API Avbd3dResult avbd3d_body_enable_hit_events(Avbd3dBodyId id, int32_t flag);

/* Get the world that owns this body */
AVBD3D_API Avbd3dResult avbd3d_body_get_world(Avbd3dBodyId id, Avbd3dWorld **out);

/* Get the number of shapes on this body */
AVBD3D_API Avbd3dResult avbd3d_body_get_shape_count(Avbd3dBodyId id, int32_t *out_count);

/* Get the shape ids for all shapes on this body, up to the provided capacity.
   @returns the number of shape ids stored in the user array */
AVBD3D_API Avbd3dResult avbd3d_body_get_shapes(Avbd3dBodyId id, Avbd3dShapeId *shapeArray, int32_t capacity,
                                               int32_t *out_count);

/* Get the number of joints on this body */
AVBD3D_API Avbd3dResult avbd3d_body_get_joint_count(Avbd3dBodyId id, int32_t *out_count);

/* Get the joint ids for all joints on this body, up to the provided capacity
   @returns the number of joint ids stored in the user array */
AVBD3D_API Avbd3dResult avbd3d_body_get_joints(Avbd3dBodyId id, Avbd3dJointId *jointArray, int32_t capacity,
                                               int32_t *out_count);

/* Get the maximum capacity required for retrieving all the touching contacts on a body */
AVBD3D_API Avbd3dResult avbd3d_body_get_contact_capacity(Avbd3dBodyId id, int32_t *out_count);

/* Get the touching contact data for a body */
AVBD3D_API Avbd3dResult avbd3d_body_get_contact_data(Avbd3dBodyId id, Avbd3dContactData *contactData,
                                                     int32_t capacity, int32_t *out_count);

/* Get the current world AABB that contains all the attached shapes. Note that this may not encompass the body origin.
   If there are no shapes attached then the returned AABB is empty and centered on the body origin. */
AVBD3D_API Avbd3dResult avbd3d_body_compute_aabb(Avbd3dBodyId id, Avbd3dAABB *out);

/* Cast a ray at a specific body using a specified body transform. Only the body's own shapes are tested. */
AVBD3D_API Avbd3dResult avbd3d_body_cast_ray(Avbd3dBodyId id, Avbd3dVec3 origin, Avbd3dVec3 translation,
                                             Avbd3dQueryFilter filter, float maxFraction,
                                             Avbd3dTransform bodyTransform, Avbd3dBodyCastResult *out);

/* Cast a shape at a specific body using a specified body transform. Only the body's own shapes are tested. */
AVBD3D_API Avbd3dResult avbd3d_body_cast_shape(Avbd3dBodyId id, Avbd3dVec3 origin,
                                               const Avbd3dShapeProxy *proxy, Avbd3dVec3 translation,
                                               Avbd3dQueryFilter filter, float maxFraction,
                                               int32_t canEncroach, Avbd3dTransform bodyTransform,
                                               Avbd3dBodyCastResult *out);

/* Overlap a shape with a specific body using a specified body transform. */
AVBD3D_API Avbd3dResult avbd3d_body_overlap_shape(Avbd3dBodyId id, Avbd3dVec3 origin,
                                                  const Avbd3dShapeProxy *proxy, Avbd3dQueryFilter filter,
                                                  Avbd3dTransform bodyTransform, int32_t *out);

/* Collide a character mover with a specific body using a specified body transform. Only convex shapes, one plane each. */
AVBD3D_API Avbd3dResult avbd3d_body_collide_mover(Avbd3dBodyId id, Avbd3dBodyPlaneResult *bodyPlanes,
                                                  int32_t planeCapacity, Avbd3dVec3 origin,
                                                  const Avbd3dCapsule *mover, Avbd3dQueryFilter filter,
                                                  Avbd3dTransform bodyTransform, int32_t *out_count);

/* 1 if id names a live body of a live world. Takes no world, so a dead world reports 0. */
AVBD3D_API int32_t avbd3d_body_is_valid(Avbd3dBodyId id);

/* The point on the body's shapes closest to target, and its distance. */
AVBD3D_API Avbd3dResult avbd3d_body_get_closest_point(Avbd3dBodyId id, Avbd3dVec3 target,
                                                      Avbd3dVec3 *out_point,
                                                      float *out_distance);

/* ---------------------------------------------------------------------------
   Shapes
   --------------------------------------------------------------------------- */

/* Create a circle shape and attach it to a body. The shape definition and geometry are fully cloned.
   Contacts are not created until the next time step.
   @return the shape id for accessing the shape */
AVBD3D_API Avbd3dResult avbd3d_create_sphere_shape(Avbd3dBodyId bodyId, const Avbd3dShapeDef *def,
                                                   const Avbd3dSphere *sphere, Avbd3dShapeId *out_id);

/* Create a capsule shape and attach it to a body. The shape definition and geometry are fully cloned.
   Contacts are not created until the next time step.
   @return the shape id for accessing the shape */
AVBD3D_API Avbd3dResult avbd3d_create_capsule_shape(Avbd3dBodyId bodyId, const Avbd3dShapeDef *def,
                                                    const Avbd3dCapsule *capsule, Avbd3dShapeId *out_id);

/* Create a convex hull shape and attach it to a body. The shape definition is fully cloned. Contacts are not created
   until the next time step.
   @return the shape id for accessing the shape */
AVBD3D_API Avbd3dResult avbd3d_create_hull_shape(Avbd3dBodyId bodyId, const Avbd3dShapeDef *def,
                                                 const Avbd3dHullData *hull, Avbd3dShapeId *out_id);

/* Create a convex hull shape and attach it to a body. The hull is cloned then transformed with scale applied first.
   Use this for non-uniform or mirrored scale or a baked local transform. The baked result is shared through the
   world hull database. The shape definition and geometry are fully cloned. Contacts are not created until the next time step.
   @return the shape id for accessing the shape */
AVBD3D_API Avbd3dResult avbd3d_create_transformed_hull_shape(Avbd3dBodyId bodyId, const Avbd3dShapeDef *def,
                                                             const Avbd3dHullData *hull,
                                                             Avbd3dTransform transform, Avbd3dVec3 scale,
                                                             Avbd3dShapeId *out_id);

/* Create a mesh hull shape and attach it to a body. The shape definition is fully cloned but the mesh is not.
   Contacts are not created until the next time step.
   Mesh collision only creates contacts on static bodies.
   Warning: this holds reference to the input mesh data which must remain valid for the lifetime of this shape
   @return the shape id for accessing the shape */
AVBD3D_API Avbd3dResult avbd3d_create_mesh_shape(Avbd3dBodyId bodyId, const Avbd3dShapeDef *def,
                                                 const Avbd3dMeshData *mesh, Avbd3dVec3 scale,
                                                 Avbd3dShapeId *out_id);

/* Create a height-field shape and attach it to a body. The shape definition is fully cloned but the height field is not.
   Contacts are not created until the next time step.
   Height field is only allowed on static bodies.
   Warning: this holds reference to the input height field which must remain valid for the lifetime of this shape
   @return the shape id for accessing the shape */
AVBD3D_API Avbd3dResult avbd3d_create_height_field_shape(Avbd3dBodyId bodyId, const Avbd3dShapeDef *def,
                                                         const Avbd3dHeightFieldData *heightField,
                                                         Avbd3dShapeId *out_id);

/* Baked compound shapes are only allowed on static bodies.
   Note: runtime compounds are achieved by adding multiple shapes to a body.
   Runtime compounds can be dynamic and/or kinematic. */
AVBD3D_API Avbd3dResult avbd3d_create_baked_compound_shape(Avbd3dBodyId bodyId, Avbd3dShapeDef *def,
                                                           const Avbd3dCompoundData *compound,
                                                           Avbd3dShapeId *out_id);

/* Destroy a shape. You may defer the body mass update which can improve performance if several shapes on a
   body are destroyed at once.
   @see avbd3d_body_apply_mass_from_shapes */
AVBD3D_API Avbd3dResult avbd3d_destroy_shape(Avbd3dShapeId shapeId, int32_t updateBodyMass);

/* Get the type of a shape */
AVBD3D_API Avbd3dResult avbd3d_shape_get_type(Avbd3dShapeId id, Avbd3dShapeType *out);

/* Get the id of the body that a shape is attached to */
AVBD3D_API Avbd3dResult avbd3d_shape_get_body(Avbd3dShapeId id, Avbd3dBodyId *out);

/* Get the world that owns this shape */
AVBD3D_API Avbd3dResult avbd3d_shape_get_world(Avbd3dShapeId id, Avbd3dWorld **out);

/* Returns true if the shape is a sensor */
AVBD3D_API Avbd3dResult avbd3d_shape_is_sensor(Avbd3dShapeId id, int32_t *out);

/* Set the user data for a shape */
AVBD3D_API Avbd3dResult avbd3d_shape_set_user_data(Avbd3dShapeId id, void *userData);

/* Get the user data for a shape. This is useful when you get a shape id
   from an event or query. */
AVBD3D_API Avbd3dResult avbd3d_shape_get_user_data(Avbd3dShapeId id, void **out);

/* Set the mass density of a shape, usually in kg/m^3.
   This will optionally update the mass properties on the parent body.
   @see Avbd3dShapeDef::density, avbd3d_body_apply_mass_from_shapes */
AVBD3D_API Avbd3dResult avbd3d_shape_set_density(Avbd3dShapeId id, float density, int32_t updateBodyMass);

/* Set the friction on a shape */
AVBD3D_API Avbd3dResult avbd3d_shape_set_friction(Avbd3dShapeId id, float friction);

/* Set the shape restitution (bounciness) */
AVBD3D_API Avbd3dResult avbd3d_shape_set_restitution(Avbd3dShapeId id, float restitution);

/* Set the shape base surface material. Does not change per triangle materials. */
AVBD3D_API Avbd3dResult avbd3d_shape_set_surface_material(Avbd3dShapeId id,
                                                          Avbd3dSurfaceMaterial surfaceMaterial);

/* Set the current filter on a shape. The new filter takes effect at the next step: contacts the
   filter now rejects end (an end touch event fires if the shape has contact events enabled) and
   pairs it now allows are created when the shapes overlap. Wakes the shape's body.
   @see Avbd3dFilter */
AVBD3D_API Avbd3dResult avbd3d_shape_set_filter(Avbd3dShapeId id, Avbd3dFilter filter);

/* Enable sensor events for this shape. Only applies to kinematic and dynamic bodies. Ignored for sensors.
   @see Avbd3dShapeDef::isSensor */
AVBD3D_API Avbd3dResult avbd3d_shape_enable_sensor_events(Avbd3dShapeId id, int32_t flag);

/* Enable contact events for this shape. Only applies to kinematic and dynamic bodies. Ignored for sensors.
   @see Avbd3dShapeDef::enableContactEvents */
AVBD3D_API Avbd3dResult avbd3d_shape_enable_contact_events(Avbd3dShapeId id, int32_t flag);

/* Enable contact hit events for this shape. Ignored for sensors.
   @see Avbd3dWorldDef.hitEventThreshold */
AVBD3D_API Avbd3dResult avbd3d_shape_enable_hit_events(Avbd3dShapeId id, int32_t flag);

/* Ray cast a shape directly. The ray runs from origin to origin + translation and the hit point
   comes back as a world position, so the cast stays precise far from the world origin. */
AVBD3D_API Avbd3dResult avbd3d_shape_ray_cast(Avbd3dShapeId id, Avbd3dVec3 origin, Avbd3dVec3 translation,
                                              Avbd3dCastOutput *out);

/* Get the maximum capacity required for retrieving all the touching contacts on a shape */
AVBD3D_API Avbd3dResult avbd3d_shape_get_contact_capacity(Avbd3dShapeId id, int32_t *out_count);

/* Get the touching contact data for a shape. The provided shapeId will be either shapeIdA or shapeIdB on the contact data.
   Note: Contacts are speculative, so some contact points may be separated.
   @returns the number of elements filled in the provided array
   Warning: do not ignore the return value, it specifies the valid number of elements */
AVBD3D_API Avbd3dResult avbd3d_shape_get_contact_data(Avbd3dShapeId id, Avbd3dContactData *contactData,
                                                      int32_t capacity, int32_t *out_count);

/* Get the maximum capacity required for retrieving all the overlapped shapes on a sensor shape.
   This returns 0 if the provided shape is not a sensor.
   @param id the id of a sensor shape
   @returns the required capacity to get all the overlaps in avbd3d_shape_get_sensor_overlaps */
AVBD3D_API Avbd3dResult avbd3d_shape_get_sensor_capacity(Avbd3dShapeId id, int32_t *out_count);

/* Get the overlap data for a sensor shape.
   @param id the id of a sensor shape
   @param visitorIds a user allocated array that is filled with the overlapping shapes (visitors)
   @param capacity the capacity of overlappedShapes
   @returns the number of elements filled in the provided array
   Warning: do not ignore the return value, it specifies the valid number of elements
   Warning: overlaps may contain destroyed shapes so use avbd3d_shape_is_valid to confirm each overlap */
AVBD3D_API Avbd3dResult avbd3d_shape_get_sensor_data(Avbd3dShapeId id, Avbd3dShapeId *visitorIds,
                                                     int32_t capacity, int32_t *out_count);

/* Get the current world AABB */
AVBD3D_API Avbd3dResult avbd3d_shape_get_aabb(Avbd3dShapeId id, Avbd3dAABB *out);

/* Compute the mass data for a shape */
AVBD3D_API Avbd3dResult avbd3d_shape_compute_mass_data(Avbd3dShapeId id, Avbd3dMassData *out);

/* 1 if id names a live shape of a live world. */
AVBD3D_API int32_t avbd3d_shape_is_valid(Avbd3dShapeId id);

/* ---------------------------------------------------------------------------
   Joints (common)
   --------------------------------------------------------------------------- */

/* Get the joint type */
AVBD3D_API Avbd3dResult avbd3d_joint_get_type(Avbd3dJointId id, Avbd3dJointType *out);

/* Get body A id on a joint */
AVBD3D_API Avbd3dResult avbd3d_joint_get_body_a(Avbd3dJointId id, Avbd3dBodyId *out);

/* Get body B id on a joint */
AVBD3D_API Avbd3dResult avbd3d_joint_get_body_b(Avbd3dJointId id, Avbd3dBodyId *out);

/* Get the world that owns this joint */
AVBD3D_API Avbd3dResult avbd3d_joint_get_world(Avbd3dJointId id, Avbd3dWorld **out);

/* Set the local frame on bodyA */
AVBD3D_API Avbd3dResult avbd3d_joint_set_local_frame_a(Avbd3dJointId id, Avbd3dTransform localFrame);

/* Set the local frame on bodyB */
AVBD3D_API Avbd3dResult avbd3d_joint_set_local_frame_b(Avbd3dJointId id, Avbd3dTransform localFrame);

/* Toggle collision between connected bodies */
AVBD3D_API Avbd3dResult avbd3d_joint_set_collide_connected(Avbd3dJointId id, int32_t shouldCollide);

/* Set the user data on a joint */
AVBD3D_API Avbd3dResult avbd3d_joint_set_user_data(Avbd3dJointId id, void *userData);

/* Get the user data on a joint */
AVBD3D_API Avbd3dResult avbd3d_joint_get_user_data(Avbd3dJointId id, void **out);

/* Wake the bodies connect to this joint */
AVBD3D_API Avbd3dResult avbd3d_joint_wake_bodies(Avbd3dJointId id);

/* Get the current constraint force for this joint */
AVBD3D_API Avbd3dResult avbd3d_joint_get_constraint_force(Avbd3dJointId id, Avbd3dVec3 *out);

/* Get the current constraint torque for this joint */
AVBD3D_API Avbd3dResult avbd3d_joint_get_constraint_torque(Avbd3dJointId id, Avbd3dVec3 *out);

/* Set the force threshold for joint events (Newtons) */
AVBD3D_API Avbd3dResult avbd3d_joint_set_force_threshold(Avbd3dJointId id, float threshold);

/* Set the torque threshold for joint events (N-m) */
AVBD3D_API Avbd3dResult avbd3d_joint_set_torque_threshold(Avbd3dJointId id, float threshold);

/* Destroys a joint. If wake_bodies is nonzero the bodies it connected are woken. */
AVBD3D_API Avbd3dResult avbd3d_destroy_joint(Avbd3dJointId id, int32_t wake_bodies);

/* 1 if id names a live joint of a live world. */
AVBD3D_API int32_t avbd3d_joint_is_valid(Avbd3dJointId id);

/* ---------------------------------------------------------------------------
   Joints
   --------------------------------------------------------------------------- */

/* Create a distance joint
   @see Avbd3dDistanceJointDef for details */
AVBD3D_API Avbd3dResult avbd3d_create_distance_joint(Avbd3dWorld *world, const Avbd3dDistanceJointDef *def,
                                                     Avbd3dJointId *out_id);

/* Set the rest length of a distance joint
   @param id The id for a distance joint
   @param length The new distance joint length */
AVBD3D_API Avbd3dResult avbd3d_distance_joint_set_length(Avbd3dJointId id, float length);

/* Enable/disable the distance joint spring. When disabled the distance joint is rigid. */
AVBD3D_API Avbd3dResult avbd3d_distance_joint_enable_spring(Avbd3dJointId id, int32_t enableSpring);

/* Set the force range for the spring. */
AVBD3D_API Avbd3dResult avbd3d_distance_joint_set_spring_force_range(Avbd3dJointId id, float lowerForce,
                                                                     float upperForce);

/* Set the spring stiffness in Hertz */
AVBD3D_API Avbd3dResult avbd3d_distance_joint_set_spring_hertz(Avbd3dJointId id, float hertz);

/* Set the spring damping ratio, non-dimensional */
AVBD3D_API Avbd3dResult avbd3d_distance_joint_set_spring_damping_ratio(Avbd3dJointId id, float dampingRatio);

/* Enable joint limit. The limit only works if the joint spring is enabled. Otherwise the joint is rigid
   and the limit has no effect. */
AVBD3D_API Avbd3dResult avbd3d_distance_joint_enable_limit(Avbd3dJointId id, int32_t enableLimit);

/* Set the minimum and maximum length parameters of a distance joint */
AVBD3D_API Avbd3dResult avbd3d_distance_joint_set_length_range(Avbd3dJointId id, float minLength,
                                                               float maxLength);

/* Enable/disable the distance joint motor */
AVBD3D_API Avbd3dResult avbd3d_distance_joint_enable_motor(Avbd3dJointId id, int32_t enableMotor);

/* Set the distance joint motor speed, usually in meters per second */
AVBD3D_API Avbd3dResult avbd3d_distance_joint_set_motor_speed(Avbd3dJointId id, float motorSpeed);

/* Set the distance joint maximum motor force, usually in newtons */
AVBD3D_API Avbd3dResult avbd3d_distance_joint_set_max_motor_force(Avbd3dJointId id, float force);

/* Get the current distance between the two anchors in meters. */
AVBD3D_API Avbd3dResult avbd3d_distance_joint_get_current_length(Avbd3dJointId id, float *out);

/* Get the force the distance joint motor applies, in newtons (positive lengthens the joint), from the newest finished step. */
AVBD3D_API Avbd3dResult avbd3d_distance_joint_get_motor_force(Avbd3dJointId id, float *out);

/* Create a motor joint
   @see Avbd3dMotorJointDef for details */
AVBD3D_API Avbd3dResult avbd3d_create_motor_joint(Avbd3dWorld *world, const Avbd3dMotorJointDef *def,
                                                  Avbd3dJointId *out_id);

/* Set the desired relative linear velocity in meters per second */
AVBD3D_API Avbd3dResult avbd3d_motor_joint_set_linear_velocity(Avbd3dJointId id, Avbd3dVec3 velocity);

/* Set the desired relative angular velocity in radians per second */
AVBD3D_API Avbd3dResult avbd3d_motor_joint_set_angular_velocity(Avbd3dJointId id, Avbd3dVec3 velocity);

/* Set the motor joint maximum force, usually in newtons */
AVBD3D_API Avbd3dResult avbd3d_motor_joint_set_max_velocity_force(Avbd3dJointId id, float maxForce);

/* Set the motor joint maximum torque, usually in newton-meters */
AVBD3D_API Avbd3dResult avbd3d_motor_joint_set_max_velocity_torque(Avbd3dJointId id, float maxTorque);

/* Set the spring linear hertz stiffness */
AVBD3D_API Avbd3dResult avbd3d_motor_joint_set_linear_hertz(Avbd3dJointId id, float hertz);

/* Set the spring linear damping ratio. Use 1.0 for critical damping. */
AVBD3D_API Avbd3dResult avbd3d_motor_joint_set_linear_damping_ratio(Avbd3dJointId id, float damping);

/* Set the spring angular hertz stiffness */
AVBD3D_API Avbd3dResult avbd3d_motor_joint_set_angular_hertz(Avbd3dJointId id, float hertz);

/* Set the spring angular damping ratio. Use 1.0 for critical damping. */
AVBD3D_API Avbd3dResult avbd3d_motor_joint_set_angular_damping_ratio(Avbd3dJointId id, float damping);

/* Set the maximum spring force in newtons. */
AVBD3D_API Avbd3dResult avbd3d_motor_joint_set_max_spring_force(Avbd3dJointId id, float maxForce);

/* Set the maximum spring torque in newtons * meters */
AVBD3D_API Avbd3dResult avbd3d_motor_joint_set_max_spring_torque(Avbd3dJointId id, float maxTorque);

/* Create a filter joint.
   @see Avbd3dFilterJointDef for details */
AVBD3D_API Avbd3dResult avbd3d_create_filter_joint(Avbd3dWorld *world, const Avbd3dFilterJointDef *def,
                                                   Avbd3dJointId *out_id);

/* Create a parallel joint
   @see Avbd3dParallelJointDef for details */
AVBD3D_API Avbd3dResult avbd3d_create_parallel_joint(Avbd3dWorld *world, const Avbd3dParallelJointDef *def,
                                                     Avbd3dJointId *out_id);

/* Set the spring stiffness in Hertz */
AVBD3D_API Avbd3dResult avbd3d_parallel_joint_set_spring_hertz(Avbd3dJointId id, float hertz);

/* Set the spring damping ratio, non-dimensional */
AVBD3D_API Avbd3dResult avbd3d_parallel_joint_set_spring_damping_ratio(Avbd3dJointId id, float dampingRatio);

/* Set the maximum spring torque, usually in newton-meters */
AVBD3D_API Avbd3dResult avbd3d_parallel_joint_set_max_torque(Avbd3dJointId id, float force);

/* Create a prismatic (slider) joint.
   @see Avbd3dPrismaticJointDef for details */
AVBD3D_API Avbd3dResult avbd3d_create_prismatic_joint(Avbd3dWorld *world, const Avbd3dPrismaticJointDef *def,
                                                      Avbd3dJointId *out_id);

/* Enable/disable the joint spring. */
AVBD3D_API Avbd3dResult avbd3d_prismatic_joint_enable_spring(Avbd3dJointId id, int32_t enableSpring);

/* Set the prismatic joint stiffness in Hertz.
   This should usually be less than a quarter of the simulation rate. For example, if the simulation
   runs at 60Hz then the joint stiffness should be 15Hz or less. */
AVBD3D_API Avbd3dResult avbd3d_prismatic_joint_set_spring_hertz(Avbd3dJointId id, float hertz);

/* Set the prismatic joint damping ratio (non-dimensional) */
AVBD3D_API Avbd3dResult avbd3d_prismatic_joint_set_spring_damping_ratio(Avbd3dJointId id, float dampingRatio);

/* Set the prismatic joint target translation. Usually in meters. */
AVBD3D_API Avbd3dResult avbd3d_prismatic_joint_set_target_translation(Avbd3dJointId id,
                                                                      float targetTranslation);

/* Enable/disable a prismatic joint limit */
AVBD3D_API Avbd3dResult avbd3d_prismatic_joint_enable_limit(Avbd3dJointId id, int32_t enableLimit);

/* Set the prismatic joint limits */
AVBD3D_API Avbd3dResult avbd3d_prismatic_joint_set_limits(Avbd3dJointId id, float lower, float upper);

/* Enable/disable a prismatic joint motor */
AVBD3D_API Avbd3dResult avbd3d_prismatic_joint_enable_motor(Avbd3dJointId id, int32_t enableMotor);

/* Set the prismatic joint motor speed, usually in meters per second */
AVBD3D_API Avbd3dResult avbd3d_prismatic_joint_set_motor_speed(Avbd3dJointId id, float motorSpeed);

/* Set the prismatic joint maximum motor force, usually in newtons */
AVBD3D_API Avbd3dResult avbd3d_prismatic_joint_set_max_motor_force(Avbd3dJointId id, float force);

/* Get the prismatic joint motor force in newtons (positive pushes body B along frame A x), from the newest finished step. */
AVBD3D_API Avbd3dResult avbd3d_prismatic_joint_get_motor_force(Avbd3dJointId id, float *out);

/* Get the current joint translation in meters: anchor B minus anchor A projected on frame A x. */
AVBD3D_API Avbd3dResult avbd3d_prismatic_joint_get_translation(Avbd3dJointId id, float *out);

/* Get the current joint translation speed in meters per second. */
AVBD3D_API Avbd3dResult avbd3d_prismatic_joint_get_speed(Avbd3dJointId id, float *out);

/* Create a revolute joint
   @see Avbd3dRevoluteJointDef for details */
AVBD3D_API Avbd3dResult avbd3d_create_revolute_joint(Avbd3dWorld *world, const Avbd3dRevoluteJointDef *def,
                                                     Avbd3dJointId *out_id);

/* Enable/disable the revolute joint spring */
AVBD3D_API Avbd3dResult avbd3d_revolute_joint_enable_spring(Avbd3dJointId id, int32_t enableSpring);

/* Set the revolute joint spring stiffness in Hertz */
AVBD3D_API Avbd3dResult avbd3d_revolute_joint_set_spring_hertz(Avbd3dJointId id, float hertz);

/* Set the revolute joint spring damping ratio, non-dimensional */
AVBD3D_API Avbd3dResult avbd3d_revolute_joint_set_spring_damping_ratio(Avbd3dJointId id, float dampingRatio);

/* Set the revolute joint target angle in radians */
AVBD3D_API Avbd3dResult avbd3d_revolute_joint_set_target_angle(Avbd3dJointId id, float targetRadians);

/* Enable/disable the revolute joint limit */
AVBD3D_API Avbd3dResult avbd3d_revolute_joint_enable_limit(Avbd3dJointId id, int32_t enableLimit);

/* Set the revolute joint limits in radians */
AVBD3D_API Avbd3dResult avbd3d_revolute_joint_set_limits(Avbd3dJointId id, float lowerLimitRadians,
                                                         float upperLimitRadians);

/* Enable/disable a revolute joint motor */
AVBD3D_API Avbd3dResult avbd3d_revolute_joint_enable_motor(Avbd3dJointId id, int32_t enableMotor);

/* Set the revolute joint motor speed in radians per second */
AVBD3D_API Avbd3dResult avbd3d_revolute_joint_set_motor_speed(Avbd3dJointId id, float motorSpeed);

/* Set the revolute joint maximum motor torque, usually in newton-meters */
AVBD3D_API Avbd3dResult avbd3d_revolute_joint_set_max_motor_torque(Avbd3dJointId id, float torque);

/* Get the revolute joint motor torque in newton-meters (positive drives body B about frame A z), from the newest finished step. */
AVBD3D_API Avbd3dResult avbd3d_revolute_joint_get_motor_torque(Avbd3dJointId id, float *out);

/* Get the current joint angle in radians, B relative to A about the hinge axis, in [-pi, pi]. */
AVBD3D_API Avbd3dResult avbd3d_revolute_joint_get_angle(Avbd3dJointId id, float *out);

/* Get the current joint angular speed in radians per second about the hinge axis. */
AVBD3D_API Avbd3dResult avbd3d_revolute_joint_get_speed(Avbd3dJointId id, float *out);

/* Create a spherical joint
   @see Avbd3dSphericalJointDef for details */
AVBD3D_API Avbd3dResult avbd3d_create_spherical_joint(Avbd3dWorld *world, const Avbd3dSphericalJointDef *def,
                                                      Avbd3dJointId *out_id);

/* Enable/disable the spherical joint cone limit */
AVBD3D_API Avbd3dResult avbd3d_spherical_joint_enable_cone_limit(Avbd3dJointId id, int32_t enableLimit);

/* Set the spherical joint limits in radians */
AVBD3D_API Avbd3dResult avbd3d_spherical_joint_set_cone_limit(Avbd3dJointId id, float angleRadians);

/* Enable/disable the spherical joint limit */
AVBD3D_API Avbd3dResult avbd3d_spherical_joint_enable_twist_limit(Avbd3dJointId id, int32_t enableLimit);

/* Set the spherical joint limits in radians */
AVBD3D_API Avbd3dResult avbd3d_spherical_joint_set_twist_limits(Avbd3dJointId id, float lowerLimitRadians,
                                                                float upperLimitRadians);

/* Enable/disable the spherical joint spring */
AVBD3D_API Avbd3dResult avbd3d_spherical_joint_enable_spring(Avbd3dJointId id, int32_t enableSpring);

/* Set the spherical joint spring stiffness in Hertz */
AVBD3D_API Avbd3dResult avbd3d_spherical_joint_set_spring_hertz(Avbd3dJointId id, float hertz);

/* Set the spherical joint spring damping ratio, non-dimensional */
AVBD3D_API Avbd3dResult avbd3d_spherical_joint_set_spring_damping_ratio(Avbd3dJointId id, float dampingRatio);

/* Enable/disable a spherical joint motor */
AVBD3D_API Avbd3dResult avbd3d_spherical_joint_enable_motor(Avbd3dJointId id, int32_t enableMotor);

/* Set the spherical joint motor velocity in radians per second. This is the relative angular
   velocity between the two bodies in world space.
   motorVelocity = angularVelocityB - angularVelocityA */
AVBD3D_API Avbd3dResult avbd3d_spherical_joint_set_motor_velocity(Avbd3dJointId id, Avbd3dVec3 motorVelocity);

/* Set the spherical joint maximum motor torque, usually in newton-meters */
AVBD3D_API Avbd3dResult avbd3d_spherical_joint_set_max_motor_torque(Avbd3dJointId id, float torque);

/* Get the spherical joint motor torque in newton-meters, in joint frame A (x, y swing, z twist), from the newest finished step. */
AVBD3D_API Avbd3dResult avbd3d_spherical_joint_get_motor_torque(Avbd3dJointId id, Avbd3dVec3 *out);

/* Get the current swing (cone) angle in radians, in [0, pi]. */
AVBD3D_API Avbd3dResult avbd3d_spherical_joint_get_cone_angle(Avbd3dJointId id, float *out);

/* Get the current twist angle in radians, in [-pi, pi]. */
AVBD3D_API Avbd3dResult avbd3d_spherical_joint_get_twist_angle(Avbd3dJointId id, float *out);

/* Set the spring target rotation, joint frame B relative to joint frame A. */
AVBD3D_API Avbd3dResult avbd3d_spherical_joint_set_target_rotation(Avbd3dJointId id,
                                                                   Avbd3dQuat targetRotation);

/* Create a weld joint
   @see Avbd3dWeldJointDef for details */
AVBD3D_API Avbd3dResult avbd3d_create_weld_joint(Avbd3dWorld *world, const Avbd3dWeldJointDef *def,
                                                 Avbd3dJointId *out_id);

/* Set the weld joint linear stiffness in Hertz. 0 is rigid. */
AVBD3D_API Avbd3dResult avbd3d_weld_joint_set_linear_hertz(Avbd3dJointId id, float hertz);

/* Set the weld joint linear damping ratio (non-dimensional) */
AVBD3D_API Avbd3dResult avbd3d_weld_joint_set_linear_damping_ratio(Avbd3dJointId id, float dampingRatio);

/* Set the weld joint angular stiffness in Hertz. 0 is rigid. */
AVBD3D_API Avbd3dResult avbd3d_weld_joint_set_angular_hertz(Avbd3dJointId id, float hertz);

/* Set weld joint angular damping ratio, non-dimensional */
AVBD3D_API Avbd3dResult avbd3d_weld_joint_set_angular_damping_ratio(Avbd3dJointId id, float dampingRatio);

/* Create a wheel joint.
   @see Avbd3dWheelJointDef for details. */
AVBD3D_API Avbd3dResult avbd3d_create_wheel_joint(Avbd3dWorld *world, const Avbd3dWheelJointDef *def,
                                                  Avbd3dJointId *out_id);

/* Enable/disable the wheel joint spring. */
AVBD3D_API Avbd3dResult avbd3d_wheel_joint_enable_suspension(Avbd3dJointId id, int32_t flag);

/* Set the wheel joint stiffness in Hertz. */
AVBD3D_API Avbd3dResult avbd3d_wheel_joint_set_suspension_hertz(Avbd3dJointId id, float hertz);

/* Set the wheel joint damping ratio, non-dimensional. */
AVBD3D_API Avbd3dResult avbd3d_wheel_joint_set_suspension_damping_ratio(Avbd3dJointId id, float dampingRatio);

/* Enable/disable the wheel joint limit. */
AVBD3D_API Avbd3dResult avbd3d_wheel_joint_enable_suspension_limit(Avbd3dJointId id, int32_t flag);

/* Set the wheel joint limits. */
AVBD3D_API Avbd3dResult avbd3d_wheel_joint_set_suspension_limits(Avbd3dJointId id, float lower, float upper);

/* Enable/disable the wheel joint motor. */
AVBD3D_API Avbd3dResult avbd3d_wheel_joint_enable_spin_motor(Avbd3dJointId id, int32_t flag);

/* Set the wheel joint motor speed in radians per second. */
AVBD3D_API Avbd3dResult avbd3d_wheel_joint_set_spin_motor_speed(Avbd3dJointId id, float speed);

/* Set the wheel joint maximum motor torque, usually in newton-meters. */
AVBD3D_API Avbd3dResult avbd3d_wheel_joint_set_max_spin_torque(Avbd3dJointId id, float torque);

/* Get the wheel joint spin motor torque in newton-meters, from the newest finished step. */
AVBD3D_API Avbd3dResult avbd3d_wheel_joint_get_spin_torque(Avbd3dJointId id, float *out);

/* Get the current wheel spin speed in radians per second about the frame B z axis. */
AVBD3D_API Avbd3dResult avbd3d_wheel_joint_get_spin_speed(Avbd3dJointId id, float *out);

/* Get the current steering angle in radians about the suspension axis. */
AVBD3D_API Avbd3dResult avbd3d_wheel_joint_get_steering_angle(Avbd3dJointId id, float *out);

/* Enable/disable wheel steering. Steering allows the wheel to rotate about the suspension axis. */
AVBD3D_API Avbd3dResult avbd3d_wheel_joint_enable_steering(Avbd3dJointId id, int32_t flag);

/* Set the wheel joint steering stiffness in Hertz. */
AVBD3D_API Avbd3dResult avbd3d_wheel_joint_set_steering_hertz(Avbd3dJointId id, float hertz);

/* Set the wheel joint steering damping ratio, non-dimensional. */
AVBD3D_API Avbd3dResult avbd3d_wheel_joint_set_steering_damping_ratio(Avbd3dJointId id, float dampingRatio);

/* Set the wheel joint maximum steering torque in N*m. */
AVBD3D_API Avbd3dResult avbd3d_wheel_joint_set_max_steering_torque(Avbd3dJointId id, float torque);

/* Enable/disable the wheel joint steering limit. */
AVBD3D_API Avbd3dResult avbd3d_wheel_joint_enable_steering_limit(Avbd3dJointId id, int32_t flag);

/* Set the wheel joint steering limits in radians. */
AVBD3D_API Avbd3dResult avbd3d_wheel_joint_set_steering_limits(Avbd3dJointId id, float lowerRadians,
                                                               float upperRadians);

/* Set the wheel joint target steering angle in radians. */
AVBD3D_API Avbd3dResult avbd3d_wheel_joint_set_target_steering_angle(Avbd3dJointId id, float radians);

/* ---------------------------------------------------------------------------
   Generic joints and springs
   --------------------------------------------------------------------------- */

AVBD3D_API Avbd3dResult avbd3d_create_generic_joint(Avbd3dWorld *world, const Avbd3dGenericJointDef *def,
                                                    Avbd3dJointId *out_id);

AVBD3D_API Avbd3dResult avbd3d_create_spring(Avbd3dWorld *world, const Avbd3dSpringDef *def,
                                             Avbd3dJointId *out_id);

/* ---------------------------------------------------------------------------
   Contacts
   --------------------------------------------------------------------------- */

/* Get the manifolds for a contact. The manifold may have no points if the contact is not touching. */
AVBD3D_API Avbd3dResult avbd3d_contact_get_data(Avbd3dContactId id, Avbd3dContactData *out);

/* 1 if id names a live contact of a live world. */
AVBD3D_API int32_t avbd3d_contact_is_valid(Avbd3dContactId id);

/* ---------------------------------------------------------------------------
   Grab
   --------------------------------------------------------------------------- */

/* Choose the bodies a grab at `seed` holds: the seed itself plus, when `followConnections`, every
   body reachable from it through joints and springs whose position lies within `radius` of
   `center`. Statics are never included.
   @return the number of ids written to `outBodies` (at most `maxCount`). */
AVBD3D_API Avbd3dResult avbd3d_grab_begin(Avbd3dWorld *world, Avbd3dBodyId seed, Avbd3dVec3 center,
                                          float radius, int32_t followConnections, Avbd3dBodyId *outBodies,
                                          int32_t maxCount, int32_t *out_count);

/* Which members of a live grab group are still connected to `seed` after joints may have broken.
   Writes 1 or 0 into `outKeep[i]` for each of `groupCount` entries.
   @return how many survived. */
AVBD3D_API Avbd3dResult avbd3d_grab_survivors(Avbd3dWorld *world, Avbd3dBodyId seed,
                                              const Avbd3dBodyId *group, int32_t groupCount,
                                              uint8_t *outKeep, int32_t *out_count);

/* Drive the grab. Call once per frame the grab is held; `count` is clamped to
   AVBD3D_MAX_GRAB_BODIES. */
AVBD3D_API Avbd3dResult avbd3d_grab_move(Avbd3dWorld *world, const Avbd3dGrabHandle *handles, int32_t count);

/* End the grab. Safe to call when nothing is held. */
AVBD3D_API Avbd3dResult avbd3d_grab_end(Avbd3dWorld *world);

#ifdef __cplusplus
}
#endif

#endif /* AVBD3D_H */
