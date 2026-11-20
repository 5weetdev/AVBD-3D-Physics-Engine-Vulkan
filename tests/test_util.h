// Test helpers: value-returning shims over the public avbd3d calls. Each t3d_<call> checks that the
// call returned AVBD3D_OK and aborts the test with a message otherwise; the raw call is used wherever
// a test checks an error result. Generated shape, kept by hand.
#pragma once

#include "avbd3d/avbd3d.h"
#include "avbd3d/avbd3d_geometry.h"
#include "avbd3d/avbd3d_vulkan.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static inline void t3d_check(Avbd3dResult r, const char *what)
{
    if (r != AVBD3D_OK)
    {
        std::fprintf(stderr, "FAIL: %s returned %s\n", what, avbd3d_result_string(r));
        std::exit(1);
    }
}

static inline Avbd3dWorld *t3d_create_world(const Avbd3dWorldDef *def)
{
    Avbd3dWorld *v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_create_world(def, &v), "avbd3d_create_world");
    return v;
}

static inline void t3d_destroy_world(Avbd3dWorld *world)
{
    t3d_check(avbd3d_destroy_world(world), "avbd3d_destroy_world");
}

static inline void t3d_step(Avbd3dWorld *world, float dt)
{
    t3d_check(avbd3d_step(world, dt), "avbd3d_step");
}

static inline void t3d_step_begin(Avbd3dWorld *world, float dt)
{
    t3d_check(avbd3d_step_begin(world, dt), "avbd3d_step_begin");
}

static inline void t3d_step_end(Avbd3dWorld *world)
{
    t3d_check(avbd3d_step_end(world), "avbd3d_step_end");
}

static inline Avbd3dStats t3d_get_stats(Avbd3dWorld *world)
{
    Avbd3dStats v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_get_stats(world, &v), "avbd3d_get_stats");
    return v;
}

static inline Avbd3dCapabilities t3d_get_capabilities(Avbd3dWorld *world)
{
    Avbd3dCapabilities v;
    std::memset(&v, 0, sizeof v);
    v.struct_size = sizeof v;
    t3d_check(avbd3d_get_capabilities(world, &v), "avbd3d_get_capabilities");
    return v;
}

static inline void t3d_world_draw(Avbd3dWorld *world, const Avbd3dDebugDraw *draw, uint64_t maskBits)
{
    t3d_check(avbd3d_world_draw(world, draw, maskBits), "avbd3d_world_draw");
}

static inline Avbd3dAABB t3d_world_get_bounds(Avbd3dWorld *world)
{
    Avbd3dAABB v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_world_get_bounds(world, &v), "avbd3d_world_get_bounds");
    return v;
}

static inline void t3d_world_enable_sleeping(Avbd3dWorld *world, int32_t flag)
{
    t3d_check(avbd3d_world_enable_sleeping(world, flag), "avbd3d_world_enable_sleeping");
}

static inline void t3d_world_enable_continuous(Avbd3dWorld *world, int32_t flag)
{
    t3d_check(avbd3d_world_enable_continuous(world, flag), "avbd3d_world_enable_continuous");
}

static inline void t3d_world_set_restitution_threshold(Avbd3dWorld *world, float value)
{
    t3d_check(avbd3d_world_set_restitution_threshold(world, value), "avbd3d_world_set_restitution_threshold");
}

static inline void t3d_world_set_hit_event_threshold(Avbd3dWorld *world, float value)
{
    t3d_check(avbd3d_world_set_hit_event_threshold(world, value), "avbd3d_world_set_hit_event_threshold");
}

static inline void t3d_world_set_gravity(Avbd3dWorld *world, Avbd3dVec3 gravity)
{
    t3d_check(avbd3d_world_set_gravity(world, gravity), "avbd3d_world_set_gravity");
}

static inline Avbd3dVec3 t3d_world_get_gravity(Avbd3dWorld *world)
{
    Avbd3dVec3 v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_world_get_gravity(world, &v), "avbd3d_world_get_gravity");
    return v;
}

static inline void t3d_world_explode(Avbd3dWorld *world, const Avbd3dExplosionDef *explosionDef)
{
    t3d_check(avbd3d_world_explode(world, explosionDef), "avbd3d_world_explode");
}

static inline int32_t t3d_is_step_in_flight(Avbd3dWorld *world)
{
    int32_t v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_is_step_in_flight(world, &v), "avbd3d_is_step_in_flight");
    return v;
}

static inline void t3d_world_set_solver_def(Avbd3dWorld *world, const Avbd3dSolverDef *def)
{
    t3d_check(avbd3d_world_set_solver_def(world, def), "avbd3d_world_set_solver_def");
}

static inline void t3d_world_set_friction_mixing(Avbd3dWorld *world, Avbd3dMixingRule rule)
{
    t3d_check(avbd3d_world_set_friction_mixing(world, rule), "avbd3d_world_set_friction_mixing");
}

static inline void t3d_world_set_restitution_mixing(Avbd3dWorld *world, Avbd3dMixingRule rule)
{
    t3d_check(avbd3d_world_set_restitution_mixing(world, rule), "avbd3d_world_set_restitution_mixing");
}

static inline Avbd3dBodyEvents t3d_world_get_body_events(Avbd3dWorld *world)
{
    Avbd3dBodyEvents v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_world_get_body_events(world, &v), "avbd3d_world_get_body_events");
    return v;
}

static inline Avbd3dSensorEvents t3d_world_get_sensor_events(Avbd3dWorld *world)
{
    Avbd3dSensorEvents v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_world_get_sensor_events(world, &v), "avbd3d_world_get_sensor_events");
    return v;
}

static inline Avbd3dContactEvents t3d_world_get_contact_events(Avbd3dWorld *world)
{
    Avbd3dContactEvents v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_world_get_contact_events(world, &v), "avbd3d_world_get_contact_events");
    return v;
}

static inline Avbd3dJointEvents t3d_world_get_joint_events(Avbd3dWorld *world)
{
    Avbd3dJointEvents v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_world_get_joint_events(world, &v), "avbd3d_world_get_joint_events");
    return v;
}

static inline Avbd3dRayResult t3d_world_cast_ray_closest(Avbd3dWorld *world, Avbd3dVec3 origin, Avbd3dVec3 translation,
                                                         Avbd3dQueryFilter filter)
{
    Avbd3dRayResult v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_world_cast_ray_closest(world, origin, translation, filter, &v), "avbd3d_world_cast_ray_closest");
    return v;
}

static inline Avbd3dBodyId t3d_create_body(Avbd3dWorld *world, const Avbd3dBodyDef *def)
{
    Avbd3dBodyId v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_create_body(world, def, &v), "avbd3d_create_body");
    return v;
}

static inline void t3d_destroy_body(Avbd3dBodyId bodyId)
{
    t3d_check(avbd3d_destroy_body(bodyId), "avbd3d_destroy_body");
}

static inline Avbd3dBodyType t3d_body_get_type(Avbd3dBodyId id)
{
    Avbd3dBodyType v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_get_type(id, &v), "avbd3d_body_get_type");
    return v;
}

static inline void t3d_body_set_type(Avbd3dBodyId id, Avbd3dBodyType type)
{
    t3d_check(avbd3d_body_set_type(id, type), "avbd3d_body_set_type");
}

static inline void t3d_body_set_user_data(Avbd3dBodyId id, void *userData)
{
    t3d_check(avbd3d_body_set_user_data(id, userData), "avbd3d_body_set_user_data");
}

static inline void *t3d_body_get_user_data(Avbd3dBodyId id)
{
    void *v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_get_user_data(id, &v), "avbd3d_body_get_user_data");
    return v;
}

static inline Avbd3dVec3 t3d_body_get_position(Avbd3dBodyId id)
{
    Avbd3dVec3 v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_get_position(id, &v), "avbd3d_body_get_position");
    return v;
}

static inline Avbd3dQuat t3d_body_get_rotation(Avbd3dBodyId id)
{
    Avbd3dQuat v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_get_rotation(id, &v), "avbd3d_body_get_rotation");
    return v;
}

static inline Avbd3dTransform t3d_body_get_transform(Avbd3dBodyId id)
{
    Avbd3dTransform v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_get_transform(id, &v), "avbd3d_body_get_transform");
    return v;
}

static inline void t3d_body_set_transform(Avbd3dBodyId id, Avbd3dVec3 position, Avbd3dQuat rotation)
{
    t3d_check(avbd3d_body_set_transform(id, position, rotation), "avbd3d_body_set_transform");
}

static inline Avbd3dVec3 t3d_body_get_local_point(Avbd3dBodyId id, Avbd3dVec3 worldPoint)
{
    Avbd3dVec3 v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_get_local_point(id, worldPoint, &v), "avbd3d_body_get_local_point");
    return v;
}

static inline Avbd3dVec3 t3d_body_get_world_point(Avbd3dBodyId id, Avbd3dVec3 localPoint)
{
    Avbd3dVec3 v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_get_world_point(id, localPoint, &v), "avbd3d_body_get_world_point");
    return v;
}

static inline Avbd3dVec3 t3d_body_get_local_vector(Avbd3dBodyId id, Avbd3dVec3 worldVector)
{
    Avbd3dVec3 v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_get_local_vector(id, worldVector, &v), "avbd3d_body_get_local_vector");
    return v;
}

static inline Avbd3dVec3 t3d_body_get_world_vector(Avbd3dBodyId id, Avbd3dVec3 localVector)
{
    Avbd3dVec3 v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_get_world_vector(id, localVector, &v), "avbd3d_body_get_world_vector");
    return v;
}

static inline Avbd3dVec3 t3d_body_get_linear_velocity(Avbd3dBodyId id)
{
    Avbd3dVec3 v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_get_linear_velocity(id, &v), "avbd3d_body_get_linear_velocity");
    return v;
}

static inline Avbd3dVec3 t3d_body_get_angular_velocity(Avbd3dBodyId id)
{
    Avbd3dVec3 v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_get_angular_velocity(id, &v), "avbd3d_body_get_angular_velocity");
    return v;
}

static inline void t3d_body_set_linear_velocity(Avbd3dBodyId id, Avbd3dVec3 linearVelocity)
{
    t3d_check(avbd3d_body_set_linear_velocity(id, linearVelocity), "avbd3d_body_set_linear_velocity");
}

static inline void t3d_body_set_angular_velocity(Avbd3dBodyId id, Avbd3dVec3 angularVelocity)
{
    t3d_check(avbd3d_body_set_angular_velocity(id, angularVelocity), "avbd3d_body_set_angular_velocity");
}

static inline void t3d_body_set_target_transform(Avbd3dBodyId id, Avbd3dTransform target, float timeStep, int32_t wake)
{
    t3d_check(avbd3d_body_set_target_transform(id, target, timeStep, wake), "avbd3d_body_set_target_transform");
}

static inline Avbd3dVec3 t3d_body_get_local_point_velocity(Avbd3dBodyId id, Avbd3dVec3 localPoint)
{
    Avbd3dVec3 v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_get_local_point_velocity(id, localPoint, &v), "avbd3d_body_get_local_point_velocity");
    return v;
}

static inline Avbd3dVec3 t3d_body_get_world_point_velocity(Avbd3dBodyId id, Avbd3dVec3 worldPoint)
{
    Avbd3dVec3 v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_get_world_point_velocity(id, worldPoint, &v), "avbd3d_body_get_world_point_velocity");
    return v;
}

static inline void t3d_body_apply_force(Avbd3dBodyId id, Avbd3dVec3 force, Avbd3dVec3 point, int32_t wake)
{
    t3d_check(avbd3d_body_apply_force(id, force, point, wake), "avbd3d_body_apply_force");
}

static inline void t3d_body_apply_force_to_center(Avbd3dBodyId id, Avbd3dVec3 force, int32_t wake)
{
    t3d_check(avbd3d_body_apply_force_to_center(id, force, wake), "avbd3d_body_apply_force_to_center");
}

static inline void t3d_body_apply_torque(Avbd3dBodyId id, Avbd3dVec3 torque, int32_t wake)
{
    t3d_check(avbd3d_body_apply_torque(id, torque, wake), "avbd3d_body_apply_torque");
}

static inline void t3d_body_apply_linear_impulse(Avbd3dBodyId id, Avbd3dVec3 impulse, Avbd3dVec3 point, int32_t wake)
{
    t3d_check(avbd3d_body_apply_linear_impulse(id, impulse, point, wake), "avbd3d_body_apply_linear_impulse");
}

static inline void t3d_body_apply_linear_impulse_to_center(Avbd3dBodyId id, Avbd3dVec3 impulse, int32_t wake)
{
    t3d_check(avbd3d_body_apply_linear_impulse_to_center(id, impulse, wake),
              "avbd3d_body_apply_linear_impulse_to_center");
}

static inline void t3d_body_apply_angular_impulse(Avbd3dBodyId id, Avbd3dVec3 impulse, int32_t wake)
{
    t3d_check(avbd3d_body_apply_angular_impulse(id, impulse, wake), "avbd3d_body_apply_angular_impulse");
}

static inline float t3d_body_get_mass(Avbd3dBodyId id)
{
    float v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_get_mass(id, &v), "avbd3d_body_get_mass");
    return v;
}

static inline Avbd3dMatrix3 t3d_body_get_local_rotational_inertia(Avbd3dBodyId id)
{
    Avbd3dMatrix3 v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_get_local_rotational_inertia(id, &v), "avbd3d_body_get_local_rotational_inertia");
    return v;
}

static inline float t3d_body_get_inverse_mass(Avbd3dBodyId id)
{
    float v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_get_inverse_mass(id, &v), "avbd3d_body_get_inverse_mass");
    return v;
}

static inline Avbd3dMatrix3 t3d_body_get_world_inverse_rotational_inertia(Avbd3dBodyId id)
{
    Avbd3dMatrix3 v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_get_world_inverse_rotational_inertia(id, &v),
              "avbd3d_body_get_world_inverse_rotational_inertia");
    return v;
}

static inline Avbd3dVec3 t3d_body_get_local_center(Avbd3dBodyId id)
{
    Avbd3dVec3 v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_get_local_center(id, &v), "avbd3d_body_get_local_center");
    return v;
}

static inline Avbd3dVec3 t3d_body_get_world_center(Avbd3dBodyId id)
{
    Avbd3dVec3 v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_get_world_center(id, &v), "avbd3d_body_get_world_center");
    return v;
}

static inline void t3d_body_set_mass_data(Avbd3dBodyId id, Avbd3dMassData massData)
{
    t3d_check(avbd3d_body_set_mass_data(id, massData), "avbd3d_body_set_mass_data");
}

static inline Avbd3dMassData t3d_body_get_mass_data(Avbd3dBodyId id)
{
    Avbd3dMassData v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_get_mass_data(id, &v), "avbd3d_body_get_mass_data");
    return v;
}

static inline void t3d_body_apply_mass_from_shapes(Avbd3dBodyId id)
{
    t3d_check(avbd3d_body_apply_mass_from_shapes(id), "avbd3d_body_apply_mass_from_shapes");
}

static inline void t3d_body_set_linear_damping(Avbd3dBodyId id, float linearDamping)
{
    t3d_check(avbd3d_body_set_linear_damping(id, linearDamping), "avbd3d_body_set_linear_damping");
}

static inline void t3d_body_set_angular_damping(Avbd3dBodyId id, float angularDamping)
{
    t3d_check(avbd3d_body_set_angular_damping(id, angularDamping), "avbd3d_body_set_angular_damping");
}

static inline void t3d_body_set_gravity_scale(Avbd3dBodyId id, float gravityScale)
{
    t3d_check(avbd3d_body_set_gravity_scale(id, gravityScale), "avbd3d_body_set_gravity_scale");
}

static inline int32_t t3d_body_is_awake(Avbd3dBodyId id)
{
    int32_t v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_is_awake(id, &v), "avbd3d_body_is_awake");
    return v;
}

static inline void t3d_body_set_awake(Avbd3dBodyId id, int32_t awake)
{
    t3d_check(avbd3d_body_set_awake(id, awake), "avbd3d_body_set_awake");
}

static inline int32_t t3d_body_is_enabled(Avbd3dBodyId id)
{
    int32_t v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_is_enabled(id, &v), "avbd3d_body_is_enabled");
    return v;
}

static inline void t3d_body_disable(Avbd3dBodyId id)
{
    t3d_check(avbd3d_body_disable(id), "avbd3d_body_disable");
}

static inline void t3d_body_enable(Avbd3dBodyId id)
{
    t3d_check(avbd3d_body_enable(id), "avbd3d_body_enable");
}

static inline void t3d_body_set_motion_locks(Avbd3dBodyId id, Avbd3dMotionLocks locks)
{
    t3d_check(avbd3d_body_set_motion_locks(id, locks), "avbd3d_body_set_motion_locks");
}

static inline void t3d_body_set_bullet(Avbd3dBodyId id, int32_t flag)
{
    t3d_check(avbd3d_body_set_bullet(id, flag), "avbd3d_body_set_bullet");
}

static inline void t3d_body_enable_hit_events(Avbd3dBodyId id, int32_t flag)
{
    t3d_check(avbd3d_body_enable_hit_events(id, flag), "avbd3d_body_enable_hit_events");
}

static inline Avbd3dWorld *t3d_body_get_world(Avbd3dBodyId id)
{
    Avbd3dWorld *v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_get_world(id, &v), "avbd3d_body_get_world");
    return v;
}

static inline int32_t t3d_body_get_shape_count(Avbd3dBodyId id)
{
    int32_t v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_get_shape_count(id, &v), "avbd3d_body_get_shape_count");
    return v;
}

static inline int32_t t3d_body_get_joint_count(Avbd3dBodyId id)
{
    int32_t v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_get_joint_count(id, &v), "avbd3d_body_get_joint_count");
    return v;
}

static inline int32_t t3d_body_get_contact_capacity(Avbd3dBodyId id)
{
    int32_t v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_get_contact_capacity(id, &v), "avbd3d_body_get_contact_capacity");
    return v;
}

static inline Avbd3dAABB t3d_body_compute_aabb(Avbd3dBodyId id)
{
    Avbd3dAABB v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_compute_aabb(id, &v), "avbd3d_body_compute_aabb");
    return v;
}

static inline Avbd3dBodyCastResult t3d_body_cast_ray(Avbd3dBodyId id, Avbd3dVec3 origin, Avbd3dVec3 translation,
                                                     Avbd3dQueryFilter filter, float maxFraction,
                                                     Avbd3dTransform bodyTransform)
{
    Avbd3dBodyCastResult v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_cast_ray(id, origin, translation, filter, maxFraction, bodyTransform, &v),
              "avbd3d_body_cast_ray");
    return v;
}

static inline Avbd3dBodyCastResult t3d_body_cast_shape(Avbd3dBodyId id, Avbd3dVec3 origin,
                                                       const Avbd3dShapeProxy *proxy, Avbd3dVec3 translation,
                                                       Avbd3dQueryFilter filter, float maxFraction, int32_t canEncroach,
                                                       Avbd3dTransform bodyTransform)
{
    Avbd3dBodyCastResult v;
    std::memset(&v, 0, sizeof v);
    t3d_check(
        avbd3d_body_cast_shape(id, origin, proxy, translation, filter, maxFraction, canEncroach, bodyTransform, &v),
        "avbd3d_body_cast_shape");
    return v;
}

static inline int32_t t3d_body_overlap_shape(Avbd3dBodyId id, Avbd3dVec3 origin, const Avbd3dShapeProxy *proxy,
                                             Avbd3dQueryFilter filter, Avbd3dTransform bodyTransform)
{
    int32_t v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_overlap_shape(id, origin, proxy, filter, bodyTransform, &v), "avbd3d_body_overlap_shape");
    return v;
}

static inline Avbd3dShapeId t3d_create_sphere_shape(Avbd3dBodyId bodyId, const Avbd3dShapeDef *def,
                                                    const Avbd3dSphere *sphere)
{
    Avbd3dShapeId v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_create_sphere_shape(bodyId, def, sphere, &v), "avbd3d_create_sphere_shape");
    return v;
}

static inline Avbd3dShapeId t3d_create_capsule_shape(Avbd3dBodyId bodyId, const Avbd3dShapeDef *def,
                                                     const Avbd3dCapsule *capsule)
{
    Avbd3dShapeId v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_create_capsule_shape(bodyId, def, capsule, &v), "avbd3d_create_capsule_shape");
    return v;
}

static inline Avbd3dShapeId t3d_create_hull_shape(Avbd3dBodyId bodyId, const Avbd3dShapeDef *def,
                                                  const Avbd3dHullData *hull)
{
    Avbd3dShapeId v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_create_hull_shape(bodyId, def, hull, &v), "avbd3d_create_hull_shape");
    return v;
}

static inline Avbd3dShapeId t3d_create_transformed_hull_shape(Avbd3dBodyId bodyId, const Avbd3dShapeDef *def,
                                                              const Avbd3dHullData *hull, Avbd3dTransform transform,
                                                              Avbd3dVec3 scale)
{
    Avbd3dShapeId v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_create_transformed_hull_shape(bodyId, def, hull, transform, scale, &v),
              "avbd3d_create_transformed_hull_shape");
    return v;
}

static inline Avbd3dShapeId t3d_create_mesh_shape(Avbd3dBodyId bodyId, const Avbd3dShapeDef *def,
                                                  const Avbd3dMeshData *mesh, Avbd3dVec3 scale)
{
    Avbd3dShapeId v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_create_mesh_shape(bodyId, def, mesh, scale, &v), "avbd3d_create_mesh_shape");
    return v;
}

static inline Avbd3dShapeId t3d_create_height_field_shape(Avbd3dBodyId bodyId, const Avbd3dShapeDef *def,
                                                          const Avbd3dHeightFieldData *heightField)
{
    Avbd3dShapeId v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_create_height_field_shape(bodyId, def, heightField, &v), "avbd3d_create_height_field_shape");
    return v;
}

static inline Avbd3dShapeId t3d_create_baked_compound_shape(Avbd3dBodyId bodyId, Avbd3dShapeDef *def,
                                                            const Avbd3dCompoundData *compound)
{
    Avbd3dShapeId v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_create_baked_compound_shape(bodyId, def, compound, &v), "avbd3d_create_baked_compound_shape");
    return v;
}

static inline void t3d_destroy_shape(Avbd3dShapeId shapeId, int32_t updateBodyMass)
{
    t3d_check(avbd3d_destroy_shape(shapeId, updateBodyMass), "avbd3d_destroy_shape");
}

static inline Avbd3dShapeType t3d_shape_get_type(Avbd3dShapeId id)
{
    Avbd3dShapeType v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_shape_get_type(id, &v), "avbd3d_shape_get_type");
    return v;
}

static inline Avbd3dBodyId t3d_shape_get_body(Avbd3dShapeId id)
{
    Avbd3dBodyId v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_shape_get_body(id, &v), "avbd3d_shape_get_body");
    return v;
}

static inline Avbd3dWorld *t3d_shape_get_world(Avbd3dShapeId id)
{
    Avbd3dWorld *v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_shape_get_world(id, &v), "avbd3d_shape_get_world");
    return v;
}

static inline int32_t t3d_shape_is_sensor(Avbd3dShapeId id)
{
    int32_t v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_shape_is_sensor(id, &v), "avbd3d_shape_is_sensor");
    return v;
}

static inline void t3d_shape_set_user_data(Avbd3dShapeId id, void *userData)
{
    t3d_check(avbd3d_shape_set_user_data(id, userData), "avbd3d_shape_set_user_data");
}

static inline void *t3d_shape_get_user_data(Avbd3dShapeId id)
{
    void *v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_shape_get_user_data(id, &v), "avbd3d_shape_get_user_data");
    return v;
}

static inline void t3d_shape_set_density(Avbd3dShapeId id, float density, int32_t updateBodyMass)
{
    t3d_check(avbd3d_shape_set_density(id, density, updateBodyMass), "avbd3d_shape_set_density");
}

static inline void t3d_shape_set_friction(Avbd3dShapeId id, float friction)
{
    t3d_check(avbd3d_shape_set_friction(id, friction), "avbd3d_shape_set_friction");
}

static inline void t3d_shape_set_restitution(Avbd3dShapeId id, float restitution)
{
    t3d_check(avbd3d_shape_set_restitution(id, restitution), "avbd3d_shape_set_restitution");
}

static inline void t3d_shape_set_surface_material(Avbd3dShapeId id, Avbd3dSurfaceMaterial surfaceMaterial)
{
    t3d_check(avbd3d_shape_set_surface_material(id, surfaceMaterial), "avbd3d_shape_set_surface_material");
}

static inline void t3d_shape_set_filter(Avbd3dShapeId id, Avbd3dFilter filter)
{
    t3d_check(avbd3d_shape_set_filter(id, filter), "avbd3d_shape_set_filter");
}

static inline void t3d_shape_enable_sensor_events(Avbd3dShapeId id, int32_t flag)
{
    t3d_check(avbd3d_shape_enable_sensor_events(id, flag), "avbd3d_shape_enable_sensor_events");
}

static inline void t3d_shape_enable_contact_events(Avbd3dShapeId id, int32_t flag)
{
    t3d_check(avbd3d_shape_enable_contact_events(id, flag), "avbd3d_shape_enable_contact_events");
}

static inline void t3d_shape_enable_hit_events(Avbd3dShapeId id, int32_t flag)
{
    t3d_check(avbd3d_shape_enable_hit_events(id, flag), "avbd3d_shape_enable_hit_events");
}

static inline Avbd3dCastOutput t3d_shape_ray_cast(Avbd3dShapeId id, Avbd3dVec3 origin, Avbd3dVec3 translation)
{
    Avbd3dCastOutput v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_shape_ray_cast(id, origin, translation, &v), "avbd3d_shape_ray_cast");
    return v;
}

static inline int32_t t3d_shape_get_contact_capacity(Avbd3dShapeId id)
{
    int32_t v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_shape_get_contact_capacity(id, &v), "avbd3d_shape_get_contact_capacity");
    return v;
}

static inline int32_t t3d_shape_get_sensor_capacity(Avbd3dShapeId id)
{
    int32_t v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_shape_get_sensor_capacity(id, &v), "avbd3d_shape_get_sensor_capacity");
    return v;
}

static inline Avbd3dAABB t3d_shape_get_aabb(Avbd3dShapeId id)
{
    Avbd3dAABB v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_shape_get_aabb(id, &v), "avbd3d_shape_get_aabb");
    return v;
}

static inline Avbd3dMassData t3d_shape_compute_mass_data(Avbd3dShapeId id)
{
    Avbd3dMassData v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_shape_compute_mass_data(id, &v), "avbd3d_shape_compute_mass_data");
    return v;
}

static inline Avbd3dJointType t3d_joint_get_type(Avbd3dJointId id)
{
    Avbd3dJointType v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_joint_get_type(id, &v), "avbd3d_joint_get_type");
    return v;
}

static inline Avbd3dBodyId t3d_joint_get_body_a(Avbd3dJointId id)
{
    Avbd3dBodyId v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_joint_get_body_a(id, &v), "avbd3d_joint_get_body_a");
    return v;
}

static inline Avbd3dBodyId t3d_joint_get_body_b(Avbd3dJointId id)
{
    Avbd3dBodyId v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_joint_get_body_b(id, &v), "avbd3d_joint_get_body_b");
    return v;
}

static inline Avbd3dWorld *t3d_joint_get_world(Avbd3dJointId id)
{
    Avbd3dWorld *v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_joint_get_world(id, &v), "avbd3d_joint_get_world");
    return v;
}

static inline void t3d_joint_set_local_frame_a(Avbd3dJointId id, Avbd3dTransform localFrame)
{
    t3d_check(avbd3d_joint_set_local_frame_a(id, localFrame), "avbd3d_joint_set_local_frame_a");
}

static inline void t3d_joint_set_local_frame_b(Avbd3dJointId id, Avbd3dTransform localFrame)
{
    t3d_check(avbd3d_joint_set_local_frame_b(id, localFrame), "avbd3d_joint_set_local_frame_b");
}

static inline void t3d_joint_set_collide_connected(Avbd3dJointId id, int32_t shouldCollide)
{
    t3d_check(avbd3d_joint_set_collide_connected(id, shouldCollide), "avbd3d_joint_set_collide_connected");
}

static inline void t3d_joint_set_user_data(Avbd3dJointId id, void *userData)
{
    t3d_check(avbd3d_joint_set_user_data(id, userData), "avbd3d_joint_set_user_data");
}

static inline void *t3d_joint_get_user_data(Avbd3dJointId id)
{
    void *v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_joint_get_user_data(id, &v), "avbd3d_joint_get_user_data");
    return v;
}

static inline void t3d_joint_wake_bodies(Avbd3dJointId id)
{
    t3d_check(avbd3d_joint_wake_bodies(id), "avbd3d_joint_wake_bodies");
}

static inline Avbd3dVec3 t3d_joint_get_constraint_force(Avbd3dJointId id)
{
    Avbd3dVec3 v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_joint_get_constraint_force(id, &v), "avbd3d_joint_get_constraint_force");
    return v;
}

static inline Avbd3dVec3 t3d_joint_get_constraint_torque(Avbd3dJointId id)
{
    Avbd3dVec3 v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_joint_get_constraint_torque(id, &v), "avbd3d_joint_get_constraint_torque");
    return v;
}

static inline void t3d_joint_set_force_threshold(Avbd3dJointId id, float threshold)
{
    t3d_check(avbd3d_joint_set_force_threshold(id, threshold), "avbd3d_joint_set_force_threshold");
}

static inline void t3d_joint_set_torque_threshold(Avbd3dJointId id, float threshold)
{
    t3d_check(avbd3d_joint_set_torque_threshold(id, threshold), "avbd3d_joint_set_torque_threshold");
}

static inline void t3d_destroy_joint(Avbd3dJointId id, int32_t wake_bodies)
{
    t3d_check(avbd3d_destroy_joint(id, wake_bodies), "avbd3d_destroy_joint");
}

static inline Avbd3dJointId t3d_create_distance_joint(Avbd3dWorld *world, const Avbd3dDistanceJointDef *def)
{
    Avbd3dJointId v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_create_distance_joint(world, def, &v), "avbd3d_create_distance_joint");
    return v;
}

static inline void t3d_distance_joint_set_length(Avbd3dJointId id, float length)
{
    t3d_check(avbd3d_distance_joint_set_length(id, length), "avbd3d_distance_joint_set_length");
}

static inline void t3d_distance_joint_enable_spring(Avbd3dJointId id, int32_t enableSpring)
{
    t3d_check(avbd3d_distance_joint_enable_spring(id, enableSpring), "avbd3d_distance_joint_enable_spring");
}

static inline void t3d_distance_joint_set_spring_force_range(Avbd3dJointId id, float lowerForce, float upperForce)
{
    t3d_check(avbd3d_distance_joint_set_spring_force_range(id, lowerForce, upperForce),
              "avbd3d_distance_joint_set_spring_force_range");
}

static inline void t3d_distance_joint_set_spring_hertz(Avbd3dJointId id, float hertz)
{
    t3d_check(avbd3d_distance_joint_set_spring_hertz(id, hertz), "avbd3d_distance_joint_set_spring_hertz");
}

static inline void t3d_distance_joint_set_spring_damping_ratio(Avbd3dJointId id, float dampingRatio)
{
    t3d_check(avbd3d_distance_joint_set_spring_damping_ratio(id, dampingRatio),
              "avbd3d_distance_joint_set_spring_damping_ratio");
}

static inline void t3d_distance_joint_enable_limit(Avbd3dJointId id, int32_t enableLimit)
{
    t3d_check(avbd3d_distance_joint_enable_limit(id, enableLimit), "avbd3d_distance_joint_enable_limit");
}

static inline void t3d_distance_joint_set_length_range(Avbd3dJointId id, float minLength, float maxLength)
{
    t3d_check(avbd3d_distance_joint_set_length_range(id, minLength, maxLength),
              "avbd3d_distance_joint_set_length_range");
}

static inline void t3d_distance_joint_enable_motor(Avbd3dJointId id, int32_t enableMotor)
{
    t3d_check(avbd3d_distance_joint_enable_motor(id, enableMotor), "avbd3d_distance_joint_enable_motor");
}

static inline void t3d_distance_joint_set_motor_speed(Avbd3dJointId id, float motorSpeed)
{
    t3d_check(avbd3d_distance_joint_set_motor_speed(id, motorSpeed), "avbd3d_distance_joint_set_motor_speed");
}

static inline void t3d_distance_joint_set_max_motor_force(Avbd3dJointId id, float force)
{
    t3d_check(avbd3d_distance_joint_set_max_motor_force(id, force), "avbd3d_distance_joint_set_max_motor_force");
}

static inline float t3d_distance_joint_get_current_length(Avbd3dJointId id)
{
    float v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_distance_joint_get_current_length(id, &v), "avbd3d_distance_joint_get_current_length");
    return v;
}

static inline float t3d_distance_joint_get_motor_force(Avbd3dJointId id)
{
    float v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_distance_joint_get_motor_force(id, &v), "avbd3d_distance_joint_get_motor_force");
    return v;
}

static inline Avbd3dJointId t3d_create_motor_joint(Avbd3dWorld *world, const Avbd3dMotorJointDef *def)
{
    Avbd3dJointId v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_create_motor_joint(world, def, &v), "avbd3d_create_motor_joint");
    return v;
}

static inline void t3d_motor_joint_set_linear_velocity(Avbd3dJointId id, Avbd3dVec3 velocity)
{
    t3d_check(avbd3d_motor_joint_set_linear_velocity(id, velocity), "avbd3d_motor_joint_set_linear_velocity");
}

static inline void t3d_motor_joint_set_angular_velocity(Avbd3dJointId id, Avbd3dVec3 velocity)
{
    t3d_check(avbd3d_motor_joint_set_angular_velocity(id, velocity), "avbd3d_motor_joint_set_angular_velocity");
}

static inline void t3d_motor_joint_set_max_velocity_force(Avbd3dJointId id, float maxForce)
{
    t3d_check(avbd3d_motor_joint_set_max_velocity_force(id, maxForce), "avbd3d_motor_joint_set_max_velocity_force");
}

static inline void t3d_motor_joint_set_max_velocity_torque(Avbd3dJointId id, float maxTorque)
{
    t3d_check(avbd3d_motor_joint_set_max_velocity_torque(id, maxTorque), "avbd3d_motor_joint_set_max_velocity_torque");
}

static inline void t3d_motor_joint_set_linear_hertz(Avbd3dJointId id, float hertz)
{
    t3d_check(avbd3d_motor_joint_set_linear_hertz(id, hertz), "avbd3d_motor_joint_set_linear_hertz");
}

static inline void t3d_motor_joint_set_linear_damping_ratio(Avbd3dJointId id, float damping)
{
    t3d_check(avbd3d_motor_joint_set_linear_damping_ratio(id, damping), "avbd3d_motor_joint_set_linear_damping_ratio");
}

static inline void t3d_motor_joint_set_angular_hertz(Avbd3dJointId id, float hertz)
{
    t3d_check(avbd3d_motor_joint_set_angular_hertz(id, hertz), "avbd3d_motor_joint_set_angular_hertz");
}

static inline void t3d_motor_joint_set_angular_damping_ratio(Avbd3dJointId id, float damping)
{
    t3d_check(avbd3d_motor_joint_set_angular_damping_ratio(id, damping),
              "avbd3d_motor_joint_set_angular_damping_ratio");
}

static inline void t3d_motor_joint_set_max_spring_force(Avbd3dJointId id, float maxForce)
{
    t3d_check(avbd3d_motor_joint_set_max_spring_force(id, maxForce), "avbd3d_motor_joint_set_max_spring_force");
}

static inline void t3d_motor_joint_set_max_spring_torque(Avbd3dJointId id, float maxTorque)
{
    t3d_check(avbd3d_motor_joint_set_max_spring_torque(id, maxTorque), "avbd3d_motor_joint_set_max_spring_torque");
}

static inline Avbd3dJointId t3d_create_filter_joint(Avbd3dWorld *world, const Avbd3dFilterJointDef *def)
{
    Avbd3dJointId v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_create_filter_joint(world, def, &v), "avbd3d_create_filter_joint");
    return v;
}

static inline Avbd3dJointId t3d_create_parallel_joint(Avbd3dWorld *world, const Avbd3dParallelJointDef *def)
{
    Avbd3dJointId v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_create_parallel_joint(world, def, &v), "avbd3d_create_parallel_joint");
    return v;
}

static inline void t3d_parallel_joint_set_spring_hertz(Avbd3dJointId id, float hertz)
{
    t3d_check(avbd3d_parallel_joint_set_spring_hertz(id, hertz), "avbd3d_parallel_joint_set_spring_hertz");
}

static inline void t3d_parallel_joint_set_spring_damping_ratio(Avbd3dJointId id, float dampingRatio)
{
    t3d_check(avbd3d_parallel_joint_set_spring_damping_ratio(id, dampingRatio),
              "avbd3d_parallel_joint_set_spring_damping_ratio");
}

static inline void t3d_parallel_joint_set_max_torque(Avbd3dJointId id, float force)
{
    t3d_check(avbd3d_parallel_joint_set_max_torque(id, force), "avbd3d_parallel_joint_set_max_torque");
}

static inline Avbd3dJointId t3d_create_prismatic_joint(Avbd3dWorld *world, const Avbd3dPrismaticJointDef *def)
{
    Avbd3dJointId v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_create_prismatic_joint(world, def, &v), "avbd3d_create_prismatic_joint");
    return v;
}

static inline void t3d_prismatic_joint_enable_spring(Avbd3dJointId id, int32_t enableSpring)
{
    t3d_check(avbd3d_prismatic_joint_enable_spring(id, enableSpring), "avbd3d_prismatic_joint_enable_spring");
}

static inline void t3d_prismatic_joint_set_spring_hertz(Avbd3dJointId id, float hertz)
{
    t3d_check(avbd3d_prismatic_joint_set_spring_hertz(id, hertz), "avbd3d_prismatic_joint_set_spring_hertz");
}

static inline void t3d_prismatic_joint_set_spring_damping_ratio(Avbd3dJointId id, float dampingRatio)
{
    t3d_check(avbd3d_prismatic_joint_set_spring_damping_ratio(id, dampingRatio),
              "avbd3d_prismatic_joint_set_spring_damping_ratio");
}

static inline void t3d_prismatic_joint_set_target_translation(Avbd3dJointId id, float targetTranslation)
{
    t3d_check(avbd3d_prismatic_joint_set_target_translation(id, targetTranslation),
              "avbd3d_prismatic_joint_set_target_translation");
}

static inline void t3d_prismatic_joint_enable_limit(Avbd3dJointId id, int32_t enableLimit)
{
    t3d_check(avbd3d_prismatic_joint_enable_limit(id, enableLimit), "avbd3d_prismatic_joint_enable_limit");
}

static inline void t3d_prismatic_joint_set_limits(Avbd3dJointId id, float lower, float upper)
{
    t3d_check(avbd3d_prismatic_joint_set_limits(id, lower, upper), "avbd3d_prismatic_joint_set_limits");
}

static inline void t3d_prismatic_joint_enable_motor(Avbd3dJointId id, int32_t enableMotor)
{
    t3d_check(avbd3d_prismatic_joint_enable_motor(id, enableMotor), "avbd3d_prismatic_joint_enable_motor");
}

static inline void t3d_prismatic_joint_set_motor_speed(Avbd3dJointId id, float motorSpeed)
{
    t3d_check(avbd3d_prismatic_joint_set_motor_speed(id, motorSpeed), "avbd3d_prismatic_joint_set_motor_speed");
}

static inline void t3d_prismatic_joint_set_max_motor_force(Avbd3dJointId id, float force)
{
    t3d_check(avbd3d_prismatic_joint_set_max_motor_force(id, force), "avbd3d_prismatic_joint_set_max_motor_force");
}

static inline float t3d_prismatic_joint_get_motor_force(Avbd3dJointId id)
{
    float v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_prismatic_joint_get_motor_force(id, &v), "avbd3d_prismatic_joint_get_motor_force");
    return v;
}

static inline float t3d_prismatic_joint_get_translation(Avbd3dJointId id)
{
    float v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_prismatic_joint_get_translation(id, &v), "avbd3d_prismatic_joint_get_translation");
    return v;
}

static inline float t3d_prismatic_joint_get_speed(Avbd3dJointId id)
{
    float v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_prismatic_joint_get_speed(id, &v), "avbd3d_prismatic_joint_get_speed");
    return v;
}

static inline Avbd3dJointId t3d_create_revolute_joint(Avbd3dWorld *world, const Avbd3dRevoluteJointDef *def)
{
    Avbd3dJointId v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_create_revolute_joint(world, def, &v), "avbd3d_create_revolute_joint");
    return v;
}

static inline void t3d_revolute_joint_enable_spring(Avbd3dJointId id, int32_t enableSpring)
{
    t3d_check(avbd3d_revolute_joint_enable_spring(id, enableSpring), "avbd3d_revolute_joint_enable_spring");
}

static inline void t3d_revolute_joint_set_spring_hertz(Avbd3dJointId id, float hertz)
{
    t3d_check(avbd3d_revolute_joint_set_spring_hertz(id, hertz), "avbd3d_revolute_joint_set_spring_hertz");
}

static inline void t3d_revolute_joint_set_spring_damping_ratio(Avbd3dJointId id, float dampingRatio)
{
    t3d_check(avbd3d_revolute_joint_set_spring_damping_ratio(id, dampingRatio),
              "avbd3d_revolute_joint_set_spring_damping_ratio");
}

static inline void t3d_revolute_joint_set_target_angle(Avbd3dJointId id, float targetRadians)
{
    t3d_check(avbd3d_revolute_joint_set_target_angle(id, targetRadians), "avbd3d_revolute_joint_set_target_angle");
}

static inline void t3d_revolute_joint_enable_limit(Avbd3dJointId id, int32_t enableLimit)
{
    t3d_check(avbd3d_revolute_joint_enable_limit(id, enableLimit), "avbd3d_revolute_joint_enable_limit");
}

static inline void t3d_revolute_joint_set_limits(Avbd3dJointId id, float lowerLimitRadians, float upperLimitRadians)
{
    t3d_check(avbd3d_revolute_joint_set_limits(id, lowerLimitRadians, upperLimitRadians),
              "avbd3d_revolute_joint_set_limits");
}

static inline void t3d_revolute_joint_enable_motor(Avbd3dJointId id, int32_t enableMotor)
{
    t3d_check(avbd3d_revolute_joint_enable_motor(id, enableMotor), "avbd3d_revolute_joint_enable_motor");
}

static inline void t3d_revolute_joint_set_motor_speed(Avbd3dJointId id, float motorSpeed)
{
    t3d_check(avbd3d_revolute_joint_set_motor_speed(id, motorSpeed), "avbd3d_revolute_joint_set_motor_speed");
}

static inline void t3d_revolute_joint_set_max_motor_torque(Avbd3dJointId id, float torque)
{
    t3d_check(avbd3d_revolute_joint_set_max_motor_torque(id, torque), "avbd3d_revolute_joint_set_max_motor_torque");
}

static inline float t3d_revolute_joint_get_motor_torque(Avbd3dJointId id)
{
    float v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_revolute_joint_get_motor_torque(id, &v), "avbd3d_revolute_joint_get_motor_torque");
    return v;
}

static inline float t3d_revolute_joint_get_angle(Avbd3dJointId id)
{
    float v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_revolute_joint_get_angle(id, &v), "avbd3d_revolute_joint_get_angle");
    return v;
}

static inline float t3d_revolute_joint_get_speed(Avbd3dJointId id)
{
    float v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_revolute_joint_get_speed(id, &v), "avbd3d_revolute_joint_get_speed");
    return v;
}

static inline Avbd3dJointId t3d_create_spherical_joint(Avbd3dWorld *world, const Avbd3dSphericalJointDef *def)
{
    Avbd3dJointId v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_create_spherical_joint(world, def, &v), "avbd3d_create_spherical_joint");
    return v;
}

static inline void t3d_spherical_joint_enable_cone_limit(Avbd3dJointId id, int32_t enableLimit)
{
    t3d_check(avbd3d_spherical_joint_enable_cone_limit(id, enableLimit), "avbd3d_spherical_joint_enable_cone_limit");
}

static inline void t3d_spherical_joint_set_cone_limit(Avbd3dJointId id, float angleRadians)
{
    t3d_check(avbd3d_spherical_joint_set_cone_limit(id, angleRadians), "avbd3d_spherical_joint_set_cone_limit");
}

static inline void t3d_spherical_joint_enable_twist_limit(Avbd3dJointId id, int32_t enableLimit)
{
    t3d_check(avbd3d_spherical_joint_enable_twist_limit(id, enableLimit), "avbd3d_spherical_joint_enable_twist_limit");
}

static inline void t3d_spherical_joint_set_twist_limits(Avbd3dJointId id, float lowerLimitRadians,
                                                        float upperLimitRadians)
{
    t3d_check(avbd3d_spherical_joint_set_twist_limits(id, lowerLimitRadians, upperLimitRadians),
              "avbd3d_spherical_joint_set_twist_limits");
}

static inline void t3d_spherical_joint_enable_spring(Avbd3dJointId id, int32_t enableSpring)
{
    t3d_check(avbd3d_spherical_joint_enable_spring(id, enableSpring), "avbd3d_spherical_joint_enable_spring");
}

static inline void t3d_spherical_joint_set_spring_hertz(Avbd3dJointId id, float hertz)
{
    t3d_check(avbd3d_spherical_joint_set_spring_hertz(id, hertz), "avbd3d_spherical_joint_set_spring_hertz");
}

static inline void t3d_spherical_joint_set_spring_damping_ratio(Avbd3dJointId id, float dampingRatio)
{
    t3d_check(avbd3d_spherical_joint_set_spring_damping_ratio(id, dampingRatio),
              "avbd3d_spherical_joint_set_spring_damping_ratio");
}

static inline void t3d_spherical_joint_enable_motor(Avbd3dJointId id, int32_t enableMotor)
{
    t3d_check(avbd3d_spherical_joint_enable_motor(id, enableMotor), "avbd3d_spherical_joint_enable_motor");
}

static inline void t3d_spherical_joint_set_motor_velocity(Avbd3dJointId id, Avbd3dVec3 motorVelocity)
{
    t3d_check(avbd3d_spherical_joint_set_motor_velocity(id, motorVelocity),
              "avbd3d_spherical_joint_set_motor_velocity");
}

static inline void t3d_spherical_joint_set_max_motor_torque(Avbd3dJointId id, float torque)
{
    t3d_check(avbd3d_spherical_joint_set_max_motor_torque(id, torque), "avbd3d_spherical_joint_set_max_motor_torque");
}

static inline Avbd3dVec3 t3d_spherical_joint_get_motor_torque(Avbd3dJointId id)
{
    Avbd3dVec3 v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_spherical_joint_get_motor_torque(id, &v), "avbd3d_spherical_joint_get_motor_torque");
    return v;
}

static inline float t3d_spherical_joint_get_cone_angle(Avbd3dJointId id)
{
    float v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_spherical_joint_get_cone_angle(id, &v), "avbd3d_spherical_joint_get_cone_angle");
    return v;
}

static inline float t3d_spherical_joint_get_twist_angle(Avbd3dJointId id)
{
    float v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_spherical_joint_get_twist_angle(id, &v), "avbd3d_spherical_joint_get_twist_angle");
    return v;
}

static inline void t3d_spherical_joint_set_target_rotation(Avbd3dJointId id, Avbd3dQuat targetRotation)
{
    t3d_check(avbd3d_spherical_joint_set_target_rotation(id, targetRotation),
              "avbd3d_spherical_joint_set_target_rotation");
}

static inline Avbd3dJointId t3d_create_weld_joint(Avbd3dWorld *world, const Avbd3dWeldJointDef *def)
{
    Avbd3dJointId v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_create_weld_joint(world, def, &v), "avbd3d_create_weld_joint");
    return v;
}

static inline void t3d_weld_joint_set_linear_hertz(Avbd3dJointId id, float hertz)
{
    t3d_check(avbd3d_weld_joint_set_linear_hertz(id, hertz), "avbd3d_weld_joint_set_linear_hertz");
}

static inline void t3d_weld_joint_set_linear_damping_ratio(Avbd3dJointId id, float dampingRatio)
{
    t3d_check(avbd3d_weld_joint_set_linear_damping_ratio(id, dampingRatio),
              "avbd3d_weld_joint_set_linear_damping_ratio");
}

static inline void t3d_weld_joint_set_angular_hertz(Avbd3dJointId id, float hertz)
{
    t3d_check(avbd3d_weld_joint_set_angular_hertz(id, hertz), "avbd3d_weld_joint_set_angular_hertz");
}

static inline void t3d_weld_joint_set_angular_damping_ratio(Avbd3dJointId id, float dampingRatio)
{
    t3d_check(avbd3d_weld_joint_set_angular_damping_ratio(id, dampingRatio),
              "avbd3d_weld_joint_set_angular_damping_ratio");
}

static inline Avbd3dJointId t3d_create_wheel_joint(Avbd3dWorld *world, const Avbd3dWheelJointDef *def)
{
    Avbd3dJointId v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_create_wheel_joint(world, def, &v), "avbd3d_create_wheel_joint");
    return v;
}

static inline void t3d_wheel_joint_enable_suspension(Avbd3dJointId id, int32_t flag)
{
    t3d_check(avbd3d_wheel_joint_enable_suspension(id, flag), "avbd3d_wheel_joint_enable_suspension");
}

static inline void t3d_wheel_joint_set_suspension_hertz(Avbd3dJointId id, float hertz)
{
    t3d_check(avbd3d_wheel_joint_set_suspension_hertz(id, hertz), "avbd3d_wheel_joint_set_suspension_hertz");
}

static inline void t3d_wheel_joint_set_suspension_damping_ratio(Avbd3dJointId id, float dampingRatio)
{
    t3d_check(avbd3d_wheel_joint_set_suspension_damping_ratio(id, dampingRatio),
              "avbd3d_wheel_joint_set_suspension_damping_ratio");
}

static inline void t3d_wheel_joint_enable_suspension_limit(Avbd3dJointId id, int32_t flag)
{
    t3d_check(avbd3d_wheel_joint_enable_suspension_limit(id, flag), "avbd3d_wheel_joint_enable_suspension_limit");
}

static inline void t3d_wheel_joint_set_suspension_limits(Avbd3dJointId id, float lower, float upper)
{
    t3d_check(avbd3d_wheel_joint_set_suspension_limits(id, lower, upper), "avbd3d_wheel_joint_set_suspension_limits");
}

static inline void t3d_wheel_joint_enable_spin_motor(Avbd3dJointId id, int32_t flag)
{
    t3d_check(avbd3d_wheel_joint_enable_spin_motor(id, flag), "avbd3d_wheel_joint_enable_spin_motor");
}

static inline void t3d_wheel_joint_set_spin_motor_speed(Avbd3dJointId id, float speed)
{
    t3d_check(avbd3d_wheel_joint_set_spin_motor_speed(id, speed), "avbd3d_wheel_joint_set_spin_motor_speed");
}

static inline void t3d_wheel_joint_set_max_spin_torque(Avbd3dJointId id, float torque)
{
    t3d_check(avbd3d_wheel_joint_set_max_spin_torque(id, torque), "avbd3d_wheel_joint_set_max_spin_torque");
}

static inline float t3d_wheel_joint_get_spin_torque(Avbd3dJointId id)
{
    float v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_wheel_joint_get_spin_torque(id, &v), "avbd3d_wheel_joint_get_spin_torque");
    return v;
}

static inline float t3d_wheel_joint_get_spin_speed(Avbd3dJointId id)
{
    float v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_wheel_joint_get_spin_speed(id, &v), "avbd3d_wheel_joint_get_spin_speed");
    return v;
}

static inline float t3d_wheel_joint_get_steering_angle(Avbd3dJointId id)
{
    float v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_wheel_joint_get_steering_angle(id, &v), "avbd3d_wheel_joint_get_steering_angle");
    return v;
}

static inline void t3d_wheel_joint_enable_steering(Avbd3dJointId id, int32_t flag)
{
    t3d_check(avbd3d_wheel_joint_enable_steering(id, flag), "avbd3d_wheel_joint_enable_steering");
}

static inline void t3d_wheel_joint_set_steering_hertz(Avbd3dJointId id, float hertz)
{
    t3d_check(avbd3d_wheel_joint_set_steering_hertz(id, hertz), "avbd3d_wheel_joint_set_steering_hertz");
}

static inline void t3d_wheel_joint_set_steering_damping_ratio(Avbd3dJointId id, float dampingRatio)
{
    t3d_check(avbd3d_wheel_joint_set_steering_damping_ratio(id, dampingRatio),
              "avbd3d_wheel_joint_set_steering_damping_ratio");
}

static inline void t3d_wheel_joint_set_max_steering_torque(Avbd3dJointId id, float torque)
{
    t3d_check(avbd3d_wheel_joint_set_max_steering_torque(id, torque), "avbd3d_wheel_joint_set_max_steering_torque");
}

static inline void t3d_wheel_joint_enable_steering_limit(Avbd3dJointId id, int32_t flag)
{
    t3d_check(avbd3d_wheel_joint_enable_steering_limit(id, flag), "avbd3d_wheel_joint_enable_steering_limit");
}

static inline void t3d_wheel_joint_set_steering_limits(Avbd3dJointId id, float lowerRadians, float upperRadians)
{
    t3d_check(avbd3d_wheel_joint_set_steering_limits(id, lowerRadians, upperRadians),
              "avbd3d_wheel_joint_set_steering_limits");
}

static inline void t3d_wheel_joint_set_target_steering_angle(Avbd3dJointId id, float radians)
{
    t3d_check(avbd3d_wheel_joint_set_target_steering_angle(id, radians),
              "avbd3d_wheel_joint_set_target_steering_angle");
}

static inline Avbd3dJointId t3d_create_generic_joint(Avbd3dWorld *world, const Avbd3dGenericJointDef *def)
{
    Avbd3dJointId v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_create_generic_joint(world, def, &v), "avbd3d_create_generic_joint");
    return v;
}

static inline Avbd3dJointId t3d_create_spring(Avbd3dWorld *world, const Avbd3dSpringDef *def)
{
    Avbd3dJointId v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_create_spring(world, def, &v), "avbd3d_create_spring");
    return v;
}

static inline Avbd3dContactData t3d_contact_get_data(Avbd3dContactId id)
{
    Avbd3dContactData v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_contact_get_data(id, &v), "avbd3d_contact_get_data");
    return v;
}

static inline void t3d_grab_move(Avbd3dWorld *world, const Avbd3dGrabHandle *handles, int32_t count)
{
    t3d_check(avbd3d_grab_move(world, handles, count), "avbd3d_grab_move");
}

static inline void t3d_grab_end(Avbd3dWorld *world)
{
    t3d_check(avbd3d_grab_end(world), "avbd3d_grab_end");
}

static inline Avbd3dHullData *t3d_create_hull(const Avbd3dVec3 *points, int32_t point_count, int32_t max_vertex_count)
{
    Avbd3dHullData *v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_create_hull(points, point_count, max_vertex_count, &v), "avbd3d_create_hull");
    return v;
}

static inline void t3d_destroy_hull(Avbd3dHullData *hull)
{
    t3d_check(avbd3d_destroy_hull(hull), "avbd3d_destroy_hull");
}

static inline Avbd3dMeshData *t3d_create_mesh(const Avbd3dMeshDef *def)
{
    Avbd3dMeshData *v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_create_mesh(def, &v), "avbd3d_create_mesh");
    return v;
}

static inline void t3d_destroy_mesh(Avbd3dMeshData *mesh)
{
    t3d_check(avbd3d_destroy_mesh(mesh), "avbd3d_destroy_mesh");
}

static inline Avbd3dHeightFieldData *t3d_create_height_field(const Avbd3dHeightFieldDef *def)
{
    Avbd3dHeightFieldData *v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_create_height_field(def, &v), "avbd3d_create_height_field");
    return v;
}

static inline void t3d_destroy_height_field(Avbd3dHeightFieldData *height_field)
{
    t3d_check(avbd3d_destroy_height_field(height_field), "avbd3d_destroy_height_field");
}

static inline Avbd3dCompoundData *t3d_create_compound(const Avbd3dCompoundDef *def)
{
    Avbd3dCompoundData *v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_create_compound(def, &v), "avbd3d_create_compound");
    return v;
}

static inline void t3d_destroy_compound(Avbd3dCompoundData *compound)
{
    t3d_check(avbd3d_destroy_compound(compound), "avbd3d_destroy_compound");
}

static inline Avbd3dGpu *t3d_create_gpu(const Avbd3dGpuDef *def)
{
    Avbd3dGpu *v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_create_gpu(def, &v), "avbd3d_create_gpu");
    return v;
}

static inline void t3d_destroy_gpu(Avbd3dGpu *gpu)
{
    t3d_check(avbd3d_destroy_gpu(gpu), "avbd3d_destroy_gpu");
}

static inline Avbd3dWorld *t3d_create_world_on_gpu(Avbd3dGpu *gpu, const Avbd3dWorldDef *def)
{
    Avbd3dWorld *v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_create_world_on_gpu(gpu, def, &v), "avbd3d_create_world_on_gpu");
    return v;
}

static inline Avbd3dVulkanContext t3d_gpu_get_vulkan_context(const Avbd3dGpu *gpu)
{
    Avbd3dVulkanContext v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_gpu_get_vulkan_context(gpu, &v), "avbd3d_gpu_get_vulkan_context");
    return v;
}

static inline uint64_t t3d_gpu_get_allocated_bytes(const Avbd3dGpu *gpu)
{
    uint64_t v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_gpu_get_allocated_bytes(gpu, &v), "avbd3d_gpu_get_allocated_bytes");
    return v;
}

static inline int32_t t3d_get_validation_message_count(void)
{
    int32_t v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_get_validation_message_count(&v), "avbd3d_get_validation_message_count");
    return v;
}

static inline void t3d_body_set_render_group(Avbd3dBodyId id, int32_t group)
{
    t3d_check(avbd3d_body_set_render_group(id, group), "avbd3d_body_set_render_group");
}

static inline int32_t t3d_body_get_render_group(Avbd3dBodyId id)
{
    int32_t v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_body_get_render_group(id, &v), "avbd3d_body_get_render_group");
    return v;
}

static inline void t3d_world_write_instances(Avbd3dWorld *world, const Avbd3dRenderDef *def)
{
    t3d_check(avbd3d_world_write_instances(world, def), "avbd3d_world_write_instances");
}

static inline uint64_t t3d_world_get_instance_buffer(Avbd3dWorld *world)
{
    uint64_t v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_world_get_instance_buffer(world, &v), "avbd3d_world_get_instance_buffer");
    return v;
}

static inline int32_t t3d_world_get_render_group_count(Avbd3dWorld *world)
{
    int32_t v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_world_get_render_group_count(world, &v), "avbd3d_world_get_render_group_count");
    return v;
}

static inline Avbd3dRenderRange t3d_world_get_render_group_range(Avbd3dWorld *world, int32_t group)
{
    Avbd3dRenderRange v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_world_get_render_group_range(world, group, &v), "avbd3d_world_get_render_group_range");
    return v;
}

static inline int32_t t3d_world_write_contact_markers(Avbd3dWorld *world, const Avbd3dContactMarkerDef *def)
{
    int32_t v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_world_write_contact_markers(world, def, &v), "avbd3d_world_write_contact_markers");
    return v;
}

static inline uint64_t t3d_world_get_contact_marker_buffer(Avbd3dWorld *world)
{
    uint64_t v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_world_get_contact_marker_buffer(world, &v), "avbd3d_world_get_contact_marker_buffer");
    return v;
}

static inline int32_t t3d_world_get_contact_markers_requested(Avbd3dWorld *world)
{
    int32_t v;
    std::memset(&v, 0, sizeof v);
    t3d_check(avbd3d_world_get_contact_markers_requested(world, &v), "avbd3d_world_get_contact_markers_requested");
    return v;
}

// Calls whose result is a count or that take callbacks: checked, returning the count.
static inline int32_t t3d_world_get_body_data(Avbd3dWorld *w, const Avbd3dBodyId *ids, int32_t n, Avbd3dBodyDataType t,
                                              void *data)
{
    int32_t v = 0;
    t3d_check(avbd3d_world_get_body_data(w, ids, n, t, data, &v), "avbd3d_world_get_body_data");
    return v;
}

static inline int32_t t3d_world_set_body_data(Avbd3dWorld *w, const Avbd3dBodyId *ids, int32_t n, Avbd3dBodyDataType t,
                                              const void *data)
{
    int32_t v = 0;
    t3d_check(avbd3d_world_set_body_data(w, ids, n, t, data, &v), "avbd3d_world_set_body_data");
    return v;
}

static inline int32_t t3d_body_get_shapes(Avbd3dBodyId id, Avbd3dShapeId *a, int32_t cap)
{
    int32_t v = 0;
    t3d_check(avbd3d_body_get_shapes(id, a, cap, &v), "avbd3d_body_get_shapes");
    return v;
}

static inline int32_t t3d_body_get_joints(Avbd3dBodyId id, Avbd3dJointId *a, int32_t cap)
{
    int32_t v = 0;
    t3d_check(avbd3d_body_get_joints(id, a, cap, &v), "avbd3d_body_get_joints");
    return v;
}

static inline int32_t t3d_body_get_contact_data(Avbd3dBodyId id, Avbd3dContactData *a, int32_t cap)
{
    int32_t v = 0;
    t3d_check(avbd3d_body_get_contact_data(id, a, cap, &v), "avbd3d_body_get_contact_data");
    return v;
}

static inline int32_t t3d_shape_get_contact_data(Avbd3dShapeId id, Avbd3dContactData *a, int32_t cap)
{
    int32_t v = 0;
    t3d_check(avbd3d_shape_get_contact_data(id, a, cap, &v), "avbd3d_shape_get_contact_data");
    return v;
}

static inline int32_t t3d_shape_get_sensor_data(Avbd3dShapeId id, Avbd3dShapeId *a, int32_t cap)
{
    int32_t v = 0;
    t3d_check(avbd3d_shape_get_sensor_data(id, a, cap, &v), "avbd3d_shape_get_sensor_data");
    return v;
}

static inline int32_t t3d_body_collide_mover(Avbd3dBodyId id, Avbd3dBodyPlaneResult *planes, int32_t cap,
                                             Avbd3dVec3 origin, const Avbd3dCapsule *mover, Avbd3dQueryFilter filter,
                                             Avbd3dTransform xf)
{
    int32_t v = 0;
    t3d_check(avbd3d_body_collide_mover(id, planes, cap, origin, mover, filter, xf, &v), "avbd3d_body_collide_mover");
    return v;
}

static inline int32_t t3d_grab_begin(Avbd3dWorld *w, Avbd3dBodyId seed, Avbd3dVec3 c, float r, int32_t follow,
                                     Avbd3dBodyId *out, int32_t max)
{
    int32_t v = 0;
    t3d_check(avbd3d_grab_begin(w, seed, c, r, follow, out, max, &v), "avbd3d_grab_begin");
    return v;
}

static inline int32_t t3d_grab_survivors(Avbd3dWorld *w, Avbd3dBodyId seed, const Avbd3dBodyId *g, int32_t n,
                                         uint8_t *keep)
{
    int32_t v = 0;
    t3d_check(avbd3d_grab_survivors(w, seed, g, n, keep, &v), "avbd3d_grab_survivors");
    return v;
}

static const Avbd3dBodyId t3d_null_body = {0, 0, 0};
static const Avbd3dShapeId t3d_null_shape = {0, 0, 0};
static const Avbd3dJointId t3d_null_joint = {0, 0, 0};
static const Avbd3dContactId t3d_null_contact = {0, 0, 0, 0};

static inline Avbd3dQuat t3d_quat_axis_angle(Avbd3dVec3 axis, float angle)
{
    const float len = std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
    const float s = len > 0.0f ? std::sin(0.5f * angle) / len : 0.0f;
    Avbd3dQuat q = {{axis.x * s, axis.y * s, axis.z * s}, std::cos(0.5f * angle)};
    return q;
}

static const Avbd3dTransform t3d_transform_identity = {{0, 0, 0}, {{0, 0, 0}, 1}};

static inline void t3d_world_overlap_aabb(Avbd3dWorld *w, Avbd3dAABB aabb, Avbd3dQueryFilter f,
                                          Avbd3dOverlapResultFcn *fcn, void *ctx)
{
    t3d_check(avbd3d_world_overlap_aabb(w, aabb, f, fcn, ctx), "avbd3d_world_overlap_aabb");
}

static inline void t3d_world_overlap_shape(Avbd3dWorld *w, Avbd3dVec3 origin, const Avbd3dShapeProxy *proxy,
                                           Avbd3dQueryFilter f, Avbd3dOverlapResultFcn *fcn, void *ctx)
{
    t3d_check(avbd3d_world_overlap_shape(w, origin, proxy, f, fcn, ctx), "avbd3d_world_overlap_shape");
}

static inline void t3d_world_cast_shape(Avbd3dWorld *w, Avbd3dVec3 origin, const Avbd3dShapeProxy *proxy,
                                        Avbd3dVec3 translation, Avbd3dQueryFilter f, Avbd3dCastResultFcn *fcn,
                                        void *ctx)
{
    t3d_check(avbd3d_world_cast_shape(w, origin, proxy, translation, f, fcn, ctx), "avbd3d_world_cast_shape");
}

static inline float t3d_world_cast_mover(Avbd3dWorld *w, Avbd3dVec3 origin, const Avbd3dCapsule *mover,
                                         Avbd3dVec3 translation, Avbd3dQueryFilter f, Avbd3dMoverFilterFcn *fcn,
                                         void *ctx)
{
    float v = 0.0f;
    t3d_check(avbd3d_world_cast_mover(w, origin, mover, translation, f, fcn, ctx, &v), "avbd3d_world_cast_mover");
    return v;
}

static inline void t3d_world_collide_mover(Avbd3dWorld *w, Avbd3dVec3 origin, const Avbd3dCapsule *mover,
                                           Avbd3dQueryFilter f, Avbd3dPlaneResultFcn *fcn, void *ctx)
{
    t3d_check(avbd3d_world_collide_mover(w, origin, mover, f, fcn, ctx), "avbd3d_world_collide_mover");
}

// Returns the distance; the closest point comes back through cp.
static inline float t3d_body_get_closest_point(Avbd3dBodyId id, Avbd3dVec3 *cp, Avbd3dVec3 target)
{
    float d = 0.0f;
    t3d_check(avbd3d_body_get_closest_point(id, target, cp, &d), "avbd3d_body_get_closest_point");
    return d;
}
