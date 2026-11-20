// The mechanical part of the public C API: one wrapper per call over the world implementation.
// The hand-written calls are in avbd3d_capi_special.cpp and avbd3d_geometry.cpp.
#include "avbd3d_capi.h"

Avbd3dWorldDef avbd3d_default_world_def(void)
{
    return pb(avbd3d_DefaultWorldDef());
}

Avbd3dBodyDef avbd3d_default_body_def(void)
{
    return pb(avbd3d_DefaultBodyDef());
}

Avbd3dFilter avbd3d_default_filter(void)
{
    return pb(avbd3d_DefaultFilter());
}

Avbd3dSurfaceMaterial avbd3d_default_surface_material(void)
{
    return pb(avbd3d_DefaultSurfaceMaterial());
}

Avbd3dShapeDef avbd3d_default_shape_def(void)
{
    return pb(avbd3d_DefaultShapeDef());
}

Avbd3dDistanceJointDef avbd3d_default_distance_joint_def(void)
{
    return pb(avbd3d_DefaultDistanceJointDef());
}

Avbd3dMotorJointDef avbd3d_default_motor_joint_def(void)
{
    return pb(avbd3d_DefaultMotorJointDef());
}

Avbd3dFilterJointDef avbd3d_default_filter_joint_def(void)
{
    return pb(avbd3d_DefaultFilterJointDef());
}

Avbd3dParallelJointDef avbd3d_default_parallel_joint_def(void)
{
    return pb(avbd3d_DefaultParallelJointDef());
}

Avbd3dPrismaticJointDef avbd3d_default_prismatic_joint_def(void)
{
    return pb(avbd3d_DefaultPrismaticJointDef());
}

Avbd3dRevoluteJointDef avbd3d_default_revolute_joint_def(void)
{
    return pb(avbd3d_DefaultRevoluteJointDef());
}

Avbd3dSphericalJointDef avbd3d_default_spherical_joint_def(void)
{
    return pb(avbd3d_DefaultSphericalJointDef());
}

Avbd3dWeldJointDef avbd3d_default_weld_joint_def(void)
{
    return pb(avbd3d_DefaultWeldJointDef());
}

Avbd3dWheelJointDef avbd3d_default_wheel_joint_def(void)
{
    return pb(avbd3d_DefaultWheelJointDef());
}

Avbd3dExplosionDef avbd3d_default_explosion_def(void)
{
    return pb(avbd3d_DefaultExplosionDef());
}

Avbd3dQueryFilter avbd3d_default_query_filter(void)
{
    return pb(avbd3d_DefaultQueryFilter());
}

Avbd3dResult avbd3d_world_get_bounds(Avbd3dWorld *world, Avbd3dAABB *out)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_AABB v_ = avbd3d_World_GetBounds(lw);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_world_get_body_events(Avbd3dWorld *world, Avbd3dBodyEvents *out)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_BodyEvents v_ = avbd3d_World_GetBodyEvents(lw);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_world_get_sensor_events(Avbd3dWorld *world, Avbd3dSensorEvents *out)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_SensorEvents v_ = avbd3d_World_GetSensorEvents(lw);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_world_get_contact_events(Avbd3dWorld *world, Avbd3dContactEvents *out)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_ContactEvents v_ = avbd3d_World_GetContactEvents(lw);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_world_get_joint_events(Avbd3dWorld *world, Avbd3dJointEvents *out)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_JointEvents v_ = avbd3d_World_GetJointEvents(lw);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_world_cast_ray_closest(Avbd3dWorld *world, Avbd3dVec3 origin, Avbd3dVec3 translation,
                                           Avbd3dQueryFilter filter, Avbd3dRayResult *out)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_RayResult v_ = avbd3d_World_CastRayClosest(lw, lg(origin), lg(translation), lg(filter));
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_world_enable_sleeping(Avbd3dWorld *world, int32_t flag)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    avbd3d_World_EnableSleeping(lw, flag != 0);
    return leave(w);
}

Avbd3dResult avbd3d_world_enable_continuous(Avbd3dWorld *world, int32_t flag)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    avbd3d_World_EnableContinuous(lw, flag != 0);
    return leave(w);
}

Avbd3dResult avbd3d_world_set_restitution_threshold(Avbd3dWorld *world, float value)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    avbd3d_World_SetRestitutionThreshold(lw, lg(value));
    return leave(w);
}

Avbd3dResult avbd3d_world_set_hit_event_threshold(Avbd3dWorld *world, float value)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    avbd3d_World_SetHitEventThreshold(lw, lg(value));
    return leave(w);
}

Avbd3dResult avbd3d_world_set_gravity(Avbd3dWorld *world, Avbd3dVec3 gravity)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    avbd3d_World_SetGravity(lw, lg(gravity));
    return leave(w);
}

Avbd3dResult avbd3d_world_get_gravity(Avbd3dWorld *world, Avbd3dVec3 *out)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_Vec3 v_ = avbd3d_World_GetGravity(lw);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_world_explode(Avbd3dWorld *world, const Avbd3dExplosionDef *explosionDef)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!explosionDef)
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_World_Explode(lw, reinterpret_cast<const avbd3d_ExplosionDef *>(explosionDef));
    return leave(w);
}

Avbd3dResult avbd3d_create_body(Avbd3dWorld *world, const Avbd3dBodyDef *def, Avbd3dBodyId *out_id)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out_id)
        return AVBD3D_ERR_INVALID_ARG;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_BodyDef l_def = lg(*def);
    const avbd3d_BodyId v_ = avbd3d_CreateBody(lw, &l_def);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    if (v_.index1 == 0)
        return AVBD3D_ERR_INVALID_ARG;
    *out_id = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_destroy_body(Avbd3dBodyId bodyId)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(bodyId, w, lid))
        return r_;
    avbd3d_DestroyBody(lid);
    return leave(w);
}

Avbd3dResult avbd3d_body_get_type(Avbd3dBodyId id, Avbd3dBodyType *out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_BodyType v_ = avbd3d_Body_GetType(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_set_type(Avbd3dBodyId id, Avbd3dBodyType type)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    avbd3d_Body_SetType(lid, lg(type));
    return leave(w);
}

Avbd3dResult avbd3d_body_set_user_data(Avbd3dBodyId id, void *userData)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    avbd3d_Body_SetUserData(lid, userData);
    return leave(w);
}

Avbd3dResult avbd3d_body_get_user_data(Avbd3dBodyId id, void **out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    void *v_ = avbd3d_Body_GetUserData(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_get_position(Avbd3dBodyId id, Avbd3dVec3 *out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_Vec3 v_ = avbd3d_Body_GetPosition(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_get_rotation(Avbd3dBodyId id, Avbd3dQuat *out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_Quat v_ = avbd3d_Body_GetRotation(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_get_transform(Avbd3dBodyId id, Avbd3dTransform *out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_Transform v_ = avbd3d_Body_GetTransform(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_set_transform(Avbd3dBodyId id, Avbd3dVec3 position, Avbd3dQuat rotation)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    avbd3d_Body_SetTransform(lid, lg(position), lg(rotation));
    return leave(w);
}

Avbd3dResult avbd3d_body_get_local_point(Avbd3dBodyId id, Avbd3dVec3 worldPoint, Avbd3dVec3 *out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_Vec3 v_ = avbd3d_Body_GetLocalPoint(lid, lg(worldPoint));
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_get_world_point(Avbd3dBodyId id, Avbd3dVec3 localPoint, Avbd3dVec3 *out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_Vec3 v_ = avbd3d_Body_GetWorldPoint(lid, lg(localPoint));
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_get_local_vector(Avbd3dBodyId id, Avbd3dVec3 worldVector, Avbd3dVec3 *out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_Vec3 v_ = avbd3d_Body_GetLocalVector(lid, lg(worldVector));
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_get_world_vector(Avbd3dBodyId id, Avbd3dVec3 localVector, Avbd3dVec3 *out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_Vec3 v_ = avbd3d_Body_GetWorldVector(lid, lg(localVector));
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_get_linear_velocity(Avbd3dBodyId id, Avbd3dVec3 *out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_Vec3 v_ = avbd3d_Body_GetLinearVelocity(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_get_angular_velocity(Avbd3dBodyId id, Avbd3dVec3 *out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_Vec3 v_ = avbd3d_Body_GetAngularVelocity(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_set_linear_velocity(Avbd3dBodyId id, Avbd3dVec3 linearVelocity)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    avbd3d_Body_SetLinearVelocity(lid, lg(linearVelocity));
    return leave(w);
}

Avbd3dResult avbd3d_body_set_angular_velocity(Avbd3dBodyId id, Avbd3dVec3 angularVelocity)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    avbd3d_Body_SetAngularVelocity(lid, lg(angularVelocity));
    return leave(w);
}

Avbd3dResult avbd3d_body_set_target_transform(Avbd3dBodyId id, Avbd3dTransform target, float timeStep, int32_t wake)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    avbd3d_Body_SetTargetTransform(lid, lg(target), lg(timeStep), wake != 0);
    return leave(w);
}

Avbd3dResult avbd3d_body_get_local_point_velocity(Avbd3dBodyId id, Avbd3dVec3 localPoint, Avbd3dVec3 *out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_Vec3 v_ = avbd3d_Body_GetLocalPointVelocity(lid, lg(localPoint));
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_get_world_point_velocity(Avbd3dBodyId id, Avbd3dVec3 worldPoint, Avbd3dVec3 *out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_Vec3 v_ = avbd3d_Body_GetWorldPointVelocity(lid, lg(worldPoint));
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_apply_force(Avbd3dBodyId id, Avbd3dVec3 force, Avbd3dVec3 point, int32_t wake)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    avbd3d_Body_ApplyForce(lid, lg(force), lg(point), wake != 0);
    return leave(w);
}

Avbd3dResult avbd3d_body_apply_force_to_center(Avbd3dBodyId id, Avbd3dVec3 force, int32_t wake)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    avbd3d_Body_ApplyForceToCenter(lid, lg(force), wake != 0);
    return leave(w);
}

Avbd3dResult avbd3d_body_apply_torque(Avbd3dBodyId id, Avbd3dVec3 torque, int32_t wake)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    avbd3d_Body_ApplyTorque(lid, lg(torque), wake != 0);
    return leave(w);
}

Avbd3dResult avbd3d_body_apply_linear_impulse(Avbd3dBodyId id, Avbd3dVec3 impulse, Avbd3dVec3 point, int32_t wake)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    avbd3d_Body_ApplyLinearImpulse(lid, lg(impulse), lg(point), wake != 0);
    return leave(w);
}

Avbd3dResult avbd3d_body_apply_linear_impulse_to_center(Avbd3dBodyId id, Avbd3dVec3 impulse, int32_t wake)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    avbd3d_Body_ApplyLinearImpulseToCenter(lid, lg(impulse), wake != 0);
    return leave(w);
}

Avbd3dResult avbd3d_body_apply_angular_impulse(Avbd3dBodyId id, Avbd3dVec3 impulse, int32_t wake)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    avbd3d_Body_ApplyAngularImpulse(lid, lg(impulse), wake != 0);
    return leave(w);
}

Avbd3dResult avbd3d_body_get_mass(Avbd3dBodyId id, float *out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const float v_ = avbd3d_Body_GetMass(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_get_local_rotational_inertia(Avbd3dBodyId id, Avbd3dMatrix3 *out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_Matrix3 v_ = avbd3d_Body_GetLocalRotationalInertia(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_get_inverse_mass(Avbd3dBodyId id, float *out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const float v_ = avbd3d_Body_GetInverseMass(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_get_world_inverse_rotational_inertia(Avbd3dBodyId id, Avbd3dMatrix3 *out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_Matrix3 v_ = avbd3d_Body_GetWorldInverseRotationalInertia(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_get_local_center(Avbd3dBodyId id, Avbd3dVec3 *out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_Vec3 v_ = avbd3d_Body_GetLocalCenter(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_get_world_center(Avbd3dBodyId id, Avbd3dVec3 *out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_Vec3 v_ = avbd3d_Body_GetWorldCenter(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_set_mass_data(Avbd3dBodyId id, Avbd3dMassData massData)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    avbd3d_Body_SetMassData(lid, lg(massData));
    return leave(w);
}

Avbd3dResult avbd3d_body_get_mass_data(Avbd3dBodyId id, Avbd3dMassData *out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_MassData v_ = avbd3d_Body_GetMassData(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_apply_mass_from_shapes(Avbd3dBodyId id)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    avbd3d_Body_ApplyMassFromShapes(lid);
    return leave(w);
}

Avbd3dResult avbd3d_body_set_linear_damping(Avbd3dBodyId id, float linearDamping)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    avbd3d_Body_SetLinearDamping(lid, lg(linearDamping));
    return leave(w);
}

Avbd3dResult avbd3d_body_set_angular_damping(Avbd3dBodyId id, float angularDamping)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    avbd3d_Body_SetAngularDamping(lid, lg(angularDamping));
    return leave(w);
}

Avbd3dResult avbd3d_body_set_gravity_scale(Avbd3dBodyId id, float gravityScale)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    avbd3d_Body_SetGravityScale(lid, lg(gravityScale));
    return leave(w);
}

Avbd3dResult avbd3d_body_is_awake(Avbd3dBodyId id, int32_t *out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const bool v_ = avbd3d_Body_IsAwake(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_ ? 1 : 0;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_set_awake(Avbd3dBodyId id, int32_t awake)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    avbd3d_Body_SetAwake(lid, awake != 0);
    return leave(w);
}

Avbd3dResult avbd3d_body_is_enabled(Avbd3dBodyId id, int32_t *out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const bool v_ = avbd3d_Body_IsEnabled(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_ ? 1 : 0;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_disable(Avbd3dBodyId id)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    avbd3d_Body_Disable(lid);
    return leave(w);
}

Avbd3dResult avbd3d_body_enable(Avbd3dBodyId id)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    avbd3d_Body_Enable(lid);
    return leave(w);
}

Avbd3dResult avbd3d_body_set_motion_locks(Avbd3dBodyId id, Avbd3dMotionLocks locks)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    avbd3d_Body_SetMotionLocks(lid, lg(locks));
    return leave(w);
}

Avbd3dResult avbd3d_body_set_bullet(Avbd3dBodyId id, int32_t flag)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    avbd3d_Body_SetBullet(lid, flag != 0);
    return leave(w);
}

Avbd3dResult avbd3d_body_enable_hit_events(Avbd3dBodyId id, int32_t flag)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    avbd3d_Body_EnableHitEvents(lid, flag != 0);
    return leave(w);
}

Avbd3dResult avbd3d_body_get_world(Avbd3dBodyId id, Avbd3dWorld **out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_WorldId v_ = avbd3d_Body_GetWorld(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = worldHandle(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_get_shape_count(Avbd3dBodyId id, int32_t *out_count)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out_count)
        return AVBD3D_ERR_INVALID_ARG;
    const int v_ = avbd3d_Body_GetShapeCount(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out_count = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_get_shapes(Avbd3dBodyId id, Avbd3dShapeId *shapeArray, int32_t capacity, int32_t *out_count)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out_count)
        return AVBD3D_ERR_INVALID_ARG;
    if (!shapeArray && capacity > 0)
        return AVBD3D_ERR_INVALID_ARG;
    const int v_ = avbd3d_Body_GetShapes(lid, reinterpret_cast<avbd3d_ShapeId *>(shapeArray), capacity);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out_count = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_get_joint_count(Avbd3dBodyId id, int32_t *out_count)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out_count)
        return AVBD3D_ERR_INVALID_ARG;
    const int v_ = avbd3d_Body_GetJointCount(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out_count = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_get_joints(Avbd3dBodyId id, Avbd3dJointId *jointArray, int32_t capacity, int32_t *out_count)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out_count)
        return AVBD3D_ERR_INVALID_ARG;
    if (!jointArray && capacity > 0)
        return AVBD3D_ERR_INVALID_ARG;
    const int v_ = avbd3d_Body_GetJoints(lid, reinterpret_cast<avbd3d_JointId *>(jointArray), capacity);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out_count = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_get_contact_capacity(Avbd3dBodyId id, int32_t *out_count)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out_count)
        return AVBD3D_ERR_INVALID_ARG;
    const int v_ = avbd3d_Body_GetContactCapacity(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out_count = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_get_contact_data(Avbd3dBodyId id, Avbd3dContactData *contactData, int32_t capacity,
                                          int32_t *out_count)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out_count)
        return AVBD3D_ERR_INVALID_ARG;
    if (!contactData && capacity > 0)
        return AVBD3D_ERR_INVALID_ARG;
    const int v_ = avbd3d_Body_GetContactData(lid, reinterpret_cast<avbd3d_ContactData *>(contactData), capacity);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out_count = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_compute_aabb(Avbd3dBodyId id, Avbd3dAABB *out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_AABB v_ = avbd3d_Body_ComputeAABB(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_cast_ray(Avbd3dBodyId id, Avbd3dVec3 origin, Avbd3dVec3 translation, Avbd3dQueryFilter filter,
                                  float maxFraction, Avbd3dTransform bodyTransform, Avbd3dBodyCastResult *out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_BodyCastResult v_ =
        avbd3d_Body_CastRay(lid, lg(origin), lg(translation), lg(filter), lg(maxFraction), lg(bodyTransform));
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_cast_shape(Avbd3dBodyId id, Avbd3dVec3 origin, const Avbd3dShapeProxy *proxy,
                                    Avbd3dVec3 translation, Avbd3dQueryFilter filter, float maxFraction,
                                    int32_t canEncroach, Avbd3dTransform bodyTransform, Avbd3dBodyCastResult *out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    if (!proxy)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_BodyCastResult v_ =
        avbd3d_Body_CastShape(lid, lg(origin), reinterpret_cast<const avbd3d_ShapeProxy *>(proxy), lg(translation),
                              lg(filter), lg(maxFraction), canEncroach != 0, lg(bodyTransform));
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_overlap_shape(Avbd3dBodyId id, Avbd3dVec3 origin, const Avbd3dShapeProxy *proxy,
                                       Avbd3dQueryFilter filter, Avbd3dTransform bodyTransform, int32_t *out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    if (!proxy)
        return AVBD3D_ERR_INVALID_ARG;
    const bool v_ = avbd3d_Body_OverlapShape(lid, lg(origin), reinterpret_cast<const avbd3d_ShapeProxy *>(proxy),
                                             lg(filter), lg(bodyTransform));
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_ ? 1 : 0;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_collide_mover(Avbd3dBodyId id, Avbd3dBodyPlaneResult *bodyPlanes, int32_t planeCapacity,
                                       Avbd3dVec3 origin, const Avbd3dCapsule *mover, Avbd3dQueryFilter filter,
                                       Avbd3dTransform bodyTransform, int32_t *out_count)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out_count)
        return AVBD3D_ERR_INVALID_ARG;
    if (!bodyPlanes && planeCapacity > 0)
        return AVBD3D_ERR_INVALID_ARG;
    if (!mover && planeCapacity > 0)
        return AVBD3D_ERR_INVALID_ARG;
    const int v_ =
        avbd3d_Body_CollideMover(lid, reinterpret_cast<avbd3d_BodyPlaneResult *>(bodyPlanes), planeCapacity, lg(origin),
                                 reinterpret_cast<const avbd3d_Capsule *>(mover), lg(filter), lg(bodyTransform));
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out_count = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_create_sphere_shape(Avbd3dBodyId bodyId, const Avbd3dShapeDef *def, const Avbd3dSphere *sphere,
                                        Avbd3dShapeId *out_id)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(bodyId, w, lid))
        return r_;
    if (!out_id)
        return AVBD3D_ERR_INVALID_ARG;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_ShapeDef l_def = lg(*def);
    if (!sphere)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_ShapeId v_ = avbd3d_CreateSphereShape(lid, &l_def, reinterpret_cast<const avbd3d_Sphere *>(sphere));
    if (Avbd3dResult r_ = leave(w))
        return r_;
    if (v_.index1 == 0)
        return AVBD3D_ERR_INVALID_ARG;
    *out_id = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_create_capsule_shape(Avbd3dBodyId bodyId, const Avbd3dShapeDef *def, const Avbd3dCapsule *capsule,
                                         Avbd3dShapeId *out_id)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(bodyId, w, lid))
        return r_;
    if (!out_id)
        return AVBD3D_ERR_INVALID_ARG;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_ShapeDef l_def = lg(*def);
    if (!capsule)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_ShapeId v_ = avbd3d_CreateCapsuleShape(lid, &l_def, reinterpret_cast<const avbd3d_Capsule *>(capsule));
    if (Avbd3dResult r_ = leave(w))
        return r_;
    if (v_.index1 == 0)
        return AVBD3D_ERR_INVALID_ARG;
    *out_id = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_create_hull_shape(Avbd3dBodyId bodyId, const Avbd3dShapeDef *def, const Avbd3dHullData *hull,
                                      Avbd3dShapeId *out_id)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(bodyId, w, lid))
        return r_;
    if (!out_id)
        return AVBD3D_ERR_INVALID_ARG;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_ShapeDef l_def = lg(*def);
    if (!hull)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_ShapeId v_ = avbd3d_CreateHullShape(lid, &l_def, reinterpret_cast<const avbd3d_HullData *>(hull));
    if (Avbd3dResult r_ = leave(w))
        return r_;
    if (v_.index1 == 0)
        return AVBD3D_ERR_INVALID_ARG;
    *out_id = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_create_transformed_hull_shape(Avbd3dBodyId bodyId, const Avbd3dShapeDef *def,
                                                  const Avbd3dHullData *hull, Avbd3dTransform transform,
                                                  Avbd3dVec3 scale, Avbd3dShapeId *out_id)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(bodyId, w, lid))
        return r_;
    if (!out_id)
        return AVBD3D_ERR_INVALID_ARG;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_ShapeDef l_def = lg(*def);
    if (!hull)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_ShapeId v_ = avbd3d_CreateTransformedHullShape(
        lid, &l_def, reinterpret_cast<const avbd3d_HullData *>(hull), lg(transform), lg(scale));
    if (Avbd3dResult r_ = leave(w))
        return r_;
    if (v_.index1 == 0)
        return AVBD3D_ERR_INVALID_ARG;
    *out_id = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_create_mesh_shape(Avbd3dBodyId bodyId, const Avbd3dShapeDef *def, const Avbd3dMeshData *mesh,
                                      Avbd3dVec3 scale, Avbd3dShapeId *out_id)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(bodyId, w, lid))
        return r_;
    if (!out_id)
        return AVBD3D_ERR_INVALID_ARG;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_ShapeDef l_def = lg(*def);
    if (!mesh)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_ShapeId v_ =
        avbd3d_CreateMeshShape(lid, &l_def, reinterpret_cast<const avbd3d_MeshData *>(mesh), lg(scale));
    if (Avbd3dResult r_ = leave(w))
        return r_;
    if (v_.index1 == 0)
        return AVBD3D_ERR_INVALID_ARG;
    *out_id = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_create_height_field_shape(Avbd3dBodyId bodyId, const Avbd3dShapeDef *def,
                                              const Avbd3dHeightFieldData *heightField, Avbd3dShapeId *out_id)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(bodyId, w, lid))
        return r_;
    if (!out_id)
        return AVBD3D_ERR_INVALID_ARG;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_ShapeDef l_def = lg(*def);
    if (!heightField)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_ShapeId v_ =
        avbd3d_CreateHeightFieldShape(lid, &l_def, reinterpret_cast<const avbd3d_HeightFieldData *>(heightField));
    if (Avbd3dResult r_ = leave(w))
        return r_;
    if (v_.index1 == 0)
        return AVBD3D_ERR_INVALID_ARG;
    *out_id = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_create_baked_compound_shape(Avbd3dBodyId bodyId, Avbd3dShapeDef *def,
                                                const Avbd3dCompoundData *compound, Avbd3dShapeId *out_id)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(bodyId, w, lid))
        return r_;
    if (!out_id)
        return AVBD3D_ERR_INVALID_ARG;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_ShapeDef l_def = lg(*def);
    if (!compound)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_ShapeId v_ =
        avbd3d_CreateBakedCompoundShape(lid, &l_def, reinterpret_cast<const avbd3d_CompoundData *>(compound));
    if (Avbd3dResult r_ = leave(w))
        return r_;
    if (v_.index1 == 0)
        return AVBD3D_ERR_INVALID_ARG;
    *out_id = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_destroy_shape(Avbd3dShapeId shapeId, int32_t updateBodyMass)
{
    World *w;
    avbd3d_ShapeId lid;
    if (Avbd3dResult r_ = enterShape(shapeId, w, lid))
        return r_;
    avbd3d_DestroyShape(lid, updateBodyMass != 0);
    return leave(w);
}

Avbd3dResult avbd3d_shape_get_type(Avbd3dShapeId id, Avbd3dShapeType *out)
{
    World *w;
    avbd3d_ShapeId lid;
    if (Avbd3dResult r_ = enterShape(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_ShapeType v_ = avbd3d_Shape_GetType(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_shape_get_body(Avbd3dShapeId id, Avbd3dBodyId *out)
{
    World *w;
    avbd3d_ShapeId lid;
    if (Avbd3dResult r_ = enterShape(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_BodyId v_ = avbd3d_Shape_GetBody(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_shape_get_world(Avbd3dShapeId id, Avbd3dWorld **out)
{
    World *w;
    avbd3d_ShapeId lid;
    if (Avbd3dResult r_ = enterShape(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_WorldId v_ = avbd3d_Shape_GetWorld(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = worldHandle(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_shape_is_sensor(Avbd3dShapeId id, int32_t *out)
{
    World *w;
    avbd3d_ShapeId lid;
    if (Avbd3dResult r_ = enterShape(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const bool v_ = avbd3d_Shape_IsSensor(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_ ? 1 : 0;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_shape_set_user_data(Avbd3dShapeId id, void *userData)
{
    World *w;
    avbd3d_ShapeId lid;
    if (Avbd3dResult r_ = enterShape(id, w, lid))
        return r_;
    avbd3d_Shape_SetUserData(lid, userData);
    return leave(w);
}

Avbd3dResult avbd3d_shape_get_user_data(Avbd3dShapeId id, void **out)
{
    World *w;
    avbd3d_ShapeId lid;
    if (Avbd3dResult r_ = enterShape(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    void *v_ = avbd3d_Shape_GetUserData(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_shape_set_density(Avbd3dShapeId id, float density, int32_t updateBodyMass)
{
    World *w;
    avbd3d_ShapeId lid;
    if (Avbd3dResult r_ = enterShape(id, w, lid))
        return r_;
    avbd3d_Shape_SetDensity(lid, lg(density), updateBodyMass != 0);
    return leave(w);
}

Avbd3dResult avbd3d_shape_set_friction(Avbd3dShapeId id, float friction)
{
    World *w;
    avbd3d_ShapeId lid;
    if (Avbd3dResult r_ = enterShape(id, w, lid))
        return r_;
    avbd3d_Shape_SetFriction(lid, lg(friction));
    return leave(w);
}

Avbd3dResult avbd3d_shape_set_restitution(Avbd3dShapeId id, float restitution)
{
    World *w;
    avbd3d_ShapeId lid;
    if (Avbd3dResult r_ = enterShape(id, w, lid))
        return r_;
    avbd3d_Shape_SetRestitution(lid, lg(restitution));
    return leave(w);
}

Avbd3dResult avbd3d_shape_set_surface_material(Avbd3dShapeId id, Avbd3dSurfaceMaterial surfaceMaterial)
{
    World *w;
    avbd3d_ShapeId lid;
    if (Avbd3dResult r_ = enterShape(id, w, lid))
        return r_;
    avbd3d_Shape_SetSurfaceMaterial(lid, lg(surfaceMaterial));
    return leave(w);
}

Avbd3dResult avbd3d_shape_set_filter(Avbd3dShapeId id, Avbd3dFilter filter)
{
    World *w;
    avbd3d_ShapeId lid;
    if (Avbd3dResult r_ = enterShape(id, w, lid))
        return r_;
    avbd3d_Shape_SetFilter(lid, lg(filter));
    return leave(w);
}

Avbd3dResult avbd3d_shape_enable_sensor_events(Avbd3dShapeId id, int32_t flag)
{
    World *w;
    avbd3d_ShapeId lid;
    if (Avbd3dResult r_ = enterShape(id, w, lid))
        return r_;
    avbd3d_Shape_EnableSensorEvents(lid, flag != 0);
    return leave(w);
}

Avbd3dResult avbd3d_shape_enable_contact_events(Avbd3dShapeId id, int32_t flag)
{
    World *w;
    avbd3d_ShapeId lid;
    if (Avbd3dResult r_ = enterShape(id, w, lid))
        return r_;
    avbd3d_Shape_EnableContactEvents(lid, flag != 0);
    return leave(w);
}

Avbd3dResult avbd3d_shape_enable_hit_events(Avbd3dShapeId id, int32_t flag)
{
    World *w;
    avbd3d_ShapeId lid;
    if (Avbd3dResult r_ = enterShape(id, w, lid))
        return r_;
    avbd3d_Shape_EnableHitEvents(lid, flag != 0);
    return leave(w);
}

Avbd3dResult avbd3d_shape_ray_cast(Avbd3dShapeId id, Avbd3dVec3 origin, Avbd3dVec3 translation, Avbd3dCastOutput *out)
{
    World *w;
    avbd3d_ShapeId lid;
    if (Avbd3dResult r_ = enterShape(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_CastOutput v_ = avbd3d_Shape_RayCast(lid, lg(origin), lg(translation));
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_shape_get_contact_capacity(Avbd3dShapeId id, int32_t *out_count)
{
    World *w;
    avbd3d_ShapeId lid;
    if (Avbd3dResult r_ = enterShape(id, w, lid))
        return r_;
    if (!out_count)
        return AVBD3D_ERR_INVALID_ARG;
    const int v_ = avbd3d_Shape_GetContactCapacity(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out_count = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_shape_get_contact_data(Avbd3dShapeId id, Avbd3dContactData *contactData, int32_t capacity,
                                           int32_t *out_count)
{
    World *w;
    avbd3d_ShapeId lid;
    if (Avbd3dResult r_ = enterShape(id, w, lid))
        return r_;
    if (!out_count)
        return AVBD3D_ERR_INVALID_ARG;
    if (!contactData && capacity > 0)
        return AVBD3D_ERR_INVALID_ARG;
    const int v_ = avbd3d_Shape_GetContactData(lid, reinterpret_cast<avbd3d_ContactData *>(contactData), capacity);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out_count = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_shape_get_sensor_capacity(Avbd3dShapeId id, int32_t *out_count)
{
    World *w;
    avbd3d_ShapeId lid;
    if (Avbd3dResult r_ = enterShape(id, w, lid))
        return r_;
    if (!out_count)
        return AVBD3D_ERR_INVALID_ARG;
    const int v_ = avbd3d_Shape_GetSensorCapacity(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out_count = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_shape_get_sensor_data(Avbd3dShapeId id, Avbd3dShapeId *visitorIds, int32_t capacity,
                                          int32_t *out_count)
{
    World *w;
    avbd3d_ShapeId lid;
    if (Avbd3dResult r_ = enterShape(id, w, lid))
        return r_;
    if (!out_count)
        return AVBD3D_ERR_INVALID_ARG;
    if (!visitorIds && capacity > 0)
        return AVBD3D_ERR_INVALID_ARG;
    const int v_ = avbd3d_Shape_GetSensorData(lid, reinterpret_cast<avbd3d_ShapeId *>(visitorIds), capacity);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out_count = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_shape_get_aabb(Avbd3dShapeId id, Avbd3dAABB *out)
{
    World *w;
    avbd3d_ShapeId lid;
    if (Avbd3dResult r_ = enterShape(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_AABB v_ = avbd3d_Shape_GetAABB(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_shape_compute_mass_data(Avbd3dShapeId id, Avbd3dMassData *out)
{
    World *w;
    avbd3d_ShapeId lid;
    if (Avbd3dResult r_ = enterShape(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_MassData v_ = avbd3d_Shape_ComputeMassData(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_joint_get_type(Avbd3dJointId id, Avbd3dJointType *out)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_JointType v_ = avbd3d_Joint_GetType(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_joint_get_body_a(Avbd3dJointId id, Avbd3dBodyId *out)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_BodyId v_ = avbd3d_Joint_GetBodyA(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_joint_get_body_b(Avbd3dJointId id, Avbd3dBodyId *out)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_BodyId v_ = avbd3d_Joint_GetBodyB(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_joint_get_world(Avbd3dJointId id, Avbd3dWorld **out)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_WorldId v_ = avbd3d_Joint_GetWorld(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = worldHandle(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_joint_set_local_frame_a(Avbd3dJointId id, Avbd3dTransform localFrame)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_Joint_SetLocalFrameA(lid, lg(localFrame));
    return leave(w);
}

Avbd3dResult avbd3d_joint_set_local_frame_b(Avbd3dJointId id, Avbd3dTransform localFrame)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_Joint_SetLocalFrameB(lid, lg(localFrame));
    return leave(w);
}

Avbd3dResult avbd3d_joint_set_collide_connected(Avbd3dJointId id, int32_t shouldCollide)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_Joint_SetCollideConnected(lid, shouldCollide != 0);
    return leave(w);
}

Avbd3dResult avbd3d_joint_set_user_data(Avbd3dJointId id, void *userData)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_Joint_SetUserData(lid, userData);
    return leave(w);
}

Avbd3dResult avbd3d_joint_get_user_data(Avbd3dJointId id, void **out)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    void *v_ = avbd3d_Joint_GetUserData(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_joint_wake_bodies(Avbd3dJointId id)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_Joint_WakeBodies(lid);
    return leave(w);
}

Avbd3dResult avbd3d_joint_get_constraint_force(Avbd3dJointId id, Avbd3dVec3 *out)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_Vec3 v_ = avbd3d_Joint_GetConstraintForce(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_joint_get_constraint_torque(Avbd3dJointId id, Avbd3dVec3 *out)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_Vec3 v_ = avbd3d_Joint_GetConstraintTorque(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_joint_set_force_threshold(Avbd3dJointId id, float threshold)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_Joint_SetForceThreshold(lid, lg(threshold));
    return leave(w);
}

Avbd3dResult avbd3d_joint_set_torque_threshold(Avbd3dJointId id, float threshold)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_Joint_SetTorqueThreshold(lid, lg(threshold));
    return leave(w);
}

Avbd3dResult avbd3d_create_parallel_joint(Avbd3dWorld *world, const Avbd3dParallelJointDef *def, Avbd3dJointId *out_id)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out_id)
        return AVBD3D_ERR_INVALID_ARG;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_ParallelJointDef l_def = lg(*def);
    const avbd3d_JointId v_ = avbd3d_CreateParallelJoint(lw, &l_def);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    if (v_.index1 == 0)
        return AVBD3D_ERR_INVALID_ARG;
    *out_id = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_parallel_joint_set_spring_hertz(Avbd3dJointId id, float hertz)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_ParallelJoint_SetSpringHertz(lid, lg(hertz));
    return leave(w);
}

Avbd3dResult avbd3d_parallel_joint_set_spring_damping_ratio(Avbd3dJointId id, float dampingRatio)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_ParallelJoint_SetSpringDampingRatio(lid, lg(dampingRatio));
    return leave(w);
}

Avbd3dResult avbd3d_parallel_joint_set_max_torque(Avbd3dJointId id, float force)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_ParallelJoint_SetMaxTorque(lid, lg(force));
    return leave(w);
}

Avbd3dResult avbd3d_create_distance_joint(Avbd3dWorld *world, const Avbd3dDistanceJointDef *def, Avbd3dJointId *out_id)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out_id)
        return AVBD3D_ERR_INVALID_ARG;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_DistanceJointDef l_def = lg(*def);
    const avbd3d_JointId v_ = avbd3d_CreateDistanceJoint(lw, &l_def);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    if (v_.index1 == 0)
        return AVBD3D_ERR_INVALID_ARG;
    *out_id = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_distance_joint_set_length(Avbd3dJointId id, float length)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_DistanceJoint_SetLength(lid, lg(length));
    return leave(w);
}

Avbd3dResult avbd3d_distance_joint_enable_spring(Avbd3dJointId id, int32_t enableSpring)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_DistanceJoint_EnableSpring(lid, enableSpring != 0);
    return leave(w);
}

Avbd3dResult avbd3d_distance_joint_set_spring_force_range(Avbd3dJointId id, float lowerForce, float upperForce)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_DistanceJoint_SetSpringForceRange(lid, lg(lowerForce), lg(upperForce));
    return leave(w);
}

Avbd3dResult avbd3d_distance_joint_set_spring_hertz(Avbd3dJointId id, float hertz)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_DistanceJoint_SetSpringHertz(lid, lg(hertz));
    return leave(w);
}

Avbd3dResult avbd3d_distance_joint_set_spring_damping_ratio(Avbd3dJointId id, float dampingRatio)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_DistanceJoint_SetSpringDampingRatio(lid, lg(dampingRatio));
    return leave(w);
}

Avbd3dResult avbd3d_distance_joint_enable_limit(Avbd3dJointId id, int32_t enableLimit)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_DistanceJoint_EnableLimit(lid, enableLimit != 0);
    return leave(w);
}

Avbd3dResult avbd3d_distance_joint_set_length_range(Avbd3dJointId id, float minLength, float maxLength)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_DistanceJoint_SetLengthRange(lid, lg(minLength), lg(maxLength));
    return leave(w);
}

Avbd3dResult avbd3d_distance_joint_enable_motor(Avbd3dJointId id, int32_t enableMotor)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_DistanceJoint_EnableMotor(lid, enableMotor != 0);
    return leave(w);
}

Avbd3dResult avbd3d_distance_joint_set_motor_speed(Avbd3dJointId id, float motorSpeed)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_DistanceJoint_SetMotorSpeed(lid, lg(motorSpeed));
    return leave(w);
}

Avbd3dResult avbd3d_distance_joint_set_max_motor_force(Avbd3dJointId id, float force)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_DistanceJoint_SetMaxMotorForce(lid, lg(force));
    return leave(w);
}

Avbd3dResult avbd3d_distance_joint_get_current_length(Avbd3dJointId id, float *out)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const float v_ = avbd3d_DistanceJoint_GetCurrentLength(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_distance_joint_get_motor_force(Avbd3dJointId id, float *out)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const float v_ = avbd3d_DistanceJoint_GetMotorForce(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_create_motor_joint(Avbd3dWorld *world, const Avbd3dMotorJointDef *def, Avbd3dJointId *out_id)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out_id)
        return AVBD3D_ERR_INVALID_ARG;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_MotorJointDef l_def = lg(*def);
    const avbd3d_JointId v_ = avbd3d_CreateMotorJoint(lw, &l_def);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    if (v_.index1 == 0)
        return AVBD3D_ERR_INVALID_ARG;
    *out_id = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_motor_joint_set_linear_velocity(Avbd3dJointId id, Avbd3dVec3 velocity)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_MotorJoint_SetLinearVelocity(lid, lg(velocity));
    return leave(w);
}

Avbd3dResult avbd3d_motor_joint_set_angular_velocity(Avbd3dJointId id, Avbd3dVec3 velocity)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_MotorJoint_SetAngularVelocity(lid, lg(velocity));
    return leave(w);
}

Avbd3dResult avbd3d_motor_joint_set_max_velocity_force(Avbd3dJointId id, float maxForce)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_MotorJoint_SetMaxVelocityForce(lid, lg(maxForce));
    return leave(w);
}

Avbd3dResult avbd3d_motor_joint_set_max_velocity_torque(Avbd3dJointId id, float maxTorque)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_MotorJoint_SetMaxVelocityTorque(lid, lg(maxTorque));
    return leave(w);
}

Avbd3dResult avbd3d_motor_joint_set_linear_hertz(Avbd3dJointId id, float hertz)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_MotorJoint_SetLinearHertz(lid, lg(hertz));
    return leave(w);
}

Avbd3dResult avbd3d_motor_joint_set_linear_damping_ratio(Avbd3dJointId id, float damping)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_MotorJoint_SetLinearDampingRatio(lid, lg(damping));
    return leave(w);
}

Avbd3dResult avbd3d_motor_joint_set_angular_hertz(Avbd3dJointId id, float hertz)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_MotorJoint_SetAngularHertz(lid, lg(hertz));
    return leave(w);
}

Avbd3dResult avbd3d_motor_joint_set_angular_damping_ratio(Avbd3dJointId id, float damping)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_MotorJoint_SetAngularDampingRatio(lid, lg(damping));
    return leave(w);
}

Avbd3dResult avbd3d_motor_joint_set_max_spring_force(Avbd3dJointId id, float maxForce)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_MotorJoint_SetMaxSpringForce(lid, lg(maxForce));
    return leave(w);
}

Avbd3dResult avbd3d_motor_joint_set_max_spring_torque(Avbd3dJointId id, float maxTorque)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_MotorJoint_SetMaxSpringTorque(lid, lg(maxTorque));
    return leave(w);
}

Avbd3dResult avbd3d_create_filter_joint(Avbd3dWorld *world, const Avbd3dFilterJointDef *def, Avbd3dJointId *out_id)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out_id)
        return AVBD3D_ERR_INVALID_ARG;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_FilterJointDef l_def = lg(*def);
    const avbd3d_JointId v_ = avbd3d_CreateFilterJoint(lw, &l_def);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    if (v_.index1 == 0)
        return AVBD3D_ERR_INVALID_ARG;
    *out_id = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_create_prismatic_joint(Avbd3dWorld *world, const Avbd3dPrismaticJointDef *def,
                                           Avbd3dJointId *out_id)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out_id)
        return AVBD3D_ERR_INVALID_ARG;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_PrismaticJointDef l_def = lg(*def);
    const avbd3d_JointId v_ = avbd3d_CreatePrismaticJoint(lw, &l_def);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    if (v_.index1 == 0)
        return AVBD3D_ERR_INVALID_ARG;
    *out_id = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_prismatic_joint_enable_spring(Avbd3dJointId id, int32_t enableSpring)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_PrismaticJoint_EnableSpring(lid, enableSpring != 0);
    return leave(w);
}

Avbd3dResult avbd3d_prismatic_joint_set_spring_hertz(Avbd3dJointId id, float hertz)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_PrismaticJoint_SetSpringHertz(lid, lg(hertz));
    return leave(w);
}

Avbd3dResult avbd3d_prismatic_joint_set_spring_damping_ratio(Avbd3dJointId id, float dampingRatio)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_PrismaticJoint_SetSpringDampingRatio(lid, lg(dampingRatio));
    return leave(w);
}

Avbd3dResult avbd3d_prismatic_joint_set_target_translation(Avbd3dJointId id, float targetTranslation)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_PrismaticJoint_SetTargetTranslation(lid, lg(targetTranslation));
    return leave(w);
}

Avbd3dResult avbd3d_prismatic_joint_enable_limit(Avbd3dJointId id, int32_t enableLimit)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_PrismaticJoint_EnableLimit(lid, enableLimit != 0);
    return leave(w);
}

Avbd3dResult avbd3d_prismatic_joint_set_limits(Avbd3dJointId id, float lower, float upper)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_PrismaticJoint_SetLimits(lid, lg(lower), lg(upper));
    return leave(w);
}

Avbd3dResult avbd3d_prismatic_joint_enable_motor(Avbd3dJointId id, int32_t enableMotor)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_PrismaticJoint_EnableMotor(lid, enableMotor != 0);
    return leave(w);
}

Avbd3dResult avbd3d_prismatic_joint_set_motor_speed(Avbd3dJointId id, float motorSpeed)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_PrismaticJoint_SetMotorSpeed(lid, lg(motorSpeed));
    return leave(w);
}

Avbd3dResult avbd3d_prismatic_joint_set_max_motor_force(Avbd3dJointId id, float force)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_PrismaticJoint_SetMaxMotorForce(lid, lg(force));
    return leave(w);
}

Avbd3dResult avbd3d_prismatic_joint_get_motor_force(Avbd3dJointId id, float *out)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const float v_ = avbd3d_PrismaticJoint_GetMotorForce(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_prismatic_joint_get_translation(Avbd3dJointId id, float *out)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const float v_ = avbd3d_PrismaticJoint_GetTranslation(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_prismatic_joint_get_speed(Avbd3dJointId id, float *out)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const float v_ = avbd3d_PrismaticJoint_GetSpeed(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_create_revolute_joint(Avbd3dWorld *world, const Avbd3dRevoluteJointDef *def, Avbd3dJointId *out_id)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out_id)
        return AVBD3D_ERR_INVALID_ARG;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_RevoluteJointDef l_def = lg(*def);
    const avbd3d_JointId v_ = avbd3d_CreateRevoluteJoint(lw, &l_def);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    if (v_.index1 == 0)
        return AVBD3D_ERR_INVALID_ARG;
    *out_id = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_revolute_joint_enable_spring(Avbd3dJointId id, int32_t enableSpring)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_RevoluteJoint_EnableSpring(lid, enableSpring != 0);
    return leave(w);
}

Avbd3dResult avbd3d_revolute_joint_set_spring_hertz(Avbd3dJointId id, float hertz)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_RevoluteJoint_SetSpringHertz(lid, lg(hertz));
    return leave(w);
}

Avbd3dResult avbd3d_revolute_joint_set_spring_damping_ratio(Avbd3dJointId id, float dampingRatio)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_RevoluteJoint_SetSpringDampingRatio(lid, lg(dampingRatio));
    return leave(w);
}

Avbd3dResult avbd3d_revolute_joint_set_target_angle(Avbd3dJointId id, float targetRadians)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_RevoluteJoint_SetTargetAngle(lid, lg(targetRadians));
    return leave(w);
}

Avbd3dResult avbd3d_revolute_joint_enable_limit(Avbd3dJointId id, int32_t enableLimit)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_RevoluteJoint_EnableLimit(lid, enableLimit != 0);
    return leave(w);
}

Avbd3dResult avbd3d_revolute_joint_set_limits(Avbd3dJointId id, float lowerLimitRadians, float upperLimitRadians)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_RevoluteJoint_SetLimits(lid, lg(lowerLimitRadians), lg(upperLimitRadians));
    return leave(w);
}

Avbd3dResult avbd3d_revolute_joint_enable_motor(Avbd3dJointId id, int32_t enableMotor)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_RevoluteJoint_EnableMotor(lid, enableMotor != 0);
    return leave(w);
}

Avbd3dResult avbd3d_revolute_joint_set_motor_speed(Avbd3dJointId id, float motorSpeed)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_RevoluteJoint_SetMotorSpeed(lid, lg(motorSpeed));
    return leave(w);
}

Avbd3dResult avbd3d_revolute_joint_set_max_motor_torque(Avbd3dJointId id, float torque)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_RevoluteJoint_SetMaxMotorTorque(lid, lg(torque));
    return leave(w);
}

Avbd3dResult avbd3d_revolute_joint_get_motor_torque(Avbd3dJointId id, float *out)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const float v_ = avbd3d_RevoluteJoint_GetMotorTorque(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_revolute_joint_get_angle(Avbd3dJointId id, float *out)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const float v_ = avbd3d_RevoluteJoint_GetAngle(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_revolute_joint_get_speed(Avbd3dJointId id, float *out)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const float v_ = avbd3d_RevoluteJoint_GetSpeed(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_create_spherical_joint(Avbd3dWorld *world, const Avbd3dSphericalJointDef *def,
                                           Avbd3dJointId *out_id)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out_id)
        return AVBD3D_ERR_INVALID_ARG;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_SphericalJointDef l_def = lg(*def);
    const avbd3d_JointId v_ = avbd3d_CreateSphericalJoint(lw, &l_def);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    if (v_.index1 == 0)
        return AVBD3D_ERR_INVALID_ARG;
    *out_id = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_spherical_joint_enable_cone_limit(Avbd3dJointId id, int32_t enableLimit)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_SphericalJoint_EnableConeLimit(lid, enableLimit != 0);
    return leave(w);
}

Avbd3dResult avbd3d_spherical_joint_set_cone_limit(Avbd3dJointId id, float angleRadians)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_SphericalJoint_SetConeLimit(lid, lg(angleRadians));
    return leave(w);
}

Avbd3dResult avbd3d_spherical_joint_enable_twist_limit(Avbd3dJointId id, int32_t enableLimit)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_SphericalJoint_EnableTwistLimit(lid, enableLimit != 0);
    return leave(w);
}

Avbd3dResult avbd3d_spherical_joint_set_twist_limits(Avbd3dJointId id, float lowerLimitRadians, float upperLimitRadians)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_SphericalJoint_SetTwistLimits(lid, lg(lowerLimitRadians), lg(upperLimitRadians));
    return leave(w);
}

Avbd3dResult avbd3d_spherical_joint_enable_spring(Avbd3dJointId id, int32_t enableSpring)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_SphericalJoint_EnableSpring(lid, enableSpring != 0);
    return leave(w);
}

Avbd3dResult avbd3d_spherical_joint_set_spring_hertz(Avbd3dJointId id, float hertz)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_SphericalJoint_SetSpringHertz(lid, lg(hertz));
    return leave(w);
}

Avbd3dResult avbd3d_spherical_joint_set_spring_damping_ratio(Avbd3dJointId id, float dampingRatio)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_SphericalJoint_SetSpringDampingRatio(lid, lg(dampingRatio));
    return leave(w);
}

Avbd3dResult avbd3d_spherical_joint_enable_motor(Avbd3dJointId id, int32_t enableMotor)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_SphericalJoint_EnableMotor(lid, enableMotor != 0);
    return leave(w);
}

Avbd3dResult avbd3d_spherical_joint_set_motor_velocity(Avbd3dJointId id, Avbd3dVec3 motorVelocity)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_SphericalJoint_SetMotorVelocity(lid, lg(motorVelocity));
    return leave(w);
}

Avbd3dResult avbd3d_spherical_joint_set_max_motor_torque(Avbd3dJointId id, float torque)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_SphericalJoint_SetMaxMotorTorque(lid, lg(torque));
    return leave(w);
}

Avbd3dResult avbd3d_spherical_joint_get_motor_torque(Avbd3dJointId id, Avbd3dVec3 *out)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_Vec3 v_ = avbd3d_SphericalJoint_GetMotorTorque(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_spherical_joint_get_cone_angle(Avbd3dJointId id, float *out)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const float v_ = avbd3d_SphericalJoint_GetConeAngle(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_spherical_joint_get_twist_angle(Avbd3dJointId id, float *out)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const float v_ = avbd3d_SphericalJoint_GetTwistAngle(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_spherical_joint_set_target_rotation(Avbd3dJointId id, Avbd3dQuat targetRotation)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_SphericalJoint_SetTargetRotation(lid, lg(targetRotation));
    return leave(w);
}

Avbd3dResult avbd3d_create_weld_joint(Avbd3dWorld *world, const Avbd3dWeldJointDef *def, Avbd3dJointId *out_id)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out_id)
        return AVBD3D_ERR_INVALID_ARG;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_WeldJointDef l_def = lg(*def);
    const avbd3d_JointId v_ = avbd3d_CreateWeldJoint(lw, &l_def);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    if (v_.index1 == 0)
        return AVBD3D_ERR_INVALID_ARG;
    *out_id = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_weld_joint_set_linear_hertz(Avbd3dJointId id, float hertz)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_WeldJoint_SetLinearHertz(lid, lg(hertz));
    return leave(w);
}

Avbd3dResult avbd3d_weld_joint_set_linear_damping_ratio(Avbd3dJointId id, float dampingRatio)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_WeldJoint_SetLinearDampingRatio(lid, lg(dampingRatio));
    return leave(w);
}

Avbd3dResult avbd3d_weld_joint_set_angular_hertz(Avbd3dJointId id, float hertz)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_WeldJoint_SetAngularHertz(lid, lg(hertz));
    return leave(w);
}

Avbd3dResult avbd3d_weld_joint_set_angular_damping_ratio(Avbd3dJointId id, float dampingRatio)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_WeldJoint_SetAngularDampingRatio(lid, lg(dampingRatio));
    return leave(w);
}

Avbd3dResult avbd3d_create_wheel_joint(Avbd3dWorld *world, const Avbd3dWheelJointDef *def, Avbd3dJointId *out_id)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out_id)
        return AVBD3D_ERR_INVALID_ARG;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_WheelJointDef l_def = lg(*def);
    const avbd3d_JointId v_ = avbd3d_CreateWheelJoint(lw, &l_def);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    if (v_.index1 == 0)
        return AVBD3D_ERR_INVALID_ARG;
    *out_id = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_wheel_joint_enable_suspension(Avbd3dJointId id, int32_t flag)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_WheelJoint_EnableSuspension(lid, flag != 0);
    return leave(w);
}

Avbd3dResult avbd3d_wheel_joint_set_suspension_hertz(Avbd3dJointId id, float hertz)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_WheelJoint_SetSuspensionHertz(lid, lg(hertz));
    return leave(w);
}

Avbd3dResult avbd3d_wheel_joint_set_suspension_damping_ratio(Avbd3dJointId id, float dampingRatio)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_WheelJoint_SetSuspensionDampingRatio(lid, lg(dampingRatio));
    return leave(w);
}

Avbd3dResult avbd3d_wheel_joint_enable_suspension_limit(Avbd3dJointId id, int32_t flag)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_WheelJoint_EnableSuspensionLimit(lid, flag != 0);
    return leave(w);
}

Avbd3dResult avbd3d_wheel_joint_set_suspension_limits(Avbd3dJointId id, float lower, float upper)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_WheelJoint_SetSuspensionLimits(lid, lg(lower), lg(upper));
    return leave(w);
}

Avbd3dResult avbd3d_wheel_joint_enable_spin_motor(Avbd3dJointId id, int32_t flag)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_WheelJoint_EnableSpinMotor(lid, flag != 0);
    return leave(w);
}

Avbd3dResult avbd3d_wheel_joint_set_spin_motor_speed(Avbd3dJointId id, float speed)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_WheelJoint_SetSpinMotorSpeed(lid, lg(speed));
    return leave(w);
}

Avbd3dResult avbd3d_wheel_joint_set_max_spin_torque(Avbd3dJointId id, float torque)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_WheelJoint_SetMaxSpinTorque(lid, lg(torque));
    return leave(w);
}

Avbd3dResult avbd3d_wheel_joint_get_spin_torque(Avbd3dJointId id, float *out)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const float v_ = avbd3d_WheelJoint_GetSpinTorque(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_wheel_joint_get_spin_speed(Avbd3dJointId id, float *out)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const float v_ = avbd3d_WheelJoint_GetSpinSpeed(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_wheel_joint_get_steering_angle(Avbd3dJointId id, float *out)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const float v_ = avbd3d_WheelJoint_GetSteeringAngle(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_wheel_joint_enable_steering(Avbd3dJointId id, int32_t flag)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_WheelJoint_EnableSteering(lid, flag != 0);
    return leave(w);
}

Avbd3dResult avbd3d_wheel_joint_set_steering_hertz(Avbd3dJointId id, float hertz)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_WheelJoint_SetSteeringHertz(lid, lg(hertz));
    return leave(w);
}

Avbd3dResult avbd3d_wheel_joint_set_steering_damping_ratio(Avbd3dJointId id, float dampingRatio)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_WheelJoint_SetSteeringDampingRatio(lid, lg(dampingRatio));
    return leave(w);
}

Avbd3dResult avbd3d_wheel_joint_set_max_steering_torque(Avbd3dJointId id, float torque)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_WheelJoint_SetMaxSteeringTorque(lid, lg(torque));
    return leave(w);
}

Avbd3dResult avbd3d_wheel_joint_enable_steering_limit(Avbd3dJointId id, int32_t flag)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_WheelJoint_EnableSteeringLimit(lid, flag != 0);
    return leave(w);
}

Avbd3dResult avbd3d_wheel_joint_set_steering_limits(Avbd3dJointId id, float lowerRadians, float upperRadians)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_WheelJoint_SetSteeringLimits(lid, lg(lowerRadians), lg(upperRadians));
    return leave(w);
}

Avbd3dResult avbd3d_wheel_joint_set_target_steering_angle(Avbd3dJointId id, float radians)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_WheelJoint_SetTargetSteeringAngle(lid, lg(radians));
    return leave(w);
}

Avbd3dResult avbd3d_contact_get_data(Avbd3dContactId id, Avbd3dContactData *out)
{
    World *w;
    avbd3d_ContactId lid;
    if (Avbd3dResult r_ = enterContact(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_ContactData v_ = avbd3d_Contact_GetData(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_is_step_in_flight(Avbd3dWorld *world, int32_t *out)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const bool v_ = avbd3d_World_IsStepInFlight(lw);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_ ? 1 : 0;
    return AVBD3D_OK;
}

Avbd3dSolverDef avbd3d_default_solver_def(void)
{
    return pb(avbd3d_DefaultSolverDef());
}

Avbd3dResult avbd3d_world_set_solver_def(Avbd3dWorld *world, const Avbd3dSolverDef *def)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_SolverDef l_def = lg(*def);
    avbd3d_World_SetSolverDef(lw, &l_def);
    return leave(w);
}

Avbd3dResult avbd3d_world_set_friction_mixing(Avbd3dWorld *world, Avbd3dMixingRule rule)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    avbd3d_World_SetFrictionMixing(lw, lg(rule));
    return leave(w);
}

Avbd3dResult avbd3d_world_set_restitution_mixing(Avbd3dWorld *world, Avbd3dMixingRule rule)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    avbd3d_World_SetRestitutionMixing(lw, lg(rule));
    return leave(w);
}

Avbd3dResult avbd3d_world_get_body_data(Avbd3dWorld *world, const Avbd3dBodyId *ids, int32_t count,
                                        Avbd3dBodyDataType type, void *data, int32_t *out)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    if (!ids && count > 0)
        return AVBD3D_ERR_INVALID_ARG;
    const int v_ = avbd3d_World_GetBodyData(lw, reinterpret_cast<const avbd3d_BodyId *>(ids), count, lg(type), data);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_world_set_body_data(Avbd3dWorld *world, const Avbd3dBodyId *ids, int32_t count,
                                        Avbd3dBodyDataType type, const void *data, int32_t *out)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    if (!ids && count > 0)
        return AVBD3D_ERR_INVALID_ARG;
    const int v_ = avbd3d_World_SetBodyData(lw, reinterpret_cast<const avbd3d_BodyId *>(ids), count, lg(type), data);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_;
    return AVBD3D_OK;
}

Avbd3dGenericJointDef avbd3d_default_generic_joint_def(void)
{
    return pb(avbd3d_DefaultAvbdJointDef());
}

Avbd3dResult avbd3d_create_generic_joint(Avbd3dWorld *world, const Avbd3dGenericJointDef *def, Avbd3dJointId *out_id)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out_id)
        return AVBD3D_ERR_INVALID_ARG;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_AvbdJointDef l_def = lg(*def);
    const avbd3d_JointId v_ = avbd3d_CreateAvbdJoint(lw, &l_def);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    if (v_.index1 == 0)
        return AVBD3D_ERR_INVALID_ARG;
    *out_id = pb(v_);
    return AVBD3D_OK;
}

Avbd3dSpringDef avbd3d_default_spring_def(void)
{
    return pb(avbd3d_DefaultAvbdSpringDef());
}

Avbd3dResult avbd3d_create_spring(Avbd3dWorld *world, const Avbd3dSpringDef *def, Avbd3dJointId *out_id)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out_id)
        return AVBD3D_ERR_INVALID_ARG;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_AvbdSpringDef l_def = lg(*def);
    const avbd3d_JointId v_ = avbd3d_CreateAvbdSpring(lw, &l_def);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    if (v_.index1 == 0)
        return AVBD3D_ERR_INVALID_ARG;
    *out_id = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_grab_begin(Avbd3dWorld *world, Avbd3dBodyId seed, Avbd3dVec3 center, float radius,
                               int32_t followConnections, Avbd3dBodyId *outBodies, int32_t maxCount, int32_t *out_count)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out_count)
        return AVBD3D_ERR_INVALID_ARG;
    if (!outBodies && maxCount > 0)
        return AVBD3D_ERR_INVALID_ARG;
    const int v_ = avbd3d_World_GrabGroup(lw, lg(seed), lg(center), lg(radius), followConnections != 0,
                                          reinterpret_cast<avbd3d_BodyId *>(outBodies), maxCount);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out_count = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_grab_survivors(Avbd3dWorld *world, Avbd3dBodyId seed, const Avbd3dBodyId *group, int32_t groupCount,
                                   uint8_t *outKeep, int32_t *out_count)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out_count)
        return AVBD3D_ERR_INVALID_ARG;
    if (!group && groupCount > 0)
        return AVBD3D_ERR_INVALID_ARG;
    if (!outKeep && groupCount > 0)
        return AVBD3D_ERR_INVALID_ARG;
    const int v_ = avbd3d_World_GrabSurvivors(lw, lg(seed), reinterpret_cast<const avbd3d_BodyId *>(group), groupCount,
                                              reinterpret_cast<uint8_t *>(outKeep));
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out_count = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_grab_move(Avbd3dWorld *world, const Avbd3dGrabHandle *handles, int32_t count)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!handles && count > 0)
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_World_GrabDrag(lw, reinterpret_cast<const avbd3d_GrabHandle *>(handles), count);
    return leave(w);
}

Avbd3dResult avbd3d_grab_end(Avbd3dWorld *world)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    avbd3d_World_GrabRelease(lw);
    return leave(w);
}

Avbd3dRenderDef avbd3d_default_render_def(void)
{
    return pb(avbd3d_DefaultRenderDef());
}

Avbd3dResult avbd3d_world_write_instances(Avbd3dWorld *world, const Avbd3dRenderDef *def)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_RenderDef l_def = lg(*def);
    avbd3d_World_WriteInstances(lw, &l_def);
    return leave(w);
}

Avbd3dResult avbd3d_world_get_instance_buffer(Avbd3dWorld *world, uint64_t *out)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const uint64_t v_ = avbd3d_World_GetInstanceBuffer(lw);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_world_get_render_group_count(Avbd3dWorld *world, int32_t *out_count)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out_count)
        return AVBD3D_ERR_INVALID_ARG;
    const int v_ = avbd3d_World_GetRenderGroupCount(lw);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out_count = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_world_get_render_group_range(Avbd3dWorld *world, int32_t group, Avbd3dRenderRange *out)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_RenderRange v_ = avbd3d_World_GetRenderGroupRange(lw, group);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = pb(v_);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_body_set_render_group(Avbd3dBodyId id, int32_t group)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    avbd3d_Body_SetRenderGroup(lid, group);
    return leave(w);
}

Avbd3dResult avbd3d_body_get_render_group(Avbd3dBodyId id, int32_t *out)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const int v_ = avbd3d_Body_GetRenderGroup(lid);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_;
    return AVBD3D_OK;
}

Avbd3dContactMarkerDef avbd3d_default_contact_marker_def(void)
{
    return pb(avbd3d_DefaultContactMarkerDef());
}

Avbd3dResult avbd3d_world_write_contact_markers(Avbd3dWorld *world, const Avbd3dContactMarkerDef *def, int32_t *out)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_ContactMarkerDef l_def = lg(*def);
    const int v_ = avbd3d_World_WriteContactMarkers(lw, &l_def);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_world_get_contact_marker_buffer(Avbd3dWorld *world, uint64_t *out)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const uint64_t v_ = avbd3d_World_GetContactMarkerBuffer(lw);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_world_get_contact_markers_requested(Avbd3dWorld *world, int32_t *out)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    const int v_ = avbd3d_World_GetContactMarkersRequested(lw);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out = v_;
    return AVBD3D_OK;
}
