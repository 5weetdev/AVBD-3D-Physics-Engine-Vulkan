// Conversions between the public structs and the implementation structs (included by avbd3d_capi.h).
#pragma once

static_assert(sizeof(Avbd3dBodyId) == sizeof(avbd3d_BodyId), "avbd3d_BodyId");
static inline avbd3d_BodyId lg(const Avbd3dBodyId &o) { avbd3d_BodyId r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dBodyId pb(const avbd3d_BodyId &o) { Avbd3dBodyId r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dShapeId) == sizeof(avbd3d_ShapeId), "avbd3d_ShapeId");
static inline avbd3d_ShapeId lg(const Avbd3dShapeId &o) { avbd3d_ShapeId r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dShapeId pb(const avbd3d_ShapeId &o) { Avbd3dShapeId r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dJointId) == sizeof(avbd3d_JointId), "avbd3d_JointId");
static inline avbd3d_JointId lg(const Avbd3dJointId &o) { avbd3d_JointId r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dJointId pb(const avbd3d_JointId &o) { Avbd3dJointId r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dContactId) == sizeof(avbd3d_ContactId), "avbd3d_ContactId");
static inline avbd3d_ContactId lg(const Avbd3dContactId &o) { avbd3d_ContactId r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dContactId pb(const avbd3d_ContactId &o) { Avbd3dContactId r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dVec3) == sizeof(avbd3d_Vec3), "avbd3d_Vec3");
static inline avbd3d_Vec3 lg(const Avbd3dVec3 &o) { avbd3d_Vec3 r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dVec3 pb(const avbd3d_Vec3 &o) { Avbd3dVec3 r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dQuat) == sizeof(avbd3d_Quat), "avbd3d_Quat");
static inline avbd3d_Quat lg(const Avbd3dQuat &o) { avbd3d_Quat r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dQuat pb(const avbd3d_Quat &o) { Avbd3dQuat r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dTransform) == sizeof(avbd3d_Transform), "avbd3d_Transform");
static inline avbd3d_Transform lg(const Avbd3dTransform &o) { avbd3d_Transform r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dTransform pb(const avbd3d_Transform &o) { Avbd3dTransform r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dMatrix3) == sizeof(avbd3d_Matrix3), "avbd3d_Matrix3");
static inline avbd3d_Matrix3 lg(const Avbd3dMatrix3 &o) { avbd3d_Matrix3 r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dMatrix3 pb(const avbd3d_Matrix3 &o) { Avbd3dMatrix3 r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dAABB) == sizeof(avbd3d_AABB), "avbd3d_AABB");
static inline avbd3d_AABB lg(const Avbd3dAABB &o) { avbd3d_AABB r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dAABB pb(const avbd3d_AABB &o) { Avbd3dAABB r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dPlane) == sizeof(avbd3d_Plane), "avbd3d_Plane");
static inline avbd3d_Plane lg(const Avbd3dPlane &o) { avbd3d_Plane r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dPlane pb(const avbd3d_Plane &o) { Avbd3dPlane r; memcpy(&r, &o, sizeof r); return r; }

static inline avbd3d_WorldDef lg(const Avbd3dWorldDef &o)
{
    avbd3d_WorldDef r = avbd3d_DefaultWorldDef();
    r.gravity = lg(o.gravity);
    r.restitutionThreshold = o.restitutionThreshold;
    r.hitEventThreshold = o.hitEventThreshold;
    r.enableSleep = (o.enableSleep) != 0;
    r.enableContinuous = (o.enableContinuous) != 0;
    return r;
}
static inline Avbd3dWorldDef pb(const avbd3d_WorldDef &o)
{
    Avbd3dWorldDef r; memset(&r, 0, sizeof r);
    r.gravity = pb(o.gravity);
    r.restitutionThreshold = o.restitutionThreshold;
    r.hitEventThreshold = o.hitEventThreshold;
    r.enableSleep = (o.enableSleep) ? 1 : 0;
    r.enableContinuous = (o.enableContinuous) ? 1 : 0;
    return r;
}

static inline avbd3d_BodyType lg(Avbd3dBodyType o) { return (avbd3d_BodyType)o; }
static inline Avbd3dBodyType pb(avbd3d_BodyType o) { return (Avbd3dBodyType)o; }

static inline avbd3d_MotionLocks lg(const Avbd3dMotionLocks &o)
{
    avbd3d_MotionLocks r; memset(&r, 0, sizeof r);
    r.linearX = (o.linearX) != 0;
    r.linearY = (o.linearY) != 0;
    r.linearZ = (o.linearZ) != 0;
    r.angularX = (o.angularX) != 0;
    r.angularY = (o.angularY) != 0;
    r.angularZ = (o.angularZ) != 0;
    return r;
}
static inline Avbd3dMotionLocks pb(const avbd3d_MotionLocks &o)
{
    Avbd3dMotionLocks r; memset(&r, 0, sizeof r);
    r.linearX = (o.linearX) ? 1 : 0;
    r.linearY = (o.linearY) ? 1 : 0;
    r.linearZ = (o.linearZ) ? 1 : 0;
    r.angularX = (o.angularX) ? 1 : 0;
    r.angularY = (o.angularY) ? 1 : 0;
    r.angularZ = (o.angularZ) ? 1 : 0;
    return r;
}

static inline avbd3d_BodyDef lg(const Avbd3dBodyDef &o)
{
    avbd3d_BodyDef r = avbd3d_DefaultBodyDef();
    r.type = (avbd3d_BodyType)o.type;
    r.position = lg(o.position);
    r.rotation = lg(o.rotation);
    r.linearVelocity = lg(o.linearVelocity);
    r.angularVelocity = lg(o.angularVelocity);
    r.linearDamping = o.linearDamping;
    r.angularDamping = o.angularDamping;
    r.gravityScale = o.gravityScale;
    r.userData = reinterpret_cast<void*>(o.userData);
    r.motionLocks = lg(o.motionLocks);
    r.isAwake = (o.isAwake) != 0;
    r.isBullet = (o.isBullet) != 0;
    r.isEnabled = (o.isEnabled) != 0;
    return r;
}
static inline Avbd3dBodyDef pb(const avbd3d_BodyDef &o)
{
    Avbd3dBodyDef r; memset(&r, 0, sizeof r);
    r.type = (int32_t)o.type;
    r.position = pb(o.position);
    r.rotation = pb(o.rotation);
    r.linearVelocity = pb(o.linearVelocity);
    r.angularVelocity = pb(o.angularVelocity);
    r.linearDamping = o.linearDamping;
    r.angularDamping = o.angularDamping;
    r.gravityScale = o.gravityScale;
    r.userData = reinterpret_cast<void*>(o.userData);
    r.motionLocks = pb(o.motionLocks);
    r.isAwake = (o.isAwake) ? 1 : 0;
    r.isBullet = (o.isBullet) ? 1 : 0;
    r.isEnabled = (o.isEnabled) ? 1 : 0;
    return r;
}

static_assert(sizeof(Avbd3dFilter) == sizeof(avbd3d_Filter), "avbd3d_Filter");
static inline avbd3d_Filter lg(const Avbd3dFilter &o) { avbd3d_Filter r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dFilter pb(const avbd3d_Filter &o) { Avbd3dFilter r; memcpy(&r, &o, sizeof r); return r; }

static inline avbd3d_SurfaceMaterial lg(const Avbd3dSurfaceMaterial &o)
{
    avbd3d_SurfaceMaterial r = avbd3d_DefaultSurfaceMaterial();
    r.friction = o.friction;
    r.restitution = o.restitution;
    r.rollingResistance = o.rollingResistance;
    r.tangentVelocity = lg(o.tangentVelocity);
    r.userMaterialId = o.userMaterialId;
    return r;
}
static inline Avbd3dSurfaceMaterial pb(const avbd3d_SurfaceMaterial &o)
{
    Avbd3dSurfaceMaterial r; memset(&r, 0, sizeof r);
    r.friction = o.friction;
    r.restitution = o.restitution;
    r.rollingResistance = o.rollingResistance;
    r.tangentVelocity = pb(o.tangentVelocity);
    r.userMaterialId = o.userMaterialId;
    return r;
}

static inline avbd3d_ShapeType lg(Avbd3dShapeType o) { return (avbd3d_ShapeType)o; }
static inline Avbd3dShapeType pb(avbd3d_ShapeType o) { return (Avbd3dShapeType)o; }

static inline avbd3d_ShapeDef lg(const Avbd3dShapeDef &o)
{
    avbd3d_ShapeDef r = avbd3d_DefaultShapeDef();
    r.userData = reinterpret_cast<void*>(o.userData);
    r.baseMaterial = lg(o.baseMaterial);
    r.density = o.density;
    r.explosionScale = o.explosionScale;
    r.filter = lg(o.filter);
    r.isSensor = (o.isSensor) != 0;
    r.enableSensorEvents = (o.enableSensorEvents) != 0;
    r.enableContactEvents = (o.enableContactEvents) != 0;
    r.enableHitEvents = (o.enableHitEvents) != 0;
    r.updateBodyMass = (o.updateBodyMass) != 0;
    return r;
}
static inline Avbd3dShapeDef pb(const avbd3d_ShapeDef &o)
{
    Avbd3dShapeDef r; memset(&r, 0, sizeof r);
    r.userData = reinterpret_cast<void*>(o.userData);
    r.baseMaterial = pb(o.baseMaterial);
    r.density = o.density;
    r.explosionScale = o.explosionScale;
    r.filter = pb(o.filter);
    r.isSensor = (o.isSensor) ? 1 : 0;
    r.enableSensorEvents = (o.enableSensorEvents) ? 1 : 0;
    r.enableContactEvents = (o.enableContactEvents) ? 1 : 0;
    r.enableHitEvents = (o.enableHitEvents) ? 1 : 0;
    r.updateBodyMass = (o.updateBodyMass) ? 1 : 0;
    return r;
}

static inline avbd3d_JointType lg(Avbd3dJointType o) { return (avbd3d_JointType)o; }
static inline Avbd3dJointType pb(avbd3d_JointType o) { return (Avbd3dJointType)o; }

static inline avbd3d_JointDef lg(const Avbd3dJointDef &o)
{
    avbd3d_JointDef r = avbd3d_DefaultFilterJointDef().base;
    r.userData = reinterpret_cast<void*>(o.userData);
    r.bodyIdA = lg(o.bodyIdA);
    r.bodyIdB = lg(o.bodyIdB);
    r.localFrameA = lg(o.localFrameA);
    r.localFrameB = lg(o.localFrameB);
    r.forceThreshold = o.forceThreshold;
    r.torqueThreshold = o.torqueThreshold;
    r.collideConnected = (o.collideConnected) != 0;
    return r;
}
static inline Avbd3dJointDef pb(const avbd3d_JointDef &o)
{
    Avbd3dJointDef r; memset(&r, 0, sizeof r);
    r.userData = reinterpret_cast<void*>(o.userData);
    r.bodyIdA = pb(o.bodyIdA);
    r.bodyIdB = pb(o.bodyIdB);
    r.localFrameA = pb(o.localFrameA);
    r.localFrameB = pb(o.localFrameB);
    r.forceThreshold = o.forceThreshold;
    r.torqueThreshold = o.torqueThreshold;
    r.collideConnected = (o.collideConnected) ? 1 : 0;
    return r;
}

static inline avbd3d_DistanceJointDef lg(const Avbd3dDistanceJointDef &o)
{
    avbd3d_DistanceJointDef r = avbd3d_DefaultDistanceJointDef();
    r.base = lg(o.base);
    r.length = o.length;
    r.enableSpring = (o.enableSpring) != 0;
    r.lowerSpringForce = o.lowerSpringForce;
    r.upperSpringForce = o.upperSpringForce;
    r.hertz = o.hertz;
    r.dampingRatio = o.dampingRatio;
    r.enableLimit = (o.enableLimit) != 0;
    r.minLength = o.minLength;
    r.maxLength = o.maxLength;
    r.enableMotor = (o.enableMotor) != 0;
    r.maxMotorForce = o.maxMotorForce;
    r.motorSpeed = o.motorSpeed;
    return r;
}
static inline Avbd3dDistanceJointDef pb(const avbd3d_DistanceJointDef &o)
{
    Avbd3dDistanceJointDef r; memset(&r, 0, sizeof r);
    r.base = pb(o.base);
    r.length = o.length;
    r.enableSpring = (o.enableSpring) ? 1 : 0;
    r.lowerSpringForce = o.lowerSpringForce;
    r.upperSpringForce = o.upperSpringForce;
    r.hertz = o.hertz;
    r.dampingRatio = o.dampingRatio;
    r.enableLimit = (o.enableLimit) ? 1 : 0;
    r.minLength = o.minLength;
    r.maxLength = o.maxLength;
    r.enableMotor = (o.enableMotor) ? 1 : 0;
    r.maxMotorForce = o.maxMotorForce;
    r.motorSpeed = o.motorSpeed;
    return r;
}

static inline avbd3d_MotorJointDef lg(const Avbd3dMotorJointDef &o)
{
    avbd3d_MotorJointDef r = avbd3d_DefaultMotorJointDef();
    r.base = lg(o.base);
    r.linearVelocity = lg(o.linearVelocity);
    r.maxVelocityForce = o.maxVelocityForce;
    r.angularVelocity = lg(o.angularVelocity);
    r.maxVelocityTorque = o.maxVelocityTorque;
    r.linearHertz = o.linearHertz;
    r.linearDampingRatio = o.linearDampingRatio;
    r.maxSpringForce = o.maxSpringForce;
    r.angularHertz = o.angularHertz;
    r.angularDampingRatio = o.angularDampingRatio;
    r.maxSpringTorque = o.maxSpringTorque;
    return r;
}
static inline Avbd3dMotorJointDef pb(const avbd3d_MotorJointDef &o)
{
    Avbd3dMotorJointDef r; memset(&r, 0, sizeof r);
    r.base = pb(o.base);
    r.linearVelocity = pb(o.linearVelocity);
    r.maxVelocityForce = o.maxVelocityForce;
    r.angularVelocity = pb(o.angularVelocity);
    r.maxVelocityTorque = o.maxVelocityTorque;
    r.linearHertz = o.linearHertz;
    r.linearDampingRatio = o.linearDampingRatio;
    r.maxSpringForce = o.maxSpringForce;
    r.angularHertz = o.angularHertz;
    r.angularDampingRatio = o.angularDampingRatio;
    r.maxSpringTorque = o.maxSpringTorque;
    return r;
}

static inline avbd3d_FilterJointDef lg(const Avbd3dFilterJointDef &o)
{
    avbd3d_FilterJointDef r = avbd3d_DefaultFilterJointDef();
    r.base = lg(o.base);
    return r;
}
static inline Avbd3dFilterJointDef pb(const avbd3d_FilterJointDef &o)
{
    Avbd3dFilterJointDef r; memset(&r, 0, sizeof r);
    r.base = pb(o.base);
    return r;
}

static inline avbd3d_ParallelJointDef lg(const Avbd3dParallelJointDef &o)
{
    avbd3d_ParallelJointDef r = avbd3d_DefaultParallelJointDef();
    r.base = lg(o.base);
    r.hertz = o.hertz;
    r.dampingRatio = o.dampingRatio;
    r.maxTorque = o.maxTorque;
    return r;
}
static inline Avbd3dParallelJointDef pb(const avbd3d_ParallelJointDef &o)
{
    Avbd3dParallelJointDef r; memset(&r, 0, sizeof r);
    r.base = pb(o.base);
    r.hertz = o.hertz;
    r.dampingRatio = o.dampingRatio;
    r.maxTorque = o.maxTorque;
    return r;
}

static inline avbd3d_PrismaticJointDef lg(const Avbd3dPrismaticJointDef &o)
{
    avbd3d_PrismaticJointDef r = avbd3d_DefaultPrismaticJointDef();
    r.base = lg(o.base);
    r.enableSpring = (o.enableSpring) != 0;
    r.hertz = o.hertz;
    r.dampingRatio = o.dampingRatio;
    r.targetTranslation = o.targetTranslation;
    r.enableLimit = (o.enableLimit) != 0;
    r.lowerTranslation = o.lowerTranslation;
    r.upperTranslation = o.upperTranslation;
    r.enableMotor = (o.enableMotor) != 0;
    r.maxMotorForce = o.maxMotorForce;
    r.motorSpeed = o.motorSpeed;
    return r;
}
static inline Avbd3dPrismaticJointDef pb(const avbd3d_PrismaticJointDef &o)
{
    Avbd3dPrismaticJointDef r; memset(&r, 0, sizeof r);
    r.base = pb(o.base);
    r.enableSpring = (o.enableSpring) ? 1 : 0;
    r.hertz = o.hertz;
    r.dampingRatio = o.dampingRatio;
    r.targetTranslation = o.targetTranslation;
    r.enableLimit = (o.enableLimit) ? 1 : 0;
    r.lowerTranslation = o.lowerTranslation;
    r.upperTranslation = o.upperTranslation;
    r.enableMotor = (o.enableMotor) ? 1 : 0;
    r.maxMotorForce = o.maxMotorForce;
    r.motorSpeed = o.motorSpeed;
    return r;
}

static inline avbd3d_RevoluteJointDef lg(const Avbd3dRevoluteJointDef &o)
{
    avbd3d_RevoluteJointDef r = avbd3d_DefaultRevoluteJointDef();
    r.base = lg(o.base);
    r.targetAngle = o.targetAngle;
    r.enableSpring = (o.enableSpring) != 0;
    r.hertz = o.hertz;
    r.dampingRatio = o.dampingRatio;
    r.enableLimit = (o.enableLimit) != 0;
    r.lowerAngle = o.lowerAngle;
    r.upperAngle = o.upperAngle;
    r.enableMotor = (o.enableMotor) != 0;
    r.maxMotorTorque = o.maxMotorTorque;
    r.motorSpeed = o.motorSpeed;
    return r;
}
static inline Avbd3dRevoluteJointDef pb(const avbd3d_RevoluteJointDef &o)
{
    Avbd3dRevoluteJointDef r; memset(&r, 0, sizeof r);
    r.base = pb(o.base);
    r.targetAngle = o.targetAngle;
    r.enableSpring = (o.enableSpring) ? 1 : 0;
    r.hertz = o.hertz;
    r.dampingRatio = o.dampingRatio;
    r.enableLimit = (o.enableLimit) ? 1 : 0;
    r.lowerAngle = o.lowerAngle;
    r.upperAngle = o.upperAngle;
    r.enableMotor = (o.enableMotor) ? 1 : 0;
    r.maxMotorTorque = o.maxMotorTorque;
    r.motorSpeed = o.motorSpeed;
    return r;
}

static inline avbd3d_SphericalJointDef lg(const Avbd3dSphericalJointDef &o)
{
    avbd3d_SphericalJointDef r = avbd3d_DefaultSphericalJointDef();
    r.base = lg(o.base);
    r.enableSpring = (o.enableSpring) != 0;
    r.hertz = o.hertz;
    r.dampingRatio = o.dampingRatio;
    r.targetRotation = lg(o.targetRotation);
    r.enableConeLimit = (o.enableConeLimit) != 0;
    r.coneAngle = o.coneAngle;
    r.enableTwistLimit = (o.enableTwistLimit) != 0;
    r.lowerTwistAngle = o.lowerTwistAngle;
    r.upperTwistAngle = o.upperTwistAngle;
    r.enableMotor = (o.enableMotor) != 0;
    r.maxMotorTorque = o.maxMotorTorque;
    r.motorVelocity = lg(o.motorVelocity);
    return r;
}
static inline Avbd3dSphericalJointDef pb(const avbd3d_SphericalJointDef &o)
{
    Avbd3dSphericalJointDef r; memset(&r, 0, sizeof r);
    r.base = pb(o.base);
    r.enableSpring = (o.enableSpring) ? 1 : 0;
    r.hertz = o.hertz;
    r.dampingRatio = o.dampingRatio;
    r.targetRotation = pb(o.targetRotation);
    r.enableConeLimit = (o.enableConeLimit) ? 1 : 0;
    r.coneAngle = o.coneAngle;
    r.enableTwistLimit = (o.enableTwistLimit) ? 1 : 0;
    r.lowerTwistAngle = o.lowerTwistAngle;
    r.upperTwistAngle = o.upperTwistAngle;
    r.enableMotor = (o.enableMotor) ? 1 : 0;
    r.maxMotorTorque = o.maxMotorTorque;
    r.motorVelocity = pb(o.motorVelocity);
    return r;
}

static inline avbd3d_WeldJointDef lg(const Avbd3dWeldJointDef &o)
{
    avbd3d_WeldJointDef r = avbd3d_DefaultWeldJointDef();
    r.base = lg(o.base);
    r.linearHertz = o.linearHertz;
    r.angularHertz = o.angularHertz;
    r.linearDampingRatio = o.linearDampingRatio;
    r.angularDampingRatio = o.angularDampingRatio;
    return r;
}
static inline Avbd3dWeldJointDef pb(const avbd3d_WeldJointDef &o)
{
    Avbd3dWeldJointDef r; memset(&r, 0, sizeof r);
    r.base = pb(o.base);
    r.linearHertz = o.linearHertz;
    r.angularHertz = o.angularHertz;
    r.linearDampingRatio = o.linearDampingRatio;
    r.angularDampingRatio = o.angularDampingRatio;
    return r;
}

static inline avbd3d_WheelJointDef lg(const Avbd3dWheelJointDef &o)
{
    avbd3d_WheelJointDef r = avbd3d_DefaultWheelJointDef();
    r.base = lg(o.base);
    r.enableSuspensionSpring = (o.enableSuspensionSpring) != 0;
    r.suspensionHertz = o.suspensionHertz;
    r.suspensionDampingRatio = o.suspensionDampingRatio;
    r.enableSuspensionLimit = (o.enableSuspensionLimit) != 0;
    r.lowerSuspensionLimit = o.lowerSuspensionLimit;
    r.upperSuspensionLimit = o.upperSuspensionLimit;
    r.enableSpinMotor = (o.enableSpinMotor) != 0;
    r.maxSpinTorque = o.maxSpinTorque;
    r.spinSpeed = o.spinSpeed;
    r.enableSteering = (o.enableSteering) != 0;
    r.steeringHertz = o.steeringHertz;
    r.steeringDampingRatio = o.steeringDampingRatio;
    r.targetSteeringAngle = o.targetSteeringAngle;
    r.maxSteeringTorque = o.maxSteeringTorque;
    r.enableSteeringLimit = (o.enableSteeringLimit) != 0;
    r.lowerSteeringLimit = o.lowerSteeringLimit;
    r.upperSteeringLimit = o.upperSteeringLimit;
    return r;
}
static inline Avbd3dWheelJointDef pb(const avbd3d_WheelJointDef &o)
{
    Avbd3dWheelJointDef r; memset(&r, 0, sizeof r);
    r.base = pb(o.base);
    r.enableSuspensionSpring = (o.enableSuspensionSpring) ? 1 : 0;
    r.suspensionHertz = o.suspensionHertz;
    r.suspensionDampingRatio = o.suspensionDampingRatio;
    r.enableSuspensionLimit = (o.enableSuspensionLimit) ? 1 : 0;
    r.lowerSuspensionLimit = o.lowerSuspensionLimit;
    r.upperSuspensionLimit = o.upperSuspensionLimit;
    r.enableSpinMotor = (o.enableSpinMotor) ? 1 : 0;
    r.maxSpinTorque = o.maxSpinTorque;
    r.spinSpeed = o.spinSpeed;
    r.enableSteering = (o.enableSteering) ? 1 : 0;
    r.steeringHertz = o.steeringHertz;
    r.steeringDampingRatio = o.steeringDampingRatio;
    r.targetSteeringAngle = o.targetSteeringAngle;
    r.maxSteeringTorque = o.maxSteeringTorque;
    r.enableSteeringLimit = (o.enableSteeringLimit) ? 1 : 0;
    r.lowerSteeringLimit = o.lowerSteeringLimit;
    r.upperSteeringLimit = o.upperSteeringLimit;
    return r;
}

static_assert(sizeof(Avbd3dExplosionDef) == sizeof(avbd3d_ExplosionDef), "avbd3d_ExplosionDef");
static inline avbd3d_ExplosionDef lg(const Avbd3dExplosionDef &o) { avbd3d_ExplosionDef r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dExplosionDef pb(const avbd3d_ExplosionDef &o) { Avbd3dExplosionDef r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dSensorBeginTouchEvent) == sizeof(avbd3d_SensorBeginTouchEvent), "avbd3d_SensorBeginTouchEvent");
static inline avbd3d_SensorBeginTouchEvent lg(const Avbd3dSensorBeginTouchEvent &o) { avbd3d_SensorBeginTouchEvent r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dSensorBeginTouchEvent pb(const avbd3d_SensorBeginTouchEvent &o) { Avbd3dSensorBeginTouchEvent r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dSensorEndTouchEvent) == sizeof(avbd3d_SensorEndTouchEvent), "avbd3d_SensorEndTouchEvent");
static inline avbd3d_SensorEndTouchEvent lg(const Avbd3dSensorEndTouchEvent &o) { avbd3d_SensorEndTouchEvent r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dSensorEndTouchEvent pb(const avbd3d_SensorEndTouchEvent &o) { Avbd3dSensorEndTouchEvent r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dSensorEvents) == sizeof(avbd3d_SensorEvents), "avbd3d_SensorEvents");
static inline avbd3d_SensorEvents lg(const Avbd3dSensorEvents &o) { avbd3d_SensorEvents r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dSensorEvents pb(const avbd3d_SensorEvents &o) { Avbd3dSensorEvents r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dContactBeginTouchEvent) == sizeof(avbd3d_ContactBeginTouchEvent), "avbd3d_ContactBeginTouchEvent");
static inline avbd3d_ContactBeginTouchEvent lg(const Avbd3dContactBeginTouchEvent &o) { avbd3d_ContactBeginTouchEvent r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dContactBeginTouchEvent pb(const avbd3d_ContactBeginTouchEvent &o) { Avbd3dContactBeginTouchEvent r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dContactEndTouchEvent) == sizeof(avbd3d_ContactEndTouchEvent), "avbd3d_ContactEndTouchEvent");
static inline avbd3d_ContactEndTouchEvent lg(const Avbd3dContactEndTouchEvent &o) { avbd3d_ContactEndTouchEvent r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dContactEndTouchEvent pb(const avbd3d_ContactEndTouchEvent &o) { Avbd3dContactEndTouchEvent r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dContactHitEvent) == sizeof(avbd3d_ContactHitEvent), "avbd3d_ContactHitEvent");
static inline avbd3d_ContactHitEvent lg(const Avbd3dContactHitEvent &o) { avbd3d_ContactHitEvent r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dContactHitEvent pb(const avbd3d_ContactHitEvent &o) { Avbd3dContactHitEvent r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dContactEvents) == sizeof(avbd3d_ContactEvents), "avbd3d_ContactEvents");
static inline avbd3d_ContactEvents lg(const Avbd3dContactEvents &o) { avbd3d_ContactEvents r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dContactEvents pb(const avbd3d_ContactEvents &o) { Avbd3dContactEvents r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dBodyMoveEvent) == sizeof(avbd3d_BodyMoveEvent), "avbd3d_BodyMoveEvent");
static inline avbd3d_BodyMoveEvent lg(const Avbd3dBodyMoveEvent &o) { avbd3d_BodyMoveEvent r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dBodyMoveEvent pb(const avbd3d_BodyMoveEvent &o) { Avbd3dBodyMoveEvent r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dBodyEvents) == sizeof(avbd3d_BodyEvents), "avbd3d_BodyEvents");
static inline avbd3d_BodyEvents lg(const Avbd3dBodyEvents &o) { avbd3d_BodyEvents r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dBodyEvents pb(const avbd3d_BodyEvents &o) { Avbd3dBodyEvents r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dJointEvent) == sizeof(avbd3d_JointEvent), "avbd3d_JointEvent");
static inline avbd3d_JointEvent lg(const Avbd3dJointEvent &o) { avbd3d_JointEvent r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dJointEvent pb(const avbd3d_JointEvent &o) { Avbd3dJointEvent r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dJointEvents) == sizeof(avbd3d_JointEvents), "avbd3d_JointEvents");
static inline avbd3d_JointEvents lg(const Avbd3dJointEvents &o) { avbd3d_JointEvents r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dJointEvents pb(const avbd3d_JointEvents &o) { Avbd3dJointEvents r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dManifoldPoint) == sizeof(avbd3d_ManifoldPoint), "avbd3d_ManifoldPoint");
static inline avbd3d_ManifoldPoint lg(const Avbd3dManifoldPoint &o) { avbd3d_ManifoldPoint r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dManifoldPoint pb(const avbd3d_ManifoldPoint &o) { Avbd3dManifoldPoint r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dManifold) == sizeof(avbd3d_Manifold), "avbd3d_Manifold");
static inline avbd3d_Manifold lg(const Avbd3dManifold &o) { avbd3d_Manifold r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dManifold pb(const avbd3d_Manifold &o) { Avbd3dManifold r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dContactData) == sizeof(avbd3d_ContactData), "avbd3d_ContactData");
static inline avbd3d_ContactData lg(const Avbd3dContactData &o) { avbd3d_ContactData r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dContactData pb(const avbd3d_ContactData &o) { Avbd3dContactData r; memcpy(&r, &o, sizeof r); return r; }

static inline avbd3d_QueryFilter lg(const Avbd3dQueryFilter &o)
{
    avbd3d_QueryFilter r = avbd3d_DefaultQueryFilter();
    r.categoryBits = o.categoryBits;
    r.maskBits = o.maskBits;
    return r;
}
static inline Avbd3dQueryFilter pb(const avbd3d_QueryFilter &o)
{
    Avbd3dQueryFilter r; memset(&r, 0, sizeof r);
    r.categoryBits = o.categoryBits;
    r.maskBits = o.maskBits;
    return r;
}

static inline avbd3d_RayResult lg(const Avbd3dRayResult &o)
{
    avbd3d_RayResult r; memset(&r, 0, sizeof r);
    r.shapeId = lg(o.shapeId);
    r.point = lg(o.point);
    r.normal = lg(o.normal);
    r.userMaterialId = o.userMaterialId;
    r.fraction = o.fraction;
    r.triangleIndex = o.triangleIndex;
    r.childIndex = o.childIndex;
    r.hit = (o.hit) != 0;
    return r;
}
static inline Avbd3dRayResult pb(const avbd3d_RayResult &o)
{
    Avbd3dRayResult r; memset(&r, 0, sizeof r);
    r.shapeId = pb(o.shapeId);
    r.point = pb(o.point);
    r.normal = pb(o.normal);
    r.userMaterialId = o.userMaterialId;
    r.fraction = o.fraction;
    r.triangleIndex = o.triangleIndex;
    r.childIndex = o.childIndex;
    r.hit = (o.hit) ? 1 : 0;
    return r;
}

static_assert(sizeof(Avbd3dShapeProxy) == sizeof(avbd3d_ShapeProxy), "avbd3d_ShapeProxy");
static inline avbd3d_ShapeProxy lg(const Avbd3dShapeProxy &o) { avbd3d_ShapeProxy r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dShapeProxy pb(const avbd3d_ShapeProxy &o) { Avbd3dShapeProxy r; memcpy(&r, &o, sizeof r); return r; }

static inline avbd3d_CastOutput lg(const Avbd3dCastOutput &o)
{
    avbd3d_CastOutput r; memset(&r, 0, sizeof r);
    r.normal = lg(o.normal);
    r.point = lg(o.point);
    r.fraction = o.fraction;
    r.iterations = o.iterations;
    r.triangleIndex = o.triangleIndex;
    r.childIndex = o.childIndex;
    r.materialIndex = o.materialIndex;
    r.hit = (o.hit) != 0;
    return r;
}
static inline Avbd3dCastOutput pb(const avbd3d_CastOutput &o)
{
    Avbd3dCastOutput r; memset(&r, 0, sizeof r);
    r.normal = pb(o.normal);
    r.point = pb(o.point);
    r.fraction = o.fraction;
    r.iterations = o.iterations;
    r.triangleIndex = o.triangleIndex;
    r.childIndex = o.childIndex;
    r.materialIndex = o.materialIndex;
    r.hit = (o.hit) ? 1 : 0;
    return r;
}

static inline avbd3d_BodyCastResult lg(const Avbd3dBodyCastResult &o)
{
    avbd3d_BodyCastResult r; memset(&r, 0, sizeof r);
    r.shapeId = lg(o.shapeId);
    r.point = lg(o.point);
    r.normal = lg(o.normal);
    r.fraction = o.fraction;
    r.triangleIndex = o.triangleIndex;
    r.userMaterialId = o.userMaterialId;
    r.iterations = o.iterations;
    r.hit = (o.hit) != 0;
    return r;
}
static inline Avbd3dBodyCastResult pb(const avbd3d_BodyCastResult &o)
{
    Avbd3dBodyCastResult r; memset(&r, 0, sizeof r);
    r.shapeId = pb(o.shapeId);
    r.point = pb(o.point);
    r.normal = pb(o.normal);
    r.fraction = o.fraction;
    r.triangleIndex = o.triangleIndex;
    r.userMaterialId = o.userMaterialId;
    r.iterations = o.iterations;
    r.hit = (o.hit) ? 1 : 0;
    return r;
}

static_assert(sizeof(Avbd3dPlaneResult) == sizeof(avbd3d_PlaneResult), "avbd3d_PlaneResult");
static inline avbd3d_PlaneResult lg(const Avbd3dPlaneResult &o) { avbd3d_PlaneResult r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dPlaneResult pb(const avbd3d_PlaneResult &o) { Avbd3dPlaneResult r; memcpy(&r, &o, sizeof r); return r; }

static inline avbd3d_CollisionPlane lg(const Avbd3dCollisionPlane &o)
{
    avbd3d_CollisionPlane r; memset(&r, 0, sizeof r);
    r.plane = lg(o.plane);
    r.pushLimit = o.pushLimit;
    r.push = o.push;
    r.clipVelocity = (o.clipVelocity) != 0;
    return r;
}
static inline Avbd3dCollisionPlane pb(const avbd3d_CollisionPlane &o)
{
    Avbd3dCollisionPlane r; memset(&r, 0, sizeof r);
    r.plane = pb(o.plane);
    r.pushLimit = o.pushLimit;
    r.push = o.push;
    r.clipVelocity = (o.clipVelocity) ? 1 : 0;
    return r;
}

static_assert(sizeof(Avbd3dPlaneSolverResult) == sizeof(avbd3d_PlaneSolverResult), "avbd3d_PlaneSolverResult");
static inline avbd3d_PlaneSolverResult lg(const Avbd3dPlaneSolverResult &o) { avbd3d_PlaneSolverResult r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dPlaneSolverResult pb(const avbd3d_PlaneSolverResult &o) { Avbd3dPlaneSolverResult r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dBodyPlaneResult) == sizeof(avbd3d_BodyPlaneResult), "avbd3d_BodyPlaneResult");
static inline avbd3d_BodyPlaneResult lg(const Avbd3dBodyPlaneResult &o) { avbd3d_BodyPlaneResult r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dBodyPlaneResult pb(const avbd3d_BodyPlaneResult &o) { Avbd3dBodyPlaneResult r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dMassData) == sizeof(avbd3d_MassData), "avbd3d_MassData");
static inline avbd3d_MassData lg(const Avbd3dMassData &o) { avbd3d_MassData r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dMassData pb(const avbd3d_MassData &o) { Avbd3dMassData r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dSphere) == sizeof(avbd3d_Sphere), "avbd3d_Sphere");
static inline avbd3d_Sphere lg(const Avbd3dSphere &o) { avbd3d_Sphere r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dSphere pb(const avbd3d_Sphere &o) { Avbd3dSphere r; memcpy(&r, &o, sizeof r); return r; }

static_assert(sizeof(Avbd3dCapsule) == sizeof(avbd3d_Capsule), "avbd3d_Capsule");
static inline avbd3d_Capsule lg(const Avbd3dCapsule &o) { avbd3d_Capsule r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dCapsule pb(const avbd3d_Capsule &o) { Avbd3dCapsule r; memcpy(&r, &o, sizeof r); return r; }

static inline avbd3d_MeshDef lg(const Avbd3dMeshDef &o)
{
    avbd3d_MeshDef r; memset(&r, 0, sizeof r);
    r.vertices = reinterpret_cast<avbd3d_Vec3*>(o.vertices);
    r.stride = o.stride;
    r.indices = reinterpret_cast<int32_t*>(o.indices);
    r.materialIndices = reinterpret_cast<uint8_t*>(o.materialIndices);
    r.weldTolerance = o.weldTolerance;
    r.vertexCount = o.vertexCount;
    r.triangleCount = o.triangleCount;
    r.weldVertices = (o.weldVertices) != 0;
    r.useMedianSplit = (o.useMedianSplit) != 0;
    r.identifyEdges = (o.identifyEdges) != 0;
    r.clockWiseWinding = (o.clockWiseWinding) != 0;
    return r;
}
static inline Avbd3dMeshDef pb(const avbd3d_MeshDef &o)
{
    Avbd3dMeshDef r; memset(&r, 0, sizeof r);
    r.vertices = reinterpret_cast<Avbd3dVec3*>(o.vertices);
    r.stride = o.stride;
    r.indices = reinterpret_cast<int32_t*>(o.indices);
    r.materialIndices = reinterpret_cast<uint8_t*>(o.materialIndices);
    r.weldTolerance = o.weldTolerance;
    r.vertexCount = o.vertexCount;
    r.triangleCount = o.triangleCount;
    r.weldVertices = (o.weldVertices) ? 1 : 0;
    r.useMedianSplit = (o.useMedianSplit) ? 1 : 0;
    r.identifyEdges = (o.identifyEdges) ? 1 : 0;
    r.clockWiseWinding = (o.clockWiseWinding) ? 1 : 0;
    return r;
}

static inline avbd3d_HeightFieldDef lg(const Avbd3dHeightFieldDef &o)
{
    avbd3d_HeightFieldDef r; memset(&r, 0, sizeof r);
    r.heights = reinterpret_cast<float*>(o.heights);
    r.materialIndices = reinterpret_cast<uint8_t*>(o.materialIndices);
    r.scale = lg(o.scale);
    r.countX = o.countX;
    r.countZ = o.countZ;
    r.globalMinimumHeight = o.globalMinimumHeight;
    r.globalMaximumHeight = o.globalMaximumHeight;
    r.clockwiseWinding = (o.clockwiseWinding) != 0;
    return r;
}
static inline Avbd3dHeightFieldDef pb(const avbd3d_HeightFieldDef &o)
{
    Avbd3dHeightFieldDef r; memset(&r, 0, sizeof r);
    r.heights = reinterpret_cast<float*>(o.heights);
    r.materialIndices = reinterpret_cast<uint8_t*>(o.materialIndices);
    r.scale = pb(o.scale);
    r.countX = o.countX;
    r.countZ = o.countZ;
    r.globalMinimumHeight = o.globalMinimumHeight;
    r.globalMaximumHeight = o.globalMaximumHeight;
    r.clockwiseWinding = (o.clockwiseWinding) ? 1 : 0;
    return r;
}

static inline avbd3d_CompoundCapsuleDef lg(const Avbd3dCompoundCapsuleDef &o)
{
    avbd3d_CompoundCapsuleDef r; memset(&r, 0, sizeof r);
    r.capsule = lg(o.capsule);
    r.material = lg(o.material);
    return r;
}
static inline Avbd3dCompoundCapsuleDef pb(const avbd3d_CompoundCapsuleDef &o)
{
    Avbd3dCompoundCapsuleDef r; memset(&r, 0, sizeof r);
    r.capsule = pb(o.capsule);
    r.material = pb(o.material);
    return r;
}

static inline avbd3d_CompoundHullDef lg(const Avbd3dCompoundHullDef &o)
{
    avbd3d_CompoundHullDef r; memset(&r, 0, sizeof r);
    r.hull = reinterpret_cast<const avbd3d_HullData*>(o.hull);
    r.transform = lg(o.transform);
    r.material = lg(o.material);
    return r;
}
static inline Avbd3dCompoundHullDef pb(const avbd3d_CompoundHullDef &o)
{
    Avbd3dCompoundHullDef r; memset(&r, 0, sizeof r);
    r.hull = reinterpret_cast<const Avbd3dHullData*>(o.hull);
    r.transform = pb(o.transform);
    r.material = pb(o.material);
    return r;
}

static_assert(sizeof(Avbd3dCompoundMeshDef) == sizeof(avbd3d_CompoundMeshDef), "avbd3d_CompoundMeshDef");
static inline avbd3d_CompoundMeshDef lg(const Avbd3dCompoundMeshDef &o) { avbd3d_CompoundMeshDef r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dCompoundMeshDef pb(const avbd3d_CompoundMeshDef &o) { Avbd3dCompoundMeshDef r; memcpy(&r, &o, sizeof r); return r; }

static inline avbd3d_CompoundSphereDef lg(const Avbd3dCompoundSphereDef &o)
{
    avbd3d_CompoundSphereDef r; memset(&r, 0, sizeof r);
    r.sphere = lg(o.sphere);
    r.material = lg(o.material);
    return r;
}
static inline Avbd3dCompoundSphereDef pb(const avbd3d_CompoundSphereDef &o)
{
    Avbd3dCompoundSphereDef r; memset(&r, 0, sizeof r);
    r.sphere = pb(o.sphere);
    r.material = pb(o.material);
    return r;
}

static_assert(sizeof(Avbd3dCompoundDef) == sizeof(avbd3d_CompoundDef), "avbd3d_CompoundDef");
static inline avbd3d_CompoundDef lg(const Avbd3dCompoundDef &o) { avbd3d_CompoundDef r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dCompoundDef pb(const avbd3d_CompoundDef &o) { Avbd3dCompoundDef r; memcpy(&r, &o, sizeof r); return r; }

static inline avbd3d_SolverDef lg(const Avbd3dSolverDef &o)
{
    avbd3d_SolverDef r = avbd3d_DefaultSolverDef();
    r.iterations = o.iterations;
    r.primalLanes = o.primalLanes;
    r.alpha = o.alpha;
    r.betaLin = o.betaLin;
    r.betaAng = o.betaAng;
    r.gamma = o.gamma;
    r.autoKStart = (o.autoKStart) != 0;
    r.dualDamping = o.dualDamping;
    r.dualDampVel = o.dualDampVel;
    r.sleepFrames = o.sleepFrames;
    r.sleepLinearVelocity = o.sleepLinearVelocity;
    r.sleepAngularVelocity = o.sleepAngularVelocity;
    r.sleepDisplacement = o.sleepDisplacement;
    r.sleepFracture = o.sleepFracture;
    r.sleepLeak = o.sleepLeak;
    r.enableKillFallen = (o.enableKillFallen) != 0;
    r.killHeight = o.killHeight;
    r.grabStrength = o.grabStrength;
    r.contactCap = o.contactCap;
    r.coarseIterations = o.coarseIterations;
    r.coarseCell = o.coarseCell;
    r.coarseDamping = o.coarseDamping;
    r.worldFrameInertia = (o.worldFrameInertia) != 0;
    return r;
}
static inline Avbd3dSolverDef pb(const avbd3d_SolverDef &o)
{
    Avbd3dSolverDef r; memset(&r, 0, sizeof r);
    r.iterations = o.iterations;
    r.primalLanes = o.primalLanes;
    r.alpha = o.alpha;
    r.betaLin = o.betaLin;
    r.betaAng = o.betaAng;
    r.gamma = o.gamma;
    r.autoKStart = (o.autoKStart) ? 1 : 0;
    r.dualDamping = o.dualDamping;
    r.dualDampVel = o.dualDampVel;
    r.sleepFrames = o.sleepFrames;
    r.sleepLinearVelocity = o.sleepLinearVelocity;
    r.sleepAngularVelocity = o.sleepAngularVelocity;
    r.sleepDisplacement = o.sleepDisplacement;
    r.sleepFracture = o.sleepFracture;
    r.sleepLeak = o.sleepLeak;
    r.enableKillFallen = (o.enableKillFallen) ? 1 : 0;
    r.killHeight = o.killHeight;
    r.grabStrength = o.grabStrength;
    r.contactCap = o.contactCap;
    r.coarseIterations = o.coarseIterations;
    r.coarseCell = o.coarseCell;
    r.coarseDamping = o.coarseDamping;
    r.worldFrameInertia = (o.worldFrameInertia) ? 1 : 0;
    return r;
}

static inline avbd3d_MixingRule lg(Avbd3dMixingRule o) { return (avbd3d_MixingRule)o; }
static inline Avbd3dMixingRule pb(avbd3d_MixingRule o) { return (Avbd3dMixingRule)o; }

static inline avbd3d_BodyDataType lg(Avbd3dBodyDataType o) { return (avbd3d_BodyDataType)o; }
static inline Avbd3dBodyDataType pb(avbd3d_BodyDataType o) { return (Avbd3dBodyDataType)o; }

static inline avbd3d_AvbdJointKind lg(Avbd3dGenericJointKind o) { return (avbd3d_AvbdJointKind)o; }
static inline Avbd3dGenericJointKind pb(avbd3d_AvbdJointKind o) { return (Avbd3dGenericJointKind)o; }

static inline avbd3d_AvbdAxisMode lg(Avbd3dGenericAxisMode o) { return (avbd3d_AvbdAxisMode)o; }
static inline Avbd3dGenericAxisMode pb(avbd3d_AvbdAxisMode o) { return (Avbd3dGenericAxisMode)o; }

static_assert(sizeof(Avbd3dGenericAxis) == sizeof(avbd3d_AvbdAxis), "avbd3d_AvbdAxis");
static inline avbd3d_AvbdAxis lg(const Avbd3dGenericAxis &o) { avbd3d_AvbdAxis r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dGenericAxis pb(const avbd3d_AvbdAxis &o) { Avbd3dGenericAxis r; memcpy(&r, &o, sizeof r); return r; }

static inline avbd3d_AvbdJointDef lg(const Avbd3dGenericJointDef &o)
{
    avbd3d_AvbdJointDef r = avbd3d_DefaultAvbdJointDef();
    r.base = lg(o.base);
    r.kind = (avbd3d_AvbdJointKind)o.kind;
    r.linearStiffness = o.linearStiffness;
    r.angularStiffness = o.angularStiffness;
    for (size_t i_ = 0; i_ < sizeof(o.axes) / sizeof(o.axes[0]); i_++) r.axes[i_] = lg(o.axes[i_]);
    r.fractureForce = o.fractureForce;
    r.fractureTorque = o.fractureTorque;
    return r;
}
static inline Avbd3dGenericJointDef pb(const avbd3d_AvbdJointDef &o)
{
    Avbd3dGenericJointDef r; memset(&r, 0, sizeof r);
    r.base = pb(o.base);
    r.kind = (int32_t)o.kind;
    r.linearStiffness = o.linearStiffness;
    r.angularStiffness = o.angularStiffness;
    for (size_t i_ = 0; i_ < sizeof(o.axes) / sizeof(o.axes[0]); i_++) r.axes[i_] = pb(o.axes[i_]);
    r.fractureForce = o.fractureForce;
    r.fractureTorque = o.fractureTorque;
    return r;
}

static inline avbd3d_AvbdSpringDef lg(const Avbd3dSpringDef &o)
{
    avbd3d_AvbdSpringDef r = avbd3d_DefaultAvbdSpringDef();
    r.base = lg(o.base);
    r.stiffness = o.stiffness;
    r.restLength = o.restLength;
    return r;
}
static inline Avbd3dSpringDef pb(const avbd3d_AvbdSpringDef &o)
{
    Avbd3dSpringDef r; memset(&r, 0, sizeof r);
    r.base = pb(o.base);
    r.stiffness = o.stiffness;
    r.restLength = o.restLength;
    return r;
}

static_assert(sizeof(Avbd3dGrabHandle) == sizeof(avbd3d_GrabHandle), "avbd3d_GrabHandle");
static inline avbd3d_GrabHandle lg(const Avbd3dGrabHandle &o) { avbd3d_GrabHandle r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dGrabHandle pb(const avbd3d_GrabHandle &o) { Avbd3dGrabHandle r; memcpy(&r, &o, sizeof r); return r; }

static inline avbd3d_GpuDef lg(const Avbd3dGpuDef &o)
{
    avbd3d_GpuDef r; memset(&r, 0, sizeof r);
    r.internalValue = kInternalValue;
    r.instanceExtensions = reinterpret_cast<const char* const*>(o.instanceExtensions);
    r.instanceExtensionCount = o.instanceExtensionCount;
    r.makeSurface = reinterpret_cast<avbd3d_MakeSurfaceFcn*>(o.makeSurface);
    r.surfaceContext = reinterpret_cast<void*>(o.surfaceContext);
    r.enableValidation = (o.enableValidation) != 0;
    r.verbose = (o.verbose) != 0;
    return r;
}
static inline Avbd3dGpuDef pb(const avbd3d_GpuDef &o)
{
    Avbd3dGpuDef r; memset(&r, 0, sizeof r);
    r.instanceExtensions = reinterpret_cast<const char* const*>(o.instanceExtensions);
    r.instanceExtensionCount = o.instanceExtensionCount;
    r.makeSurface = reinterpret_cast<Avbd3dMakeSurfaceFcn*>(o.makeSurface);
    r.surfaceContext = reinterpret_cast<void*>(o.surfaceContext);
    r.enableValidation = (o.enableValidation) ? 1 : 0;
    r.verbose = (o.verbose) ? 1 : 0;
    return r;
}

static_assert(sizeof(Avbd3dVulkanContext) == sizeof(avbd3d_VulkanContext), "avbd3d_VulkanContext");
static inline avbd3d_VulkanContext lg(const Avbd3dVulkanContext &o) { avbd3d_VulkanContext r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dVulkanContext pb(const avbd3d_VulkanContext &o) { Avbd3dVulkanContext r; memcpy(&r, &o, sizeof r); return r; }

static inline avbd3d_Overlay lg(Avbd3dOverlay o) { return (avbd3d_Overlay)o; }
static inline Avbd3dOverlay pb(avbd3d_Overlay o) { return (Avbd3dOverlay)o; }

static inline avbd3d_RenderDef lg(const Avbd3dRenderDef &o)
{
    avbd3d_RenderDef r = avbd3d_DefaultRenderDef();
    r.frustumPlanes = reinterpret_cast<const float*>(o.frustumPlanes);
    r.overlay = (avbd3d_Overlay)o.overlay;
    r.speedMax = o.speedMax;
    r.loadMax = o.loadMax;
    r.jointLoadMax = o.jointLoadMax;
    r.penetrationMax = o.penetrationMax;
    r.degreeMax = o.degreeMax;
    r.jointDegreeMax = o.jointDegreeMax;
    r.colorStatics = (o.colorStatics) != 0;
    return r;
}
static inline Avbd3dRenderDef pb(const avbd3d_RenderDef &o)
{
    Avbd3dRenderDef r; memset(&r, 0, sizeof r);
    r.frustumPlanes = reinterpret_cast<const float*>(o.frustumPlanes);
    r.overlay = (int32_t)o.overlay;
    r.speedMax = o.speedMax;
    r.loadMax = o.loadMax;
    r.jointLoadMax = o.jointLoadMax;
    r.penetrationMax = o.penetrationMax;
    r.degreeMax = o.degreeMax;
    r.jointDegreeMax = o.jointDegreeMax;
    r.colorStatics = (o.colorStatics) ? 1 : 0;
    return r;
}

static_assert(sizeof(Avbd3dRenderRange) == sizeof(avbd3d_RenderRange), "avbd3d_RenderRange");
static inline avbd3d_RenderRange lg(const Avbd3dRenderRange &o) { avbd3d_RenderRange r; memcpy(&r, &o, sizeof r); return r; }
static inline Avbd3dRenderRange pb(const avbd3d_RenderRange &o) { Avbd3dRenderRange r; memcpy(&r, &o, sizeof r); return r; }

static inline avbd3d_ContactTint lg(Avbd3dContactTint o) { return (avbd3d_ContactTint)o; }
static inline Avbd3dContactTint pb(avbd3d_ContactTint o) { return (Avbd3dContactTint)o; }

static inline avbd3d_ContactMarkerDef lg(const Avbd3dContactMarkerDef &o)
{
    avbd3d_ContactMarkerDef r = avbd3d_DefaultContactMarkerDef();
    r.points = (o.points) != 0;
    r.normals = (o.normals) != 0;
    r.manifoldLinks = (o.manifoldLinks) != 0;
    r.tint = (avbd3d_ContactTint)o.tint;
    r.pointSize = o.pointSize;
    r.normalLength = o.normalLength;
    r.forceMax = o.forceMax;
    r.maxMarkers = o.maxMarkers;
    return r;
}
static inline Avbd3dContactMarkerDef pb(const avbd3d_ContactMarkerDef &o)
{
    Avbd3dContactMarkerDef r; memset(&r, 0, sizeof r);
    r.points = (o.points) ? 1 : 0;
    r.normals = (o.normals) ? 1 : 0;
    r.manifoldLinks = (o.manifoldLinks) ? 1 : 0;
    r.tint = (int32_t)o.tint;
    r.pointSize = o.pointSize;
    r.normalLength = o.normalLength;
    r.forceMax = o.forceMax;
    r.maxMarkers = o.maxMarkers;
    return r;
}

