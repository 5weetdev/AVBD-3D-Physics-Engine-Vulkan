// Host-side implementations of the avbd3d API that the world layer answers without the solver:
// names and flags, mass and geometry reads, joint frames and thresholds, and typed joint
// properties. A typed joint property is a field of the joint's stored typed def (JointRec::def).
// A setter writes the field and queues a row rebuild at the next flush, so the change reaches the
// simulation on the next step. A getter returns the creation value or the last value set.
//
// Invalid ids return the default for the return type.
#include "avbd3d_world.h"

#include "box3d/math_functions.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

using namespace avbd3d_impl;

namespace
{

// Equal to kInternalWorld in avbd3d_api.cpp. Every def struct carries it in internalValue.
constexpr int kInternalWorld = 1152023;

// The stored def of a live joint, read as its typed struct. The typed def is the source of truth:
// a setter writes the field back and the next flush rebuilds the rows from these bytes.
template <class Def> bool loadDef(avbd3d_JointId id, avbd3d_JointType type, Def &d, World *&w, int &r)
{
    w = worldOfJoint(id, r);
    if (!w || w->joints[r].type != static_cast<int>(type))
        return false;
    std::memcpy(&d, w->joints[r].def.data(), sizeof d);
    return true;
}

// Writes a typed def back and queues a row rebuild (the rows depend on the def).
template <class Def> void storeDef(World *w, int r, const Def &d)
{
    std::memcpy(w->joints[r].def.data(), &d, sizeof d);
    w->markJoint(r);
}

// The base part every typed def starts with (base is the first member of each def).
bool loadBase(avbd3d_JointId id, avbd3d_JointDef &base, World *&w, int &r)
{
    w = worldOfJoint(id, r);
    if (!w)
        return false;
    std::memcpy(&base, w->joints[r].def.data(), sizeof base);
    return true;
}

// Writes the base back. Thresholds do not need a row rebuild, only a device write.
void storeBase(World *w, int r, const avbd3d_JointDef &base, bool rebuild)
{
    std::memcpy(w->joints[r].def.data(), &base, sizeof base);
    if (rebuild)
        w->markJoint(r);
    else
        w->touchJointThresholds(r);
}

// Live joint ids of a body, in record order.
std::vector<avbd3d_JointId> liveJointIds(const World *w, int b)
{
    std::vector<avbd3d_JointId> out;
    for (int r : w->bodies[b].joints)
    {
        if (r < 0 || r >= (int)w->joints.size() || !w->joints[r].alive)
            continue;
        avbd3d_JointId id;
        id.index1 = r + 1;
        id.world0 = (uint16_t)w->worldIndex;
        id.generation = w->joints[r].gen;
        out.push_back(id);
    }
    return out;
}

// World properties kept on the host (World::props). Unset keys read as zero.
void setWorldProp(avbd3d_WorldId id, const char *key, double v)
{
    if (World *w = settled(id))
        w->props[key] = v;
}

double worldProp(avbd3d_WorldId id, const char *key)
{
    World *w = settled(id);
    if (!w)
        return 0.0;
    const auto it = w->props.find(key);
    return it == w->props.end() ? 0.0 : it->second;
}

// ---- typed joint accessors: each reads or writes one field of the stored typed def -----------

#define JOINT_SET(FN, TYPE, DEF, FIELD, T)                                                                             \
    void FN(avbd3d_JointId id, T value)                                                                                \
    {                                                                                                                  \
        DEF d;                                                                                                         \
        World *w;                                                                                                      \
        int r;                                                                                                         \
        if (!loadDef(id, TYPE, d, w, r))                                                                               \
            return;                                                                                                    \
        static_assert(std::is_same<decltype(d.FIELD), T>::value, #FIELD " has another type");                          \
        d.FIELD = value;                                                                                               \
        storeDef(w, r, d);                                                                                             \
    }
#define JOINT_SET2(FN, TYPE, DEF, F0, F1)                                                                              \
    void FN(avbd3d_JointId id, float a, float b)                                                                       \
    {                                                                                                                  \
        DEF d;                                                                                                         \
        World *w;                                                                                                      \
        int r;                                                                                                         \
        if (!loadDef(id, TYPE, d, w, r))                                                                               \
            return;                                                                                                    \
        d.F0 = a;                                                                                                      \
        d.F1 = b;                                                                                                      \
        storeDef(w, r, d);                                                                                             \
    }

} // namespace

// ---- joint state readout: the two bodies' current poses in the def's own A/B order -------------

namespace
{

// Body poses and frames of a joint as the caller named them (A and B as in the def, even when the
// rows were built with the world side swapped to A). A world side has the identity pose.
struct JointView
{
    World *w = nullptr;
    int r = 0;
    vec3 pA{0, 0, 0}, pB{0, 0, 0}, vA{0, 0, 0}, vB{0, 0, 0}, wA{0, 0, 0}, wB{0, 0, 0};
    quat qA{0, 0, 0, 1}, qB{0, 0, 0, 1};
    vec3 fpA{0, 0, 0}, fpB{0, 0, 0};
    quat fqA{0, 0, 0, 1}, fqB{0, 0, 0, 1};
    quat frameA() const { return qA * fqA; }
    quat frameB() const { return qB * fqB; }
};

bool loadView(avbd3d_JointId id, avbd3d_JointType type, JointView &v)
{
    v.w = worldOfJoint(id, v.r);
    if (!v.w || v.w->joints[v.r].type != static_cast<int>(type))
        return false;
    const JointRec &j = v.w->joints[v.r];
    avbd3d_JointDef base;
    std::memcpy(&base, j.def.data(), sizeof base);
    const SceneDesc &d = v.w->desc;
    const int a = j.swapped ? j.b : j.a, b = j.swapped ? j.a : j.b;
    if (a >= 0)
    {
        v.pA = d.pos[a];
        v.qA = d.rot[a];
        v.vA = d.velLin[a];
        v.wA = d.velAng[a];
    }
    if (b >= 0)
    {
        v.pB = d.pos[b];
        v.qB = d.rot[b];
        v.vB = d.velLin[b];
        v.wB = d.velAng[b];
    }
    v.fpA = toV(base.localFrameA.p);
    v.fpB = toV(base.localFrameB.p);
    v.fqA = toQ(base.localFrameA.q);
    v.fqB = toQ(base.localFrameB.q);
    return true;
}

// B's frame relative to A's, on the shorter arc.
quat relativeFrame(const JointView &v)
{
    const quat qa = v.frameA();
    quat qb = v.frameB();
    if (qa.x * qb.x + qa.y * qb.y + qa.z * qb.z + qa.w * qb.w < 0.0f)
        qb = quat{-qb.x, -qb.y, -qb.z, -qb.w};
    return conjugate(qa) * qb;
}

float twistAngleOf(quat q)
{
    return 2.0f * (q.w < 0.0f ? std::atan2(-q.z, -q.w) : std::atan2(q.z, q.w));
}

// Prismatic: separation of the anchors, and the axis it is measured on (frame A x, world).
vec3 prismaticSeparation(const JointView &v, vec3 &axis)
{
    axis = rotate(v.frameA(), vec3{1, 0, 0});
    return (v.pB - v.pA) + rotate(v.qB, v.fpB) - rotate(v.qA, v.fpA);
}

} // namespace

extern "C"
{

    // ---- joint: generic --------------------------------------------------------------------------

    avbd3d_JointType avbd3d_Joint_GetType(avbd3d_JointId id)
    {
        int r;
        World *w = worldOfJoint(id, r);
        return w ? static_cast<avbd3d_JointType>(w->joints[r].type) : avbd3d_parallelJoint;
    }

    avbd3d_BodyId avbd3d_Joint_GetBodyA(avbd3d_JointId id)
    {
        int r;
        World *w = worldOfJoint(id, r);
        return w && w->joints[r].a >= 0 ? w->bodyId(w->joints[r].a) : avbd3d_nullBodyId;
    }

    avbd3d_BodyId avbd3d_Joint_GetBodyB(avbd3d_JointId id)
    {
        int r;
        World *w = worldOfJoint(id, r);
        return w ? w->bodyId(w->joints[r].b) : avbd3d_nullBodyId;
    }

    avbd3d_WorldId avbd3d_Joint_GetWorld(avbd3d_JointId id)
    {
        int r;
        World *w = worldOfJoint(id, r);
        return w ? worldIdOf(w) : avbd3d_nullWorldId;
    }

    // The frames are the def's, as the caller named them: a swapped joint swaps them at build time.
    void avbd3d_Joint_SetLocalFrameA(avbd3d_JointId id, avbd3d_Transform localFrame)
    {
        avbd3d_JointDef base;
        World *w;
        int r;
        if (loadBase(id, base, w, r))
        {
            base.localFrameA = localFrame;
            storeBase(w, r, base, true);
        }
    }

    void avbd3d_Joint_SetLocalFrameB(avbd3d_JointId id, avbd3d_Transform localFrame)
    {
        avbd3d_JointDef base;
        World *w;
        int r;
        if (loadBase(id, base, w, r))
        {
            base.localFrameB = localFrame;
            storeBase(w, r, base, true);
        }
    }

    // Changes which body pairs the joint ignores, so the rows are rebuilt.
    void avbd3d_Joint_SetCollideConnected(avbd3d_JointId id, bool shouldCollide)
    {
        avbd3d_JointDef base;
        World *w;
        int r;
        if (loadBase(id, base, w, r))
        {
            base.collideConnected = shouldCollide;
            storeBase(w, r, base, true);
        }
    }

    void avbd3d_Joint_SetUserData(avbd3d_JointId id, void *userData)
    {
        avbd3d_JointDef base;
        World *w;
        int r;
        if (loadBase(id, base, w, r))
        {
            base.userData = userData;
            storeBase(w, r, base, false);
        }
    }

    void *avbd3d_Joint_GetUserData(avbd3d_JointId id)
    {
        avbd3d_JointDef base;
        World *w;
        int r;
        return loadBase(id, base, w, r) ? base.userData : nullptr;
    }

    void avbd3d_Joint_WakeBodies(avbd3d_JointId id)
    {
        int r;
        if (World *w = worldOfJoint(id, r))
        {
            const JointRec &j = w->joints[r];
            if (j.a >= 0)
                w->setAwake(j.a, true);
            w->setAwake(j.b, true);
        }
    }

    // The force and torque on body B (world frame) from the multipliers of the newest finished step,
    // pulled from the joint snapshot on first use (World::jointForce). Zero for a joint without rows
    // (filter joint, avbd spring) or one created since that step.
    avbd3d_Vec3 avbd3d_Joint_GetConstraintForce(avbd3d_JointId id)
    {
        int r;
        vec3 f{0, 0, 0}, t{0, 0, 0};
        if (World *w = worldOfJoint(id, r))
            w->jointForce(r, f, t);
        return avbd3d_Vec3{f.x, f.y, f.z};
    }

    avbd3d_Vec3 avbd3d_Joint_GetConstraintTorque(avbd3d_JointId id)
    {
        int r;
        vec3 f{0, 0, 0}, t{0, 0, 0};
        if (World *w = worldOfJoint(id, r))
            w->jointForce(r, f, t);
        return avbd3d_Vec3{t.x, t.y, t.z};
    }

    // The fracture thresholds live in the def; a change is a device write, not a row rebuild.
    void avbd3d_Joint_SetForceThreshold(avbd3d_JointId id, float threshold)
    {
        avbd3d_JointDef base;
        World *w;
        int r;
        if (loadBase(id, base, w, r))
        {
            base.forceThreshold = threshold;
            storeBase(w, r, base, false);
        }
    }

    void avbd3d_Joint_SetTorqueThreshold(avbd3d_JointId id, float threshold)
    {
        avbd3d_JointDef base;
        World *w;
        int r;
        if (loadBase(id, base, w, r))
        {
            base.torqueThreshold = threshold;
            storeBase(w, r, base, false);
        }
    }

    // ---- joint: typed accessors ------------------------------------------------------------------

    // Parallel
    JOINT_SET(avbd3d_ParallelJoint_SetSpringHertz, avbd3d_parallelJoint, avbd3d_ParallelJointDef, hertz, float)
    JOINT_SET(avbd3d_ParallelJoint_SetSpringDampingRatio, avbd3d_parallelJoint, avbd3d_ParallelJointDef, dampingRatio,
              float)
    JOINT_SET(avbd3d_ParallelJoint_SetMaxTorque, avbd3d_parallelJoint, avbd3d_ParallelJointDef, maxTorque, float)

    // Distance
    JOINT_SET(avbd3d_DistanceJoint_SetLength, avbd3d_distanceJoint, avbd3d_DistanceJointDef, length, float)
    JOINT_SET(avbd3d_DistanceJoint_EnableSpring, avbd3d_distanceJoint, avbd3d_DistanceJointDef, enableSpring, bool)
    JOINT_SET2(avbd3d_DistanceJoint_SetSpringForceRange, avbd3d_distanceJoint, avbd3d_DistanceJointDef,
               lowerSpringForce, upperSpringForce)
    JOINT_SET(avbd3d_DistanceJoint_SetSpringHertz, avbd3d_distanceJoint, avbd3d_DistanceJointDef, hertz, float)
    JOINT_SET(avbd3d_DistanceJoint_SetSpringDampingRatio, avbd3d_distanceJoint, avbd3d_DistanceJointDef, dampingRatio,
              float)
    JOINT_SET(avbd3d_DistanceJoint_EnableLimit, avbd3d_distanceJoint, avbd3d_DistanceJointDef, enableLimit, bool)
    JOINT_SET2(avbd3d_DistanceJoint_SetLengthRange, avbd3d_distanceJoint, avbd3d_DistanceJointDef, minLength, maxLength)
    JOINT_SET(avbd3d_DistanceJoint_EnableMotor, avbd3d_distanceJoint, avbd3d_DistanceJointDef, enableMotor, bool)
    JOINT_SET(avbd3d_DistanceJoint_SetMotorSpeed, avbd3d_distanceJoint, avbd3d_DistanceJointDef, motorSpeed, float)
    JOINT_SET(avbd3d_DistanceJoint_SetMaxMotorForce, avbd3d_distanceJoint, avbd3d_DistanceJointDef, maxMotorForce,
              float)

    // Motor
    JOINT_SET(avbd3d_MotorJoint_SetLinearVelocity, avbd3d_motorJoint, avbd3d_MotorJointDef, linearVelocity, avbd3d_Vec3)
    JOINT_SET(avbd3d_MotorJoint_SetAngularVelocity, avbd3d_motorJoint, avbd3d_MotorJointDef, angularVelocity,
              avbd3d_Vec3)
    JOINT_SET(avbd3d_MotorJoint_SetMaxVelocityForce, avbd3d_motorJoint, avbd3d_MotorJointDef, maxVelocityForce, float)
    JOINT_SET(avbd3d_MotorJoint_SetMaxVelocityTorque, avbd3d_motorJoint, avbd3d_MotorJointDef, maxVelocityTorque, float)
    JOINT_SET(avbd3d_MotorJoint_SetLinearHertz, avbd3d_motorJoint, avbd3d_MotorJointDef, linearHertz, float)
    JOINT_SET(avbd3d_MotorJoint_SetLinearDampingRatio, avbd3d_motorJoint, avbd3d_MotorJointDef, linearDampingRatio,
              float)
    JOINT_SET(avbd3d_MotorJoint_SetAngularHertz, avbd3d_motorJoint, avbd3d_MotorJointDef, angularHertz, float)
    JOINT_SET(avbd3d_MotorJoint_SetAngularDampingRatio, avbd3d_motorJoint, avbd3d_MotorJointDef, angularDampingRatio,
              float)
    JOINT_SET(avbd3d_MotorJoint_SetMaxSpringForce, avbd3d_motorJoint, avbd3d_MotorJointDef, maxSpringForce, float)
    JOINT_SET(avbd3d_MotorJoint_SetMaxSpringTorque, avbd3d_motorJoint, avbd3d_MotorJointDef, maxSpringTorque, float)

    // Prismatic
    JOINT_SET(avbd3d_PrismaticJoint_EnableSpring, avbd3d_prismaticJoint, avbd3d_PrismaticJointDef, enableSpring, bool)
    JOINT_SET(avbd3d_PrismaticJoint_SetSpringHertz, avbd3d_prismaticJoint, avbd3d_PrismaticJointDef, hertz, float)
    JOINT_SET(avbd3d_PrismaticJoint_SetSpringDampingRatio, avbd3d_prismaticJoint, avbd3d_PrismaticJointDef,
              dampingRatio, float)
    JOINT_SET(avbd3d_PrismaticJoint_SetTargetTranslation, avbd3d_prismaticJoint, avbd3d_PrismaticJointDef,
              targetTranslation, float)
    JOINT_SET(avbd3d_PrismaticJoint_EnableLimit, avbd3d_prismaticJoint, avbd3d_PrismaticJointDef, enableLimit, bool)
    JOINT_SET2(avbd3d_PrismaticJoint_SetLimits, avbd3d_prismaticJoint, avbd3d_PrismaticJointDef, lowerTranslation,
               upperTranslation)
    JOINT_SET(avbd3d_PrismaticJoint_EnableMotor, avbd3d_prismaticJoint, avbd3d_PrismaticJointDef, enableMotor, bool)
    JOINT_SET(avbd3d_PrismaticJoint_SetMotorSpeed, avbd3d_prismaticJoint, avbd3d_PrismaticJointDef, motorSpeed, float)
    JOINT_SET(avbd3d_PrismaticJoint_SetMaxMotorForce, avbd3d_prismaticJoint, avbd3d_PrismaticJointDef, maxMotorForce,
              float)

    // Revolute
    JOINT_SET(avbd3d_RevoluteJoint_EnableSpring, avbd3d_revoluteJoint, avbd3d_RevoluteJointDef, enableSpring, bool)
    JOINT_SET(avbd3d_RevoluteJoint_SetSpringHertz, avbd3d_revoluteJoint, avbd3d_RevoluteJointDef, hertz, float)
    JOINT_SET(avbd3d_RevoluteJoint_SetSpringDampingRatio, avbd3d_revoluteJoint, avbd3d_RevoluteJointDef, dampingRatio,
              float)
    JOINT_SET(avbd3d_RevoluteJoint_SetTargetAngle, avbd3d_revoluteJoint, avbd3d_RevoluteJointDef, targetAngle, float)
    JOINT_SET(avbd3d_RevoluteJoint_EnableLimit, avbd3d_revoluteJoint, avbd3d_RevoluteJointDef, enableLimit, bool)
    JOINT_SET2(avbd3d_RevoluteJoint_SetLimits, avbd3d_revoluteJoint, avbd3d_RevoluteJointDef, lowerAngle, upperAngle)
    JOINT_SET(avbd3d_RevoluteJoint_EnableMotor, avbd3d_revoluteJoint, avbd3d_RevoluteJointDef, enableMotor, bool)
    JOINT_SET(avbd3d_RevoluteJoint_SetMotorSpeed, avbd3d_revoluteJoint, avbd3d_RevoluteJointDef, motorSpeed, float)
    JOINT_SET(avbd3d_RevoluteJoint_SetMaxMotorTorque, avbd3d_revoluteJoint, avbd3d_RevoluteJointDef, maxMotorTorque,
              float)

    // Spherical
    JOINT_SET(avbd3d_SphericalJoint_EnableConeLimit, avbd3d_sphericalJoint, avbd3d_SphericalJointDef, enableConeLimit,
              bool)
    JOINT_SET(avbd3d_SphericalJoint_SetConeLimit, avbd3d_sphericalJoint, avbd3d_SphericalJointDef, coneAngle, float)
    JOINT_SET(avbd3d_SphericalJoint_EnableTwistLimit, avbd3d_sphericalJoint, avbd3d_SphericalJointDef, enableTwistLimit,
              bool)
    JOINT_SET2(avbd3d_SphericalJoint_SetTwistLimits, avbd3d_sphericalJoint, avbd3d_SphericalJointDef, lowerTwistAngle,
               upperTwistAngle)
    JOINT_SET(avbd3d_SphericalJoint_EnableSpring, avbd3d_sphericalJoint, avbd3d_SphericalJointDef, enableSpring, bool)
    JOINT_SET(avbd3d_SphericalJoint_SetSpringHertz, avbd3d_sphericalJoint, avbd3d_SphericalJointDef, hertz, float)
    JOINT_SET(avbd3d_SphericalJoint_SetSpringDampingRatio, avbd3d_sphericalJoint, avbd3d_SphericalJointDef,
              dampingRatio, float)
    JOINT_SET(avbd3d_SphericalJoint_EnableMotor, avbd3d_sphericalJoint, avbd3d_SphericalJointDef, enableMotor, bool)
    JOINT_SET(avbd3d_SphericalJoint_SetMotorVelocity, avbd3d_sphericalJoint, avbd3d_SphericalJointDef, motorVelocity,
              avbd3d_Vec3)
    JOINT_SET(avbd3d_SphericalJoint_SetMaxMotorTorque, avbd3d_sphericalJoint, avbd3d_SphericalJointDef, maxMotorTorque,
              float)

    // Weld
    JOINT_SET(avbd3d_WeldJoint_SetLinearHertz, avbd3d_weldJoint, avbd3d_WeldJointDef, linearHertz, float)
    JOINT_SET(avbd3d_WeldJoint_SetLinearDampingRatio, avbd3d_weldJoint, avbd3d_WeldJointDef, linearDampingRatio, float)
    JOINT_SET(avbd3d_WeldJoint_SetAngularHertz, avbd3d_weldJoint, avbd3d_WeldJointDef, angularHertz, float)
    JOINT_SET(avbd3d_WeldJoint_SetAngularDampingRatio, avbd3d_weldJoint, avbd3d_WeldJointDef, angularDampingRatio,
              float)

    // Wheel
    JOINT_SET(avbd3d_WheelJoint_EnableSuspension, avbd3d_wheelJoint, avbd3d_WheelJointDef, enableSuspensionSpring, bool)
    JOINT_SET(avbd3d_WheelJoint_SetSuspensionHertz, avbd3d_wheelJoint, avbd3d_WheelJointDef, suspensionHertz, float)
    JOINT_SET(avbd3d_WheelJoint_SetSuspensionDampingRatio, avbd3d_wheelJoint, avbd3d_WheelJointDef,
              suspensionDampingRatio, float)
    JOINT_SET(avbd3d_WheelJoint_EnableSuspensionLimit, avbd3d_wheelJoint, avbd3d_WheelJointDef, enableSuspensionLimit,
              bool)
    JOINT_SET2(avbd3d_WheelJoint_SetSuspensionLimits, avbd3d_wheelJoint, avbd3d_WheelJointDef, lowerSuspensionLimit,
               upperSuspensionLimit)
    JOINT_SET(avbd3d_WheelJoint_EnableSpinMotor, avbd3d_wheelJoint, avbd3d_WheelJointDef, enableSpinMotor, bool)
    JOINT_SET(avbd3d_WheelJoint_SetSpinMotorSpeed, avbd3d_wheelJoint, avbd3d_WheelJointDef, spinSpeed, float)
    JOINT_SET(avbd3d_WheelJoint_SetMaxSpinTorque, avbd3d_wheelJoint, avbd3d_WheelJointDef, maxSpinTorque, float)
    JOINT_SET(avbd3d_WheelJoint_EnableSteering, avbd3d_wheelJoint, avbd3d_WheelJointDef, enableSteering, bool)
    JOINT_SET(avbd3d_WheelJoint_SetSteeringHertz, avbd3d_wheelJoint, avbd3d_WheelJointDef, steeringHertz, float)
    JOINT_SET(avbd3d_WheelJoint_SetSteeringDampingRatio, avbd3d_wheelJoint, avbd3d_WheelJointDef, steeringDampingRatio,
              float)
    JOINT_SET(avbd3d_WheelJoint_SetMaxSteeringTorque, avbd3d_wheelJoint, avbd3d_WheelJointDef, maxSteeringTorque, float)
    JOINT_SET(avbd3d_WheelJoint_EnableSteeringLimit, avbd3d_wheelJoint, avbd3d_WheelJointDef, enableSteeringLimit, bool)
    JOINT_SET2(avbd3d_WheelJoint_SetSteeringLimits, avbd3d_wheelJoint, avbd3d_WheelJointDef, lowerSteeringLimit,
               upperSteeringLimit)
    JOINT_SET(avbd3d_WheelJoint_SetTargetSteeringAngle, avbd3d_wheelJoint, avbd3d_WheelJointDef, targetSteeringAngle,
              float)

    // ---- body --------------------------------------------------------------------------------------

    void avbd3d_Body_SetBullet(avbd3d_BodyId id, bool flag)
    {
        int b;
        if (World *w = worldOfBody(id, b))
        {
            w->bodies[b].bullet = flag;
            w->syncProps(b);
        }
    }

    int avbd3d_Body_GetJointCount(avbd3d_BodyId id)
    {
        int b;
        World *w = worldOfBody(id, b);
        return w ? (int)liveJointIds(w, b).size() : 0;
    }

    int avbd3d_Body_GetJoints(avbd3d_BodyId id, avbd3d_JointId *jointArray, int capacity)
    {
        int b;
        World *w = worldOfBody(id, b);
        if (!w)
            return 0;
        const std::vector<avbd3d_JointId> ids = liveJointIds(w, b);
        const int n = std::min((int)ids.size(), capacity);
        for (int i = 0; jointArray && i < n; i++)
            jointArray[i] = ids[i];
        return n > 0 ? n : 0;
    }

    avbd3d_MassData avbd3d_Body_GetMassData(avbd3d_BodyId id)
    {
        int b;
        avbd3d_MassData m{};
        if (World *w = worldOfBody(id, b))
        {
            m.mass = w->desc.mass[b];
            m.center = fromV(w->desc.bodyLocalCenter[b]);
            m.inertia = principalToMatrix(w->desc.moment[b], w->desc.bodyPrincipalRot[b]);
        }
        return m;
    }

    avbd3d_AABB avbd3d_Body_ComputeAABB(avbd3d_BodyId id)
    {
        int b;
        avbd3d_AABB out{};
        World *w = worldOfBody(id, b);
        if (!w)
            return out;
        bool any = false;
        b3AABB box{};
        for (int s : w->bodies[b].shapes)
        {
            const int proxy = w->shapes[s].proxy;
            if (proxy < 0)
                continue;
            w->ensureProxies();
            const b3AABB a = w->shapes[s].aabb;
            if (!any)
            {
                box = a;
                any = true;
                continue;
            }
            box.lowerBound.x = std::min(box.lowerBound.x, a.lowerBound.x);
            box.lowerBound.y = std::min(box.lowerBound.y, a.lowerBound.y);
            box.lowerBound.z = std::min(box.lowerBound.z, a.lowerBound.z);
            box.upperBound.x = std::max(box.upperBound.x, a.upperBound.x);
            box.upperBound.y = std::max(box.upperBound.y, a.upperBound.y);
            box.upperBound.z = std::max(box.upperBound.z, a.upperBound.z);
        }
        return any ? avbd3d_bit<avbd3d_AABB>(box) : out;
    }

    avbd3d_Matrix3 avbd3d_Body_GetWorldInverseRotationalInertia(avbd3d_BodyId id)
    {
        int b;
        avbd3d_Matrix3 out{};
        if (World *w = worldOfBody(id, b))
        {
            auto inv = [](float v) { return v > 0.0f ? 1.0f / v : 0.0f; };
            const vec3 i = w->desc.moment[b];
            b3Matrix3 d{};
            d.cx = b3Vec3{inv(i.x), 0.0f, 0.0f};
            d.cy = b3Vec3{0.0f, inv(i.y), 0.0f};
            d.cz = b3Vec3{0.0f, 0.0f, inv(i.z)};
            const b3Matrix3 r = b3MakeMatrixFromQuat(avbd3d_bit<b3Quat>(fromQ(w->desc.rot[b])));
            out = avbd3d_bit<avbd3d_Matrix3>(b3MulMM(b3MulMM(r, d), b3Transpose(r)));
        }
        return out;
    }

    // A dynamic body only, as Box3D. The inertia (about `center`) is diagonalised into the principal
    // moments the solver uses; a non-positive mass is ignored, since mass 0 is static on the device.
    void avbd3d_Body_SetMassData(avbd3d_BodyId id, avbd3d_MassData massData)
    {
        int b;
        World *w = worldOfBody(id, b);
        if (!w || w->bodies[b].type != avbd3d_dynamicBody || !(massData.mass > 0.0f))
            return;
        const avbd3d_Matrix3 &I = massData.inertia;
        const float a[3][3] = {{I.cx.x, I.cy.x, I.cz.x}, {I.cx.y, I.cy.y, I.cz.y}, {I.cx.z, I.cy.z, I.cz.z}};
        vec3 moment;
        quat rot;
        principalInertia(a, moment, rot);
        // A zero moment would leave the angular block of the primal system singular.
        const float floor = 1e-6f * massData.mass;
        moment = vec3{std::max(moment.x, floor), std::max(moment.y, floor), std::max(moment.z, floor)};
        w->applyMassFrame(b, massData.mass, toV(massData.center), moment, rot);
    }

    void avbd3d_Body_ApplyMassFromShapes(avbd3d_BodyId id)
    {
        int b;
        if (World *w = worldOfBody(id, b))
            w->updateMass(b);
    }

    // ---- shape -----------------------------------------------------------------------------------

    void avbd3d_Shape_SetDensity(avbd3d_ShapeId id, float density, bool updateBodyMass)
    {
        int s;
        World *w = worldOfShape(id, s);
        if (!w || !(density >= 0.0f))
            return;
        w->forShapeGroup(s, [&](int k) { w->shapes[k].density = density; });
        if (updateBodyMass)
            w->updateMass(w->shapes[s].body);
    }

    avbd3d_AABB avbd3d_Shape_GetAABB(avbd3d_ShapeId id)
    {
        int s;
        avbd3d_AABB out{};
        World *w = worldOfShape(id, s);
        if (!w || w->shapes[s].proxy < 0)
            return out;
        w->ensureProxies();
        b3AABB box = w->shapes[s].aabb;
        for (int m : w->shapes[s].members) // a compound: the union of its children
            if (w->shapes[m].proxy >= 0)
                box = b3AABB_Union(box, w->shapes[m].aabb);
        return avbd3d_bit<avbd3d_AABB>(box);
    }

    // The shape alone at its density: centre in the body frame, inertia about that centre. A mesh, a
    // height field or a hull before its geometry has none.
    avbd3d_MassData avbd3d_Shape_ComputeMassData(avbd3d_ShapeId id)
    {
        int s;
        avbd3d_MassData m{};
        World *w = worldOfShape(id, s);
        if (!w)
            return m;
        const ShapeRec &r = w->shapes[s];
        ShapeMassInput in;
        in.kind = w->desc.shapeKind[s];
        in.half = w->desc.shapeHalf[s];
        in.localPos = r.localPos;
        in.localRot = r.localRot;
        in.density = r.density;
        if (in.kind == AVBD_SHAPE_HULL)
            in.hull = (r.geom.hull || r.geom.box) ? r.geom.hullData() : nullptr;
        if (in.kind == AVBD_SHAPE_MESH)
            in.mesh = r.geom.mesh.get();
        float mass;
        vec3 center;
        float a[3][3];
        computeShapeMass(in, r.density, mass, center, a);
        m.mass = mass;
        m.center = fromV(center);
        m.inertia.cx = avbd3d_Vec3{a[0][0], a[1][0], a[2][0]};
        m.inertia.cy = avbd3d_Vec3{a[0][1], a[1][1], a[2][1]};
        m.inertia.cz = avbd3d_Vec3{a[0][2], a[1][2], a[2][2]};
        return m;
    }

    // ---- world -----------------------------------------------------------------------------------

    avbd3d_AABB avbd3d_World_GetBounds(avbd3d_WorldId id)
    {
        World *w = settled(id);
        return w ? avbd3d_bit<avbd3d_AABB>(b3DynamicTree_GetRootBounds(&w->queryTree())) : avbd3d_AABB{};
    }

    avbd3d_JointEvents avbd3d_World_GetJointEvents(avbd3d_WorldId id)
    {
        avbd3d_JointEvents e{nullptr, 0};
        if (World *w = settled(id))
        {
            std::vector<avbd3d_JointEvent> &ev = w->jointEvents();
            e.jointEvents = ev.data();
            e.count = (int)ev.size();
        }
        return e;
    }

    void avbd3d_World_EnableContinuous(avbd3d_WorldId id, bool flag)
    {
        setWorldProp(id, "continuous", flag ? 1.0 : 0.0);
    }

    void avbd3d_World_SetRestitutionThreshold(avbd3d_WorldId id, float value)
    {
        if (World *w = settled(id))
            w->setRestitutionThreshold(value);
    }

    void avbd3d_World_SetFrictionMixing(avbd3d_WorldId id, avbd3d_MixingRule rule)
    {
        if (World *w = settled(id))
            w->setMixing((int)rule, w->restitutionMixing);
    }

    void avbd3d_World_SetRestitutionMixing(avbd3d_WorldId id, avbd3d_MixingRule rule)
    {
        if (World *w = settled(id))
            w->setMixing(w->frictionMixing, (int)rule);
    }

    // ---- joint: state readout (Box3D semantics) ---------------------------------------------------

    // Motor readouts come from the newest finished step's multiplier of the joint's velocity-motor
    // row. A linear lambda is a force on the solver's coordinate (anchor A minus anchor B), so the
    // force on B along Box3D's axis has the same sign; an angular lambda is torque / arm on the
    // coordinate "B relative to A", so the torque on B is -lambda * arm.
    float avbd3d_RevoluteJoint_GetAngle(avbd3d_JointId id)
    {
        JointView v;
        return loadView(id, avbd3d_revoluteJoint, v) ? twistAngleOf(relativeFrame(v)) : 0.0f;
    }

    float avbd3d_RevoluteJoint_GetSpeed(avbd3d_JointId id)
    {
        JointView v;
        if (!loadView(id, avbd3d_revoluteJoint, v))
            return 0.0f;
        return dot(v.wB - v.wA, rotate(v.frameA(), vec3{0, 0, 1}));
    }

    float avbd3d_RevoluteJoint_GetMotorTorque(avbd3d_JointId id)
    {
        int r;
        float lam, arm;
        World *w = worldOfJoint(id, r);
        if (!w || w->joints[r].type != avbd3d_revoluteJoint || !w->jointMotorLambda(r, 5, lam, arm))
            return 0.0f;
        return -lam * arm;
    }

    float avbd3d_PrismaticJoint_GetTranslation(avbd3d_JointId id)
    {
        JointView v;
        if (!loadView(id, avbd3d_prismaticJoint, v))
            return 0.0f;
        vec3 axis;
        const vec3 d = prismaticSeparation(v, axis);
        return dot(d, axis);
    }

    float avbd3d_PrismaticJoint_GetSpeed(avbd3d_JointId id)
    {
        JointView v;
        if (!loadView(id, avbd3d_prismaticJoint, v))
            return 0.0f;
        vec3 axis;
        const vec3 d = prismaticSeparation(v, axis);
        const vec3 rA = rotate(v.qA, v.fpA), rB = rotate(v.qB, v.fpB);
        const vec3 vRel = (v.vB + cross(v.wB, rB)) - (v.vA + cross(v.wA, rA));
        return dot(d, cross(v.wA, axis)) + dot(axis, vRel);
    }

    float avbd3d_PrismaticJoint_GetMotorForce(avbd3d_JointId id)
    {
        int r;
        float lam, arm;
        World *w = worldOfJoint(id, r);
        if (!w || w->joints[r].type != avbd3d_prismaticJoint || !w->jointMotorLambda(r, 0, lam, arm))
            return 0.0f;
        return lam;
    }

    float avbd3d_SphericalJoint_GetConeAngle(avbd3d_JointId id)
    {
        JointView v;
        if (!loadView(id, avbd3d_sphericalJoint, v))
            return 0.0f;
        const quat q = relativeFrame(v);
        return 2.0f * std::atan2(std::sqrt(q.x * q.x + q.y * q.y), std::sqrt(q.z * q.z + q.w * q.w));
    }

    float avbd3d_SphericalJoint_GetTwistAngle(avbd3d_JointId id)
    {
        JointView v;
        return loadView(id, avbd3d_sphericalJoint, v) ? twistAngleOf(relativeFrame(v)) : 0.0f;
    }

    avbd3d_Vec3 avbd3d_SphericalJoint_GetMotorTorque(avbd3d_JointId id)
    {
        int r;
        avbd3d_Vec3 out{0, 0, 0};
        World *w = worldOfJoint(id, r);
        if (!w || w->joints[r].type != avbd3d_sphericalJoint)
            return out;
        float t[3] = {0, 0, 0};
        for (int k = 0; k < 3; k++)
        {
            float lam, arm;
            if (w->jointMotorLambda(r, 3 + k, lam, arm))
                t[k] = -lam * arm;
        }
        return avbd3d_Vec3{t[0], t[1], t[2]};
    }

    void avbd3d_SphericalJoint_SetTargetRotation(avbd3d_JointId id, avbd3d_Quat q)
    {
        avbd3d_SphericalJointDef d;
        World *w;
        int r;
        if (!loadDef(id, avbd3d_sphericalJoint, d, w, r))
            return;
        d.targetRotation = q;
        storeDef(w, r, d);
        // A new target only matters to a body that can move: wake both ends, as Box3D does.
        const JointRec &jr = w->joints[r];
        if (jr.a >= 0)
            w->setAwake(jr.a, true);
        if (jr.b >= 0)
            w->setAwake(jr.b, true);
    }

    float avbd3d_WheelJoint_GetSpinSpeed(avbd3d_JointId id)
    {
        JointView v;
        if (!loadView(id, avbd3d_wheelJoint, v))
            return 0.0f;
        return dot(v.wB - v.wA, rotate(v.frameB(), vec3{0, 0, 1}));
    }

    float avbd3d_WheelJoint_GetSpinTorque(avbd3d_JointId id)
    {
        int r;
        float lam, arm;
        World *w = worldOfJoint(id, r);
        if (!w || w->joints[r].type != avbd3d_wheelJoint || !w->jointMotorLambda(r, 5, lam, arm))
            return 0.0f;
        return -lam * arm;
    }

    float avbd3d_WheelJoint_GetSteeringAngle(avbd3d_JointId id)
    {
        JointView v;
        if (!loadView(id, avbd3d_wheelJoint, v))
            return 0.0f;
        const vec3 zA = rotate(v.frameA(), vec3{0, 0, 1}), yA = rotate(v.frameA(), vec3{0, 1, 0});
        const vec3 zB = rotate(v.frameB(), vec3{0, 0, 1});
        return std::atan2(-dot(zB, yA), dot(zB, zA));
    }

    float avbd3d_DistanceJoint_GetCurrentLength(avbd3d_JointId id)
    {
        JointView v;
        if (!loadView(id, avbd3d_distanceJoint, v))
            return 0.0f;
        return length((v.pB + rotate(v.qB, v.fpB)) - (v.pA + rotate(v.qA, v.fpA)));
    }

    // The motor row is the distance joint's third scalar slot. Its coordinate is the length itself, so
    // the force on B along A -> B is -lambda.
    float avbd3d_DistanceJoint_GetMotorForce(avbd3d_JointId id)
    {
        int r;
        float lam, arm;
        World *w = worldOfJoint(id, r);
        if (!w || w->joints[r].type != avbd3d_distanceJoint || !w->jointMotorLambda(r, 2, lam, arm))
            return 0.0f;
        return -lam;
    }

} // extern "C"
