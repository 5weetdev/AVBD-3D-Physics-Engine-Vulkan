#pragma once
// Box3D-style joint presets over SceneBuilder (docs/SOLVER_3D_ROADMAP.md, Phase 3). Host only,
// header only, no dependency on Box3D: the parameter structs repeat Box3D's field names and
// meanings so a caller holding a b3*JointDef can fill them one to one, and each preset lowers
// to addJoint6 rows (plus, where Box3D layers several behaviours on one axis, a companion
// joint, see "Layers").
//
// Frames. localFrameA/B are body-local (position + rotation), exactly Box3D's: they map to
// JointDesc rA/frameA and rB/frameB. bodyA < 0 anchors to the world and its frame is then a
// world frame. The two frames should coincide in world space at authoring time (AGENTS.md
// gotcha 10: C0 ~ 0); a preset does not move anything.
//
// Axis convention. The solver's six axes are linear x/y/z then angular x/y/z of frame A.
//
//   Box3D                              solver axis
//   ---------------------------------  ---------------------------------------------------
//   revolute hinge  (frame z)          angular 5 (z): twist.   Angular 3,4 locked.
//   prismatic slide (frame A x)        linear 0 (x).           Linear 1,2 and angular 0-2 locked.
//   wheel suspension (frame A x)       linear 0.               Linear 1,2 and angular 4 locked.
//   wheel spin      (frame B z)        angular 5 (twist).
//   wheel steering  (frame A x)        angular 3 (swing about x).
//   spherical cone  (frame A z)        angular 3 as ONE cone row on the swing angle
//                                      (JointAxisCone); angular 4 is then unused.
//   spherical twist (frame B z)        angular 5 (twist), about the same z as the cone axis.
//   parallel (z of A vs z of B)        angular 3,4 (swing x, y).
//   distance                           generic 2: linear 0 length / 1 min-max / 2 motor, all
//                                      scalar rows along the anchor-to-anchor direction.
//
// Signs. The solver's linear coordinate is anchorA - anchorB in frame A; Box3D's translation is
// anchorB - anchorA. The prismatic, wheel and motor presets negate linear targets, limits (which
// swap) and speeds so callers keep Box3D's meaning. Angles already agree (B relative to A).
//
// Why the distance joint is a joint row and not a SpringSoA entry: it needs a multiplier and a
// ramped penalty (rigid length), one-sided limit rows and a force-bounded motor on the same
// anchors, and the readout and fracture of a joint. A SpringSoA entry is a stateless soft spring.
//
// Presets that name a z axis (revolute, wheel, spherical, parallel) use the swing-twist
// decomposition (generic 3): angular x,y are the swing's rotation vector and z is the exact
// twist angle in (-pi, pi], so a hinge limit or cone does not couple with the other rotation.
// Weld, prismatic and the motor joint use the rotation vector (generic 1).
//
// Layers. A solver axis has one mode, Box3D lets spring, limit and motor act together. Rows are
// placed motor first, then spring, then limit; a row whose axis is already taken goes to a
// companion joint between the same bodies with the same frames and every other axis free. The
// companion's index follows the primary's, and jointCount() grows by two (or three) rather than
// one. Thresholds and collideConnected apply to the primary; collideConnected is undone once per
// joint.
//
// A spring's dampingRatio zeta is honoured: c = 2 zeta sqrt(k m_eff), applied as an implicit-Euler
// damper on the row's change over the step (forces.slang, jointDamperCoef). zeta is stored in
// 1/16 steps (0 .. 1.94).
//   * drawScale / internalValue (not present here).
// Spring stiffness is k = m_eff (2 pi f)^2 with m_eff the two-body effective mass or inertia
// along the row, from the authored masses (rotational coupling at the anchors is ignored).
// forceThreshold / torqueThreshold are event thresholds: apply them with
// VkWorld::setJointThresholds(joint, forceThreshold, torqueThreshold) after build.
// Other documented limits: the spherical motor velocity is in joint frame A (x, y = swing, z =
// twist; it equals Box3D's world-space vector when the frames are aligned with the world); a
// wheel's steering is the swing about frame A x and is a good model for small angles; the motor
// joint's velocities are in frame A coordinates.

#include "scene_builder.h"

#include <cmath>
#include <vector>

namespace jointpresets
{

struct JointFrame
{
    vec3 p{0, 0, 0};
    quat q{0, 0, 0, 1};
};

struct JointBase
{
    int bodyA = -1; // < 0: the world
    int bodyB = -1;
    JointFrame localFrameA, localFrameB;
    float forceThreshold = 0.0f, torqueThreshold = 0.0f;
    bool collideConnected = false;
};

struct WeldDef
{
    JointBase base;
    float linearHertz = 0.0f, angularHertz = 0.0f; // 0 = rigid
    float linearDampingRatio = 0.0f, angularDampingRatio = 0.0f;
};

struct RevoluteDef
{
    JointBase base;
    float targetAngle = 0.0f; // spring target
    bool enableSpring = false;
    float hertz = 0.0f, dampingRatio = 0.0f;
    bool enableLimit = false;
    float lowerAngle = 0.0f, upperAngle = 0.0f;
    bool enableMotor = false;
    float maxMotorTorque = 0.0f, motorSpeed = 0.0f;
};

struct PrismaticDef
{
    JointBase base;
    bool enableSpring = false;
    float hertz = 0.0f, dampingRatio = 0.0f, targetTranslation = 0.0f;
    bool enableLimit = false;
    float lowerTranslation = 0.0f, upperTranslation = 0.0f;
    bool enableMotor = false;
    float maxMotorForce = 0.0f, motorSpeed = 0.0f;
};

struct WheelDef
{
    JointBase base;
    bool enableSuspensionSpring = false;
    float suspensionHertz = 0.0f, suspensionDampingRatio = 0.0f;
    bool enableSuspensionLimit = false;
    float lowerSuspensionLimit = 0.0f, upperSuspensionLimit = 0.0f;
    bool enableSpinMotor = false;
    float maxSpinTorque = 0.0f, spinSpeed = 0.0f;
    bool enableSteering = false;
    float steeringHertz = 0.0f, steeringDampingRatio = 0.0f, targetSteeringAngle = 0.0f;
    float maxSteeringTorque = 0.0f;
    bool enableSteeringLimit = false;
    float lowerSteeringLimit = 0.0f, upperSteeringLimit = 0.0f;
};

struct SphericalDef
{
    JointBase base;
    bool enableSpring = false; // pulls frame B toward targetRotation (B relative to A, identity = aligned)
    float hertz = 0.0f, dampingRatio = 0.0f;
    quat targetRotation{0, 0, 0, 1};
    bool enableConeLimit = false;
    float coneAngle = 0.0f; // [0, pi]
    bool enableTwistLimit = false;
    float lowerTwistAngle = 0.0f, upperTwistAngle = 0.0f;
    bool enableMotor = false;
    float maxMotorTorque = 0.0f;
    vec3 motorVelocity{0, 0, 0}; // joint frame A: x, y swing, z twist, all three driven
};

struct DistanceDef
{
    JointBase base;
    float length = -1.0f; // < 0: the authored anchor separation
    bool enableSpring = false;
    float lowerSpringForce = -1e30f; // Box3D: lower bounds tension (negative), upper compression
    float upperSpringForce = 1e30f;
    float hertz = 0.0f, dampingRatio = 0.0f;
    bool enableLimit = false;
    float minLength = 0.0f, maxLength = 1e30f;
    bool enableMotor = false;
    float maxMotorForce = 0.0f, motorSpeed = 0.0f;
};

struct MotorDef
{
    JointBase base;
    vec3 linearVelocity{0, 0, 0};
    float maxVelocityForce = 0.0f;
    vec3 angularVelocity{0, 0, 0};
    float maxVelocityTorque = 0.0f;
    float linearHertz = 0.0f, linearDampingRatio = 0.0f, maxSpringForce = 0.0f;
    float angularHertz = 0.0f, angularDampingRatio = 0.0f, maxSpringTorque = 0.0f;
};

struct ParallelDef
{
    JointBase base;
    float hertz = 0.0f; // 0 = rigid
    float dampingRatio = 0.0f;
    float maxTorque = 0.0f; // <= 0: unbounded
};

struct FilterDef
{
    JointBase base;
};

namespace detail
{

constexpr float kTwoPi = 6.28318530717958647692f;
constexpr int kGenericRotvec = 1, kGenericDistance = 2, kGenericSwingTwist = 3;

inline float finiteOrInf(float v) { return v > 1e29f ? INFINITY : v; }

inline vec3 unit(int k) { return vec3{k == 0 ? 1.0f : 0.0f, k == 1 ? 1.0f : 0.0f, k == 2 ? 1.0f : 0.0f}; }

inline quat worldFrame(const SceneBuilder &sb, int body, const JointFrame &f)
{
    return body >= 0 ? sb.bodyRot(body) * f.q : f.q;
}

// Effective mass of the pair along a world direction (translation only).
inline float linearMass(const SceneBuilder &sb, const JointBase &b)
{
    const float iA = b.bodyA >= 0 && sb.bodyMass(b.bodyA) > 0.0f ? 1.0f / sb.bodyMass(b.bodyA) : 0.0f;
    const float iB = b.bodyB >= 0 && sb.bodyMass(b.bodyB) > 0.0f ? 1.0f / sb.bodyMass(b.bodyB) : 0.0f;
    return iA + iB > 0.0f ? 1.0f / (iA + iB) : 1.0f;
}

inline float invInertiaAbout(const SceneBuilder &sb, int body, vec3 axisWorld)
{
    if (body < 0)
        return 0.0f;
    const vec3 m = sb.bodyMoment(body);
    if (!(sb.bodyMass(body) > 0.0f) || !(m.x > 0.0f))
        return 0.0f;
    const vec3 a = rotate(conjugate(sb.bodyRot(body)), axisWorld);
    return a.x * a.x / m.x + a.y * a.y / m.y + a.z * a.z / m.z;
}

inline float angularInertia(const SceneBuilder &sb, const JointBase &b, vec3 axisWorld)
{
    const float inv = invInertiaAbout(sb, b.bodyA, axisWorld) + invInertiaAbout(sb, b.bodyB, axisWorld);
    return inv > 0.0f ? 1.0f / inv : 1.0f;
}

inline float arm(const SceneBuilder &sb, const JointBase &b)
{
    const float a = sb.jointTorqueArm(b.bodyA, b.bodyB);
    return a > 0.0f ? a : 1.0f;
}

// hertz -> the solver's stored stiffness. Linear: k = m (2 pi f)^2. Angular: the stamp sees
// C' = arm * angle and torque = F * arm, so stored = I (2 pi f)^2 / arm^2.
inline float linStiffness(const SceneBuilder &sb, const JointBase &b, float hertz)
{
    const float w = kTwoPi * hertz;
    return linearMass(sb, b) * w * w;
}

inline float angStiffness(const SceneBuilder &sb, const JointBase &b, int frameAxis, float hertz)
{
    const quat qf = worldFrame(sb, b.bodyA, b.localFrameA);
    const float w = kTwoPi * hertz, a = arm(sb, b);
    return angularInertia(sb, b, rotate(qf, unit(frameAxis))) * w * w / (a * a);
}

inline float angBound(const SceneBuilder &sb, const JointBase &b, float maxTorque)
{
    return maxTorque > 0.0f ? maxTorque / arm(sb, b) : INFINITY;
}

// Rows placed on a stack of joints (see "Layers" above).
struct Rows
{
    struct Layer
    {
        JointDesc d;
        float boundLin = INFINITY, boundAng = INFINITY;
    };

    SceneBuilder &sb;
    const JointBase &base;
    int kind;
    std::vector<Layer> layers;

    Rows(SceneBuilder &s, const JointBase &b, int k) : sb(s), base(b), kind(k) { addLayer(); }

    void addLayer()
    {
        Layer l;
        l.d.rA = base.localFrameA.p;
        l.d.rB = base.localFrameB.p;
        l.d.frameA = base.localFrameA.q;
        l.d.frameB = base.localFrameB.q;
        layers.push_back(l);
    }

    // First layer whose axis is still free, adding one if none is.
    void place(int axis, int mode, float lo, float hi, float bound = INFINITY, float zeta = 0.0f)
    {
        size_t li = 0;
        while (li < layers.size() && layers[li].d.axis[axis].mode != JointAxisFree)
            li++;
        if (li == layers.size())
            addLayer();
        Layer &l = layers[li];
        l.d.axis[axis].mode = mode;
        l.d.axis[axis].lo = lo;
        l.d.axis[axis].hi = hi;
        l.d.axis[axis].damping = zeta;
        if (mode == JointAxisSpringF)
            (axis < 3 ? l.boundLin : l.boundAng) = bound;
    }

    void lock(int axis) { place(axis, JointAxisLocked, 0.0f, 0.0f); }

    int commit()
    {
        int first = -1;
        for (const Layer &l : layers)
        {
            const int j = sb.addJoint6(base.bodyA, base.bodyB, l.d);
            if (kind != kGenericRotvec)
                sb.setJointGeneric(j, kind);
            sb.setJointForceBounds(j, l.boundLin, l.boundAng);
            if (base.collideConnected)
                sb.allowCollision(base.bodyA, base.bodyB);
            if (first < 0)
                first = j;
        }
        return first;
    }
};

} // namespace detail

// Rows below: the rigid ones first, then motor, spring, limit.

inline int addWeld(SceneBuilder &sb, const WeldDef &def)
{
    detail::Rows r(sb, def.base, detail::kGenericRotvec);
    for (int k = 0; k < 3; k++)
    {
        if (def.linearHertz > 0.0f)
            r.place(k, JointAxisSpring, 0.0f, detail::linStiffness(sb, def.base, def.linearHertz), INFINITY, def.linearDampingRatio);
        else
            r.lock(k);
        if (def.angularHertz > 0.0f)
            r.place(3 + k, JointAxisSpring, 0.0f, detail::angStiffness(sb, def.base, k, def.angularHertz), INFINITY,
                    def.angularDampingRatio);
        else
            r.lock(3 + k);
    }
    return r.commit();
}

inline int addRevolute(SceneBuilder &sb, const RevoluteDef &def)
{
    detail::Rows r(sb, def.base, detail::kGenericSwingTwist);
    for (int k = 0; k < 3; k++)
        r.lock(k);
    r.lock(3);
    r.lock(4);
    if (def.enableMotor)
        r.place(5, JointAxisVelocityMotor, def.motorSpeed, detail::angBound(sb, def.base, def.maxMotorTorque));
    if (def.enableSpring && def.hertz > 0.0f)
        r.place(5, JointAxisSpring, def.targetAngle, detail::angStiffness(sb, def.base, 2, def.hertz), INFINITY,
                def.dampingRatio);
    if (def.enableLimit)
        r.place(5, JointAxisLimited, def.lowerAngle, def.upperAngle);
    return r.commit();
}

inline int addPrismatic(SceneBuilder &sb, const PrismaticDef &def)
{
    detail::Rows r(sb, def.base, detail::kGenericRotvec);
    r.lock(1);
    r.lock(2);
    for (int k = 3; k < 6; k++)
        r.lock(k);
    // Box3D's translation is anchorB - anchorA along frame A x; the solver's linear coordinate is
    // anchorA - anchorB, so every linear target, limit and speed changes sign (limits swap).
    if (def.enableMotor)
        r.place(0, JointAxisVelocityMotor, -def.motorSpeed, def.maxMotorForce > 0.0f ? def.maxMotorForce : INFINITY);
    if (def.enableSpring && def.hertz > 0.0f)
        r.place(0, JointAxisSpring, -def.targetTranslation, detail::linStiffness(sb, def.base, def.hertz), INFINITY,
                def.dampingRatio);
    if (def.enableLimit)
        r.place(0, JointAxisLimited, -def.upperTranslation, -def.lowerTranslation);
    return r.commit();
}

inline int addWheel(SceneBuilder &sb, const WheelDef &def)
{
    detail::Rows r(sb, def.base, detail::kGenericSwingTwist);
    r.lock(1);
    r.lock(2);
    r.lock(4);
    // Spin: the twist axis, free unless motorised.
    if (def.enableSpinMotor)
        r.place(5, JointAxisVelocityMotor, def.spinSpeed, detail::angBound(sb, def.base, def.maxSpinTorque));
    // Steering: swing about A's x. Fixed forward unless enabled.
    if (!def.enableSteering)
        r.lock(3);
    else
    {
        if (def.steeringHertz > 0.0f)
            r.place(3, JointAxisSpringF, def.targetSteeringAngle,
                    detail::angStiffness(sb, def.base, 0, def.steeringHertz),
                    detail::angBound(sb, def.base, def.maxSteeringTorque), def.steeringDampingRatio);
        if (def.enableSteeringLimit)
            r.place(3, JointAxisLimited, def.lowerSteeringLimit, def.upperSteeringLimit);
    }
    // Suspension along A's x.
    if (def.enableSuspensionSpring && def.suspensionHertz > 0.0f)
        r.place(0, JointAxisSpring, 0.0f, detail::linStiffness(sb, def.base, def.suspensionHertz), INFINITY,
                def.suspensionDampingRatio);
    if (def.enableSuspensionLimit)
        r.place(0, JointAxisLimited, -def.upperSuspensionLimit, -def.lowerSuspensionLimit); // see addPrismatic
    return r.commit();
}

// Swing-twist coordinates of a rotation of frame B relative to frame A, as the solver measures them
// (forces.slang jointRawAt, generic 3): x, y = rotation vector of the swing, z = twist angle.
inline vec3 swingTwistCoords(quat q)
{
    const float n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (!(n > 1e-6f))
        return vec3{0, 0, 0};
    q = quat{q.x / n, q.y / n, q.z / n, q.w / n};
    if (q.w < 0.0f)
        q = quat{-q.x, -q.y, -q.z, -q.w};
    const float hn = std::sqrt(q.z * q.z + q.w * q.w);
    const quat tw = hn > 1e-7f ? quat{0, 0, q.z / hn, q.w / hn} : quat{0, 0, 0, 1};
    quat sw = q * quat{-tw.x, -tw.y, -tw.z, tw.w};
    if (sw.w < 0.0f)
        sw = quat{-sw.x, -sw.y, -sw.z, -sw.w};
    const float s = std::sqrt(sw.x * sw.x + sw.y * sw.y + sw.z * sw.z);
    const float k = s < 1e-7f ? 2.0f : 2.0f * std::atan2(s, sw.w) / s;
    return vec3{sw.x * k, sw.y * k, 2.0f * std::atan2(q.z, q.w)};
}

inline int addSpherical(SceneBuilder &sb, const SphericalDef &def)
{
    detail::Rows r(sb, def.base, detail::kGenericSwingTwist);
    for (int k = 0; k < 3; k++)
        r.lock(k);
    // Motor on all three angular axes (frame A: x, y swing, z twist), then the spring toward
    // targetRotation, then the limits; an axis already taken moves the row to a companion joint.
    if (def.enableMotor)
    {
        const float v[3] = {def.motorVelocity.x, def.motorVelocity.y, def.motorVelocity.z};
        for (int k = 0; k < 3; k++)
            r.place(3 + k, JointAxisVelocityMotor, v[k], detail::angBound(sb, def.base, def.maxMotorTorque));
    }
    if (def.enableSpring && def.hertz > 0.0f)
    {
        const vec3 t = swingTwistCoords(def.targetRotation);
        const float tv[3] = {t.x, t.y, t.z};
        for (int k = 0; k < 3; k++)
            r.place(3 + k, JointAxisSpring, tv[k], detail::angStiffness(sb, def.base, k, def.hertz), INFINITY,
                    def.dampingRatio);
    }
    if (def.enableConeLimit)
        r.place(3, JointAxisCone, 0.0f, def.coneAngle);
    if (def.enableTwistLimit)
        r.place(5, JointAxisLimited, def.lowerTwistAngle, def.upperTwistAngle);
    return r.commit();
}

inline int addParallel(SceneBuilder &sb, const ParallelDef &def)
{
    detail::Rows r(sb, def.base, detail::kGenericSwingTwist);
    for (int k = 0; k < 2; k++)
    {
        if (def.hertz > 0.0f)
            r.place(3 + k, JointAxisSpringF, 0.0f, detail::angStiffness(sb, def.base, k, def.hertz),
                    detail::angBound(sb, def.base, def.maxTorque), def.dampingRatio);
        else
            r.lock(3 + k);
    }
    return r.commit();
}

inline int addMotor(SceneBuilder &sb, const MotorDef &def)
{
    detail::Rows r(sb, def.base, detail::kGenericRotvec);
    // Linear velocities change sign: the solver's linear coordinate is anchorA - anchorB (see addPrismatic).
    const float vl[3] = {-def.linearVelocity.x, -def.linearVelocity.y, -def.linearVelocity.z};
    const float va[3] = {def.angularVelocity.x, def.angularVelocity.y, def.angularVelocity.z};
    for (int k = 0; k < 3; k++)
    {
        if (def.maxVelocityForce > 0.0f)
            r.place(k, JointAxisVelocityMotor, vl[k], def.maxVelocityForce);
        if (def.maxVelocityTorque > 0.0f)
            r.place(3 + k, JointAxisVelocityMotor, va[k], detail::angBound(sb, def.base, def.maxVelocityTorque));
    }
    for (int k = 0; k < 3; k++)
    {
        if (def.linearHertz > 0.0f)
            r.place(k, JointAxisSpringF, 0.0f, detail::linStiffness(sb, def.base, def.linearHertz),
                    def.maxSpringForce > 0.0f ? def.maxSpringForce : INFINITY, def.linearDampingRatio);
        if (def.angularHertz > 0.0f)
            r.place(3 + k, JointAxisSpringF, 0.0f, detail::angStiffness(sb, def.base, k, def.angularHertz),
                    detail::angBound(sb, def.base, def.maxSpringTorque), def.angularDampingRatio);
    }
    return r.commit();
}

inline int addDistance(SceneBuilder &sb, const DistanceDef &def)
{
    const JointBase &b = def.base;
    JointDesc d;
    d.rA = b.localFrameA.p;
    d.rB = b.localFrameB.p;
    d.frameA = b.localFrameA.q;
    d.frameB = b.localFrameB.q;

    const vec3 pA = b.bodyA >= 0 ? transform(sb.bodyPos(b.bodyA), sb.bodyRot(b.bodyA), d.rA) : d.rA;
    const vec3 pB = transform(sb.bodyPos(b.bodyB), sb.bodyRot(b.bodyB), d.rB);
    const float rest = def.length >= 0.0f ? def.length : length(pA - pB);

    float boundLin = INFINITY, boundAng = INFINITY;
    if (def.enableSpring && def.hertz > 0.0f && def.minLength < def.maxLength)
    {
        // Box3D's lower/upperSpringForce are signed: lower <= 0 bounds tension, upper >= 0 compression.
        d.axis[0] = {JointAxisSpringF, rest, detail::linStiffness(sb, b, def.hertz), def.dampingRatio};
        boundLin = detail::finiteOrInf(std::fmax(0.0f, -def.lowerSpringForce));
        boundAng = detail::finiteOrInf(std::fmax(0.0f, def.upperSpringForce));
        if (def.enableMotor)
            d.axis[2] = {JointAxisVelocityMotor, def.motorSpeed, def.maxMotorForce > 0.0f ? def.maxMotorForce : INFINITY};
        if (def.enableLimit)
            d.axis[1] = {JointAxisLimited, def.minLength, def.maxLength};
    }
    else
    {
        // Rigid: overrides the limit and the motor, as in Box3D. MOTOR with an unbounded force is a
        // hard row at `rest`.
        d.axis[0] = {JointAxisMotor, rest, INFINITY};
    }

    const int j = sb.addJoint6(b.bodyA, b.bodyB, d);
    sb.setJointGeneric(j, detail::kGenericDistance);
    sb.setJointForceBounds(j, boundLin, boundAng);
    if (b.collideConnected)
        sb.allowCollision(b.bodyA, b.bodyB);
    return j;
}

// A filter joint is only a collision filter: no constraint, so no joint index (-1).
inline int addFilter(SceneBuilder &sb, const FilterDef &def)
{
    sb.ignoreCollision(def.base.bodyA, def.base.bodyB);
    return -1;
}

} // namespace jointpresets
