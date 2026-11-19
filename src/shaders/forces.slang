// Ported from engine/core/gpu_forces.cuh -- the persistent-force (joint/spring) primal
// stamp and dual update: the ball-socket weld (`j.generic[i] == 0`), the generic 6-DOF
// joint (per-axis free/locked/limited/motor/spring, `j.generic[i] != 0`) and the plain
// distance spring. Kernels call jointInitialize/stampJoint/jointDual, which dispatch on
// the generic flag exactly as gpu_forces.cuh does. Same term order as gpu_forces.cuh
// throughout so the two read side by side.

module forces;

import maths;
import types;

public static const int FORCE_MANIFOLD = 0;
public static const int FORCE_JOINT = 1;
public static const int FORCE_SPRING = 2;

public int graphType(uint64_t k) { return (int)((uint)(k & 0xFFFFFFFFu) >> 29); }
public int graphIndex(uint64_t k) { return (int)((uint)(k & 0xFFFFFFFFu) & ((1u << 29) - 1u)); }

// World-space anchor point. A world anchor (body < 0) is already in world space.
public vec3 jointAnchor(BodySoA b, int body, vec3 r)
{
    return body >= 0 ? transformQ(b.pos[body], b.rot[body], r) : r;
}

public quat jointRot(BodySoA b, int body)
{
    return body >= 0 ? b.rot[body] : quat(0.0f, 0.0f, 0.0f, 1.0f);
}

// H = outer(r, F) - dot(r, F) * I, column-norm diagonal approximation (gpu_forces.cuh).
public float3x3 geometricStiffnessBallSocketDiag(vec3 r, vec3 F)
{
    float d = dot3(r, F);
    float d2 = d * d;
    float rSq = lengthSq3(r);
    return diagonalM(
        sqrt(max(0.0f, rSq * F.x * F.x - 2.0f * r.x * F.x * d + d2)),
        sqrt(max(0.0f, rSq * F.y * F.y - 2.0f * r.y * F.y * d + d2)),
        sqrt(max(0.0f, rSq * F.z * F.z - 2.0f * r.z * F.z * d + d2)));
}

// PENALTY_MIN/MAX (collision.h) and JOINT_PENALTY_MAX (gpu_forces.cuh) -- kept local so
// forces.slang does not have to import dual.slang/narrowphase.slang just for constants.
static const float PENALTY_MIN_J = 1.0f;
static const float PENALTY_MAX_J = 10000000000.0f;
static const float JOINT_PENALTY_MAX = 500000.0f;

// Effective mass along a world-space linear axis: J M^-1 J^T for the two bodies' linear
// and angular response (gpu_forces.cuh:jointEffMassLin). body A may be a world anchor
// (ia < 0), which contributes nothing.
float jointEffMassLin(BodySoA b, int ia, int ib, vec3 rAW, vec3 rBW, vec3 axis, float dt)
{
    float invMA = ia >= 0 && b.mass[ia] > 0.0f ? 1.0f / b.mass[ia] : 0.0f;
    float invMB = b.mass[ib] > 0.0f ? 1.0f / b.mass[ib] : 0.0f;
    float w = invMA + invMB;

    if (ia >= 0)
    {
        vec3 momA = b.moment[ia];
        if (momA.x > 0.0f)
        {
            vec3 cA = cross3(rAW, axis);
            w += cA.x * cA.x / momA.x + cA.y * cA.y / momA.y + cA.z * cA.z / momA.z;
        }
    }
    vec3 momB = b.moment[ib];
    if (momB.x > 0.0f)
    {
        vec3 cB = cross3(rBW, axis);
        w += cB.x * cB.x / momB.x + cB.y * cB.y / momB.y + cB.z * cB.z / momB.z;
    }
    return w > 0.0f ? clampf(1.0f / (w * dt * dt), PENALTY_MIN_J, PENALTY_MAX_J) : PENALTY_MIN_J;
}

// Effective inertia along a world-space rotation axis (gpu_forces.cuh:jointEffMassAng).
float jointEffMassAng(BodySoA b, int ia, int ib, vec3 axis, float torqueArm, float dt)
{
    float w = 0.0f;
    if (ia >= 0)
    {
        vec3 momA = b.moment[ia];
        if (momA.x > 0.0f)
        {
            vec3 aBody = rotateQ(conjugateQ(b.rot[ia]), axis);
            w += aBody.x * aBody.x / momA.x + aBody.y * aBody.y / momA.y + aBody.z * aBody.z / momA.z;
        }
    }
    vec3 momB = b.moment[ib];
    if (momB.x > 0.0f)
    {
        vec3 bBody = rotateQ(conjugateQ(b.rot[ib]), axis);
        w += bBody.x * bBody.x / momB.x + bBody.y * bBody.y / momB.y + bBody.z * bBody.z / momB.z;
    }
    w *= torqueArm * torqueArm;
    return w > 0.0f ? clampf(1.0f / (w * dt * dt), PENALTY_MIN_J, PENALTY_MAX_J) : PENALTY_MIN_J;
}

// Scales a warm-started lambda down where its matching penalty component has ramped
// above the effective-mass baseline, gated by relative speed (gpu_forces.cuh:jointDampLambda).
// dualDamping <= 0 is a no-op and bit-exact against the pre-existing behaviour.
vec3 jointDampLambda(vec3 lambda, vec3 penalty, vec3 kAuto, vec3 vRel, float dualDamping, float dualDampVel)
{
    if (dualDamping <= 0.0f)
        return lambda;
    float gate = dualDampVel > 0.0f ? max(0.0f, 1.0f - length3(vRel) / dualDampVel) : 1.0f;
    float strength = dualDamping * gate;
    if (strength <= 0.0f)
        return lambda;
    vec3 out = lambda;
    if (penalty.x > kAuto.x && kAuto.x > 0.0f)
        out.x *= 1.0f - strength * (1.0f - kAuto.x / penalty.x);
    if (penalty.y > kAuto.y && kAuto.y > 0.0f)
        out.y *= 1.0f - strength * (1.0f - kAuto.y / penalty.y);
    if (penalty.z > kAuto.z && kAuto.z > 0.0f)
        out.z *= 1.0f - strength * (1.0f - kAuto.z / penalty.z);
    return out;
}

// Transliterated from Joint::initialize (weld path only). Now ports the effective-mass
// auto-clamp: a scalar maxKLin/maxKAng taken from the max of the three world-axis
// baselines, since the weld path's penalty is diagonal in world space rather than a joint
// frame. Without it a warm-started weld penalty could ratchet past what the body masses
// involved can support.
public void jointInitializeWeld(JointSoA j, int i, BodySoA b, float alpha, float gamma, float dt,
                                float dualDamping, float dualDampVel)
{
    int ia = j.bodyA[i], ib = j.bodyB[i];

    j.C0Lin[i] = sub(jointAnchor(b, ia, j.rA[i]), transformQ(b.pos[ib], b.rot[ib], j.rB[i]));
    j.C0Ang[i] = scale(subQQ(mulQQ(jointRot(b, ia), j.restAng[i]), b.rot[ib]), j.torqueArm[i]);

    vec3 rAW = ia >= 0 ? rotateQ(b.rot[ia], j.rA[i]) : vec3(0.0f, 0.0f, 0.0f);
    vec3 rBW = rotateQ(b.rot[ib], j.rB[i]);
    float torqueArm = j.torqueArm[i];

    vec3 kAutoLin = vec3(jointEffMassLin(b, ia, ib, rAW, rBW, vec3(1, 0, 0), dt),
                         jointEffMassLin(b, ia, ib, rAW, rBW, vec3(0, 1, 0), dt),
                         jointEffMassLin(b, ia, ib, rAW, rBW, vec3(0, 0, 1), dt));
    vec3 kAutoAng = vec3(jointEffMassAng(b, ia, ib, vec3(1, 0, 0), torqueArm, dt),
                         jointEffMassAng(b, ia, ib, vec3(0, 1, 0), torqueArm, dt),
                         jointEffMassAng(b, ia, ib, vec3(0, 0, 1), torqueArm, dt));

    j.lambdaLin[i] = scale(j.lambdaLin[i], alpha * gamma);
    j.lambdaAng[i] = scale(j.lambdaAng[i], alpha * gamma);

    vec3 penLin = j.penaltyLin[i];
    vec3 penAng = j.penaltyAng[i];
    penLin = vec3(clampf(penLin.x * gamma, PENALTY_MIN_J, JOINT_PENALTY_MAX), clampf(penLin.y * gamma, PENALTY_MIN_J, JOINT_PENALTY_MAX), clampf(penLin.z * gamma, PENALTY_MIN_J, JOINT_PENALTY_MAX));
    penAng = vec3(clampf(penAng.x * gamma, PENALTY_MIN_J, JOINT_PENALTY_MAX), clampf(penAng.y * gamma, PENALTY_MIN_J, JOINT_PENALTY_MAX), clampf(penAng.z * gamma, PENALTY_MIN_J, JOINT_PENALTY_MAX));

    // Clamp to material stiffness AND the effective-mass baseline (prevents ratcheting
    // to infinity) -- the exact reference rule, a single scalar cap per block, not per axis.
    float maxKLin = min(max(10.0f * max(max(kAutoLin.x, kAutoLin.y), kAutoLin.z), 1000.0f), JOINT_PENALTY_MAX);
    float maxKAng = min(max(10.0f * max(max(kAutoAng.x, kAutoAng.y), kAutoAng.z), 1000.0f), JOINT_PENALTY_MAX);
    float capLin = min(j.stiffnessLin[i], maxKLin);
    float capAng = min(j.stiffnessAng[i], maxKAng);
    penLin = vec3(min(penLin.x, capLin), min(penLin.y, capLin), min(penLin.z, capLin));
    penAng = vec3(min(penAng.x, capAng), min(penAng.y, capAng), min(penAng.z, capAng));
    j.penaltyLin[i] = penLin;
    j.penaltyAng[i] = penAng;

    if (dualDamping > 0.0f)
    {
        vec3 vA = ia >= 0 ? add(b.velLin[ia], cross3(b.velAng[ia], rAW)) : vec3(0.0f, 0.0f, 0.0f);
        vec3 vB = add(b.velLin[ib], cross3(b.velAng[ib], rBW));
        vec3 wRel = sub(ia >= 0 ? b.velAng[ia] : vec3(0.0f, 0.0f, 0.0f), b.velAng[ib]);

        j.lambdaLin[i] = jointDampLambda(j.lambdaLin[i], penLin, kAutoLin, sub(vA, vB), dualDamping, dualDampVel);
        j.lambdaAng[i] = jointDampLambda(j.lambdaAng[i], penAng, kAutoAng, wRel, dualDamping, dualDampVel);
    }
}

// Transliterated from Joint::updatePrimal (weld path only). `isA` selects the Jacobian.
public void stampJointWeld(JointSoA j, int i, bool isA, BodySoA b, float alpha,
                            inout float3x3 lhsLin, inout float3x3 lhsAng, inout float3x3 lhsCross,
                            inout vec3 rhsLin, inout vec3 rhsAng)
{
    if (j.broken[i] != 0)
        return;

    int ia = j.bodyA[i], ib = j.bodyB[i];
    vec3 rA = j.rA[i], rB = j.rB[i];
    float torqueArm = j.torqueArm[i];

    vec3 penaltyLin = j.penaltyLin[i];
    if (lengthSq3(penaltyLin) > 0.0f)
    {
        float3x3 K = diagonalV(penaltyLin);
        vec3 C = sub(jointAnchor(b, ia, rA), transformQ(b.pos[ib], b.rot[ib], rB));

        if (isinf(j.stiffnessLin[i]))
            C = sub(C, scale(j.C0Lin[i], alpha));

        vec3 F = add(mulMV(K, C), j.lambdaLin[i]);

        float3x3 jAng = isA ? skewM(neg(rotateQ(b.rot[ia], rA))) : skewM(rotateQ(b.rot[ib], rB));
        float3x3 jAngT = transposeM(jAng);
        float3x3 jAngTk = mulMM(jAngT, K);

        lhsLin = addM(lhsLin, K);
        lhsAng = addM(lhsAng, mulMM(jAngTk, jAng));
        lhsCross = addM(lhsCross, isA ? jAngTk : negM(jAngTk));

        vec3 r = isA ? rotateQ(b.rot[ia], rA) : neg(rotateQ(b.rot[ib], rB));
        lhsAng = addM(lhsAng, geometricStiffnessBallSocketDiag(r, F));

        rhsLin = add(rhsLin, isA ? F : neg(F));
        rhsAng = add(rhsAng, mulMV(jAngT, F));
    }

    vec3 penaltyAng = j.penaltyAng[i];
    if (lengthSq3(penaltyAng) > 0.0f)
    {
        float3x3 K = diagonalV(penaltyAng);
        vec3 C = scale(subQQ(mulQQ(jointRot(b, ia), j.restAng[i]), b.rot[ib]), torqueArm);

        if (isinf(j.stiffnessAng[i]))
            C = sub(C, scale(j.C0Ang[i], alpha));

        vec3 F = add(mulMV(K, C), j.lambdaAng[i]);

        float arm2 = torqueArm * torqueArm;
        lhsAng = addM(lhsAng, scaleM(K, arm2));
        rhsAng = add(rhsAng, isA ? scale(F, torqueArm) : scale(F, -torqueArm));
    }
}

// Fracture test, transliterated from jointFracture.
public void jointFracture(JointSoA j, int i, BodySoA b)
{
    int ia = j.bodyA[i], ib = j.bodyB[i];
    float fracture = j.fracture[i];
    bool isBroken = false;

    if (fracture < 3.402823466e38f)
    {
        // The angular multiplier is force-scaled: the joint's torque is |lambdaAng| * torqueArm.
        // fractureAng (N*m) is the bending capacity; < 0 keeps the legacy test, whose implied
        // torque capacity is fracture * torqueArm (an unrelated number for a real section).
        const float capAng = j.fractureAng != nullptr ? j.fractureAng[i] : -1.0f;
        const float angTorque = length3(j.lambdaAng[i]) * j.torqueArm[i];
        const bool angBroken = capAng >= 0.0f ? angTorque > capAng : lengthSq3(j.lambdaAng[i]) > fracture * fracture;
        if (angBroken)
        {
            isBroken = true;
        }
        else
        {
            vec3 pA = ia >= 0 ? transformQ(b.pos[ia], b.rot[ia], j.rA[i]) : j.rA[i];
            vec3 pB = transformQ(b.pos[ib], b.rot[ib], j.rB[i]);
            vec3 d = sub(pA, pB);
            float len = length3(d);
            float axial = 0.0f;
            if (len >= 1e-6f)
                axial = dot3(j.lambdaLin[i], divs(d, len));

            vec3 shear = len >= 1e-6f ? sub(j.lambdaLin[i], scale(divs(d, len), axial)) : j.lambdaLin[i];

            if (axial > fracture || axial < -fracture * 10.0f || lengthSq3(shear) > fracture * fracture)
                isBroken = true;
        }
    }

    if (isBroken)
    {
        j.penaltyLin[i] = vec3(0.0f, 0.0f, 0.0f);
        j.penaltyAng[i] = vec3(0.0f, 0.0f, 0.0f);
        j.lambdaLin[i] = vec3(0.0f, 0.0f, 0.0f);
        j.lambdaAng[i] = vec3(0.0f, 0.0f, 0.0f);
        j.broken[i] = 1;
    }
}

// Transliterated from Joint::updateDual (weld path only), fracture test included.
public void jointDualWeld(JointSoA j, int i, BodySoA b, float alpha, float betaLin, float betaAng)
{
    if (j.broken[i] != 0)
        return;

    int ia = j.bodyA[i], ib = j.bodyB[i];
    float torqueArm = j.torqueArm[i];

    vec3 penaltyLin = j.penaltyLin[i];
    if (lengthSq3(penaltyLin) > 0.0f)
    {
        vec3 C = sub(jointAnchor(b, ia, j.rA[i]), transformQ(b.pos[ib], b.rot[ib], j.rB[i]));

        if (isinf(j.stiffnessLin[i]))
        {
            C = sub(C, scale(j.C0Lin[i], alpha));
            j.lambdaLin[i] = add(mulMV(diagonalV(penaltyLin), C), j.lambdaLin[i]);
        }

        vec3 absC = absv(C);
        vec3 grown = add(penaltyLin, scale(absC, betaLin));
        j.penaltyLin[i] = vec3(min(grown.x, min(j.stiffnessLin[i], 500000.0f)),
                                min(grown.y, min(j.stiffnessLin[i], 500000.0f)),
                                min(grown.z, min(j.stiffnessLin[i], 500000.0f)));
    }

    vec3 penaltyAng = j.penaltyAng[i];
    if (lengthSq3(penaltyAng) > 0.0f)
    {
        vec3 C = scale(subQQ(mulQQ(jointRot(b, ia), j.restAng[i]), b.rot[ib]), torqueArm);

        if (isinf(j.stiffnessAng[i]))
        {
            C = sub(C, scale(j.C0Ang[i], alpha));
            j.lambdaAng[i] = add(mulMV(diagonalV(penaltyAng), C), j.lambdaAng[i]);
        }

        vec3 absC = absv(C);
        vec3 grown = add(penaltyAng, scale(absC, betaAng));
        j.penaltyAng[i] = vec3(min(grown.x, min(j.stiffnessAng[i], 500000.0f)),
                                min(grown.y, min(j.stiffnessAng[i], 500000.0f)),
                                min(grown.z, min(j.stiffnessAng[i], 500000.0f)));
    }

    jointFracture(j, i, b);
}

// ---------------------------------------------------------------------------
// Generic 6-DOF joint (gpu_forces.cuh's rotationVector .. jointDual6). A joint with
// generic[i] == 0 is the weld above and never reaches this code. For a generic joint the
// dual state (C0/penalty/lambda) lives in joint-frame A's coordinates, not world.
// ---------------------------------------------------------------------------

// JointAxisMode (gpu_forces.cuh). FREE is 0 so a zeroed mode array is six free axes.
static const int JOINT_AXIS_FREE = 0;
static const int JOINT_AXIS_LOCKED = 1;
static const int JOINT_AXIS_LIMITED = 2;
static const int JOINT_AXIS_MOTOR = 3;
static const int JOINT_AXIS_SPRING = 4;
// Phase 3 rows (docs/SOLVER_3D_ROADMAP.md), all inside the same per-axis model:
//  SPRING_F  a spring whose force is bounded: lo = target, hi = stiffness, bound = the joint's
//            stiffnessLin (linear axes) / stiffnessAng (angular), which are unread on the generic
//            path. A distance joint (generic == 2) reads [-stiffnessAng, +stiffnessLin] instead.
//  VMOTOR    a velocity motor: lo = speed, hi = max force. Drives the axis to x(x^t) + speed*dt
//            and bounds the total force k*C + lambda, like MOTOR.
//  CONE      angular slot 3 of a swing-twist joint (generic == 3): one one-sided row on the swing
//            angle, hi = half angle. Clamped exactly as LIMITED, so its multiplier is released
//            inside the cone rather than warm-started into the opposite side.
static const int JOINT_AXIS_SPRING_F = 5;
static const int JOINT_AXIS_VMOTOR = 6;
static const int JOINT_AXIS_CONE = 7;
// j.generic values: 0 weld, 1 6-DOF (rotation-vector angles), 2 distance (all three linear
// slots are scalar rows along the anchor-to-anchor direction), 3 6-DOF with swing-twist angles.
static const int JOINT_GENERIC_DISTANCE = 2;
static const int JOINT_GENERIC_SWINGTWIST = 3;
static const float JOINT_PI = 3.14159265358979f;

float infF() { return asfloat(0x7f800000u); }

vec3 withComp(vec3 v, int k, float x)
{
    if (k == 0)
        v.x = x;
    else if (k == 1)
        v.y = x;
    else
        v.z = x;
    return v;
}

vec3 unitAxis(int k) { return vec3(k == 0 ? 1.0f : 0.0f, k == 1 ? 1.0f : 0.0f, k == 2 ? 1.0f : 0.0f); }

// The exact log map, not the small-angle quaternion difference the weld uses: limits are
// authored in radians and must sit where they were asked for at large angles.
vec3 rotationVector(quat q)
{
    // Shortest arc, so an angle past 180 degrees does not jump the long way round.
    if (q.w < 0.0f)
        q = scaleQ(q, -1.0f);
    vec3 v = vec3(q.x, q.y, q.z);
    float s = length3(v);
    if (s < 1.0e-7f)
        return scale(v, 2.0f);
    return scale(v, 2.0f * atan2(s, q.w) / s);
}

// One axis of the 6: the residual to drive to zero, and the axis's stiffness -- INFINITY
// for a hard axis, so the weld's isinf test still decides stabilization and multiplier.
float jointWrapAngle(float d) { return d - 2.0f * JOINT_PI * floor(d / (2.0f * JOINT_PI) + 0.5f); }

float jointAxisResidual(int mode, float lo, float hi, float x, float x0, float dt, bool angular,
                        out float stiffnessOut)
{
    stiffnessOut = infF();
    if (mode == JOINT_AXIS_LOCKED)
        return x;
    if (mode == JOINT_AXIS_LIMITED)
    {
        if (x < lo)
            return x - lo;
        if (x > hi)
            return x - hi;
        return 0.0f;
    }
    if (mode == JOINT_AXIS_MOTOR)
        return x - lo;
    if (mode == JOINT_AXIS_SPRING || mode == JOINT_AXIS_SPRING_F)
    {
        stiffnessOut = hi; // finite: soft, so no stabilization and no multiplier
        return x - lo;
    }
    if (mode == JOINT_AXIS_VMOTOR)
    {
        float d = x - x0; // x0 is the axis value at x^t
        return (angular ? jointWrapAngle(d) : d) - lo * dt;
    }
    if (mode == JOINT_AXIS_CONE)
        return x > hi ? x - hi : 0.0f;
    stiffnessOut = 0.0f; // JOINT_AXIS_FREE
    return 0.0f;
}

// The raw axis values of a generic joint at a given pair of poses: linear in frame A, angular
// either as the rotation vector (generic 1) or as swing-twist (generic 3: x,y = the swing's
// rotation vector, z = the twist angle in (-pi, pi]).
struct JointRaw
{
    quat qfA;
    vec3 lin, ang;
    vec3 coneDir; // unit swing direction in frame A's xy plane (generic 3)
    float swing;  // swing angle |(ang.x, ang.y)|
    vec3 n;       // unit anchor A - anchor B in world (distance joint)
    float len;
};

JointRaw jointRawAt(JointSoA j, int i, vec3 pA, quat qA, vec3 pB, quat qB)
{
    int ia = j.bodyA[i];
    JointRaw r;
    r.qfA = mulQQ(ia >= 0 ? qA : quat(0.0f, 0.0f, 0.0f, 1.0f), j.frameA[i]);
    quat qfB = mulQQ(qB, j.frameB[i]);
    quat inv = conjugateQ(normalizeQ(r.qfA));

    vec3 anchorA = ia >= 0 ? transformQ(pA, qA, j.rA[i]) : j.rA[i];
    vec3 dWorld = sub(anchorA, transformQ(pB, qB, j.rB[i]));
    r.lin = rotateQ(inv, dWorld);
    r.len = length3(dWorld);
    r.n = r.len > 1.0e-9f ? divs(dWorld, r.len) : vec3(1.0f, 0.0f, 0.0f);
    r.coneDir = vec3(1.0f, 0.0f, 0.0f);
    r.swing = 0.0f;

    quat qrel = mulQQ(inv, qfB);
    if ((int)j.generic[i] == JOINT_GENERIC_SWINGTWIST)
    {
        if (qrel.w < 0.0f)
            qrel = scaleQ(qrel, -1.0f);
        float hn = sqrt(qrel.z * qrel.z + qrel.w * qrel.w);
        quat tw = hn > 1.0e-7f ? quat(0.0f, 0.0f, qrel.z / hn, qrel.w / hn) : quat(0.0f, 0.0f, 0.0f, 1.0f);
        vec3 sv = rotationVector(mulQQ(qrel, conjugateQ(tw)));
        r.ang = vec3(sv.x, sv.y, 2.0f * atan2(qrel.z, qrel.w));
        r.swing = sqrt(sv.x * sv.x + sv.y * sv.y);
        if (r.swing > 1.0e-6f)
            r.coneDir = vec3(sv.x / r.swing, sv.y / r.swing, 0.0f);
    }
    else
        r.ang = rotationVector(qrel);
    return r;
}

// Pose at x^t: only a dynamic, awake body has a meaningful initialLin/Ang; a static or
// sleeping one is where it is.
vec3 jointPos0(BodySoA b, int body) { return b.mass[body] > 0.0f && b.awake[body] != 0 ? b.initialLin[body] : b.pos[body]; }
quat jointRot0(BodySoA b, int body) { return b.mass[body] > 0.0f && b.awake[body] != 0 ? b.initialAng[body] : b.rot[body]; }

// Everything the generic path needs from the current pose, gathered once so the primal
// stamp and the dual update cannot disagree about it.
struct JointAxisState
{
    quat qfA;        // joint frame A, in world
    vec3 cLin, cAng; // residual per axis, in frame A
    vec3 kLin, kAng; // per-axis stiffness (INFINITY for a hard axis)
    vec3 coneDir;    // generic 3: swing direction in frame A
    vec3 n;          // generic 2: unit direction A - B
    float len;       // generic 2: anchor distance
    bool damped;     // some soft spring axis has a damping ratio
    vec3 dLin, dAng; // C - C(x^t) on damped spring axes (angular wrapped), 0 elsewhere
};

// The axisMode byte is the mode in its low 3 bits and, for SPRING / SPRING_F, the damping
// ratio in the upper 5 bits (zeta = code / 16, so 0 .. 1.94 in steps of 1/16).
int axisBase(int m) { return m & 7; }
float axisZeta(int m)
{
    int b = m & 7;
    return (b == JOINT_AXIS_SPRING || b == JOINT_AXIS_SPRING_F) ? (float)(m >> 3) * 0.0625f : 0.0f;
}

JointAxisState jointAxisState(JointSoA j, int i, BodySoA b)
{
    int ia = j.bodyA[i], ib = j.bodyB[i];
    JointAxisState st;
    int gen = (int)j.generic[i];
    int base = i * 6;
    vec3 origin = vec3(0.0f, 0.0f, 0.0f);

    JointRaw cur = jointRawAt(j, i, ia >= 0 ? b.pos[ia] : origin, jointRot(b, ia), b.pos[ib], b.rot[ib]);
    JointRaw ini = cur;
    float dt = 0.0f;
    bool hasVel = false;
    bool hasDamp = false;
    for (int k = 0; k < 6; k++)
    {
        hasVel = hasVel || axisBase((int)j.axisMode[base + k]) == JOINT_AXIS_VMOTOR;
        hasDamp = hasDamp || axisZeta((int)j.axisMode[base + k]) > 0.0f;
    }
    st.damped = hasDamp;
    st.dLin = vec3(0.0f, 0.0f, 0.0f);
    st.dAng = vec3(0.0f, 0.0f, 0.0f);
    if (hasVel || hasDamp)
    {
        ini = jointRawAt(j, i, ia >= 0 ? jointPos0(b, ia) : origin,
                         ia >= 0 ? jointRot0(b, ia) : quat(0.0f, 0.0f, 0.0f, 1.0f), jointPos0(b, ib), jointRot0(b, ib));
        dt = j.restAng[i].x; // jointInitialize6 parks the step's dt here for a velocity motor or damper
    }

    st.qfA = cur.qfA;
    st.coneDir = cur.coneDir;
    st.n = cur.n;
    st.len = cur.len;

    vec3 zero = vec3(0.0f, 0.0f, 0.0f);
    st.cLin = zero;
    st.cAng = zero;
    st.kLin = zero;
    st.kAng = zero;
    for (int k = 0; k < 3; k++)
    {
        float kl, ka;
        float xl = gen == JOINT_GENERIC_DISTANCE ? cur.len : compAt(cur.lin, k);
        float xl0 = gen == JOINT_GENERIC_DISTANCE ? ini.len : compAt(ini.lin, k);
        int ml = (int)j.axisMode[base + k];
        float cl = jointAxisResidual(axisBase(ml), j.axisLo[base + k], j.axisHi[base + k],
                                     xl, xl0, dt, false, kl);
        if (axisZeta(ml) > 0.0f)
            st.dLin = withComp(st.dLin, k, cl - (xl0 - j.axisLo[base + k]));
        int mw = (int)j.axisMode[base + k + 3];
        int ma = axisBase(mw);
        float xa = (k == 0 && ma == JOINT_AXIS_CONE) ? cur.swing : compAt(cur.ang, k);
        // Negated: `ang` is B's offset from A (what a limit is authored against) while the
        // weld's angular C, whose Jacobians these are, is target - actual.
        float ca = -jointAxisResidual(ma, j.axisLo[base + k + 3], j.axisHi[base + k + 3],
                                      xa, compAt(ini.ang, k), dt, true, ka);
        if (axisZeta(mw) > 0.0f)
            st.dAng = withComp(st.dAng, k, jointWrapAngle(ca + (compAt(ini.ang, k) - j.axisLo[base + k + 3])));
        st.cLin = withComp(st.cLin, k, cl);
        st.kLin = withComp(st.kLin, k, kl);
        st.cAng = withComp(st.cAng, k, ca);
        st.kAng = withComp(st.kAng, k, ka);
    }
    return st;
}

// Per-axis force bounds, applied to the primal force and to the multiplier alike. MOTOR
// is a two-sided magnitude bound; LIMITED a one-sided one that releases inside range.
vec3 jointClampForce(JointSoA j, int i, int base, vec3 c, vec3 f)
{
    for (int k = 0; k < 3; k++)
    {
        int m = axisBase((int)j.axisMode[i * 6 + base + k]);
        float fk = compAt(f, k);
        if (m == JOINT_AXIS_MOTOR || m == JOINT_AXIS_VMOTOR)
        {
            float lim = j.axisHi[i * 6 + base + k];
            fk = min(max(fk, -lim), lim);
        }
        else if (m == JOINT_AXIS_SPRING_F)
        {
            float up = base == 0 ? j.stiffnessLin[i] : j.stiffnessAng[i];
            float dn = up;
            if ((int)j.generic[i] == JOINT_GENERIC_DISTANCE)
                dn = j.stiffnessAng[i]; // asymmetric: [-stiffnessAng, +stiffnessLin]
            fk = min(max(fk, -dn), up);
        }
        else if (m == JOINT_AXIS_LIMITED || m == JOINT_AXIS_CONE)
        {
            float ck = compAt(c, k);
            if (ck > 0.0f)
                fk = max(fk, 0.0f);
            else if (ck < 0.0f)
                fk = min(fk, 0.0f);
            else
                fk = 0.0f;
        }
        f = withComp(f, k, fk);
    }
    return f;
}

// Subtract C0 only on the hard axes -- the weld's scalar isinf test, three times over.
vec3 jointStabilize(vec3 c, vec3 c0, vec3 stiffness, float alpha)
{
    for (int k = 0; k < 3; k++)
        if (isinf(compAt(stiffness, k)))
            c = withComp(c, k, compAt(c, k) - compAt(c0, k) * alpha);
    return c;
}

// R * diag(d) * R^T, with the isotropic fast path.
float3x3 rotatedDiagonal(quat q, vec3 d)
{
    if (d.x == d.y && d.y == d.z)
        return diagonalM(d.x, d.x, d.x);

    float x2 = q.x + q.x, y2 = q.y + q.y, z2 = q.z + q.z;
    float xx = q.x * x2, yy = q.y * y2, zz = q.z * z2;
    float xy = q.x * y2, xz = q.x * z2, yz = q.y * z2;
    float wx = q.w * x2, wy = q.w * y2, wz = q.w * z2;

    vec3 c0 = vec3(1.0f - (yy + zz), xy + wz, xz - wy);
    vec3 c1 = vec3(xy - wz, 1.0f - (xx + zz), yz + wx);
    vec3 c2 = vec3(xz + wy, yz - wx, 1.0f - (xx + yy));

    return addM(addM(scaleM(outerM(c0, c0), d.x), scaleM(outerM(c1, c1), d.y)), scaleM(outerM(c2, c2), d.z));
}

// jointInitialize6: the weld's three jobs (freeze C(x-), decay the multipliers, ramp the
// penalties), per axis and in the joint frame.
public void jointInitialize6(JointSoA j, int i, BodySoA b, float alpha, float gamma, float dt,
                             float dualDamping, float dualDampVel)
{
    int ia = j.bodyA[i], ib = j.bodyB[i];

    // A velocity motor needs the step's dt in the residual, and stamp/dual are not given it;
    // restAng is read by the weld path only, so a generic joint's is free to carry it.
    for (int k = 0; k < 6; k++)
    {
        int m = (int)j.axisMode[i * 6 + k];
        if (m == JOINT_AXIS_VMOTOR || axisZeta(m) > 0.0f)
            j.restAng[i] = quat(dt, 0.0f, 0.0f, 1.0f);
    }

    JointAxisState st = jointAxisState(j, i, b);

    j.C0Lin[i] = st.cLin;
    j.C0Ang[i] = scale(st.cAng, j.torqueArm[i]);
    // A velocity target is not an error to regularise away (C - alpha*C0 would cancel it).
    for (int k = 0; k < 6; k++)
        if ((int)j.axisMode[i * 6 + k] == JOINT_AXIS_VMOTOR)
        {
            if (k < 3)
                j.C0Lin[i] = withComp(j.C0Lin[i], k, 0.0f);
            else
                j.C0Ang[i] = withComp(j.C0Ang[i], k - 3, 0.0f);
        }

    j.lambdaLin[i] = scale(j.lambdaLin[i], alpha * gamma);
    j.lambdaAng[i] = scale(j.lambdaAng[i], alpha * gamma);

    quat R = normalizeQ(st.qfA);
    vec3 rAW = ia >= 0 ? rotateQ(b.rot[ia], j.rA[i]) : vec3(0.0f, 0.0f, 0.0f);
    vec3 rBW = rotateQ(b.rot[ib], j.rB[i]);
    float torqueArm = j.torqueArm[i];

    vec3 kAutoLin = vec3(0.0f, 0.0f, 0.0f), kAutoAng = vec3(0.0f, 0.0f, 0.0f);
    for (int k = 0; k < 3; k++)
    {
        vec3 ek = rotateQ(R, unitAxis(k));
        // A distance joint's three rows all act along the anchor-to-anchor direction.
        vec3 ekLin = (int)j.generic[i] == JOINT_GENERIC_DISTANCE ? st.n : ek;
        kAutoLin = withComp(kAutoLin, k, jointEffMassLin(b, ia, ib, rAW, rBW, ekLin, dt));
        kAutoAng = withComp(kAutoAng, k, jointEffMassAng(b, ia, ib, ek, torqueArm, dt));
    }

    vec3 penLin = j.penaltyLin[i];
    vec3 penAng = j.penaltyAng[i];
    for (int k = 0; k < 3; k++)
    {
        float pl = clampf(compAt(penLin, k) * gamma, PENALTY_MIN_J, JOINT_PENALTY_MAX);
        float pa = clampf(compAt(penAng, k) * gamma, PENALTY_MIN_J, JOINT_PENALTY_MAX);
        // Cap at the axis's own stiffness and its effective-mass baseline.
        float maxKLin = min(max(10.0f * compAt(kAutoLin, k), 1000.0f), JOINT_PENALTY_MAX);
        float maxKAng = min(max(10.0f * compAt(kAutoAng, k), 1000.0f), JOINT_PENALTY_MAX);
        penLin = withComp(penLin, k, min(pl, min(compAt(st.kLin, k), maxKLin)));
        penAng = withComp(penAng, k, min(pa, min(compAt(st.kAng, k), maxKAng)));
    }
    j.penaltyLin[i] = penLin;
    j.penaltyAng[i] = penAng;

    if (dualDamping > 0.0f)
    {
        vec3 vA = ia >= 0 ? add(b.velLin[ia], cross3(b.velAng[ia], rAW)) : vec3(0.0f, 0.0f, 0.0f);
        vec3 vB = add(b.velLin[ib], cross3(b.velAng[ib], rBW));
        vec3 wRel = sub(ia >= 0 ? b.velAng[ia] : vec3(0.0f, 0.0f, 0.0f), b.velAng[ib]);

        j.lambdaLin[i] = jointDampLambda(j.lambdaLin[i], penLin, kAutoLin, sub(vA, vB), dualDamping, dualDampVel);
        j.lambdaAng[i] = jointDampLambda(j.lambdaAng[i], penAng, kAutoAng, wRel, dualDamping, dualDampVel);
    }
}

// Spring damper coefficients per axis, as c/dt in the same units as the axis's penalty:
// c = 2 zeta sqrt(k m_eff), and kAuto = m_eff / dt^2, so c/dt = 2 zeta sqrt(k kAuto). The
// damper is the implicit-Euler energy 1/2 (c/dt)(C - C(x^t))^2: Hessian c/dt, force (c/dt) dC.
vec3 jointDamperCoef(JointSoA j, int i, BodySoA b, JointAxisState st, quat R, bool angular)
{
    int ia = j.bodyA[i], ib = j.bodyB[i];
    float dt = j.restAng[i].x;
    vec3 rAW = ia >= 0 ? rotateQ(b.rot[ia], j.rA[i]) : vec3(0.0f, 0.0f, 0.0f);
    vec3 rBW = rotateQ(b.rot[ib], j.rB[i]);
    vec3 cd = vec3(0.0f, 0.0f, 0.0f);
    for (int k = 0; k < 3; k++)
    {
        float z = axisZeta((int)j.axisMode[i * 6 + (angular ? 3 : 0) + k]);
        if (z <= 0.0f)
            continue;
        vec3 ek = rotateQ(R, unitAxis(k));
        float kAuto = angular ? jointEffMassAng(b, ia, ib, ek, j.torqueArm[i], dt)
                              : jointEffMassLin(b, ia, ib, rAW, rBW,
                                                (int)j.generic[i] == JOINT_GENERIC_DISTANCE ? st.n : ek, dt);
        float ks = angular ? compAt(st.kAng, k) : compAt(st.kLin, k);
        cd = withComp(cd, k, 2.0f * z * sqrt(ks * kAuto));
    }
    return cd;
}

// stampJoint6: the weld stamp with a joint-frame penalty tensor and per-axis residuals;
// the force is clamped per axis before it is rotated into world.
public void stampJoint6(JointSoA j, int i, bool isA, BodySoA b, float alpha,
                        inout float3x3 lhsLin, inout float3x3 lhsAng, inout float3x3 lhsCross,
                        inout vec3 rhsLin, inout vec3 rhsAng)
{
    int ia = j.bodyA[i], ib = j.bodyB[i];
    vec3 rA = j.rA[i], rB = j.rB[i];
    float torqueArm = j.torqueArm[i];
    JointAxisState st = jointAxisState(j, i, b);
    quat R = normalizeQ(st.qfA);

    vec3 penLin = j.penaltyLin[i];
    if (lengthSq3(penLin) > 0.0f)
    {
        vec3 cdLin = st.damped ? jointDamperCoef(j, i, b, st, R, false) : vec3(0.0f, 0.0f, 0.0f);
        const vec3 penLinK = add(penLin, cdLin); // Hessian: spring penalty + damper c/dt
        float3x3 K = rotatedDiagonal(R, penLinK);
        vec3 C = jointStabilize(st.cLin, j.C0Lin[i], st.kLin, alpha);

        vec3 lam = j.lambdaLin[i];
        vec3 Ffr = vec3(penLin.x * C.x + lam.x + cdLin.x * st.dLin.x, penLin.y * C.y + lam.y + cdLin.y * st.dLin.y,
                        penLin.z * C.z + lam.z + cdLin.z * st.dLin.z);
        Ffr = jointClampForce(j, i, 0, st.cLin, Ffr);
        vec3 F = rotateQ(R, Ffr);
        float3x3 Kdir = diagonalM(0.0f, 0.0f, 0.0f);
        const bool dist = (int)j.generic[i] == JOINT_GENERIC_DISTANCE;
        if (dist)
        {
            // Three scalar rows (length / limits / motor) on one direction: forces add, penalties
            // add, and the direction's own turning contributes a PSD diagonal f/len (I - n n^T).
            float f = Ffr.x + Ffr.y + Ffr.z;
            F = scale(st.n, f);
            K = scaleM(outerM(st.n, st.n), penLinK.x + penLinK.y + penLinK.z);
            float g = max(f, 0.0f) / max(st.len, 1.0e-6f);
            Kdir = diagonalM(g * (1.0f - st.n.x * st.n.x), g * (1.0f - st.n.y * st.n.y), g * (1.0f - st.n.z * st.n.z));
        }

        float3x3 jAng = isA ? skewM(neg(rotateQ(b.rot[ia], rA))) : skewM(rotateQ(b.rot[ib], rB));
        float3x3 jAngT = transposeM(jAng);
        float3x3 jAngTk = mulMM(jAngT, K);

        lhsLin = addM(lhsLin, K);
        if (dist)
            lhsLin = addM(lhsLin, Kdir);
        lhsAng = addM(lhsAng, mulMM(jAngTk, jAng));
        lhsCross = addM(lhsCross, isA ? jAngTk : negM(jAngTk));

        vec3 r = isA ? rotateQ(b.rot[ia], rA) : neg(rotateQ(b.rot[ib], rB));
        lhsAng = addM(lhsAng, geometricStiffnessBallSocketDiag(r, F));

        rhsLin = add(rhsLin, isA ? F : neg(F));
        rhsAng = add(rhsAng, mulMV(jAngT, F));
    }

    vec3 penAng = j.penaltyAng[i];
    if (lengthSq3(penAng) > 0.0f)
    {
        vec3 cdAng = st.damped ? jointDamperCoef(j, i, b, st, R, true) : vec3(0.0f, 0.0f, 0.0f);
        const vec3 penAngK = add(penAng, cdAng);
        float3x3 K = rotatedDiagonal(R, penAngK);
        vec3 C = jointStabilize(scale(st.cAng, torqueArm), j.C0Ang[i], st.kAng, alpha);

        vec3 lam = j.lambdaAng[i];
        vec3 dA = scale(st.dAng, torqueArm);
        vec3 Ffr = vec3(penAng.x * C.x + lam.x + cdAng.x * dA.x, penAng.y * C.y + lam.y + cdAng.y * dA.y,
                        penAng.z * C.z + lam.z + cdAng.z * dA.z);
        Ffr = jointClampForce(j, i, 3, st.cAng, Ffr);
        vec3 F = rotateQ(R, Ffr);
        if ((int)j.axisMode[i * 6 + 3] == JOINT_AXIS_CONE)
        {
            // The cone row acts about the swing direction d (slot x), not about frame A's x
            // axis: rebuild the basis (d, d-perp, z) so K and F follow it. Slot y is free.
            vec3 d = vec3(st.coneDir.x, st.coneDir.y, 0.0f);
            vec3 e0 = rotateQ(R, d), e1 = rotateQ(R, vec3(-d.y, d.x, 0.0f)), e2 = rotateQ(R, vec3(0.0f, 0.0f, 1.0f));
            K = addM(addM(scaleM(outerM(e0, e0), penAngK.x), scaleM(outerM(e1, e1), penAngK.y)),
                     scaleM(outerM(e2, e2), penAngK.z));
            F = add(add(scale(e0, Ffr.x), scale(e1, Ffr.y)), scale(e2, Ffr.z));
        }

        float arm2 = torqueArm * torqueArm;
        lhsAng = addM(lhsAng, scaleM(K, arm2));
        rhsAng = add(rhsAng, isA ? scale(F, torqueArm) : scale(F, -torqueArm));
    }
}

// jointDual6: per-axis multiplier and penalty update. The motor limit is a plain
// per-axis clamp because the multiplier is stored in the joint frame.
public void jointDual6(JointSoA j, int i, BodySoA b, float alpha, float betaLin, float betaAng)
{
    JointAxisState st = jointAxisState(j, i, b);
    float torqueArm = j.torqueArm[i];

    vec3 penLin = j.penaltyLin[i];
    if (lengthSq3(penLin) > 0.0f)
    {
        vec3 C = jointStabilize(st.cLin, j.C0Lin[i], st.kLin, alpha);
        vec3 lam = j.lambdaLin[i];
        for (int k = 0; k < 3; k++)
        {
            float kk = compAt(st.kLin, k), pk = compAt(penLin, k), ck = compAt(C, k);
            if (isinf(kk))
                lam = withComp(lam, k, compAt(lam, k) + pk * ck);
            penLin = withComp(penLin, k, min(pk + abs(ck) * betaLin, min(kk, JOINT_PENALTY_MAX)));
        }
        j.lambdaLin[i] = jointClampForce(j, i, 0, st.cLin, lam);
        j.penaltyLin[i] = penLin;
    }

    vec3 penAng = j.penaltyAng[i];
    if (lengthSq3(penAng) > 0.0f)
    {
        vec3 C = jointStabilize(scale(st.cAng, torqueArm), j.C0Ang[i], st.kAng, alpha);
        vec3 lam = j.lambdaAng[i];
        for (int k = 0; k < 3; k++)
        {
            float kk = compAt(st.kAng, k), pk = compAt(penAng, k), ck = compAt(C, k);
            if (isinf(kk))
                lam = withComp(lam, k, compAt(lam, k) + pk * ck);
            penAng = withComp(penAng, k, min(pk + abs(ck) * betaAng, min(kk, JOINT_PENALTY_MAX)));
        }
        j.lambdaAng[i] = jointClampForce(j, i, 3, st.cAng, lam);
        j.penaltyAng[i] = penAng;
    }
}

bool jointIsGeneric(JointSoA j, int i) { return j.generic != nullptr && j.generic[i] != 0; }

// The entry points the kernels call. They dispatch as gpu_forces.cuh's jointInitialize,
// stampJoint and jointDual do: a generic joint takes the 6-DOF path, a weld takes the
// weld path, which is untouched.
public void jointInitialize(JointSoA j, int i, BodySoA b, float alpha, float gamma, float dt,
                            float dualDamping, float dualDampVel)
{
    if (jointIsGeneric(j, i))
        jointInitialize6(j, i, b, alpha, gamma, dt, dualDamping, dualDampVel);
    else
        jointInitializeWeld(j, i, b, alpha, gamma, dt, dualDamping, dualDampVel);
}

public void stampJoint(JointSoA j, int i, bool isA, BodySoA b, float alpha,
                       inout float3x3 lhsLin, inout float3x3 lhsAng, inout float3x3 lhsCross,
                       inout vec3 rhsLin, inout vec3 rhsAng)
{
    if (j.broken[i] != 0)
        return;
    if (jointIsGeneric(j, i))
        stampJoint6(j, i, isA, b, alpha, lhsLin, lhsAng, lhsCross, rhsLin, rhsAng);
    else
        stampJointWeld(j, i, isA, b, alpha, lhsLin, lhsAng, lhsCross, rhsLin, rhsAng);
}

public void jointDual(JointSoA j, int i, BodySoA b, float alpha, float betaLin, float betaAng)
{
    if (j.broken[i] != 0)
        return;
    if (jointIsGeneric(j, i))
    {
        jointDual6(j, i, b, alpha, betaLin, betaAng);
        jointFracture(j, i, b);
    }
    else
        jointDualWeld(j, i, b, alpha, betaLin, betaAng);
}

// Transliterated from Spring::updatePrimal. No dual state.
public void stampSpring(SpringSoA s, int i, bool isA, BodySoA b,
                         inout float3x3 lhsLin, inout float3x3 lhsAng, inout float3x3 lhsCross,
                         inout vec3 rhsLin, inout vec3 rhsAng)
{
    stampSpringA(s, i, isA, b, 0.0f, lhsLin, lhsAng, lhsCross, rhsLin, rhsAng);
}

public void stampSpringA(SpringSoA s, int i, bool isA, BodySoA b, float alpha,
                         inout float3x3 lhsLin, inout float3x3 lhsAng, inout float3x3 lhsCross,
                         inout vec3 rhsLin, inout vec3 rhsAng)
{
    int ia = s.bodyA[i], ib = s.bodyB[i];
    vec3 rA = s.rA[i], rB = s.rB[i];

    vec3 pA = jointAnchor(b, ia, rA);
    vec3 pB = transformQ(b.pos[ib], b.rot[ib], rB);
    vec3 d = sub(pA, pB);
    float dLen = length3(d);
    if (dLen <= 1.0e-6f)
        return;

    float stiffness = s.stiffness[i];
    vec3 n = divs(d, dLen);
    float C = dLen - s.rest[i];
    float f = stiffness * C;

    vec3 rWorld;
    vec3 jLin;
    vec3 jAng;
    if (isA)
    {
        rWorld = rotateQ(b.rot[ia], rA);
        jLin = n;
        jAng = cross3(rWorld, n);
    }
    else
    {
        rWorld = rotateQ(b.rot[ib], rB);
        jLin = neg(n);
        jAng = neg(cross3(rWorld, n));
    }

    vec3 F = scale(jLin, f);
    vec3 Tau = scale(jAng, f);
    float3x3 Kll = scaleM(outerM(jLin, jLin), stiffness);
    float3x3 Kla = scaleM(outerM(jAng, jLin), stiffness);
    float3x3 Kaa = scaleM(outerM(jAng, jAng), stiffness);

    lhsLin = addM(lhsLin, Kll);
    lhsAng = addM(lhsAng, Kaa);
    lhsCross = addM(lhsCross, Kla);
    rhsLin = add(rhsLin, F);
    rhsAng = add(rhsAng, Tau);
}
