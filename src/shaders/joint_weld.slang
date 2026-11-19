// Ball-socket weld joints (`j.generic[i] == 0`): the persistent-force primal stamp, dual update and fracture, plus the constraint-graph key helpers shared by every force type.

module joint_weld;

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
public static const float PENALTY_MIN_J = 1.0f;
static const float PENALTY_MAX_J = 10000000000.0f;
public static const float JOINT_PENALTY_MAX = 500000.0f;

// Effective mass along a world-space linear axis: J M^-1 J^T for the two bodies' linear
// and angular response (gpu_forces.cuh:jointEffMassLin). body A may be a world anchor
// (ia < 0), which contributes nothing.
public float jointEffMassLin(BodySoA b, int ia, int ib, vec3 rAW, vec3 rBW, vec3 axis, float dt)
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
public float jointEffMassAng(BodySoA b, int ia, int ib, vec3 axis, float torqueArm, float dt)
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
public vec3 jointDampLambda(vec3 lambda, vec3 penalty, vec3 kAuto, vec3 vRel, float dualDamping, float dualDampVel)
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
