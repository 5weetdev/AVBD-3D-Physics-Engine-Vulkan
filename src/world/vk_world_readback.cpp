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
#include "vk_world.h"
#include "vk_world_internal.h"
#include "vk_capacity.h"
#include "vk_small_world.h"
#include "collision.h" // MAX_MANIFOLD_CONTACTS
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace avbdvk
{



void VkWorld::downloadPositions(std::vector<vec3> &out) const
{
    assertIdle("VkWorld::downloadPositions");
    out.resize(m_bodyCount);
    if (m_bodyCount > 0)
        pos.readback(out.data(), m_bodyCount * sizeof(vec3));
}


void VkWorld::downloadRotations(std::vector<quat> &out) const
{
    assertIdle("VkWorld::downloadRotations");
    out.resize(m_bodyCount);
    if (m_bodyCount > 0)
        rot.readback(out.data(), m_bodyCount * sizeof(quat));
}



void VkWorld::downloadAwake(std::vector<unsigned char> &out) const
{
    assertIdle("VkWorld::downloadAwake");
    out.resize(m_bodyCount);
    if (m_bodyCount > 0)
        awake.readback(out.data(), m_bodyCount * sizeof(unsigned char));
}


FullSpringSoAGpu VkWorld::springView() const
{
    FullSpringSoAGpu sp{};
    sp.bodyA = sBodyA.address();
    sp.bodyB = sBodyB.address();
    sp.rA = sRA.address();
    sp.rB = sRB.address();
    sp.rest = sRest.address();
    sp.stiffness = sStiffness.address();
    sp.count = (uint32_t)m_springCount;
    return sp;
}


FullManifoldSoAGpu VkWorld::manifoldView() const
{
    FullManifoldSoAGpu m{};
    const VkManifoldSet &cur = narrow.manifolds();
    m.key = cur.key;
    m.bodyA = cur.bodyA;
    m.bodyB = cur.bodyB;
    m.basis = cur.basis;
    m.shapeA = cur.shapeA;
    m.shapeB = cur.shapeB;
    m.friction = cur.friction;
    m.contactOffset = cur.contactOffset;
    m.contactCount = cur.contactCount;
    m.count = narrow.activeManifolds();
    m.capacity = narrow.manifoldSlots();
    m.feature = cur.feature;
    m.rA = cur.rA;
    m.rB = cur.rB;
    m.C0 = cur.C0;
    m.penalty = cur.penalty;
    m.lambda = cur.lambda;
    m.stick = cur.stick;
    m.contactTotal = narrow.contactTotal();
    m.contactCapacity = m_contactCap;
    return m;
}


FullBodySoAGpu VkWorld::bodyView() const
{
    FullBodySoAGpu b{};
    b.pos = pos.address();
    b.rot = rot.address();
    b.initialLin = initialLin.address();
    b.initialAng = initialAng.address();
    b.inertialLin = inertialLin.address();
    b.inertialAng = inertialAng.address();
    b.velLin = velLin.address();
    b.velAng = velAng.address();
    b.prevVelLin = prevVelLin.address();
    b.halfExtents = halfExtents.address();
    b.shapeType = shapeType.address();
    b.moment = moment.address();
    b.mass = mass.address();
    b.friction = friction.address();
    b.awake = awake.address();
    b.sleepTimer = sleepTimer.address();
    b.restPos = restPos.address();
    b.restRot = restRot.address();
    b.gyration = gyration.address();
    b.props = propsAddress();
    b.count = (uint32_t)m_bodyCount;
    return b;
}


ShapeSoAGpu VkWorld::shapeView() const
{
    ShapeSoAGpu v{};
    v.body = shpBody.address();
    v.localPos = shpLocalPos.address();
    v.localRot = shpLocalRot.address();
    v.halfExtents = shpHalf.address();
    v.shapeType = shpType.address();
    v.friction = shpFriction.address();
    v.category = shpCategory.address();
    v.mask = shpMask.address();
    v.group = shpGroup.address();
    v.flags = shpFlags.address();
    v.geomOffset = shpGeomOffset.address();
    v.count = (uint32_t)m_shapeCount;
    return v;
}


BroadphaseShapes VkWorld::broadphaseShapes() const
{
    BroadphaseShapes v;
    v.shapeBody = shpBody.address();
    v.localPos = shpLocalPos.address();
    v.localRot = shpLocalRot.address();
    v.halfExtents = shpHalf.address();
    v.flags = shpFlags.address();
    return v;
}


FullJointSoAGpu VkWorld::jointView() const
{
    FullJointSoAGpu j{};
    j.bodyA = jBodyA.address();
    j.bodyB = jBodyB.address();
    j.rA = jRA.address();
    j.rB = jRB.address();
    j.C0Lin = jC0Lin.address();
    j.C0Ang = jC0Ang.address();
    j.penaltyLin = jPenaltyLin.address();
    j.penaltyAng = jPenaltyAng.address();
    j.lambdaLin = jLambdaLin.address();
    j.lambdaAng = jLambdaAng.address();
    j.stiffnessLin = jStiffnessLin.address();
    j.stiffnessAng = jStiffnessAng.address();
    j.fracture = jFracture.address();
    j.fractureAng = jFractureAng.address();
    j.torqueArm = jTorqueArm.address();
    j.restAng = jRestAng.address();
    j.broken = jBroken.address();
    j.frameA = jFrameA.address();
    j.frameB = jFrameB.address();
    j.axisMode = jAxisMode.address();
    j.axisLo = jAxisLo.address();
    j.axisHi = jAxisHi.address();
    j.generic = jGeneric.address();
    j.count = (uint32_t)m_jointSlots;
    return j;
}


BodyProps VkWorld::bodyProps(int body) const
{
    if (body < 0 || body >= (int)m_hostProps.size())
        return BodyProps{};
    return m_hostProps[body];
}



void VkWorld::downloadJointBroken(std::vector<unsigned char> &out) const
{
    assertIdle("VkWorld::downloadJointBroken");
    out.resize(m_jointCount);
    if (m_jointCount <= 0)
        return;
    // Fractures only happen inside a step, so the last step's copy is current.
    if (m_brokenMirror)
        memcpy(out.data(), brokenReadout.mapped(), m_jointCount * sizeof(unsigned char));
    else
        jBroken.readback(out.data(), m_jointCount * sizeof(unsigned char));
}





JointForce VkWorld::jointForce(int gen, const vec3 &lamLin, const vec3 &lamAng, float arm, const quat &frameA,
                               const quat &frameB, const vec3 &rA, const vec3 &rB, bool hasA, const vec3 &posA,
                               const quat &rotA, const vec3 &posB, const quat &rotB, bool cone)
{
    const quat qA = hasA ? rotA : quat{0, 0, 0, 1};
    const quat qB = rotB;
    const quat qfA = normalize(qA * frameA);
    vec3 force, torque;
    if (gen == 0)
    {
        // Weld: multipliers are world-space.
        force = lamLin;
        torque = lamAng * arm;
    }
    else if (gen == 2)
    {
        // Distance: three scalar rows along n = unit(anchorA - anchorB).
        const vec3 pA = hasA ? transform(posA, qA, rA) : rA;
        const vec3 pB = transform(posB, qB, rB);
        const vec3 d = pA - pB;
        const float len = length(d);
        const vec3 nrm = len > 1e-9f ? d / len : vec3{1, 0, 0};
        force = nrm * (lamLin.x + lamLin.y + lamLin.z);
        torque = vec3{0, 0, 0};
    }
    else
    {
        force = rotate(qfA, lamLin);
        vec3 t = rotate(qfA, lamAng);
        if (cone)
        {
            // Slot x acts about the swing direction (see forces.slang), not frame A's x.
            quat qrel = conjugate(qfA) * (qB * frameB);
            if (qrel.w < 0.0f)
                qrel = quat{-qrel.x, -qrel.y, -qrel.z, -qrel.w};
            const float hn = std::sqrt(qrel.z * qrel.z + qrel.w * qrel.w);
            const quat tw = hn > 1e-7f ? quat{0, 0, qrel.z / hn, qrel.w / hn} : quat{0, 0, 0, 1};
            const quat sw = qrel * conjugate(tw);
            vec3 dir{1, 0, 0};
            const float s = std::sqrt(sw.x * sw.x + sw.y * sw.y);
            if (s > 1e-6f)
                dir = vec3{sw.x / s, sw.y / s, 0.0f};
            const vec3 e0 = rotate(qfA, dir), e1 = rotate(qfA, vec3{-dir.y, dir.x, 0.0f});
            const vec3 e2 = rotate(qfA, vec3{0, 0, 1});
            t = e0 * lamAng.x + e1 * lamAng.y + e2 * lamAng.z;
        }
        torque = t * arm;
    }
    JointForce jf;
    jf.force = force;
    jf.torque = torque;
    jf.events = 0;
    return jf;
}



namespace
{
// The live-slot header of the current manifold set, shared by downloadManifolds and
// downloadManifoldDual.
struct ManifoldHeader
{
    std::vector<int> count, offset, bodyA, bodyB;
};
ManifoldHeader readManifoldHeader(const VkNarrowphase &np)
{
    ManifoldHeader h;
    const int slots = np.manifoldSlots();
    const bool B = np.curIsB();
    h.count.resize(slots);
    h.offset.resize(slots);
    h.bodyA.resize(slots);
    h.bodyB.resize(slots);
    np.bufCount(B).readback(h.count.data(), slots * sizeof(int));
    np.bufOffset(B).readback(h.offset.data(), slots * sizeof(int));
    np.bufBodyA(B).readback(h.bodyA.data(), slots * sizeof(int));
    np.bufBodyB(B).readback(h.bodyB.data(), slots * sizeof(int));
    return h;
}
} // namespace






} // namespace avbdvk
