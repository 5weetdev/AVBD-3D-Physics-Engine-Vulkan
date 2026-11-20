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


void VkWorld::releaseDragSlots(int first, int count)
{
    if (count <= 0)
        return;
    DragReleaseArgsGpu args{};
    args.j = jointView();
    args.slot0 = m_jointCount + first;
    args.count = count;

    CommandList cmd;
    HostRingBuffer ring;
    cmd.init(dev);
    ring.init(dev, 1 << 14);
    cmd.begin();
    launchDragRelease(cmd, ring, interactPl, args, count);
    cmd.submit();
    cmd.wait();
    cmd.destroy();
    ring.destroy();
}


// the drag path: the per-body stiffness
// (5000 * mass * grabStrength * weight), the same "fresh grab" rule (the group changed
// size, or nothing was held), and the same release of slots the group no longer uses.
void VkWorld::dragBodies(const int *bodies, const vec3 *targets, const vec3 *locals, int count, const float *weights,
                         float grabStrength)
{
    assertIdle("VkWorld::dragBodies");
    if (!bodies || count <= 0)
        return;
    if (count > kMaxDragJoints)
        count = kMaxDragJoints;

    std::vector<int> hBodies;
    std::vector<vec3> hTargets, hLocals;
    std::vector<float> hStiff;
    for (int k = 0; k < count; k++)
    {
        const int idx = bodies[k];
        if (idx < 0 || idx >= m_bodyCount)
            continue;
        hBodies.push_back(idx);
        hTargets.push_back(targets[k]);
        hLocals.push_back(locals[k]);
        const float m = idx < (int)m_hostMass.size() ? m_hostMass[idx] : 1.0f;
        const float w = weights ? weights[k] : 1.0f;
        hStiff.push_back(5000.0f * (m > 0.0f ? m : 1.0f) * grabStrength * (w > 0.0f ? w : 0.0f));
    }
    const int n = (int)hBodies.size();
    if (n == 0)
        return;

    const bool grab = !m_dragActive || n != m_dragCount;
    if (m_dragActive && n < m_dragCount)
        releaseDragSlots(n, m_dragCount - n);
    m_dragActive = true;
    m_dragCount = n;

    // Queued, not submitted: the next step records the update (recordDragUpdate). Calls
    // between two steps keep the last targets; a fresh grab among them still resets the duals.
    char *stage = (char *)dragStage.mapped();
    memcpy(stage + kDragBodiesOff, hBodies.data(), n * sizeof(int));
    memcpy(stage + kDragTargetsOff, hTargets.data(), n * sizeof(vec3));
    memcpy(stage + kDragLocalsOff, hLocals.data(), n * sizeof(vec3));
    memcpy(stage + kDragStiffOff, hStiff.data(), n * sizeof(float));
    m_dragPendingGrab = (m_dragPending && m_dragPendingGrab) || grab;
    m_dragPending = true;
    m_dragPendingCount = n;
}


void VkWorld::recordDragUpdate(CommandList &cmd, HostRingBuffer &ring)
{
    if (!m_dragPending)
        return;
    const int n = m_dragPendingCount;
    cmd.copy(dragStage, dragBodies_, n * sizeof(int), kDragBodiesOff);
    cmd.copy(dragStage, dragTargets, n * sizeof(vec3), kDragTargetsOff);
    cmd.copy(dragStage, dragLocals, n * sizeof(vec3), kDragLocalsOff);
    cmd.copy(dragStage, dragStiffness, n * sizeof(float), kDragStiffOff);
    cmd.barrier();

    DragUpdateArgsGpu args{};
    args.b = bodyView();
    args.j = jointView();
    args.slot0 = m_jointCount;
    args.count = n;
    args.bodies = dragBodies_.address();
    args.targets = dragTargets.address();
    args.locals = dragLocals.address();
    args.stiffness = dragStiffness.address();
    args.grab = m_dragPendingGrab ? 1 : 0;
    launchDragUpdate(cmd, ring, interactPl, args, n);
    cmd.barrier();
    m_dragPending = false;
    m_dragPendingGrab = false;
}


void VkWorld::releaseDrag()
{
    assertIdle("VkWorld::releaseDrag");
    m_dragPending = false; // a queued update must not re-arm the slots released here
    if (!m_dragActive)
        return;
    releaseDragSlots(0, m_dragCount);
    m_dragActive = false;
    m_dragCount = 0;
}





void VkWorld::wakeAll()
{
    assertIdle("VkWorld::wakeAll");
    if (m_dynCount == 0)
        return;
    WakeAllArgsGpu args{};
    args.b = bodyView();
    args.dyn = dynamicIdx.address();
    args.dynCount = m_dynCount;

    CommandList cmd;
    HostRingBuffer ring;
    cmd.init(dev);
    ring.init(dev, 1 << 14);
    cmd.begin();
    launchWakeAll(cmd, ring, wakePl, args, m_dynCount);
    cmd.submit();
    cmd.wait();
    cmd.destroy();
    ring.destroy();
    m_activeCount = m_dynCount;
}


namespace
{
// Payload bytes per write/read type -- the mutation path's mutStride/readStride.
int mutStride(BodyWrite type)
{
    switch (type)
    {
    case BodyWrite::Transform:
        return (int)sizeof(BodyTransform);
    case BodyWrite::Impulse:
    case BodyWrite::Force:
        return (int)sizeof(BodyPointVector);
    case BodyWrite::LinearVelocity:
    case BodyWrite::AngularVelocity:
    case BodyWrite::Torque:
        return (int)sizeof(vec3);
    case BodyWrite::Awake:
        return 1;
    case BodyWrite::Props:
        return (int)sizeof(BodyProps);
    case BodyWrite::Target:
        return (int)sizeof(BodyTarget);
    }
    return 0;
}

int readStride(BodyRead type)
{
    switch (type)
    {
    case BodyRead::Pose:
        return (int)sizeof(BodyTransform);
    case BodyRead::LinearVelocity:
    case BodyRead::AngularVelocity:
        return (int)sizeof(vec3);
    case BodyRead::Awake:
        return 1;
    }
    return 0;
}
} // namespace


void VkWorld::setBodyData(const int *indices, const void *data, BodyWrite type, int count)
{
    assertIdle("VkWorld::setBodyData");
    if (!indices || !data || count <= 0)
        return;
    const int stride = mutStride(type);
    if (stride <= 0)
        return;

    if (type == BodyWrite::Props)
    {
        // Mirror on the host so the next step knows whether the props path is needed; the
        // KINEMATIC bit is authoring-only and survives the write (the shader does the same).
        const BodyProps *src = (const BodyProps *)data;
        for (int k = 0; k < count; k++)
        {
            const int b = indices[k];
            if (b < 0 || b >= (int)m_hostProps.size())
                continue;
            BodyProps np = src[k];
            np.flags = (np.flags & ~(unsigned)BODY_KINEMATIC) | (m_hostProps[b].flags & BODY_KINEMATIC);
            m_hostProps[b] = np;
            if (!bodyPropsDefault(np))
                m_anyProps = true;
            if (np.restitution > 0.0f)
                m_anyRestitution = true;
            if ((np.flags & BODY_BULLET) != 0)
                m_anyBullet = true;
        }
    }

    // Each batch's data starts 16-byte aligned: an Awake batch is 1 byte per body, and a
    // vec3/quat batch after it would otherwise be read from a misaligned device address.
    m_mutData.resize((m_mutData.size() + 15) & ~(size_t)15);
    MutBatch batch;
    batch.type = (int)type;
    batch.offset = (int)m_mutIdx.size();
    batch.dataOffset = m_mutData.size();
    batch.count = count;
    m_mutIdx.insert(m_mutIdx.end(), indices, indices + count);
    const unsigned char *bytes = (const unsigned char *)data;
    m_mutData.insert(m_mutData.end(), bytes, bytes + (size_t)count * stride);
    m_mutBatches.push_back(batch);

    // Conservative, as in the mutation path: an Awake(false)-only batch seeds nothing, and the
    // flood then merges an empty seed set.
    m_seedsPending = true;
    // The batch may move a static (applied in-step by the mutation pass, before the
    // broadphase): rebuild the static table in that step. Conservative, and cheap.
    broad.markStaticDirty();
}


void VkWorld::getBodyData(const int *indices, void *data, BodyRead type, int count) const
{
    assertIdle("VkWorld::getBodyData");
    if (!indices || !data || count <= 0)
        return;
    const int stride = readStride(type);
    if (stride <= 0)
        return;

    const size_t idxBytes = (size_t)count * sizeof(int), outBytes = (size_t)count * stride;
    if (readIdxBuf.size() < idxBytes)
        readIdxBuf.resize(idxBytes);
    if (readDataBuf.size() < outBytes)
        readDataBuf.resize(outBytes);
    readIdxBuf.upload(indices, idxBytes);

    BodyGatherArgsGpu args{};
    args.b = bodyView();
    args.type = (int)type;
    args.idx = readIdxBuf.address();
    args.out = readDataBuf.address();
    args.count = count;

    VkWorld &self = const_cast<VkWorld &>(*this); // the pipelines are not const
    CommandList cmd;
    HostRingBuffer ring;
    cmd.init(self.dev);
    ring.init(self.dev, 1 << 14);
    cmd.begin();
    launchBodyGather(cmd, ring, self.mutatePl, args, count);
    cmd.submit();
    cmd.wait();
    cmd.destroy();
    ring.destroy();
    readDataBuf.readback(data, outBytes);
}


void VkWorld::setTransform(int body, vec3 p, quat r)
{
    BodyTransform x{p, r};
    setBodyData(&body, &x, BodyWrite::Transform, 1);
}


void VkWorld::setLinearVelocity(int body, vec3 v)
{
    setBodyData(&body, &v, BodyWrite::LinearVelocity, 1);
}


void VkWorld::setAngularVelocity(int body, vec3 w)
{
    setBodyData(&body, &w, BodyWrite::AngularVelocity, 1);
}


void VkWorld::applyImpulse(int body, vec3 impulse, vec3 worldPoint)
{
    BodyPointVector pv{impulse, worldPoint};
    setBodyData(&body, &pv, BodyWrite::Impulse, 1);
}


void VkWorld::applyForce(int body, vec3 force, vec3 worldPoint)
{
    BodyPointVector pv{force, worldPoint};
    setBodyData(&body, &pv, BodyWrite::Force, 1);
}


void VkWorld::applyTorque(int body, vec3 torque)
{
    setBodyData(&body, &torque, BodyWrite::Torque, 1);
}


void VkWorld::setAwake(int body, bool on)
{
    unsigned char v = on ? 1 : 0;
    setBodyData(&body, &v, BodyWrite::Awake, 1);
}


void VkWorld::setTargetTransform(int body, vec3 p, quat r, float dt)
{
    BodyTarget t{p, r, dt};
    setBodyData(&body, &t, BodyWrite::Target, 1);
}


void VkWorld::wakeRegion(vec3 aabbMin, vec3 aabbMax)
{
    assertIdle("VkWorld::wakeRegion");
    if (m_dynCount == 0)
        return;
    // Seeded straight into the device buffer rather than queued: a wake is idempotent
    // and the flood that consumes it runs after every batch has landed (the mutation path).
    WakeSeedRegionArgsGpu args{};
    args.b = bodyView();
    args.dyn = dynamicIdx.address();
    args.dynCount = m_dynCount;
    args.loX = aabbMin.x;
    args.loY = aabbMin.y;
    args.loZ = aabbMin.z;
    args.hiX = aabbMax.x;
    args.hiY = aabbMax.y;
    args.hiZ = aabbMax.z;
    args.seed = wakeSeed.address();
    args.changed = floodChanged.address();
    floodChanged.zero(sizeof(int));

    CommandList cmd;
    HostRingBuffer ring;
    cmd.init(dev);
    ring.init(dev, 1 << 14);
    cmd.begin();
    launchWakeSeedRegion(cmd, ring, mutatePl, args, m_dynCount);
    cmd.submit();
    cmd.wait();
    cmd.destroy();
    ring.destroy();
    m_seedsPending = true;
}


void VkWorld::setJointBroken(int joint, bool broken)
{
    assertIdle("VkWorld::setJointBroken");
    if (joint < 0 || joint >= m_jointCount)
        return;
    const unsigned char v = broken ? 1 : 0;
    jBroken.upload(&v, 1, (VkDeviceSize)joint);
    m_brokenMirror = false;
}


void VkWorld::setJointRows(const JointWriteRecGpu *recs, int count)
{
    assertIdle("VkWorld::setJointRows");
    for (int i = 0; i < count; i++)
        if (recs[i].row < (uint32_t)m_jointCount)
            m_jwRecs.push_back(recs[i]);
}


void VkWorld::setJointThresholds(int joint, float force, float torque)
{
    assertIdle("VkWorld::setJointThresholds");
    if (joint < 0 || joint >= m_jointCount)
        return;
    if ((int)m_jointForceLimit.size() < m_jointCount)
    {
        m_jointForceLimit.resize(m_jointCount, 0.0f);
        m_jointTorqueLimit.resize(m_jointCount, 0.0f);
    }
    m_jointForceLimit[joint] = force;
    m_jointTorqueLimit[joint] = torque;
}

} // namespace avbdvk
