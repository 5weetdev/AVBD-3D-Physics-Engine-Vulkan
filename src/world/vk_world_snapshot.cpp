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


// --- Pose snapshot ------------------------------------------------------------------------

VkDeviceSize VkWorld::snapOffset(int region, int cap)
{
    // Bytes per body of each region, in order: pos, rot, velLin, velAng, awake, prevAwake.
    static const int kBytes[6] = {12, 16, 12, 12, 1, 1};
    VkDeviceSize off = 0;
    for (int r = 0; r < region && r < 6; r++)
        off += (VkDeviceSize)kBytes[r] * (VkDeviceSize)cap;
    return off;
}


void VkWorld::enableSnapshot(bool on)
{
    assertIdle("VkWorld::enableSnapshot");
    m_snapOn = on;
}


void VkWorld::snapshotWaitCopy()
{
    if (m_copySlot < 0)
        return;
    VK_CHECK(vkWaitForFences(dev.device(), 1, &m_copyFence, VK_TRUE, UINT64_MAX));
    m_copySlot = -1;
}


void VkWorld::snapshotGrow()
{
    // Called from stepBeginImpl, with no step in flight. The newest snapshot is moved to the
    // host first (its slot is about to be freed), and the host copy keeps it across the grow.
    if (m_snapLatest != 0 && m_hostSerial != m_snapLatest)
        requestSnapshotCopy();
    snapshotWaitCopy();
    const int cap = (std::max(m_bodyCount, m_snapCap + m_snapCap / 2) + 15) & ~15; // 16-byte regions
    if (m_copyPool == VK_NULL_HANDLE)
    {
        // The copy queue when the device has one; it must not be the step queue's family
        // for the DMA to run beside a step, so the fallbacks only keep a 1-family device working.
        uint32_t family = dev.copyFamily();
        m_copyQueue = dev.copyQueue();
        if (m_copyQueue == VK_NULL_HANDLE)
        {
            family = dev.sideComputeFamily();
            m_copyQueue = dev.sideComputeQueue();
        }
        if (m_copyQueue == VK_NULL_HANDLE)
        {
            family = dev.computeQueueFamily();
            m_copyQueue = dev.computeQueue();
        }
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = family;
        VK_CHECK(vkCreateCommandPool(dev.device(), &pci, nullptr, &m_copyPool));
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = m_copyPool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(dev.device(), &ai, &m_copyCb));
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VK_CHECK(vkCreateFence(dev.device(), &fi, nullptr, &m_copyFence));
    }
    const VkDeviceSize bytes = snapOffset(6, cap);
    for (Buffer &slot : snapSlot)
    {
        slot.init(dev);
        slot.setConcurrent(true);
        slot.resize(bytes);
    }
    m_slotSerial[0] = m_slotSerial[1] = 0;
    // The host copy is regrown with its contents moved region by region (offsets are cap-based).
    std::vector<unsigned char> old;
    if (m_hostCount > 0 && snapHost.mapped())
    {
        old.resize((size_t)snapOffset(6, m_snapCap));
        memcpy(old.data(), snapHost.mapped(), old.size());
    }
    snapHost.destroy();
    snapHost.init(dev, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MappedBuffer::HostAccess::Read);
    if (!old.empty())
        for (int r = 0; r < 6; r++)
            memcpy((unsigned char *)snapHost.mapped() + snapOffset(r, cap), old.data() + snapOffset(r, m_snapCap),
                   (size_t)(snapOffset(r + 1, 1) - snapOffset(r, 1)) * (size_t)m_hostCount);
    m_snapCap = cap;
}


void VkWorld::requestSnapshotCopy()
{
    if (m_snapLatest == 0 || m_hostSerial == m_snapLatest || m_deviceError != VK_SUCCESS)
        return;
    const int s = m_slotSerial[0] == m_snapLatest ? 0 : m_slotSerial[1] == m_snapLatest ? 1 : -1;
    if (s < 0)
        return;
    try
    {
        // One host copy: an older DMA into it lands first (it is ~0.1 ms, never a step).
        snapshotWaitCopy();
        VK_CHECK(vkResetCommandBuffer(m_copyCb, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(m_copyCb, &bi));
        const int n = m_slotCount[s];
        VkBufferCopy regions[6];
        for (int r = 0; r < 6; r++)
        {
            const VkDeviceSize off = snapOffset(r, m_snapCap);
            regions[r] = {off, off, (snapOffset(r + 1, 1) - snapOffset(r, 1)) * (VkDeviceSize)n};
        }
        vkCmdCopyBuffer(m_copyCb, snapSlot[s].handle(), snapHost.handle(), 6, regions);
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        mb.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(m_copyCb, &dep);
        VK_CHECK(vkEndCommandBuffer(m_copyCb));

        // Ordered after the step that wrote the slot by its timeline value (already reached
        // whenever stepEnd() returned, so this never waits on a later step).
        const VkSemaphore timeline = m_stepCmd.timeline();
        const uint64_t value = m_slotTimeline[s];
        VkTimelineSemaphoreSubmitInfo ti{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        ti.waitSemaphoreValueCount = 1;
        ti.pWaitSemaphoreValues = &value;
        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.pNext = &ti;
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &timeline;
        si.pWaitDstStageMask = &waitStage;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &m_copyCb;
        VK_CHECK(vkResetFences(dev.device(), 1, &m_copyFence));
        VK_CHECK(vkQueueSubmit(m_copyQueue, 1, &si, m_copyFence));
        m_copySlot = s;
        m_hostSerial = m_snapLatest;
        m_hostCount = n;
    }
    catch (const VkError &e)
    {
        failDevice(e);
    }
}


VkWorld::SnapshotView VkWorld::snapshotView()
{
    requestSnapshotCopy();
    SnapshotView v;
    try
    {
        snapshotWaitCopy();
    }
    catch (const VkError &e)
    {
        failDevice(e);
        return v;
    }
    if (m_hostSerial == 0 || m_hostCount == 0)
        return v;
    unsigned char *base = (unsigned char *)snapHost.mapped();
    v.pos = (const vec3 *)(base + snapOffset(0, m_snapCap));
    v.rot = (const quat *)(base + snapOffset(1, m_snapCap));
    v.velLin = (const vec3 *)(base + snapOffset(2, m_snapCap));
    v.velAng = (const vec3 *)(base + snapOffset(3, m_snapCap));
    v.awake = base + snapOffset(4, m_snapCap);
    v.prevAwake = base + snapOffset(5, m_snapCap);
    v.count = m_hostCount;
    v.serial = m_hostSerial;
    return v;
}


void VkWorld::snapshotJointGrow()
{
    // Called from stepBeginImpl with no step in flight and m_jointCount > 0. The newest joint
    // data is moved to the host first when it is about to be freed and the game has read it.
    if (m_jointCount <= m_snapJCap)
        return;
    if (m_snapJCap > 0 && m_jHostSerial != 0 && m_jHostSerial != m_snapLatest)
        snapshotJointPull();
    snapshotWaitCopy();
    const int cap = (std::max(m_jointCount, m_snapJCap + m_snapJCap / 2) + 15) & ~15;
    const VkDeviceSize bytes = (VkDeviceSize)cap * 2 * sizeof(vec3);
    for (Buffer &slot : snapJSlot)
    {
        slot.init(dev);
        slot.setConcurrent(true);
        slot.resize(bytes);
    }
    m_jSlotSerial[0] = m_jSlotSerial[1] = 0;
    std::vector<unsigned char> lamLin, lamAng;
    if (m_jHostCount > 0 && snapJHost.mapped())
    {
        const unsigned char *h = (const unsigned char *)snapJHost.mapped();
        lamLin.assign(h, h + (size_t)m_jHostCount * sizeof(vec3));
        const unsigned char *a = h + (size_t)m_snapJCap * sizeof(vec3);
        lamAng.assign(a, a + (size_t)m_jHostCount * sizeof(vec3));
    }
    snapJHost.destroy();
    snapJHost.init(dev, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MappedBuffer::HostAccess::Read);
    if (!lamLin.empty())
    {
        memcpy(snapJHost.mapped(), lamLin.data(), lamLin.size());
        memcpy((unsigned char *)snapJHost.mapped() + (size_t)cap * sizeof(vec3), lamAng.data(), lamAng.size());
    }
    m_snapJCap = cap;
}


// DMA of the newest committed snapshot's joint multipliers to snapJHost, on the copy queue,
// waiting for that copy only. Shares the body DMA's command buffer and fence.
void VkWorld::snapshotJointPull()
{
    if (m_snapLatest == 0 || m_jHostSerial == m_snapLatest || m_deviceError != VK_SUCCESS ||
        m_copyPool == VK_NULL_HANDLE)
        return;
    const int s = m_jSlotSerial[0] == m_snapLatest ? 0 : m_jSlotSerial[1] == m_snapLatest ? 1 : -1;
    if (s < 0)
        return;
    try
    {
        snapshotWaitCopy();
        VK_CHECK(vkResetCommandBuffer(m_copyCb, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(m_copyCb, &bi));
        const int n = m_jSlotCount[s];
        const VkDeviceSize lin = (VkDeviceSize)n * sizeof(vec3);
        const VkDeviceSize angOff = (VkDeviceSize)m_snapJCap * sizeof(vec3);
        VkBufferCopy regions[2] = {{0, 0, lin}, {angOff, angOff, lin}};
        vkCmdCopyBuffer(m_copyCb, snapJSlot[s].handle(), snapJHost.handle(), 2, regions);
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        mb.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(m_copyCb, &dep);
        VK_CHECK(vkEndCommandBuffer(m_copyCb));
        const VkSemaphore timeline = m_stepCmd.timeline();
        const uint64_t value = m_slotTimeline[s];
        VkTimelineSemaphoreSubmitInfo ti{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        ti.waitSemaphoreValueCount = 1;
        ti.pWaitSemaphoreValues = &value;
        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.pNext = &ti;
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &timeline;
        si.pWaitDstStageMask = &waitStage;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &m_copyCb;
        VK_CHECK(vkResetFences(dev.device(), 1, &m_copyFence));
        VK_CHECK(vkQueueSubmit(m_copyQueue, 1, &si, m_copyFence));
        VK_CHECK(vkWaitForFences(dev.device(), 1, &m_copyFence, VK_TRUE, UINT64_MAX));
        m_jHostSerial = m_snapLatest;
        m_jHostCount = n;
    }
    catch (const VkError &e)
    {
        failDevice(e);
    }
}


VkWorld::JointSnapshotView VkWorld::jointSnapshotView()
{
    snapshotJointPull();
    JointSnapshotView v;
    if (m_jHostSerial == 0 || m_jHostCount == 0 || !snapJHost.mapped())
        return v;
    unsigned char *base = (unsigned char *)snapJHost.mapped();
    v.lambdaLin = (const vec3 *)base;
    v.lambdaAng = (const vec3 *)(base + (size_t)m_snapJCap * sizeof(vec3));
    v.count = m_jHostCount;
    v.serial = m_jHostSerial;
    return v;
}


void VkWorld::snapshotDestroy(bool all)
{
    if (m_copySlot >= 0 && m_copyFence)
        vkWaitForFences(dev.device(), 1, &m_copyFence, VK_TRUE, UINT64_MAX); // no throw on teardown
    m_copySlot = -1;
    snapSlot[0].destroy();
    snapSlot[1].destroy();
    snapHost.destroy();
    snapJSlot[0].destroy();
    snapJSlot[1].destroy();
    snapJHost.destroy();
    m_snapJCap = 0;
    m_jSlotSerial[0] = m_jSlotSerial[1] = 0;
    m_jHostSerial = 0;
    m_jHostCount = 0;
    m_stepJSlot = -1;
    m_snapCap = 0;
    m_snapLatest = m_hostSerial = 0;
    m_slotSerial[0] = m_slotSerial[1] = 0;
    m_hostCount = 0;
    m_stepSlot = -1;
    if (all && m_copyPool)
    {
        vkDestroyFence(dev.device(), m_copyFence, nullptr);
        vkDestroyCommandPool(dev.device(), m_copyPool, nullptr);
        m_copyFence = VK_NULL_HANDLE;
        m_copyPool = VK_NULL_HANDLE;
        m_copyCb = VK_NULL_HANDLE;
        m_copyQueue = VK_NULL_HANDLE;
    }
}


RestitutionArgsGpu VkWorld::restitutionArgs(int slots, const ConstraintGraphGpu &g)
{
    const VkManifoldSet cur = narrow.pendingOutput();
    RestitutionArgsGpu a{};
    a.pos = pos.address();
    a.rot = rot.address();
    a.velLin = velLin.address();
    a.velAng = velAng.address();
    a.mass = mass.address();
    a.moment = moment.address();
    a.props = props.address();
    a.bodyA = cur.bodyA;
    a.bodyB = cur.bodyB;
    a.basis = cur.basis;
    a.contactOffset = cur.contactOffset;
    a.contactCount = cur.contactCount;
    a.rA = cur.rA;
    a.rB = cur.rB;
    a.lambda = cur.lambda;
    a.gKey = g.key;
    a.gIsA = g.isA;
    a.gStart = g.start;
    a.gEnd = g.end;
    a.dyn = dynamicIdx.address();
    a.dynCount = (uint32_t)m_dynCount;
    a.out = restSlots.address();
    a.shapeA = cur.shapeA;
    a.shapeB = cur.shapeB;
    a.shpRestitution = shpRestitution.address();
    a.slots = (uint32_t)slots;
    a.threshold = m_restitutionThreshold;
    return a;
}


void VkWorld::enableContactReport(bool on)
{
    assertIdle("VkWorld::enableContactReport");
    m_reportOn = on;
    if (!on)
    {
        m_reportCount = 0;
        m_reportTruncated = false;
    }
}


// Grows only (on enable, and after a step that reported more than the buffer held), so a
// steady world allocates nothing per step. Called from stepBeginImpl, before recording.
void VkWorld::ensureReportCapacity()
{
    if (!m_reportOn || m_reportCap >= m_reportWant)
        return;
    const int cap = m_reportWant;
    const VkBufferUsageFlags dst = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    reportCounter.resize(sizeof(uint32_t));
    reportRecords.resize((VkDeviceSize)cap * sizeof(ContactReportRecordGpu));
    reportCountReadout.destroy();
    reportCountReadout.init(dev, sizeof(uint32_t), dst, MappedBuffer::HostAccess::Read);
    reportRecReadout.destroy();
    reportRecReadout.init(dev, (VkDeviceSize)cap * sizeof(ContactReportRecordGpu), dst, MappedBuffer::HostAccess::Read);
    m_reportCap = cap;
}


VkWorld::ContactReportView VkWorld::contactReportView() const
{
    ContactReportView v;
    if (m_reportOn && m_reportCount > 0)
    {
        v.records = reportRecReadout.mappedAs<ContactReportRecordGpu>();
        v.count = m_reportCount;
    }
    v.truncated = m_reportOn && m_reportTruncated;
    return v;
}

} // namespace avbdvk
