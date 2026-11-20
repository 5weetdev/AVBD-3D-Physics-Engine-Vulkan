#include "vk_world.h"
#include "vk_world_internal.h"
#include "vk_capacity.h"
#include "vk_small_world.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace avbdvk
{



// A capacity overflow made phases 4-8 a no-op on device: grow what was too small so the pass can
// be discarded and recorded again.
void VkWorld::growAfterCapacityOverflow()
{
    StepCtx &S = m_step;
    StepResult &result = S.result;
    auto &capAttempt = S.capAttempt;
    // Capacity overflow: phases 4-8 had no effect on device. Grow, discard the
    // pass, and record phases 2-8 again with phase 4 restarted from the
    // device state this step began with (the seeds were never merged).
    const int pairCap0 = m_pairCap, contactCap0 = m_contactCap, hullCap0 = narrow.hullCapacity();
    const int childCap0 = narrow.childCapacity();
    if (broad.overflowed())
        grow(broad.producedPairs(), m_contactCap);
    else
    {
        narrow.finish(broad.pairCount(), false);
        grow(broad.pairCount(), narrow.requiredContacts());
        if (narrow.hullPairs() > narrow.hullCapacity())
            narrow.growHull(narrow.hullPairs());
        if (narrow.meshRecorded() && narrow.childOverflowed())
        {
            narrow.growChild(narrow.childRaw());
            syncManifoldCap();
        }
    }
    constraints.finishRecorded();
    capAttempt++;
    result.capacityReruns++;
    // At a ceiling nothing grew: another identical pass would overflow again,
    // so the next one is the clamped commit.
    if (m_pairCap == pairCap0 && m_contactCap == contactCap0 && narrow.hullCapacity() == hullCap0 &&
        narrow.childCapacity() == childCap0)
        capAttempt = std::max(capAttempt, kMaxCapAttempts);
    if (capAttempt > 64)
    {
        fprintf(stderr, "[avbd_vk] step: capacity reruns without converging\n");
        abort();
    }
}


// The front overflowed a hard ceiling and the step commits clamped results: say which resource.
void VkWorld::reportFrontTruncation()
{
    StepResult &result = m_step.result;
    if (broad.overflowed())
    {
        result.ok = false;
        result.truncated = true;
        result.failPhase = "broadphase";
        result.failResource = "pairs";
        result.required = broad.producedPairs();
        result.capacity = m_pairCap;
    }
    else if (narrow.overflowed())
    {
        result.ok = false;
        result.truncated = true;
        result.failPhase = "narrowphase";
        result.failResource = "contacts";
        result.required = narrow.requiredContacts();
        result.capacity = m_contactCap;
    }
    else if (narrow.hullRecorded() && narrow.hullPairs() > narrow.hullCapacity())
    {
        result.ok = false;
        result.truncated = true;
        result.failPhase = "narrowphase";
        result.failResource = "hullPairs";
        result.required = narrow.hullPairs();
        result.capacity = narrow.hullCapacity();
    }
    else if (narrow.meshRecorded() && narrow.childOverflowed())
    {
        result.ok = false;
        result.truncated = true;
        result.failPhase = "narrowphase";
        result.failResource = "meshChildren";
        result.required = narrow.childRaw();
        result.capacity = narrow.childCapacity();
    }
}


// Reads back the sleep first half's convergence flags: whether the wake flood and the island flood
// finished within their recorded rounds.
void VkWorld::readPhase4Convergence()
{
    StepCtx &S = m_step;
    const int32_t *readout = wakeReadout.mappedAs<int32_t>();
    auto &edgeCount = S.edgeCount;
    auto &floodDone = S.floodDone;
    auto &floodRec = S.floodRec;
    auto &islandRec = S.islandRec;
    auto &phase4Done = S.phase4Done;
    auto &floodCause = S.floodCause;
    auto &islandCause = S.islandCause;
    m_activeCount = readout[0];

    bool converged = true;
    floodCause = islandCause = false;
    if (!floodDone)
    {
        const int at = convergedAt(readout + 1, floodRec);
        floodDone = at > 0 || floodRec >= kFloodMaxRounds;
        m_lastFloodRounds = at > 0 ? at : floodRec;
        converged = floodDone;
        floodCause = !floodDone;
    }
    if (edgeCount > 0)
    {
        const int at = convergedAt(readout + 1 + kFloodMaxRounds, islandRec);
        m_lastIslandRounds = at > 0 ? at : islandRec;
        islandCause = !(at > 0 || islandRec >= kIslandMaxRounds);
        converged = converged && !islandCause;
    }
    phase4Done = converged;
}


// The GPU-timestamp split of a committed pass.
void VkWorld::recordPassTimings()
{
    StepCtx &S = m_step;
    CommandList &cmd = m_stepCmd;
    auto &iters = S.iters;
    auto &tsStart = S.tsStart;
    auto &tsEnd = S.tsEnd;
    auto &frontMs = S.frontMs;
    auto &front = S.front;
    m_timing.iterMs = m_timing.primalMs = m_timing.dualMs = -1.0f;
    m_timing.tailMs = m_timing.gpuMs = m_timing.frontMs = -1.0f;
    m_timing.gpuMs = (float)(cmd.timestampDeltaNs(tsStart, tsEnd) * 1e-6);
    m_timing.tailMs = (float)(cmd.timestampDeltaNs((uint32_t)(3 + 2 * iters), tsEnd) * 1e-6);
    m_timing.frontMs = frontMs;
    // Phases 4-5: end of narrowphase (or, on a rerun pass, the top of the buffer) to the
    // iteration loop, readouts included.
    m_timing.gpuBuildMs = (float)(cmd.timestampDeltaNs(front ? 2u : tsStart, 3) * 1e-6);
    if (iters > 0 && cmd.timestampDeltaNs(3, (uint32_t)(3 + 2 * iters)) >= 0.0)
    {
        double primal = 0.0, dual = 0.0;
        for (int it = 0; it < iters; it++)
        {
            primal += cmd.timestampDeltaNs((uint32_t)(3 + 2 * it), (uint32_t)(3 + 2 * it + 1));
            dual += cmd.timestampDeltaNs((uint32_t)(3 + 2 * it + 1), (uint32_t)(3 + 2 * it + 2));
        }
        m_timing.iterMs = (float)(cmd.timestampDeltaNs(3, (uint32_t)(3 + 2 * iters)) * 1e-6);
        m_timing.primalMs = (float)(primal * 1e-6);
        m_timing.dualMs = (float)(dual * 1e-6);
    }
}


// Books a flagged pass against its causes and widens the bound that was too small.
void VkWorld::countRerunCauses(bool jpShort, bool colorShort)
{
    StepCtx &S = m_step;
    StepResult &result = S.result;
    auto &jpRounds = S.jpRounds;
    auto &colorBound = S.colorBound;
    auto &floodCause = S.floodCause;
    auto &islandCause = S.islandCause;
    if (floodCause)
        result.floodReruns++;
    if (islandCause)
        result.islandReruns++;
    if (jpShort)
        result.jpReruns++;
    if (colorShort)
        result.colorReruns++;
    floodCause = islandCause = false;
    if (jpShort)
        jpRounds = widenJpRounds(jpRounds, (int)VkConstraints::kMaxJpRounds);
    if (colorShort)
        colorBound = widenColorBound(constraints.colorCount(), (int)AVBD_MAX_COLORS_VK);
}


bool VkWorld::finishPass()
{
    StepCtx &S = m_step;
    CommandList &cmd = m_stepCmd;
    const int32_t *stepFlag = stepFlagReadout.mappedAs<int32_t>();
    auto &tsStart = S.tsStart;
    auto &frontMs = S.frontMs;
    auto &phase4Done = S.phase4Done;
    auto &jpRounds = S.jpRounds;
    auto &colorBound = S.colorBound;
    auto &hostWaitMs = S.hostWaitMs;
    auto &frontPending = S.frontPending;
    auto &front = S.front;
    auto &recordedP1 = S.recordedP1;
    auto &phase4ThisPass = S.phase4ThisPass;
    {
        const Clock::time_point tW = Clock::now();
        cmd.wait();
        hostWaitMs += msSince(tW, Clock::now());
    }

    if (front)
    {
        m_timing.broadMs = (float)(cmd.timestampDeltaNs(0, 1) * 1e-6);
        m_timing.narrowMs = (float)(cmd.timestampDeltaNs(1, 2) * 1e-6);
        if (recordedP1)
            frontMs = (float)(cmd.timestampDeltaNs(tsStart, 0) * 1e-6);
        broad.finish();
        if (stepFlag[1])
        {
            growAfterCapacityOverflow();
            S.pass++;
            return false;
        }
        narrow.finish(broad.pairCount());
        frontPending = false;
        reportFrontTruncation();
    }

    if (phase4ThisPass)
        readPhase4Convergence();
    constraints.finishRecorded();
    m_lastColors = constraints.colorCount();

    if (!stepFlag[0])
    {
        recordPassTimings();
        return true;
    }

    // Flagged: grow whichever bound was too small and go again.
    const bool jpShort = jpBoundShort(constraints.jpConverged(), jpRounds, VkConstraints::kMaxJpRounds);
    const bool colorShort = colorBoundShort(colorBound, constraints.colorCount(), AVBD_MAX_COLORS_VK);
    if (phase4Done && !jpShort && !colorShort)
    {
        fprintf(stderr, "[avbd_vk] step: flag set but phase 4, JP and colours are all within bounds\n");
        abort();
    }
    countRerunCauses(jpShort, colorShort);
    S.pass++;
    return false;
}



uint64_t VkWorld::stepBegin(const SolverParams &params)
{
    if (m_stepInFlight)
    {
        fprintf(stderr, "[avbd_vk] stepBegin() called while a step is already in flight\n");
        abort();
    }
    if (m_deviceError != VK_SUCCESS)
    {
        // Dead world: no Vulkan call. stepEnd() hands back the sticky error.
        m_stepInFlight = true;
        return 0;
    }
    try
    {
        return stepBeginImpl(params);
    }
    catch (const VkError &e)
    {
        failDevice(e);
        m_stepInFlight = true; // stepEnd() consumes it and reports the error
        return 0;
    }
}



uint64_t VkWorld::stepBeginImpl(const SolverParams &params)
{
    StepCtx &S = m_step;
    S = StepCtx{};
    S.submits0 = Device::submitCount();
    S.tStep = Clock::now();
    m_params = params;
    S.p = packSolverParams(params);
    m_stepSerial++;
    m_stepSlot = -1;
    if (m_snapOn && m_bodyCount > 0)
    {
        if (m_snapCap < m_bodyCount)
            snapshotGrow(); // grows only (with headroom): a steady world allocates nothing here
        // This step overwrites the older slot: a DMA still reading it lands first.
        const int s = (int)(m_stepSerial & 1);
        if (m_copySlot == s)
            snapshotWaitCopy();
        m_slotSerial[s] = 0;
        m_stepSlot = s;
        m_stepJSlot = -1;
        if (m_jointCount > 0)
        {
            snapshotJointGrow();
            m_jSlotSerial[s] = 0;
            m_stepJSlot = s;
        }
    }
    ensureReportCapacity();

    FullBodySoAGpu &b = S.b;
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

    FullJointSoAGpu &j = S.j;
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

    S.sp = springView();

    S.sleepOn = params.sleepEnabled && params.sleepFrames > 0 && m_dynCount > 0;

    // Phases 2+3 inputs (see recordAndSubmitPass for the capacity-rerun contract).
    S.npb.pos = pos.address();
    S.npb.rot = rot.address();
    S.npb.shapeBody = shpBody.address();
    S.npb.localPos = shpLocalPos.address();
    S.npb.localRot = shpLocalRot.address();
    S.npb.halfExtents = shpHalf.address();
    S.npb.shapeType = shpType.address();
    S.npb.friction = shpFriction.address();
    S.npb.flags = shpFlags.address();
    S.npb.mass = mass.address();
    S.npb.moment = moment.address();
    S.npb.velLin = velLin.address();
    S.npb.velAng = velAng.address();
    S.npb.props = propsAddress();
    S.npb.category = shpCategory.address();
    S.npb.mask = shpMask.address();
    S.npb.group = shpGroup.address();
    S.hasHulls = m_hullShapes > 0 && geomHeap.size() > 0;
    S.nph.geomHeap = geomHeap.address();
    S.nph.geomOffset = shpGeomOffset.address();
    S.nph.geomHeapBytes = geomHeap.size();
    S.hasMeshes = m_meshShapes > 0 && geomHeap.size() > 0;
    S.nmh.geom = S.nph;
    S.nmh.prim = &primPl;
    S.nmh.radix = &radix;
    S.npp.dt = params.dt;
    S.npp.alpha = params.alpha;
    S.npp.gamma = params.gamma;
    S.npp.autoKStart = params.autoKStart;
    S.npp.dualDamping = params.dualDamping;
    S.npp.dualDampVel = params.dualDampVel;
    S.edgeCount = m_jointSlots + m_springCount;

    // Timestamp layout: see recordAndSubmitPass.
    S.iters = std::max(params.iterations, 0);
    S.tsStart = (uint32_t)(2 * S.iters + 4);
    S.tsEnd = (uint32_t)(2 * S.iters + 5);
    if (S.tsEnd + 1 > m_stepCmd.maxTimestamps())
    {
        m_stepCmd.destroy();
        m_stepCmd.init(dev, S.tsEnd + 1);
    }
    S.frontMs = -1.0f;

    S.npp.contactCap = effectiveContactCap(params.contactCap);

    S.cbr.mass = mass.address();
    S.cbr.awake = b.awake;
    S.phase4Done = !S.sleepOn;

    S.jpRounds = m_dbg.jpBound > 0 ? std::min(m_dbg.jpBound, (int)VkConstraints::kMaxJpRounds) : constraints.jpBound();
    S.colorBound = std::min(std::max(4, m_lastColors + 2), (int)AVBD_MAX_COLORS_VK);
    if (m_dbg.colorBound > 0)
        S.colorBound = std::min(m_dbg.colorBound, (int)AVBD_MAX_COLORS_VK);
    S.tSolve = Clock::now();
    S.tLoop = S.tSolve;

    m_stepInFlight = true;
    recordAndSubmitPass();
    return m_stepCmd.lastSubmitValue();
}



VkWorld::StepResult VkWorld::stepEnd()
{
    if (!m_stepInFlight)
    {
        fprintf(stderr, "[avbd_vk] stepEnd() called without a step in flight\n");
        abort();
    }
    if (m_deviceError != VK_SUCCESS)
    {
        m_stepInFlight = false;
        return m_stepResult = deviceFailResult();
    }
    try
    {
        return stepEndImpl();
    }
    catch (const VkError &e)
    {
        failDevice(e);
        m_stepInFlight = false;
        return m_stepResult = deviceFailResult();
    }
}



void VkWorld::failDevice(const VkError &e)
{
    if (m_deviceError == VK_SUCCESS)
    {
        m_deviceError = e.result() != VK_SUCCESS ? e.result() : VK_ERROR_UNKNOWN;
        m_deviceErrorWhat = e.what();
        fprintf(stderr, "[avbd_vk] world is dead after a device error: %s\n", e.what());
    }
}



VkWorld::StepResult VkWorld::deviceFailResult() const
{
    StepResult r;
    r.ok = false;
    r.failPhase = "device";
    r.vkError = (int)m_deviceError;
    r.deviceLost = m_deviceError == VK_ERROR_DEVICE_LOST;
    return r;
}



VkWorld::StepResult VkWorld::stepEndImpl()
{
    StepCtx &S = m_step;
    while (!finishPass())
        recordAndSubmitPass();

    // Phase 5 is no longer a host-timed submission of its own: buildMs is the host time
    // spent recording it on the final pass.
    m_timing.buildMs = S.recordBuildMs;
    const Clock::time_point tEnd = Clock::now();
    m_timing.solveMs = msSince(S.tSolve, tEnd);
    m_timing.stepMs = msSince(S.tStep, tEnd);
    m_timing.hostSubmitMs = S.hostSubmitMs;
    m_timing.hostWaitMs = S.hostWaitMs;
    m_timing.hostRecordMs = S.hostRecordMs;
    m_timing.hostPreMs = msSince(S.tStep, S.tLoop);
    S.result.submits = (int)(Device::submitCount() - S.submits0);
    m_brokenMirror = m_brokenRecorded; // as recorded by the committed (last) pass
    if (m_reportOn && m_reportCap > 0)
    {
        // required counts every reported manifold; the records hold the first m_reportCap.
        // A truncated step keeps its committed results and grows the buffer for the next one.
        const int32_t req = reportCountReadout.mappedAs<int32_t>()[0];
        m_reportCount = req < m_reportCap ? req : m_reportCap;
        m_reportTruncated = req > m_reportCap;
        if (m_reportTruncated)
            m_reportWant = std::max(2 * req, 2 * m_reportCap);
    }
    else
    {
        m_reportCount = 0;
        m_reportTruncated = false;
    }
    if (m_stepSlot >= 0)
    {
        // The committed (last) pass wrote the slot; publish it.
        m_slotSerial[m_stepSlot] = m_stepSerial;
        m_slotCount[m_stepSlot] = m_bodyCount;
        m_slotTimeline[m_stepSlot] = m_stepCmd.lastSubmitValue();
        if (m_stepJSlot >= 0)
        {
            m_jSlotSerial[m_stepJSlot] = m_stepSerial;
            m_jSlotCount[m_stepJSlot] = m_jointCount;
        }
        m_snapLatest = m_stepSerial;
        m_stepSlot = -1;
        m_stepJSlot = -1;
    }
    m_stepResult = S.result;
    m_stepInFlight = false;
    return S.result;
}



VkWorld::StepResult VkWorld::step(const SolverParams &params)
{
    stepBegin(params);
    return stepEnd();
}

} // namespace avbdvk
