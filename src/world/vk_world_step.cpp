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


void VkWorld::beginStepRecording()
{
    m_stepRing.reset();
    m_stepCmd.begin();
}


void VkWorld::assertIdle(const char *who) const
{
    if (m_stepInFlight)
    {
        fprintf(stderr, "[avbd_vk] %s called between stepBegin() and stepEnd()\n", who);
        abort();
    }
}


namespace
{
constexpr int kMaxCapAttempts = 5;  // the pass after this many capacity reruns commits clamped
constexpr int kCapSlotsBase = 2048; // phase 4 <= ~670 dispatches, phase 5 ~100 + JP rounds
// First zero slot + 1 (the convergence round), or 0 if every slot changed.
int convergedAt(const int32_t *flags, int n)
{
    for (int r = 0; r < n; r++)
        if (flags[r] == 0)
            return r + 1;
    return 0;
}
} // namespace


void VkWorld::recordPhase1()
{
    StepCtx &S = m_step;
    const SolverParams &params = m_params;
    CommandList &cmd = m_stepCmd;
    HostRingBuffer &ring = m_stepRing;
    auto &b = S.b;
    auto &j = S.j;
    auto &p = S.p;
    auto &sleepOn = S.sleepOn;
    auto &seedsPendingAtStart = S.seedsPendingAtStart;
    // The snapshot's prevAwake: awake as the step began, before queued wakes apply.
    if (m_stepSlot >= 0)
    {
        cmd.copy(awake, snapSlot[m_stepSlot], (VkDeviceSize)m_bodyCount, 0, snapOffset(5, m_snapCap));
        cmd.barrier();
    }
    // Geometry added since the last step (addGeometry): land the staged bytes in the heap.
    // The stage was sized in addGeometry, so nothing allocates here. Idempotent, so a rerun
    // pass that records this again would only repeat a copy (the list is cleared first time).
    if (!m_geomCopies.empty())
    {
        memcpy(geomStage.mapped(), m_geomPending.data(), m_geomPending.size());
        for (const GeomCopy &c : m_geomCopies)
            cmd.copy(geomStage, geomHeap, c.bytes, c.stageOffset, c.heapOffset);
        m_geomCopies.clear();
        m_geomPending.clear();
    }
    // Apply everything queued since the last step, before anything reads a pose:
    // one upload for the frame's writes, then one launch per batch in submission
    // order (VkWorld::step()). A frame that mutated nothing issues no launches here.
    if (!m_mutBatches.empty())
    {
        const size_t idxBytes = m_mutIdx.size() * sizeof(int);
        if (mutIdxBuf.size() < idxBytes)
            mutIdxBuf.resize(idxBytes);
        if (mutDataBuf.size() < m_mutData.size())
            mutDataBuf.resize(m_mutData.size());
        // Host-filled staging, landed by recorded copies (no submission of their own).
        // Every submission is waited, so regrowing here cannot free one in flight.
        auto growStage = [&](MappedBuffer &stage, VkDeviceSize bytes)
        {
            if (stage.size() >= bytes)
                return;
            const VkDeviceSize cap = std::max<VkDeviceSize>(bytes, stage.size() * 2);
            stage.destroy();
            stage.init(dev, cap, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MappedBuffer::HostAccess::Write);
        };
        growStage(mutIdxStage, idxBytes);
        growStage(mutDataStage, m_mutData.size());
        if (idxBytes > 0)
            memcpy(mutIdxStage.mapped(), m_mutIdx.data(), idxBytes);
        if (!m_mutData.empty())
            memcpy(mutDataStage.mapped(), m_mutData.data(), m_mutData.size());
        cmd.copy(mutIdxStage, mutIdxBuf, idxBytes);
        cmd.copy(mutDataStage, mutDataBuf, m_mutData.size());
        for (const MutBatch &batch : m_mutBatches)
        {
            MutateApplyArgsGpu ma{};
            ma.b = b;
            ma.type = batch.type;
            ma.idx = mutIdxBuf.address() + (VkDeviceAddress)batch.offset * sizeof(int);
            ma.data = mutDataBuf.address() + (VkDeviceAddress)batch.dataOffset;
            ma.count = batch.count;
            ma.dt = params.dt;
            ma.seed = wakeSeed.address();
            launchMutateApply(cmd, ring, mutatePl, ma, batch.count);
            cmd.barrier();
        }
        m_mutBatches.clear();
        m_mutIdx.clear();
        m_mutData.clear();
    }
    // Joint row rewrites queued since the last step: one upload, one launch. Recorded after the
    // mutate batches and before joint init, which rebuilds each row's C0 from the new fields.
    if (!m_jwRecs.empty())
    {
        const size_t bytes = m_jwRecs.size() * sizeof(JointWriteRecGpu);
        if (jwBuf.size() < bytes)
            jwBuf.resize(bytes);
        if (jwStage.size() < bytes)
        {
            jwStage.destroy();
            jwStage.init(dev, std::max<VkDeviceSize>(bytes, jwStage.size() * 2), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         MappedBuffer::HostAccess::Write);
        }
        memcpy(jwStage.mapped(), m_jwRecs.data(), bytes);
        cmd.copy(jwStage, jwBuf, bytes);
        JointWriteArgsGpu ja{};
        ja.j = jointView();
        ja.recs = jwBuf.address();
        ja.count = (int32_t)m_jwRecs.size();
        launchJointWrite(cmd, ring, mutatePl, ja, ja.count);
        cmd.barrier();
        m_jwRecs.clear();
    }
    recordDragUpdate(cmd, ring); // ahead of joint init, which derives C0 for the drag slots
    // With deactivation off nothing consumes wake seeds -- every body is already
    // awake -- but they must not fire spuriously if it is switched back on. Zeroed
    // after the mutate batches (which seed) and before anything reads them.
    if (!sleepOn && m_seedsPending)
    {
        cmd.fill(wakeSeed, 0u, (VkDeviceSize)m_bodyCount * sizeof(unsigned char));
        m_seedsPending = false;
    }
    {
        // Always at least kMaxDragJoints slots (the drag slot), matching the reference's
        // jointsTotal-sized joint_init dispatch -- k_joint_init/jointInitialize()
        // itself is broken-agnostic, so this also (harmlessly) re-derives C0/kAuto
        // for a broken drag slot every step, exactly as VkWorld::step() does.
        JointInitArgsGpu ja{};
        ja.b = b;
        ja.j = j;
        ja.p = p;
        launchJointInit(cmd, ring, solverPl, ja, m_jointSlots);
        cmd.barrier();
    }
    IntegrateBodySoAGpu ib{};
    ib.pos = pos.address();
    ib.rot = rot.address();
    ib.initialLin = initialLin.address();
    ib.initialAng = initialAng.address();
    ib.inertialLin = inertialLin.address();
    ib.inertialAng = inertialAng.address();
    ib.velLin = velLin.address();
    ib.velAng = velAng.address();
    ib.prevVelLin = prevVelLin.address();
    ib.mass = mass.address();
    ib.awake = b.awake;
    ib.count = (uint32_t)m_bodyCount;
    IntegrateArgsGpu ia{};
    ia.b = ib;
    ia.p.dt = params.dt;
    ia.p.gravityX = params.gravity.x;
    ia.p.gravityY = params.gravity.y;
    ia.p.gravityZ = params.gravity.z;
    if (m_anyProps)
    {
        IntegratePropsArgsGpu pa{};
        pa.b = ib;
        pa.p = ia.p;
        pa.props = props.address();
        launchIntegrateProps(cmd, ring, integratePl, pa, m_bodyCount);
    }
    else
        launchIntegrate(cmd, ring, integratePl, ia, m_bodyCount);
    // With deactivation off every dynamic body is solved: the active list is the
    // dynamic list. Nothing reads activeIdx before the constraint build, so the copy
    // rides along here instead of costing a readback and an upload of its own.
    if (!sleepOn)
    {
        if (m_dynCount > 0)
            cmd.copy(dynamicIdx, activeIdx, (VkDeviceSize)m_dynCount * sizeof(int));
        // Phase 4 is skipped, so nothing else writes the device-side live count the
        // constraint build and the sleep commit read.
        cmd.fill(activeCounter, (uint32_t)m_dynCount, sizeof(int32_t));
    }
    // What phase 4 will find pending, for a capacity rerun to restart from.
    seedsPendingAtStart = m_seedsPending;
}




void VkWorld::setupFront()
{
    StepCtx &S = m_step;
    const SolverParams &params = m_params;
    auto &b = S.b;
    auto &j = S.j;
    auto &sp = S.sp;
    auto &sleepOn = S.sleepOn;
    auto &seedsPendingAtStart = S.seedsPendingAtStart;
    auto &edgeCount = S.edgeCount;
    auto &wd = S.wd;
    auto &fc = S.fc;
    auto &fj = S.fj;
    auto &fi = S.fi;
    auto &merge = S.merge;
    auto &mergeIsland = S.mergeIsland;
    auto &isArgs = S.isArgs;
    auto &slots = S.slots;
    auto &m = S.m;
    auto &floodDone = S.floodDone;
    auto &floodRec = S.floodRec;
    auto &islandRec = S.islandRec;
    auto &floodBound = S.floodBound;
    auto &islandBound = S.islandBound;
    auto &cbr = S.cbr;
    auto &cmr = S.cmr;
    auto &cjr = S.cjr;
    auto &csr = S.csr;
    auto &phase4Done = S.phase4Done;
    auto &p4Fresh = S.p4Fresh;
    const VkManifoldSet cur = narrow.pendingOutput();
    slots = narrow.pendingSlots();
    m = FullManifoldSoAGpu{};
    m.key = cur.key;
    m.bodyA = cur.bodyA;
    m.bodyB = cur.bodyB;
    m.basis = cur.basis;
    m.shapeA = cur.shapeA;
    m.shapeB = cur.shapeB;
    m.friction = cur.friction;
    m.contactOffset = cur.contactOffset;
    m.contactCount = cur.contactCount;
    // count/contactTotal are host bookkeeping no shader reads (every consumer walks
    // `slots`, skipping contactCount <= 0): the record-time capacities stand in.
    m.count = slots;
    m.capacity = slots;
    m.feature = cur.feature;
    m.rA = cur.rA;
    m.rB = cur.rB;
    m.C0 = cur.C0;
    m.penalty = cur.penalty;
    m.lambda = cur.lambda;
    m.stick = cur.stick;
    m.contactTotal = m_contactCap;
    m.contactCapacity = m_contactCap;
    wd = WakeDetectCall{};
    fc = WakeFloodContactsArgsGpu{};
    fj = WakeFloodJointsArgsGpu{};
    fi = WakeFloodJointsArgsGpu{};
    merge = WakeSeedMergeArgsGpu{};
    mergeIsland = WakeSeedMergeArgsGpu{};
    isArgs = WakeIslandSeedArgsGpu{};
    floodDone = true;
    floodRec = islandRec = 0;
    floodBound = islandBound = 0;
    phase4Done = !sleepOn;
    p4Fresh = true;
    if (sleepOn)
    {
        wd.clear.dyn = dynamicIdx.address();
        wd.clear.dynCount = m_dynCount;
        wd.clear.count = touchCount.address();
        wd.clear.hash = touchHash.address();
        wd.accum.b = b;
        wd.accum.m = m;
        wd.accum.slots = slots;
        wd.accum.count = touchCount.address();
        wd.accum.hash = touchHash.address();
        wd.accumJ.b = b;
        wd.accumJ.j = j;
        wd.accumJ.sp = sp;
        wd.accumJ.count = touchCount.address();
        wd.accumJ.hash = touchHash.address();
        wd.request.b = b;
        wd.request.m = m;
        wd.request.slots = slots;
        wd.request.wakeLinVel = params.sleepLinVel;
        wd.request.wakeAngVel = params.sleepAngVel;
        wd.request.request = wakeRequest.address();
        wd.requestJ.b = b;
        wd.requestJ.j = j;
        wd.requestJ.sp = sp;
        wd.requestJ.request = wakeRequest.address();
        wd.lostSupport.b = b;
        wd.lostSupport.dyn = dynamicIdx.address();
        wd.lostSupport.dynCount = m_dynCount;
        wd.lostSupport.count = touchCount.address();
        wd.lostSupport.hash = touchHash.address();
        wd.lostSupport.snapCount = snapCount.address();
        wd.lostSupport.snapHash = snapHash.address();
        wd.lostSupport.request = wakeRequest.address();
        wd.dynCount = m_dynCount;
        wd.slots = slots;
        wd.edgeCount = edgeCount;

        // Phase 4 is recorded blind to a bound (the last step's convergence round + 2, at
        // least 4) instead of looping on a readback. Each round flags into its own slot,
        // and each merge is gated on the last recorded round having changed nothing, so an
        // unconverged flood keeps its seeds rather than merging a partial wake.
        //
        // Why the result is the old loops' exactly: the floods are monotone closures, so
        // rounds past the fixed point are no-ops. If a bound proves too small, the next
        // pass continues the explicit flood from device state and redoes the island
        // flood. Its seeds (awake || request) then lie between the correct seed set and
        // that set's closure -- a first-pass partial wake only ever adds bodies of the
        // final closure, and WakeApply is idempotent -- so it closes to the same set.
        // At a cap the merge is ungated, as the old loops merged whatever they had.
        fc.b = b;
        fc.m = m;
        fc.slots = slots;
        fc.seed = wakeSeed.address();
        fj.b = b;
        fj.j = j;
        fj.sp = sp;
        fj.seed = wakeSeed.address();
        fi = fj;
        fi.seed = islandSeed.address();
        merge.dyn = dynamicIdx.address();
        merge.dynCount = m_dynCount;
        merge.seed = wakeSeed.address();
        merge.request = wakeRequest.address();
        mergeIsland = merge;
        mergeIsland.seed = islandSeed.address();
        isArgs.b = b;
        isArgs.dyn = dynamicIdx.address();
        isArgs.dynCount = m_dynCount;
        isArgs.request = wakeRequest.address();
        isArgs.seed = islandSeed.address();
        isArgs.any = wakeAny.address();

        floodDone = !seedsPendingAtStart;
        m_seedsPending = false;
        floodBound = std::min(std::max(4, m_lastFloodRounds + 2), (int)kFloodMaxRounds);
        islandBound = std::min(std::max(4, m_lastIslandRounds + 2), (int)kIslandMaxRounds);
        if (m_dbg.floodBound > 0)
            floodBound = std::min(m_dbg.floodBound, (int)kFloodMaxRounds);
        if (m_dbg.islandBound > 0)
            islandBound = std::min(m_dbg.islandBound, (int)kIslandMaxRounds);
    }
    else
    {
        m_activeCount = m_dynCount;
        // (Wake seeds were already zeroed in phase 1 if pending.)
    }

    cmr.bodyA = m.bodyA;
    cmr.bodyB = m.bodyB;
    cmr.contactCount = m.contactCount;
    cmr.slots = slots;
    cjr.bodyA = j.bodyA;
    cjr.bodyB = j.bodyB;
    cjr.broken = j.broken;
    cjr.count = m_jointSlots;
    csr.bodyA = sp.bodyA;
    csr.bodyB = sp.bodyB;
    csr.count = m_springCount;
}


void VkWorld::recordFront()
{
    StepCtx &S = m_step;
    const SolverParams &params = m_params;
    CommandList &cmd = m_stepCmd;
    HostRingBuffer &ring = m_stepRing;
    auto &npb = S.npb;
    auto &npp = S.npp;
    auto &tsStart = S.tsStart;
    auto &slots = S.slots;
    auto &jpRounds = S.jpRounds;
    auto &colorBound = S.colorBound;
    auto &tSolve = S.tSolve;
    auto &frontPending = S.frontPending;
    auto &phase1Done = S.phase1Done;
    auto &capAttempt = S.capAttempt;
    auto &pass = S.pass;
    auto &tPass = S.tPass;
    auto &coarseIters = S.coarseIters;
    auto &gateSlots = S.gateSlots;
    auto &front = S.front;
    auto &recordedP1 = S.recordedP1;
    tPass = Clock::now();
    if (pass > kMaxRerunPasses)
    {
        fprintf(stderr, "[avbd_vk] step: %d reruns without converging\n", pass);
        abort();
    }
    tSolve = Clock::now();
    // Everything that can reallocate is grown before the first recorded command.
    // +25: the tail's fixed dispatches.
    coarseIters = std::min(std::max(params.coarseIterations, 0), params.iterations);
    gateSlots =
        params.iterations * (colorBound + 2) + 25 + 24 * coarseIters + 2; // coarse: reset, assign, 8 parities x 3
    // Reserved for the worst-case colour / JP-round bounds, not this pass's: a bound
    // that grows between steps must not allocate.
    m_gate.reserve(params.iterations * (AVBD_MAX_COLORS_VK + 2) + 24 + 128 + 24 * params.iterations + 2);
    m_gateCap.reserve(kCapSlotsBase + (int)VkConstraints::kMaxJpRounds);
    constraints.prepare(radix, m_dynCount);
    beginStepRecording();
    cmd.writeTimestamp(tsStart);
    cmd.fill(stepState, 0u);

    front = frontPending;
    S.front = front;
    recordedP1 = false;
    if (front)
    {
        if (!phase1Done)
        {
            recordPhase1();
            phase1Done = true;
            recordedP1 = true;
        }
        // The last attempt commits whatever it produced, clamped: no flag then.
        const bool last = capAttempt >= kMaxCapAttempts;
        cmd.writeTimestamp(0);
        BroadphaseOversize bo;
        bo.gridIdx = &gridIdx;
        bo.gridCount = m_gridCount;
        bo.oversizeIdx = &oversizeIdx;
        bo.oversizeCount = m_oversizeCount;
        broad.record(cmd, ring, bphPl, primPl, radix, pos.address(), rot.address(), broadphaseShapes(), dynShapeIdx,
                     m_dynShapeCount, staticShapeIdx, m_staticShapeCount, m_gridCell, &ignorePairs, m_ignoreCount, &bo);
        cmd.barrier();
        cmd.writeTimestamp(1);
        if (S.hasMeshes && narrow.childCapacity() == 0)
        {
            // First mesh step: the child slots exist before the constraint graph is built from them.
            narrow.growChild(64);
            syncManifoldCap();
        }
        narrow.record(cmd, ring, npPl, npb, broad.pairs(), broad.capacity(), &broad.counter(), npp,
                      S.hasHulls ? &S.nph : nullptr, S.hasMeshes ? &S.nmh : nullptr);
        cmd.writeTimestamp(2);
        broad.recordReadout(cmd);
        narrow.recordReadout(cmd);
        cmd.barrier();
        setupFront();
        if (!last)
        {
            launchStepFlagIfGreater(cmd, ring, solverPl, broad.counter().address(), stepState.address(),
                                    broad.capacity());
            cmd.barrier();
            launchStepFlagIfGreater(cmd, ring, solverPl, narrow.counterAddress(), stepState.address(),
                                    narrow.contactCapacity());
            if (narrow.hullRecorded())
            {
                cmd.barrier();
                // More hull pairs than scratch slots: grow the scratch and rerun, never a silent min.
                launchStepFlagIfGreater(cmd, ring, solverPl, narrow.hullCounterAddress(), stepState.address(),
                                        narrow.hullCapacity());
            }
            if (narrow.meshRecorded())
            {
                cmd.barrier();
                // More child keys than child slots: grow the child capacity and rerun, never a silent min.
                launchStepFlagIfGreater(cmd, ring, solverPl, narrow.childCounterAddress(), stepState.address(),
                                        narrow.childCapacity());
            }
        }
        cmd.barrier();
        // Phases 4-5 are skipped on device if either counter outgrew its capacity.
        m_gateCap.begin(cmd, ring, solverPl.gateDispatchArgs, stepState.address() + sizeof(int32_t),
                        kCapSlotsBase + jpRounds);
        cmd.setGate(&m_gateCap);
    }

}

void VkWorld::recordSleepFirstHalf()
{
    StepCtx &S = m_step;
    CommandList &cmd = m_stepCmd;
    HostRingBuffer &ring = m_stepRing;
    auto &b = S.b;
    auto &edgeCount = S.edgeCount;
    auto &wd = S.wd;
    auto &slots = S.slots;
    auto &floodDone = S.floodDone;
    auto &floodRec = S.floodRec;
    auto &islandRec = S.islandRec;
    auto &floodBound = S.floodBound;
    auto &islandBound = S.islandBound;
    auto &phase4Done = S.phase4Done;
    auto &p4Fresh = S.p4Fresh;
    auto &fc = S.fc;
    auto &fj = S.fj;
    auto &fi = S.fi;
    auto &merge = S.merge;
    auto &mergeIsland = S.mergeIsland;
    auto &isArgs = S.isArgs;
    auto &pass = S.pass;
    auto &phase4ThisPass = S.phase4ThisPass;
    phase4ThisPass = !phase4Done;
    S.phase4ThisPass = phase4ThisPass;
    if (phase4ThisPass)
    {
        const bool p4First = p4Fresh;
        p4Fresh = false;
        if (p4First)
        {
            cmd.fill(wakeRequest, 0u, m_bodyCount * sizeof(unsigned char));
            launchWakeDetect(cmd, ring, wakePl, wd);
            cmd.barrier();
        }
        const bool floodThisPass = !floodDone;
        if (floodThisPass)
        {
            if (p4First)
                cmd.fill(floodChanged, 0u);
            const int more = p4First ? floodBound : std::min(floodRec, kFloodMaxRounds - floodRec);
            recordWakeFloodRounds(cmd, ring, mutatePl, fc, fj, slots, edgeCount, floodChanged.address(), floodRec,
                                  more);
            floodRec += more;
            merge.gate = floodRec < kFloodMaxRounds
                             ? floodChanged.address() + (VkDeviceAddress)(floodRec - 1) * sizeof(int32_t)
                             : 0;
            launchWakeSeedMerge(cmd, ring, mutatePl, merge, m_dynCount);
            cmd.barrier();
        }

        // The island flood restarts whenever the requests may have grown under it
        // (the first pass, or a pass whose explicit merge may have fired); otherwise
        // it continues from its seeds on device, since its merge was held back.
        const bool islandFresh = p4First || floodThisPass;
        int islandFirst = 0, islandMore = 0;
        if (edgeCount > 0)
        {
            if (islandFresh)
            {
                cmd.fill(islandSeed, 0u, m_bodyCount * sizeof(unsigned char));
                launchWakeIslandSeed(cmd, ring, wakePl, isArgs, m_dynCount);
                cmd.barrier();
                cmd.fill(wakeChanged, 0u);
                islandMore =
                    p4First ? islandBound : std::max(islandBound, std::min(2 * islandRec, (int)kIslandMaxRounds));
            }
            else
            {
                islandFirst = islandRec;
                islandMore = std::min(islandRec, kIslandMaxRounds - islandRec);
            }
            recordWakeFloodRounds(cmd, ring, mutatePl, fc, fi, 0, edgeCount, wakeChanged.address(), islandFirst,
                                  islandMore);
            islandRec = islandFirst + islandMore;
            mergeIsland.gate = islandRec < kIslandMaxRounds
                                   ? wakeChanged.address() + (VkDeviceAddress)(islandRec - 1) * sizeof(int32_t)
                                   : 0;
            launchWakeSeedMerge(cmd, ring, mutatePl, mergeIsland, m_dynCount);
            cmd.barrier();
        }

        WakeApplyArgsGpu wa{};
        wa.b = b;
        wa.dyn = dynamicIdx.address();
        wa.dynCount = m_dynCount;
        wa.request = wakeRequest.address();
        launchWakeApply(cmd, ring, wakePl, wa, m_dynCount);
        cmd.barrier();

        ActiveFlagsArgsGpu af{};
        af.b = b;
        af.dyn = dynamicIdx.address();
        af.dynCount = m_dynCount;
        af.flags = activeFlags.address();
        launchActiveFlags(cmd, ring, sleepPl, af, m_dynCount);
        cmd.barrier();
        selectFlagged(cmd, ring, primPl, m_select, dynamicIdx, activeFlags, activeIdx, activeCounter,
                      (uint32_t)m_dynCount);

        // A bound too small leaves the last recorded round still changing something;
        // flag it so this pass's phases 6-8 are gated off and the pass reruns.
        if (floodThisPass && floodRec < kFloodMaxRounds)
            launchStepFlagIfSet(cmd, ring, solverPl,
                                floodChanged.address() + (VkDeviceAddress)(floodRec - 1) * sizeof(int32_t),
                                stepState.address());
        if (edgeCount > 0 && islandRec < kIslandMaxRounds)
            launchStepFlagIfSet(cmd, ring, solverPl,
                                wakeChanged.address() + (VkDeviceAddress)(islandRec - 1) * sizeof(int32_t),
                                stepState.address());
        cmd.barrier();

        cmd.copyToHost(activeCounter, wakeReadout, sizeof(int32_t));
        if (floodThisPass)
            cmd.copyToHost(floodChanged, wakeReadout, (VkDeviceSize)floodRec * sizeof(int32_t), 0, sizeof(int32_t));
        if (edgeCount > 0)
            cmd.copyToHost(wakeChanged, wakeReadout, (VkDeviceSize)islandRec * sizeof(int32_t), 0,
                           (VkDeviceSize)(1 + kFloodMaxRounds) * sizeof(int32_t));
    }

}

void VkWorld::recordConstraintsBuild()
{
    StepCtx &S = m_step;
    CommandList &cmd = m_stepCmd;
    HostRingBuffer &ring = m_stepRing;
    auto &slots = S.slots;
    auto &cbr = S.cbr;
    auto &cmr = S.cmr;
    auto &cjr = S.cjr;
    auto &csr = S.csr;
    auto &jpRounds = S.jpRounds;
    auto &colorBound = S.colorBound;
    auto &recordBuildMs = S.recordBuildMs;
    auto &tRec = S.tRec;
    auto &g = S.g;
    auto &restitutionOn = S.restitutionOn;
    auto &front = S.front;
    // --- Phase 5: constraint graph build, recorded into this submission ----------
    // m_dynCount is only the bound; the live count is read from activeCounter on device.
    tRec = Clock::now();
    const int recordedJp = constraints.recordBuild(cmd, ring, csPl, primPl, radix, cbr, cmr, cjr, csr, activeIdx,
                                                   m_dynCount, activeCounter.address(), jpRounds);
    if (m_dynCount > 0 && recordedJp > 0 && recordedJp < VkConstraints::kMaxJpRounds)
        launchStepFlagIfSet(cmd, ring, solverPl, constraints.remainingSlotAddr(recordedJp - 1), stepState.address());
    if (colorBound < AVBD_MAX_COLORS_VK)
        launchStepColorBound(cmd, ring, solverPl, constraints.colorStartAddr(), constraints.colorEndAddr(),
                             stepState.address(), colorBound, AVBD_MAX_COLORS_VK);
    cmd.barrier();
    recordBuildMs = msSince(tRec, Clock::now());
    if (front)
        cmd.setGate(nullptr);

    VkConstraintGraph gRaw = constraints.graph();
    g = {};
    g.key = gRaw.key;
    g.isA = gRaw.isA;
    g.start = gRaw.start;
    g.end = gRaw.end;

    restitutionOn = (m_anyRestitution || m_anyShapeRestitution) && slots > 0;
    if (restitutionOn)
    {
        // The approach speed of every manifold, before the iterations move anything: the
        // velocities still hold v^t. Grown with headroom only when the slot count outgrew it,
        // so a steady scene allocates nothing.
        const VkDeviceSize need = (VkDeviceSize)narrow.slotCapacity() * sizeof(RestitutionSlotGpu);
        if (restSlots.size() < need)
            restSlots.resize(need + need / 2);
        launchRestitutionPre(cmd, ring, restPl, restitutionArgs(slots, g));
        cmd.barrier();
    }

}

void VkWorld::recordIterations()
{
    StepCtx &S = m_step;
    const SolverParams &params = m_params;
    CommandList &cmd = m_stepCmd;
    auto &edgeCount = S.edgeCount;
    auto &slots = S.slots;
    auto &pass = S.pass;
    auto &coarseIters = S.coarseIters;
    auto &withForces = S.withForces;
    auto &warp4 = S.warp4;
    auto &fusedIter = S.fusedIter;
    // --- Phase 6: iteration loop --------------------------------------------------
    // GPU timestamps around each iteration's primal sweep and dual pass: slot 0 at
    // the start, 2it+1 after the primal colours, 2it+2 after the duals. Primal and
    // dual are summed over the iterations; iterMs is the whole loop.
    cmd.writeTimestamp(3);
    withForces = edgeCount > 0;
    warp4 = params.primalLanes > 1;
    // Small-world fusion (vk_small_world.h): the whole loop as one single-workgroup dispatch.
    // Decided on the host from counts known at record time. Anything the fused kernel does not
    // run (the coarse correction, thread-per-body primal) keeps the
    // multi-dispatch path below, as does a world past the workgroup's reach.
    fusedIter = smallWorldEnabled() && warp4 && params.iterations > 0 && coarseIters == 0 && m_dynCount > 0 &&
                m_dynCount <= kFusedIterMaxBodies && slots <= kFusedIterMaxManifoldSlots &&
                m_jointSlots <= kFusedIterMaxJointSlots;
    m_lastFusedIter = fusedIter;
    recordIterationLoop();
}

void VkWorld::recordIterationLoop()
{
    StepCtx &S = m_step;
    const SolverParams &params = m_params;
    CommandList &cmd = m_stepCmd;
    HostRingBuffer &ring = m_stepRing;
    auto &b = S.b;
    auto &j = S.j;
    auto &sp = S.sp;
    auto &p = S.p;
    auto &slots = S.slots;
    auto &m = S.m;
    auto &colorBound = S.colorBound;
    auto &pass = S.pass;
    auto &coarseIters = S.coarseIters;
    auto &gateSlots = S.gateSlots;
    auto &g = S.g;
    auto &withForces = S.withForces;
    auto &warp4 = S.warp4;
    auto &fusedIter = S.fusedIter;
    auto &cStart = S.cStart;
    auto &cEnd = S.cEnd;
    auto &ca = S.ca;
    if (fusedIter)
        m_fusedIterPasses++;
    if (!fusedIter)
        cmd.fill(primalIndirect, 0u);
    m_gate.begin(cmd, ring, solverPl.gateDispatchArgs, stepState.address(), gateSlots);
    cmd.setGate(&m_gate);

    // The colour ranges stay on device: the primal sweep reads its range and its
    // group count from them, so the only host input left is how many colours to
    // record. An empty colour dispatches zero groups.
    cStart = constraints.colorStartAddr();
    cEnd = constraints.colorEndAddr();
    if (!fusedIter)
        launchPrimalDispatchArgs(cmd, ring, solverPl, cStart, cEnd, primalIndirect, colorBound, warp4);
    if (!fusedIter)
        cmd.barrierIndirect();

    ca = {};
    if (coarseIters > 0)
    {
        ca.pa.b = b;
        ca.pa.g = g;
        ca.pa.m = m;
        ca.pa.joints = j;
        ca.pa.springs = sp;
        ca.pa.p = p;
        ca.active = activeIdx.address();
        ca.activeCount = activeCounter.address();
        ca.aggOf = coarseAggOf.address();
        ca.keys = coarseKeys.address();
        ca.acc = coarseAcc.address();
        ca.delta = coarseDelta.address();
        ca.stats = coarseStats.address();
        ca.maxActive = m_dynCount;
        ca.bodyCount = m_bodyCount;
        ca.tableSize = m_coarseTable;
        ca.cell = params.coarseCell > 0.0f ? params.coarseCell : 4.0f * std::max(m_cellSize, 1e-3f);
        ca.damping = params.coarseDamping;
        ca.withForces = withForces ? 1 : 0;
        cmd.fill(coarseStats, 0u);
        launchCoarse(cmd, ring, solverPl.coarseReset, ca, std::max(m_coarseTable, m_bodyCount), 256);
        cmd.barrier();
        launchCoarse(cmd, ring, solverPl.coarseAssign, ca, m_dynCount, 256);
        cmd.barrier();
    }
    if (fusedIter)
    {
        PrimalArgsGpu pa{};
        pa.b = b;
        pa.g = g;
        pa.m = m;
        pa.joints = j;
        pa.springs = sp;
        pa.colorBodies = constraints.colorBodies().address();
        pa.p = p;
        pa.colorStart = cStart;
        pa.colorEnd = cEnd;
        pa.shapeRolling = m_anyRolling ? shpRolling.address() : 0;
        DualArgsGpu da{};
        da.b = b;
        da.m = m;
        da.slots = slots;
        da.p = p;
        JointDualArgsGpu jd{};
        jd.b = b;
        jd.j = j;
        jd.p = p;
        launchIterateSmall(cmd, ring, solverPl, pa, da, jd, params.iterations, colorBound, withForces, m_jointSlots);
        cmd.barrier();
        // The per-iteration stamps finishPass reads; the loop is one dispatch, so they all land
        // after it and primalMs / dualMs report the loop as a whole in the first interval.
        for (int it = 0; it < params.iterations; it++)
        {
            cmd.writeTimestamp((uint32_t)(3 + 2 * it + 1));
            cmd.writeTimestamp((uint32_t)(3 + 2 * it + 2));
        }
    }
    else
        for (int it = 0; it < params.iterations; it++)
        {
            for (int c = 0; c < colorBound; c++)
            {
                PrimalArgsGpu pa{};
                pa.b = b;
                pa.g = g;
                pa.m = m;
                pa.joints = j;
                pa.springs = sp;
                pa.colorBodies = constraints.colorBodies().address();
                pa.p = p;
                pa.colorStart = cStart;
                pa.colorEnd = cEnd;
                pa.color = c;
                pa.shapeRolling = m_anyRolling ? shpRolling.address() : 0;
                launchPrimalColorIndirect(cmd, ring, solverPl, pa, warp4, withForces, primalIndirect);
                cmd.barrier();
            }
            for (int c = 0; it < coarseIters && c < 8; c++)
            {
                ca.color = c; // Gauss-Seidel over the 8 cell parities: neighbours never move together
                launchCoarse(cmd, ring, solverPl.coarseAssemble, ca, m_dynCount, 128);
                cmd.barrier();
                launchCoarse(cmd, ring, solverPl.coarseSolve, ca, m_coarseTable, 128);
                cmd.barrier();
                launchCoarse(cmd, ring, solverPl.coarseApply, ca, m_dynCount, 256);
                cmd.barrier();
            }
            cmd.writeTimestamp((uint32_t)(3 + 2 * it + 1));
            DualArgsGpu da{};
            da.b = b;
            da.m = m;
            da.slots = slots;
            da.p = p;
            launchDual(cmd, ring, solverPl, da, slots);
            cmd.barrier();
            {
                JointDualArgsGpu jd{};
                jd.b = b;
                jd.j = j;
                jd.p = p;
                launchJointDual(cmd, ring, solverPl, jd, m_jointSlots);
                cmd.barrier();
            }
            cmd.writeTimestamp((uint32_t)(3 + 2 * it + 2));
        }

}

void VkWorld::recordTail()
{
    StepCtx &S = m_step;
    const SolverParams &params = m_params;
    CommandList &cmd = m_stepCmd;
    HostRingBuffer &ring = m_stepRing;
    auto &b = S.b;
    auto &p = S.p;
    auto &slots = S.slots;
    auto &m = S.m;
    auto &pass = S.pass;
    auto &g = S.g;
    auto &restitutionOn = S.restitutionOn;
    auto &ca = S.ca;
    // --- Phase 7: velocity, explosion guard, kill-fallen -------------------------
    IntegrateBodySoAGpu ib{};
    ib.pos = pos.address();
    ib.rot = rot.address();
    ib.initialLin = initialLin.address();
    ib.initialAng = initialAng.address();
    ib.inertialLin = inertialLin.address();
    ib.inertialAng = inertialAng.address();
    ib.velLin = velLin.address();
    ib.velAng = velAng.address();
    ib.prevVelLin = prevVelLin.address();
    ib.mass = mass.address();
    ib.awake = awake.address();
    ib.count = (uint32_t)m_bodyCount;
    VelocityArgsGpu va{};
    va.b = ib;
    va.p.dt = params.dt;
    va.p.gravityX = params.gravity.x;
    va.p.gravityY = params.gravity.y;
    va.p.gravityZ = params.gravity.z;
    va.p.spinExact = params.exactSpinVelocity ? 1u : 0u;
    if (m_anyProps)
    {
        VelocityPropsArgsGpu pa{};
        pa.b = ib;
        pa.p = va.p;
        pa.props = props.address();
        launchVelocityProps(cmd, ring, integratePl, pa, m_bodyCount);
    }
    else
        launchVelocity(cmd, ring, integratePl, va, m_bodyCount);
    cmd.barrier();

    if (restitutionOn)
    {
        // Bounce: one normal impulse per manifold that approached faster than the threshold,
        // then each dynamic body sums its own (restitution.slang).
        const RestitutionArgsGpu ra = restitutionArgs(slots, g);
        launchRestitutionImpulse(cmd, ring, restPl, ra);
        cmd.barrier();
        launchRestitutionApply(cmd, ring, restPl, ra);
        cmd.barrier();
    }

    if (m_reportOn && m_reportCap > 0)
    {
        // Contact report (contact_report.slang): post-solve manifolds, pre-step velocities.
        // The counter is zeroed per pass, so a rerun never adds to a discarded pass.
        cmd.fill(reportCounter, 0u, sizeof(uint32_t));
        cmd.barrier();
        ContactReportArgsGpu ra{};
        ra.pos = b.pos;
        ra.rot = b.rot;
        ra.prevVelLin = b.prevVelLin;
        ra.bodyA = m.bodyA;
        ra.bodyB = m.bodyB;
        ra.shapeA = m.shapeA;
        ra.shapeB = m.shapeB;
        ra.basis = m.basis;
        ra.contactOffset = m.contactOffset;
        ra.contactCount = m.contactCount;
        ra.rA = m.rA;
        ra.rB = m.rB;
        ra.C0 = m.C0;
        ra.lambda = m.lambda;
        ra.shapeFlags = shpFlags.address();
        ra.counter = reportCounter.address();
        ra.records = reportRecords.address();
        ra.capacity = (uint32_t)m_reportCap;
        ra.slots = (uint32_t)slots;
        launchContactReport(cmd, ring, reportPl, ra);
        cmd.barrier();
        // Nothing after this writes the report buffers, so the copies can ride here.
        cmd.copyToHost(reportCounter, reportCountReadout, sizeof(uint32_t));
        cmd.copyToHost(reportRecords, reportRecReadout, (VkDeviceSize)m_reportCap * sizeof(ContactReportRecordGpu));
    }

    if (m_dynCount > 0)
    {
        ExplosionGuardArgsGpu ega{};
        ega.b = ib;
        ega.dyn = dynamicIdx.address();
        ega.dynCount = m_dynCount;
        ega.sleepTimer = sleepTimer.address();
        ega.restPos = restPos.address();
        ega.restRot = restRot.address();
        launchExplosionGuard(cmd, ring, integratePl, ega, m_dynCount);
        cmd.barrier();

        if (params.enableContinuous && m_dynShapeCount > 0)
        {
            // Continuous collision (ccd.slang): pose clamp to the earliest time of impact.
            CcdArgsGpu ca{};
            ca.pos = ib.pos;
            ca.rot = ib.rot;
            ca.velLin = ib.velLin;
            ca.initLin = ib.initialLin;
            ca.initAng = ib.initialAng;
            ca.mass = mass.address();
            ca.awake = awake.address();
            ca.sleepTimer = sleepTimer.address();
            ca.props = propsAddress();
            ca.shpBody = shpBody.address();
            ca.shpLocalPos = shpLocalPos.address();
            ca.shpLocalRot = shpLocalRot.address();
            ca.shpHalf = shpHalf.address();
            ca.shpType = shpType.address();
            ca.shpFlags = shpFlags.address();
            ca.shpCategory = shpCategory.address();
            ca.shpMask = shpMask.address();
            ca.shpGroup = shpGroup.address();
            ca.dynShape = dynShapeIdx.address();
            ca.staticShape = staticShapeIdx.address();
            ca.ignore = ignorePairs.address();
            ca.ignoreCount = (uint32_t)m_ignoreCount;
            ca.geomHeap = geomHeap.address();
            ca.geomOffset = shpGeomOffset.address();
            ca.geomHeapBytes = geomHeap.size();
            ca.dynCount = (uint32_t)m_dynShapeCount;
            ca.staticCount = (uint32_t)m_staticShapeCount;
            ccd.record(cmd, ring, ca, m_bodyCap, m_anyBullet);
        }

        if (params.killFallenEnabled)
        {
            KillFallenArgsGpu kfa{};
            kfa.b = ib;
            kfa.dyn = dynamicIdx.address();
            kfa.dynCount = m_dynCount;
            {
                // Kill height is measured along up = -normalize(gravity); zero gravity falls back to +Y.
                const float gl = std::sqrt(params.gravity.x * params.gravity.x + params.gravity.y * params.gravity.y +
                                           params.gravity.z * params.gravity.z);
                kfa.killHeight = params.killHeight;
                kfa.upX = gl > 0.0f ? -params.gravity.x / gl : 0.0f;
                kfa.upY = gl > 0.0f ? -params.gravity.y / gl : 1.0f;
                kfa.upZ = gl > 0.0f ? -params.gravity.z / gl : 0.0f;
            }
            kfa.sleepTimer = sleepTimer.address();
            kfa.restPos = restPos.address();
            kfa.restRot = restRot.address();
            launchKillFallen(cmd, ring, integratePl, kfa, m_dynCount);
            cmd.barrier();
        }
    }

}

void VkWorld::recordSleepCommit()
{
    StepCtx &S = m_step;
    const SolverParams &params = m_params;
    CommandList &cmd = m_stepCmd;
    HostRingBuffer &ring = m_stepRing;
    auto &b = S.b;
    auto &j = S.j;
    auto &sp = S.sp;
    auto &sleepOn = S.sleepOn;
    auto &m = S.m;
    auto &g = S.g;
    // --- Phase 8: sleep commit --------------------------------------------------
    if (sleepOn)
    {
        SleepCommitCall sc{};
        sc.fracture.b = b;
        sc.fracture.j = j;
        sc.fracture.ratio = params.sleepFracture;
        // csSleepBlockFracture stores block[body] for every joint loaded past sleepFracture of
        // its break force; a null block here lost the device the first time one was (a grab or
        // an impact on a jointed structure).
        sc.fracture.block = sleepBlock.address();
        sc.drag.dragIdx = 0;
        sc.drag.dragCount = 0;
        sc.drag.block = sleepBlock.address();
        sc.jointCount = m_jointSlots;
        sc.dragCount = 0;
        sc.dynCount = m_dynCount;
        sc.solveCount = m_dynCount;
        sc.commit.b = b;
        sc.commit.g = g;
        sc.commit.m = m;
        sc.commit.j = j;
        sc.commit.sp = sp;
        sc.commit.active = activeIdx.address();
        sc.commit.activeCount = m_dynCount;
        sc.commit.activeCountPtr = activeCounter.address();
        sc.commit.candidate = sleepCandidate.address();
        sc.commit.touchCount = touchCount.address();
        sc.commit.touchHash = touchHash.address();
        sc.commit.snapCount = snapCount.address();
        sc.commit.snapHash = snapHash.address();
        sc.mark.b = b;
        sc.mark.dyn = dynamicIdx.address();
        sc.mark.dynCount = m_dynCount;
        sc.mark.candidate = sleepCandidate.address();
        sc.mark.frames = params.sleepFrames;
        sc.mark.linVel = params.sleepLinVel;
        sc.mark.angVel = params.sleepAngVel;
        sc.mark.disp = params.sleepDisp;
        sc.mark.leak = params.sleepLeak;
        sc.mark.block = sleepBlock.address();
        sc.mark.hasBlock = 1u;
        // Zeroed by a recorded fill, not by launchSleepCommit's own synchronous
        // one-shot (block == nullptr skips it): that would be a second submission.
        cmd.fill(sleepBlock, 0u, m_bodyCount * sizeof(unsigned char));
        cmd.barrier();
        sc.block = nullptr;
        launchSleepCommit(cmd, ring, sleepPl, sc);
        cmd.barrier();
    }
}

void VkWorld::submitPass()
{
    StepCtx &S = m_step;
    CommandList &cmd = m_stepCmd;
    HostRingBuffer &ring = m_stepRing;
    auto &tsEnd = S.tsEnd;
    auto &hostSubmitMs = S.hostSubmitMs;
    auto &hostRecordMs = S.hostRecordMs;
    auto &tPass = S.tPass;
    auto &recordedP1 = S.recordedP1;
    cmd.setGate(nullptr);
    cmd.writeTimestamp(tsEnd);
    cmd.copyToHost(stepState, stepFlagReadout, 2 * sizeof(int32_t));
    // The pose snapshot: a VRAM copy into this step's slot (sized in stepBeginImpl). The
    // host copy, if anyone wants one, is a separate DMA (requestSnapshotCopy).
    if (m_stepSlot >= 0)
    {
        const VkDeviceSize n = (VkDeviceSize)m_bodyCount;
        const int cap = m_snapCap;
        const CommandList::CopyOp ops[5] = {
            {&pos, 0, snapOffset(0, cap), n * sizeof(vec3)},
            {&rot, 0, snapOffset(1, cap), n * sizeof(quat)},
            {&velLin, 0, snapOffset(2, cap), n * sizeof(vec3)},
            {&velAng, 0, snapOffset(3, cap), n * sizeof(vec3)},
            {&awake, 0, snapOffset(4, cap), n},
        };
        cmd.copyBatch(ops, 5, snapSlot[m_stepSlot]);
        if (m_stepJSlot >= 0)
        {
            const VkDeviceSize jn = (VkDeviceSize)m_jointCount * sizeof(vec3);
            const CommandList::CopyOp jops[2] = {
                {&jLambdaLin, 0, 0, jn},
                {&jLambdaAng, 0, (VkDeviceSize)m_snapJCap * sizeof(vec3), jn},
            };
            cmd.copyBatch(jops, 2, snapJSlot[m_stepJSlot]);
        }
    }
    // A held group re-checks which of its joints broke every frame; ride the copy along.
    // brokenReadout is sized by uploadJoints, so the step allocates nothing here.
    const VkDeviceSize brokenBytes = (VkDeviceSize)m_jointCount * sizeof(unsigned char);
    m_brokenRecorded = m_dragActive && m_jointCount > 0 && brokenReadout.size() >= brokenBytes;
    if (m_brokenRecorded)
        cmd.copyToHost(jBroken, brokenReadout, brokenBytes);
    if (ring.wrapped())
    {
        // Phases 4-8 are one submission between ring resets, so a wrap has
        // overwritten argument structs it still reads.
        fprintf(stderr, "[avbd_vk] step: argument ring overflowed within one submission\n");
        abort();
    }
    const Clock::time_point tSub = Clock::now();
    m_lastOps = cmd.opCounts();
    cmd.submit();
    S.tWait = Clock::now();
    S.recordedP1 = recordedP1;
    hostSubmitMs += msSince(tSub, S.tWait);
    hostRecordMs += msSince(tPass, tSub);
}


void VkWorld::recordAndSubmitPass()
{
    recordFront();
    recordSleepFirstHalf();
    recordConstraintsBuild();
    recordIterations();
    recordTail();
    recordSleepCommit();
    submitPass();
}


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
    StepResult &result = S.result;
    const int32_t *readout = wakeReadout.mappedAs<int32_t>();
    const int32_t *stepFlag = stepFlagReadout.mappedAs<int32_t>();
    auto &edgeCount = S.edgeCount;
    auto &iters = S.iters;
    auto &tsStart = S.tsStart;
    auto &tsEnd = S.tsEnd;
    auto &frontMs = S.frontMs;
    auto &slots = S.slots;
    auto &floodDone = S.floodDone;
    auto &floodRec = S.floodRec;
    auto &islandRec = S.islandRec;
    auto &phase4Done = S.phase4Done;
    auto &jpRounds = S.jpRounds;
    auto &colorBound = S.colorBound;
    auto &floodCause = S.floodCause;
    auto &islandCause = S.islandCause;
    auto &hostWaitMs = S.hostWaitMs;
    auto &frontPending = S.frontPending;
    auto &capAttempt = S.capAttempt;
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
