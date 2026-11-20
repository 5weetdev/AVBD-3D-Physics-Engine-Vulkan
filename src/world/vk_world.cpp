#include "vk_world.h"

#include "vk_capacity.h"
#include "vk_small_world.h"
#include "vk_world_internal.h"

#include "collision.h" // MAX_MANIFOLD_CONTACTS

#include "integrate_spirv.h"
#include "broadphase_spirv.h"
#include "narrowphase_spirv.h"
#include "narrowphase_hull_spirv.h"
#include "narrowphase_mesh_spirv.h"
#include "constraints_spirv.h"
#include "primal_spirv.h"
#include "step_control_spirv.h"
#include "coarse_spirv.h"
#include "dual_spirv.h"
#include "sleep_spirv.h"
#include "contact_report_spirv.h"
#include "restitution_spirv.h"
#include "wake_spirv.h"
#include "mutate_spirv.h"
#include "render_spirv.h"
#include "render_overlay_spirv.h"
#include "render_debug_spirv.h"
#include "drag_spirv.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace avbdvk
{

void VkWorld::init()
{
    if (m_init)
        return;
    if (m_ownDev)
        dev.init();
    layout.init(dev);
    integratePl.init(dev, layout, integrate_spirv, integrate_spirv_words);
    bphPl.init(dev, layout, broadphase_spirv, broadphase_spirv_words);
    npPl.init(dev, layout, narrowphase_spirv, narrowphase_spirv_words);
    npPl.initHull(dev, layout, narrowphase_hull_spirv, narrowphase_hull_spirv_words);
    npPl.initMesh(dev, layout, narrowphase_mesh_spirv, narrowphase_mesh_spirv_words);
    csPl.init(dev, layout, constraints_spirv, constraints_spirv_words);
    solverPl.init(dev, layout, primal_spirv, primal_spirv_words, dual_spirv, dual_spirv_words, step_control_spirv,
                  step_control_spirv_words, coarse_spirv, coarse_spirv_words);
    sleepPl.init(dev, layout, sleep_spirv, sleep_spirv_words);
    reportPl.init(dev, layout, contact_report_spirv, contact_report_spirv_words);
    ccd.init(dev, layout);
    restPl.init(dev, layout, restitution_spirv, restitution_spirv_words);
    wakePl.init(dev, layout, wake_spirv, wake_spirv_words);
    mutatePl.init(dev, layout, mutate_spirv, mutate_spirv_words);
    primPl.init(dev, layout);
    renderPl.init(dev, layout, {render_spirv, render_spirv_words, render_overlay_spirv, render_overlay_spirv_words,
                                render_debug_spirv, render_debug_spirv_words});
    interactPl.init(dev, layout, drag_spirv, drag_spirv_words);
    dev.savePipelineCache(); // keep the compiled kernels even if this process is killed before shutdown

    broad.init(dev);
    narrow.init(dev);
    constraints.init(dev);
    radix.init(dev);
    m_select.init(dev);
    m_stepCmd.init(dev, 2 * 16 + 6);
    m_stepRing.init(dev, 1 << 23);

    pos.init(dev);
    rot.init(dev);
    initialLin.init(dev);
    initialAng.init(dev);
    inertialLin.init(dev);
    inertialAng.init(dev);
    velLin.init(dev);
    velAng.init(dev);
    prevVelLin.init(dev);
    halfExtents.init(dev);
    shapeType.init(dev);
    moment.init(dev);
    mass.init(dev);
    friction.init(dev);
    awake.init(dev);
    sleepTimer.init(dev);
    restPos.init(dev);
    restRot.init(dev);
    gyration.init(dev);
    props.init(dev);
    dynamicIdx.init(dev);
    staticIdx.init(dev);
    ignorePairs.init(dev);
    gridIdx.init(dev);
    oversizeIdx.init(dev);
    shpBody.init(dev);
    shpLocalPos.init(dev);
    shpLocalRot.init(dev);
    shpHalf.init(dev);
    shpType.init(dev);
    shpFriction.init(dev);
    shpRestitution.init(dev);
    shpRolling.init(dev);
    shpCategory.init(dev);
    shpMask.init(dev);
    shpGroup.init(dev);
    shpFlags.init(dev);
    shpGeomOffset.init(dev);
    geomHeap.init(dev);
    dynShapeIdx.init(dev);
    staticShapeIdx.init(dev);

    jBodyA.init(dev);
    jBodyB.init(dev);
    jRA.init(dev);
    jRB.init(dev);
    jC0Lin.init(dev);
    jC0Ang.init(dev);
    jPenaltyLin.init(dev);
    jPenaltyAng.init(dev);
    jLambdaLin.init(dev);
    jLambdaAng.init(dev);
    jStiffnessLin.init(dev);
    jStiffnessAng.init(dev);
    jFracture.init(dev);
    jFractureAng.init(dev);
    jTorqueArm.init(dev);
    jRestAng.init(dev);
    jBroken.init(dev);
    jFrameA.init(dev);
    jFrameB.init(dev);
    jAxisMode.init(dev);
    jAxisLo.init(dev);
    jAxisHi.init(dev);
    jGeneric.init(dev);

    sBodyA.init(dev);
    sBodyB.init(dev);
    sRA.init(dev);
    sRB.init(dev);
    sRest.init(dev);
    sStiffness.init(dev);

    activeIdx.init(dev);
    activeFlags.init(dev);
    activeCounter.init(dev);
    coarseAggOf.init(dev);
    coarseKeys.init(dev);
    coarseAcc.init(dev);
    coarseDelta.init(dev);
    coarseStats.init(dev);
    touchCount.init(dev);
    touchHash.init(dev);
    snapCount.init(dev);
    snapHash.init(dev);
    wakeRequest.init(dev);
    wakeSeed.init(dev);
    wakeAny.init(dev);
    wakeChanged.init(dev);
    floodChanged.init(dev);
    islandSeed.init(dev);
    primalIndirect.init(dev);
    stepState.init(dev);
    m_gate.init(dev);
    m_gateCap.init(dev);
    mutIdxBuf.init(dev);
    mutDataBuf.init(dev);
    readIdxBuf.init(dev);
    readDataBuf.init(dev);
    jwBuf.init(dev);
    sleepCandidate.init(dev);
    sleepBlock.init(dev);
    reportCounter.init(dev);
    reportRecords.init(dev);
    restSlots.init(dev);

    instances.init(dev);
    debugInstances.init(dev);
    frustumBuf.init(dev);
    shapeBaseBuf.init(dev);
    shapeVisBuf.init(dev);
    visibleCounterBuf.init(dev);
    debugCounterBuf.init(dev);
    groupIdx.init(dev);
    overlayLoad.init(dev);
    overlayPush.init(dev);
    overlayPull.init(dev);
    overlayStress.init(dev);
    overlayCone.init(dev);
    overlayPen.init(dev);
    overlayJointDeg.init(dev);
    shapeBaseBuf.resize(3 * sizeof(int32_t));
    shapeVisBuf.resize(3 * sizeof(int32_t));
    // writeInstances() uploads into this unconditionally when culling is on, so it
    // must exist even if the first caller is the only one -- an upload() into a
    // never-resized Buffer is a vkCmdCopyBuffer with a null dst (driver fault).
    frustumBuf.resize(24 * sizeof(float));
    visibleCounterBuf.resize(sizeof(int32_t));
    debugCounterBuf.resize(sizeof(int32_t));
    dragBodies_.init(dev);
    dragTargets.init(dev);
    dragLocals.init(dev);
    dragStiffness.init(dev);
    dragBodies_.resize(kMaxDragJoints * sizeof(int));
    dragTargets.resize(kMaxDragJoints * sizeof(vec3));
    dragLocals.resize(kMaxDragJoints * sizeof(vec3));
    dragStiffness.resize(kMaxDragJoints * sizeof(float));
    dragStage.init(dev, kDragStageBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MappedBuffer::HostAccess::Write);

    m_init = true;
}

void VkWorld::reset()
{
    assertIdle("VkWorld::reset");
    // Buffer::destroy() (called by resize(0) or the destructor) frees the
    // vkAllocateMemory allocation outright -- this is the "does not grow across scene
    // reset" gate. Re-`init()`ing each Buffer with a fresh 0-byte state is enough;
    // build() will resize() them again on the next call.
    pos.destroy();
    rot.destroy();
    initialLin.destroy();
    initialAng.destroy();
    inertialLin.destroy();
    inertialAng.destroy();
    velLin.destroy();
    velAng.destroy();
    prevVelLin.destroy();
    halfExtents.destroy();
    shapeType.destroy();
    moment.destroy();
    mass.destroy();
    friction.destroy();
    awake.destroy();
    sleepTimer.destroy();
    restPos.destroy();
    restRot.destroy();
    gyration.destroy();
    props.destroy();
    dynamicIdx.destroy();
    staticIdx.destroy();
    ignorePairs.destroy();
    gridIdx.destroy();
    oversizeIdx.destroy();
    shpBody.destroy();
    shpLocalPos.destroy();
    shpLocalRot.destroy();
    shpHalf.destroy();
    shpType.destroy();
    shpFriction.destroy();
    shpRestitution.destroy();
    shpRolling.destroy();
    shpCategory.destroy();
    shpMask.destroy();
    shpGroup.destroy();
    shpFlags.destroy();
    shpGeomOffset.destroy();
    // The geometry heap goes with the scene (the memory-reset gate), blobs and all.
    geomHeap.destroy();
    geomStage.destroy();
    m_geomPending.clear();
    m_geomCopies.clear();
    m_geomIndex.clear();
    m_geomSceneOffset.clear();
    m_geomCursor = 0;
    dynShapeIdx.destroy();
    staticShapeIdx.destroy();
    ccd.release();

    jBodyA.destroy();
    jBodyB.destroy();
    jRA.destroy();
    jRB.destroy();
    jC0Lin.destroy();
    jC0Ang.destroy();
    jPenaltyLin.destroy();
    jPenaltyAng.destroy();
    jLambdaLin.destroy();
    jLambdaAng.destroy();
    jStiffnessLin.destroy();
    jStiffnessAng.destroy();
    jFracture.destroy();
    jFractureAng.destroy();
    jTorqueArm.destroy();
    jRestAng.destroy();
    jBroken.destroy();
    jFrameA.destroy();
    jFrameB.destroy();
    jAxisMode.destroy();
    jAxisLo.destroy();
    jAxisHi.destroy();
    jGeneric.destroy();

    sBodyA.destroy();
    sBodyB.destroy();
    sRA.destroy();
    sRB.destroy();
    sRest.destroy();
    sStiffness.destroy();

    activeIdx.destroy();
    activeFlags.destroy();
    activeCounter.destroy();
    coarseAggOf.destroy();
    coarseKeys.destroy();
    coarseAcc.destroy();
    coarseDelta.destroy();
    coarseStats.destroy();
    touchCount.destroy();
    touchHash.destroy();
    snapCount.destroy();
    snapHash.destroy();
    wakeRequest.destroy();
    wakeSeed.destroy();
    wakeAny.destroy();
    wakeChanged.destroy();
    floodChanged.destroy();
    islandSeed.destroy();
    wakeReadout.destroy();
    stepFlagReadout.destroy();
    snapshotDestroy(false);
    primalIndirect.destroy();
    stepState.destroy();
    m_gate.destroy();
    m_gateCap.destroy();
    mutIdxBuf.destroy();
    mutDataBuf.destroy();
    mutIdxStage.destroy();
    mutDataStage.destroy();
    jwBuf.destroy();
    jwStage.destroy();
    m_jwRecs.clear();
    readIdxBuf.destroy();
    readDataBuf.destroy();
    m_mutBatches.clear();
    m_mutIdx.clear();
    m_mutData.clear();
    m_jwRecs.clear();
    m_seedsPending = false;
    sleepCandidate.destroy();
    sleepBlock.destroy();

    // The phases' capacity-sized buffers and the sort scratch go too: reserve() and
    // the radix/scan ensure-paths rebuild them for the next scene, and without this a
    // 100k-body scene's pair, manifold and sort buffers outlive the switch to a small
    // one (radix scratch never shrinks on its own).
    narrow.reset();
    narrow.release();
    broad.release();
    constraints.release();
    radix.destroy();
    m_select.destroy();

    instances.destroy();
    debugInstances.destroy();
    groupIdx.destroy();
    overlayLoad.destroy();
    overlayPush.destroy();
    overlayPull.destroy();
    overlayStress.destroy();
    overlayCone.destroy();
    overlayPen.destroy();
    overlayJointDeg.destroy();
    m_instanceCount = m_debugInstanceCount = m_debugRequested = 0;
    m_groupBase.clear();
    m_groupVisible.clear();
    m_dragActive = false;
    m_dragPending = false;
    brokenReadout.destroy(); // scene-sized
    m_brokenRecorded = m_brokenMirror = false;

    m_bodyCount = m_dynCount = m_staticCount = 0;
    m_shapeCount = m_dynShapeCount = m_staticShapeCount = m_shapeCap = 0;
    m_hullShapes = 0;
    m_meshShapes = 0;
    m_jointCount = m_springCount = 0;
    m_jointSlots = 0;
    m_activeCount = 0;
    m_bodyCap = m_jointCap = m_springCap = 0;
    m_pairCap = m_contactCap = m_manifoldCap = 0;
}

void VkWorld::destroy()
{
    if (!m_init)
        return;
    reset();
    instances.destroy();
    debugInstances.destroy();
    frustumBuf.destroy();
    shapeBaseBuf.destroy();
    shapeVisBuf.destroy();
    visibleCounterBuf.destroy();
    debugCounterBuf.destroy();
    dragBodies_.destroy();
    dragTargets.destroy();
    dragLocals.destroy();
    dragStiffness.destroy();
    dragStage.destroy();
    radix.destroy();
    m_select.destroy();
    snapshotDestroy(true);
    m_stepCmd.destroy();
    m_stepRing.destroy();
    constraints.destroy();
    narrow.destroy();
    broad.destroy();
    interactPl.destroy();
    renderPl.destroy();
    primPl.destroy();
    mutatePl.destroy();
    wakePl.destroy();
    reportPl.destroy();
    ccd.destroy();
    restPl.destroy();
    restSlots.destroy();
    reportCounter.destroy();
    reportRecords.destroy();
    reportCountReadout.destroy();
    reportRecReadout.destroy();
    sleepPl.destroy();
    solverPl.destroy();
    csPl.destroy();
    npPl.destroy();
    bphPl.destroy();
    integratePl.destroy();
    layout.destroy();
    if (m_ownDev)
        dev.shutdown();
    m_init = false;
}

// Rebuilt every step, so contents need not survive growth.
void VkWorld::sizeCoarse(int bodyCap)
{
    int t = 64;
    while (t < 2 * bodyCap)
        t *= 2;
    m_coarseTable = t;
    coarseAggOf.resize((VkDeviceSize)std::max(bodyCap, 1) * sizeof(int));
    coarseKeys.resize((VkDeviceSize)t * sizeof(uint32_t));
    coarseAcc.resize((VkDeviceSize)t * 36 * sizeof(float));
    coarseDelta.resize((VkDeviceSize)t * 6 * sizeof(float));
    coarseStats.resize(4 * sizeof(int32_t));
}

void VkWorld::reserve(int bodyCap, int jointCap, int springCap, int pairCap, int contactCap, int manifoldCap,
                      int shapeCap)
{
    auto vb = [](Buffer &b, size_t bytes) { b.resize((VkDeviceSize)bytes); };

    vb(pos, bodyCap * sizeof(vec3));
    vb(rot, bodyCap * sizeof(quat));
    vb(initialLin, bodyCap * sizeof(vec3));
    vb(initialAng, bodyCap * sizeof(quat));
    vb(inertialLin, bodyCap * sizeof(vec3));
    vb(inertialAng, bodyCap * sizeof(quat));
    vb(velLin, bodyCap * sizeof(vec3));
    vb(velAng, bodyCap * sizeof(vec3));
    vb(prevVelLin, bodyCap * sizeof(vec3));
    vb(halfExtents, bodyCap * sizeof(vec3));
    vb(shapeType, bodyCap * sizeof(unsigned char));
    vb(moment, bodyCap * sizeof(vec3));
    vb(mass, bodyCap * sizeof(float));
    vb(friction, bodyCap * sizeof(float));
    vb(awake, bodyCap * sizeof(unsigned char));
    vb(sleepTimer, bodyCap * sizeof(int));
    vb(restPos, bodyCap * sizeof(vec3));
    vb(restRot, bodyCap * sizeof(quat));
    vb(gyration, bodyCap * sizeof(vec3));
    vb(props, bodyCap * sizeof(BodyProps));
    vb(dynamicIdx, bodyCap * sizeof(int));
    vb(staticIdx, bodyCap * sizeof(int));
    vb(gridIdx, shapeCap * sizeof(int));
    vb(oversizeIdx, kMaxOversizeBodies * sizeof(int));
    vb(dynShapeIdx, shapeCap * sizeof(int));
    vb(staticShapeIdx, shapeCap * sizeof(int));
    vb(shpBody, shapeCap * sizeof(int));
    vb(shpLocalPos, shapeCap * sizeof(vec3));
    vb(shpLocalRot, shapeCap * sizeof(quat));
    vb(shpHalf, shapeCap * sizeof(vec3));
    vb(shpType, shapeCap * sizeof(unsigned char));
    vb(shpFriction, shapeCap * sizeof(float));
    vb(shpRestitution, shapeCap * sizeof(float));
    vb(shpRolling, shapeCap * sizeof(float));
    vb(shpCategory, shapeCap * sizeof(uint64_t));
    vb(shpMask, shapeCap * sizeof(uint64_t));
    vb(shpGroup, shapeCap * sizeof(int));
    vb(shpFlags, shapeCap * sizeof(unsigned));
    vb(shpGeomOffset, shapeCap * sizeof(unsigned));

    vb(activeIdx, bodyCap * sizeof(int));
    vb(activeFlags, bodyCap * sizeof(unsigned char));
    vb(activeCounter, sizeof(int));
    sizeCoarse(bodyCap);
    vb(touchCount, bodyCap * sizeof(int));
    vb(touchHash, bodyCap * sizeof(unsigned int));
    vb(snapCount, bodyCap * sizeof(int));
    vb(snapHash, bodyCap * sizeof(unsigned int));
    vb(wakeRequest, bodyCap * sizeof(unsigned char));
    vb(wakeSeed, bodyCap * sizeof(unsigned char));
    vb(wakeAny, sizeof(int));
    vb(wakeChanged, kIslandMaxRounds * sizeof(int));
    vb(floodChanged, kFloodMaxRounds * sizeof(int));
    vb(islandSeed, bodyCap * sizeof(unsigned char));
    if (!wakeReadout.mapped())
        wakeReadout.init(dev, (VkDeviceSize)(1 + kFloodMaxRounds + kIslandMaxRounds) * sizeof(int32_t),
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT, MappedBuffer::HostAccess::Read);
    if (!stepFlagReadout.mapped())
        stepFlagReadout.init(dev, 2 * sizeof(int32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             MappedBuffer::HostAccess::Read);
    primalIndirect.resize((VkDeviceSize)AVBD_MAX_COLORS_VK * 3 * sizeof(uint32_t), VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
    if (stepState.size() == 0)
        stepState.resize(16);
    vb(sleepCandidate, bodyCap * sizeof(unsigned char));
    vb(sleepBlock, bodyCap * sizeof(unsigned char));

    int jc = std::max(jointCap, 1);
    vb(jBodyA, jc * sizeof(int));
    vb(jBodyB, jc * sizeof(int));
    vb(jRA, jc * sizeof(vec3));
    vb(jRB, jc * sizeof(vec3));
    vb(jC0Lin, jc * sizeof(vec3));
    vb(jC0Ang, jc * sizeof(vec3));
    vb(jPenaltyLin, jc * sizeof(vec3));
    vb(jPenaltyAng, jc * sizeof(vec3));
    vb(jLambdaLin, jc * sizeof(vec3));
    vb(jLambdaAng, jc * sizeof(vec3));
    vb(jStiffnessLin, jc * sizeof(float));
    vb(jStiffnessAng, jc * sizeof(float));
    vb(jFracture, jc * sizeof(float));
    vb(jFractureAng, jc * sizeof(float));
    vb(jTorqueArm, jc * sizeof(float));
    vb(jRestAng, jc * sizeof(quat));
    vb(jBroken, jc * sizeof(unsigned char));
    vb(jFrameA, jc * sizeof(quat));
    vb(jFrameB, jc * sizeof(quat));
    vb(jAxisMode, jc * 6 * sizeof(unsigned char));
    vb(jAxisLo, jc * 6 * sizeof(float));
    vb(jAxisHi, jc * 6 * sizeof(float));
    vb(jGeneric, jc * sizeof(unsigned char));

    int sc = std::max(springCap, 1);
    vb(sBodyA, sc * sizeof(int));
    vb(sBodyB, sc * sizeof(int));
    vb(sRA, sc * sizeof(vec3));
    vb(sRB, sc * sizeof(vec3));
    vb(sRest, sc * sizeof(float));
    vb(sStiffness, sc * sizeof(float));

    broad.reserve(shapeCap, std::max(pairCap, 1));
    narrow.reserve(std::max(pairCap, 1), std::max(contactCap, 1));
    constraints.reserve(bodyCap, std::max(manifoldCap, 1), jc + sc);

    m_bodyCap = bodyCap;
    m_shapeCap = shapeCap;
    m_jointCap = jointCap;
    m_springCap = springCap;
    m_pairCap = pairCap;
    m_contactCap = contactCap;
    m_manifoldCap = manifoldCap;
}

void VkWorld::grow(int neededPairs, int neededContacts)
{
    // Headroom constants of the two overflow-retry sites
    // (broadphase pairs: +1024, narrowphase contacts: +4096).
    int newPairCap = std::max(m_pairCap, cappedGrow(neededPairs, 1024, m_dbg.hardPairCap > 0 ? m_dbg.hardPairCap : kDefaultCapacityCeiling));
    int newContactCap = std::max(m_contactCap, cappedGrow(neededContacts, 4096, m_dbg.hardContactCap > 0 ? m_dbg.hardContactCap : kDefaultCapacityCeiling));
    bool pairGrew = newPairCap > m_pairCap;
    bool contactGrew = newContactCap > m_contactCap;
    if (pairGrew)
    {
        broad.reserve(m_shapeCap, newPairCap);
        m_pairCap = newPairCap;
    }
    if (contactGrew || pairGrew)
    {
        // VkNarrowphase::grow() preserves the persistent contact cache (feature keys,
        // lambda, penalty, stick) across this resize via Buffer::resizePreserve(),
        // mirroring the capacity code's reserveKeep() -- warm starting survives a mid-run
        // overflow instead of resetting to zero.
        narrow.grow(newPairCap, newContactCap);
        m_contactCap = newContactCap;
    }
    if (pairGrew && newPairCap > m_manifoldCap)
    {
        // BUG FIX (coordinator point 2): the original capacity growth always
        // re-reserves the constraint graph to the new pair count in lockstep -- one
        // manifold-worth of CSR entries per pair, at most two per manifold. This call
        // was missing entirely: pairCap/contactCap grew but constraints.reserve() never
        // did, so manifoldCap stayed at whatever a forced-tiny (or even the initial)
        // capacity left it. Once narrowphase produced more live manifolds than that
        // stale manifoldCap, VkConstraints::build() silently truncated the CSR graph to
        // the first manifoldCap of them (see vk_constraints.cpp) -- every dynamic body
        // whose only contact fell past that cut never got a primal/dual entry at all,
        // so it free-fell through the rest of the pile untouched: no NaN, no explosion
        // guard trip, just a body the solver never knew was in contact with anything.
        // Re-reserving here on every pair-capacity growth, not just the initial
        // uploadScene() sizing, is the fix; it costs one more CSR buffer resize on an
        // already-rare growth path, not a per-step cost.
        constraints.reserve(m_bodyCap, newPairCap, m_jointCap + m_springCap);
        m_manifoldCap = newPairCap;
    }
    if (narrow.childCapacity() > 0)
        syncManifoldCap();
}

void VkWorld::syncManifoldCap()
{
    // Mesh child slots sit after the pair slots, so the constraint graph needs pairs + child slots.
    const int want = narrow.slotCapacity();
    if (want > m_manifoldCap)
    {
        constraints.reserve(m_bodyCap, want, m_jointCap + m_springCap);
        m_manifoldCap = want;
    }
}

uint64_t VkWorld::deviceBytesAllocated() const
{
    const Buffer *all[] = {
        &pos,
        &rot,
        &initialLin,
        &initialAng,
        &inertialLin,
        &inertialAng,
        &velLin,
        &velAng,
        &prevVelLin,
        &halfExtents,
        &shapeType,
        &moment,
        &mass,
        &friction,
        &awake,
        &sleepTimer,
        &restPos,
        &restRot,
        &gyration,
        &props,
        &dynamicIdx,
        &staticIdx,
        &gridIdx,
        &oversizeIdx,
        &shpBody,
        &shpLocalPos,
        &shpLocalRot,
        &shpHalf,
        &shpType,
        &shpFriction,
        &shpRestitution,
        &shpRolling,
        &shpCategory,
        &shpMask,
        &shpGroup,
        &shpFlags,
        &shpGeomOffset,
        &geomHeap,
        &dynShapeIdx,
        &staticShapeIdx,
        &jBodyA,
        &jBodyB,
        &jRA,
        &jRB,
        &jC0Lin,
        &jC0Ang,
        &jPenaltyLin,
        &jPenaltyAng,
        &jLambdaLin,
        &jLambdaAng,
        &jStiffnessLin,
        &jStiffnessAng,
        &jFracture,
        &jFractureAng,
        &jTorqueArm,
        &jRestAng,
        &jBroken,
        &jFrameA,
        &jFrameB,
        &jAxisMode,
        &jAxisLo,
        &jAxisHi,
        &jGeneric,
        &sBodyA,
        &sBodyB,
        &sRA,
        &sRB,
        &sRest,
        &sStiffness,
        &activeIdx,
        &activeFlags,
        &activeCounter,
        &touchCount,
        &touchHash,
        &snapCount,
        &snapHash,
        &wakeRequest,
        &wakeSeed,
        &wakeAny,
        &wakeChanged,
        &sleepCandidate,
        &sleepBlock,
        &ignorePairs,
        &floodChanged,
        &islandSeed,
        &primalIndirect,
        &stepState,
        &mutIdxBuf,
        &mutDataBuf,
        &readIdxBuf,
        &readDataBuf,
        &instances,
        &debugInstances,
        &groupIdx,
        &overlayLoad,
        &overlayPush,
        &overlayPull,
        &overlayStress,
        &overlayCone,
        &overlayPen,
        &overlayJointDeg,
        &frustumBuf,
        &shapeBaseBuf,
        &shapeVisBuf,
        &visibleCounterBuf,
        &debugCounterBuf,
        &dragBodies_,
        &dragTargets,
        &dragLocals,
        &dragStiffness,
    };
    uint64_t total = 0;
    for (const Buffer *buf : all)
        total += (uint64_t)buf->size();

    // Point 3: the above is only VkWorld's own body/joint/spring/sleep-wake buffers.
    // broad/narrow/constraints/radix each own a private buffer set (double-buffered
    // manifolds, CSR/coloring scratch, radix sort ping-pong) that this list cannot see
    // and previously silently omitted -- add each subsystem's own transitive total.
    total += (uint64_t)mutIdxStage.size() + (uint64_t)mutDataStage.size() + (uint64_t)geomStage.size();
    total += (uint64_t)dragStage.size() + (uint64_t)brokenReadout.size();
    total += broad.bytesAllocated();
    total += narrow.bytesAllocated();
    total += constraints.bytesAllocated();
    total += radix.bytesAllocated();
    return total;
}

} // namespace avbdvk
