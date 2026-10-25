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

#include "vk_small_world.h"

#include "collision.h" // MAX_MANIFOLD_CONTACTS

#include "integrate_spirv.h"
#include "broadphase_spirv.h"
#include "narrowphase_spirv.h"
#include "narrowphase_hull_spirv.h"
#include "narrowphase_mesh_spirv.h"
#include "constraints_spirv.h"
#include "primal_spirv.h"
#include "dual_spirv.h"
#include "sleep_spirv.h"
#include "contact_report_spirv.h"
#include "restitution_spirv.h"
#include "wake_spirv.h"
#include "mutate_spirv.h"
#include "render_spirv.h"
#include "drag_spirv.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace avbdvk
{

// Drag slots beyond the authored joints, always present -- solver_impl's
// kMaxDragJoints, the same pool size, so a multi-body grab holds as many bodies here as
// it does on the reference.
static const int kMaxDragJoints = 256;
// dragStage layout: bodies, targets, locals, stiffness, each kMaxDragJoints long.
static const VkDeviceSize kDragBodiesOff = 0;
static const VkDeviceSize kDragTargetsOff = kDragBodiesOff + kMaxDragJoints * sizeof(int);
static const VkDeviceSize kDragLocalsOff = kDragTargetsOff + kMaxDragJoints * sizeof(vec3);
static const VkDeviceSize kDragStiffOff = kDragLocalsOff + kMaxDragJoints * sizeof(vec3);
static const VkDeviceSize kDragStageBytes = kDragStiffOff + kMaxDragJoints * sizeof(float);
// Above this many oversized bodies the broadphase falls back to the legacy cell (brute force is O(L*N)).
static const int kMaxOversizeBodies = 64;

// Exact mirror of VkWorld::step()'s private cappedGrow(): a single-shot
// need + need/4 + slack, clamped to a hard ceiling so a bad pose (broadphase
// reporting "everything overlaps everything") cannot run away trying to
// vkAllocateMemory an unbounded amount -- see
// explosion-freezes-machine-is-broadphase-not-numerics in the project memory.
// `need < 0` (a counter that wrapped around) is treated as worst-case, same as
// VkWorld::step(). Duplicated here rather than exported from VkWorld::step(), which is a private
// TU function; same policy constant, same shape -- `have` is unused, kept for
// call-site symmetry with the old signature.
// A contact cap below 4 is raised to 4. Measured on the settled 10k pile:
// the reduction seeds on the deepest point, which moves
// between corners as a box rocks, so a cap of 2 or 3 drops a different corner of a face
// each step, the returning one starts cold, and the pile never settles (cap 3: 3x the
// manifolds, 1.3x the step; cap 1 4.5x). Four keeps a box face's whole quad.
static int effectiveContactCap(int cap)
{
    return cap > 0 ? std::max(cap, 4) : 0;
}

static int cappedGrow(int have, int need, int slack, int ceiling = 0)
{
    (void)have;
    const int kMaxCapacity = ceiling > 0 ? ceiling : 20 * 1000 * 1000;
    if (need < 0)
        return kMaxCapacity;
    long long grown = (long long)need + need / 4 + slack;
    if (grown > kMaxCapacity)
        grown = kMaxCapacity;
    return (int)grown;
}

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
    solverPl.init(dev, layout, primal_spirv, primal_spirv_words, dual_spirv, dual_spirv_words);
    sleepPl.init(dev, layout, sleep_spirv, sleep_spirv_words);
    reportPl.init(dev, layout, contact_report_spirv, contact_report_spirv_words);
    ccd.init(dev, layout);
    restPl.init(dev, layout, restitution_spirv, restitution_spirv_words);
    wakePl.init(dev, layout, wake_spirv, wake_spirv_words);
    mutatePl.init(dev, layout, mutate_spirv, mutate_spirv_words);
    primPl.init(dev, layout);
    renderPl.init(dev, layout, render_spirv, render_spirv_words);
    interactPl.init(dev, layout, drag_spirv, drag_spirv_words);
    dev.savePipelineCache(); // keep the compiled kernels even if this process is killed before shutdown

    broad.init(dev);
    narrow.init(dev);
    constraints.init(dev);
    radix.init(dev);
    m_select.init(dev);
    m_stepCmd.init(dev, 2 * 16 + 6);
    m_stepRing.init(dev, 1 << 23);

    pos.init(dev); rot.init(dev); initialLin.init(dev); initialAng.init(dev);
    inertialLin.init(dev); inertialAng.init(dev);
    velLin.init(dev); velAng.init(dev); prevVelLin.init(dev);
    halfExtents.init(dev); shapeType.init(dev); moment.init(dev); mass.init(dev); friction.init(dev);
    awake.init(dev); sleepTimer.init(dev); restPos.init(dev); restRot.init(dev); gyration.init(dev);
    props.init(dev);
    dynamicIdx.init(dev); staticIdx.init(dev); ignorePairs.init(dev);
    gridIdx.init(dev); oversizeIdx.init(dev);
    shpBody.init(dev); shpLocalPos.init(dev); shpLocalRot.init(dev); shpHalf.init(dev); shpType.init(dev);
    shpFriction.init(dev); shpRestitution.init(dev); shpRolling.init(dev); shpCategory.init(dev); shpMask.init(dev); shpGroup.init(dev); shpFlags.init(dev);
    shpGeomOffset.init(dev); geomHeap.init(dev);
    dynShapeIdx.init(dev); staticShapeIdx.init(dev);

    jBodyA.init(dev); jBodyB.init(dev); jRA.init(dev); jRB.init(dev);
    jC0Lin.init(dev); jC0Ang.init(dev); jPenaltyLin.init(dev); jPenaltyAng.init(dev);
    jLambdaLin.init(dev); jLambdaAng.init(dev);
    jStiffnessLin.init(dev); jStiffnessAng.init(dev); jFracture.init(dev); jFractureAng.init(dev);
    jTorqueArm.init(dev); jRestAng.init(dev); jBroken.init(dev);
    jFrameA.init(dev); jFrameB.init(dev); jAxisMode.init(dev); jAxisLo.init(dev); jAxisHi.init(dev); jGeneric.init(dev);

    sBodyA.init(dev); sBodyB.init(dev); sRA.init(dev); sRB.init(dev); sRest.init(dev); sStiffness.init(dev);

    activeIdx.init(dev); activeFlags.init(dev); activeCounter.init(dev);
    coarseAggOf.init(dev); coarseKeys.init(dev); coarseAcc.init(dev); coarseDelta.init(dev); coarseStats.init(dev);
    touchCount.init(dev); touchHash.init(dev); snapCount.init(dev); snapHash.init(dev);
    wakeRequest.init(dev); wakeSeed.init(dev); wakeAny.init(dev); wakeChanged.init(dev);
    floodChanged.init(dev); islandSeed.init(dev); primalIndirect.init(dev); stepState.init(dev); m_gate.init(dev); m_gateCap.init(dev); mutIdxBuf.init(dev); mutDataBuf.init(dev);
    readIdxBuf.init(dev); readDataBuf.init(dev); jwBuf.init(dev);
    sleepCandidate.init(dev); sleepBlock.init(dev);
    reportCounter.init(dev); reportRecords.init(dev); restSlots.init(dev);

    instances.init(dev); debugInstances.init(dev);
    frustumBuf.init(dev); shapeBaseBuf.init(dev); shapeVisBuf.init(dev);
    visibleCounterBuf.init(dev); debugCounterBuf.init(dev);
    groupIdx.init(dev);
    overlayLoad.init(dev); overlayPush.init(dev); overlayPull.init(dev); overlayStress.init(dev);
    overlayCone.init(dev); overlayPen.init(dev); overlayJointDeg.init(dev);
    shapeBaseBuf.resize(3 * sizeof(int32_t));
    shapeVisBuf.resize(3 * sizeof(int32_t));
    // writeInstances() uploads into this unconditionally when culling is on, so it
    // must exist even if the first caller is the only one -- an upload() into a
    // never-resized Buffer is a vkCmdCopyBuffer with a null dst (driver fault).
    frustumBuf.resize(24 * sizeof(float));
    visibleCounterBuf.resize(sizeof(int32_t));
    debugCounterBuf.resize(sizeof(int32_t));
    dragBodies_.init(dev); dragTargets.init(dev); dragLocals.init(dev); dragStiffness.init(dev);
    dragBodies_.resize(kMaxDragJoints * sizeof(int)); dragTargets.resize(kMaxDragJoints * sizeof(vec3));
    dragLocals.resize(kMaxDragJoints * sizeof(vec3)); dragStiffness.resize(kMaxDragJoints * sizeof(float));
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
    pos.destroy(); rot.destroy(); initialLin.destroy(); initialAng.destroy();
    inertialLin.destroy(); inertialAng.destroy();
    velLin.destroy(); velAng.destroy(); prevVelLin.destroy();
    halfExtents.destroy(); shapeType.destroy(); moment.destroy(); mass.destroy(); friction.destroy();
    awake.destroy(); sleepTimer.destroy(); restPos.destroy(); restRot.destroy(); gyration.destroy();
    props.destroy();
    dynamicIdx.destroy(); staticIdx.destroy(); ignorePairs.destroy();
    gridIdx.destroy(); oversizeIdx.destroy();
    shpBody.destroy(); shpLocalPos.destroy(); shpLocalRot.destroy(); shpHalf.destroy(); shpType.destroy();
    shpFriction.destroy(); shpRestitution.destroy(); shpRolling.destroy(); shpCategory.destroy(); shpMask.destroy(); shpGroup.destroy(); shpFlags.destroy();
    shpGeomOffset.destroy();
    // The geometry heap goes with the scene (the memory-reset gate), blobs and all.
    geomHeap.destroy(); geomStage.destroy();
    m_geomPending.clear(); m_geomCopies.clear(); m_geomIndex.clear(); m_geomSceneOffset.clear();
    m_geomCursor = 0;
    dynShapeIdx.destroy(); staticShapeIdx.destroy();
    ccd.release();

    jBodyA.destroy(); jBodyB.destroy(); jRA.destroy(); jRB.destroy();
    jC0Lin.destroy(); jC0Ang.destroy(); jPenaltyLin.destroy(); jPenaltyAng.destroy();
    jLambdaLin.destroy(); jLambdaAng.destroy();
    jStiffnessLin.destroy(); jStiffnessAng.destroy(); jFracture.destroy(); jFractureAng.destroy();
    jTorqueArm.destroy(); jRestAng.destroy(); jBroken.destroy();
    jFrameA.destroy(); jFrameB.destroy(); jAxisMode.destroy(); jAxisLo.destroy(); jAxisHi.destroy(); jGeneric.destroy();

    sBodyA.destroy(); sBodyB.destroy(); sRA.destroy(); sRB.destroy(); sRest.destroy(); sStiffness.destroy();

    activeIdx.destroy(); activeFlags.destroy(); activeCounter.destroy();
    coarseAggOf.destroy(); coarseKeys.destroy(); coarseAcc.destroy(); coarseDelta.destroy(); coarseStats.destroy();
    touchCount.destroy(); touchHash.destroy(); snapCount.destroy(); snapHash.destroy();
    wakeRequest.destroy(); wakeSeed.destroy(); wakeAny.destroy(); wakeChanged.destroy();
    floodChanged.destroy(); islandSeed.destroy(); wakeReadout.destroy(); stepFlagReadout.destroy(); snapshotDestroy(false); primalIndirect.destroy(); stepState.destroy(); m_gate.destroy(); m_gateCap.destroy(); mutIdxBuf.destroy(); mutDataBuf.destroy(); mutIdxStage.destroy(); mutDataStage.destroy();
    jwBuf.destroy(); jwStage.destroy(); m_jwRecs.clear();
    readIdxBuf.destroy(); readDataBuf.destroy();
    m_mutBatches.clear(); m_mutIdx.clear(); m_mutData.clear();
    m_jwRecs.clear();
    m_seedsPending = false;
    sleepCandidate.destroy(); sleepBlock.destroy();

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

    instances.destroy(); debugInstances.destroy();
    groupIdx.destroy();
    overlayLoad.destroy(); overlayPush.destroy(); overlayPull.destroy(); overlayStress.destroy();
    overlayCone.destroy(); overlayPen.destroy(); overlayJointDeg.destroy();
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
    instances.destroy(); debugInstances.destroy();
    frustumBuf.destroy(); shapeBaseBuf.destroy(); shapeVisBuf.destroy();
    visibleCounterBuf.destroy(); debugCounterBuf.destroy();
    dragBodies_.destroy(); dragTargets.destroy(); dragLocals.destroy(); dragStiffness.destroy();
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
    reportCounter.destroy(); reportRecords.destroy(); reportCountReadout.destroy(); reportRecReadout.destroy();
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

    vb(pos, bodyCap * sizeof(vec3)); vb(rot, bodyCap * sizeof(quat));
    vb(initialLin, bodyCap * sizeof(vec3)); vb(initialAng, bodyCap * sizeof(quat));
    vb(inertialLin, bodyCap * sizeof(vec3)); vb(inertialAng, bodyCap * sizeof(quat));
    vb(velLin, bodyCap * sizeof(vec3)); vb(velAng, bodyCap * sizeof(vec3)); vb(prevVelLin, bodyCap * sizeof(vec3));
    vb(halfExtents, bodyCap * sizeof(vec3)); vb(shapeType, bodyCap * sizeof(unsigned char));
    vb(moment, bodyCap * sizeof(vec3));
    vb(mass, bodyCap * sizeof(float)); vb(friction, bodyCap * sizeof(float));
    vb(awake, bodyCap * sizeof(unsigned char)); vb(sleepTimer, bodyCap * sizeof(int));
    vb(restPos, bodyCap * sizeof(vec3)); vb(restRot, bodyCap * sizeof(quat));
    vb(gyration, bodyCap * sizeof(vec3));
    vb(props, bodyCap * sizeof(BodyProps));
    vb(dynamicIdx, bodyCap * sizeof(int)); vb(staticIdx, bodyCap * sizeof(int));
    vb(gridIdx, shapeCap * sizeof(int)); vb(oversizeIdx, kMaxOversizeBodies * sizeof(int));
    vb(dynShapeIdx, shapeCap * sizeof(int)); vb(staticShapeIdx, shapeCap * sizeof(int));
    vb(shpBody, shapeCap * sizeof(int));
    vb(shpLocalPos, shapeCap * sizeof(vec3)); vb(shpLocalRot, shapeCap * sizeof(quat));
    vb(shpHalf, shapeCap * sizeof(vec3)); vb(shpType, shapeCap * sizeof(unsigned char));
    vb(shpFriction, shapeCap * sizeof(float));
    vb(shpRestitution, shapeCap * sizeof(float)); vb(shpRolling, shapeCap * sizeof(float));
    vb(shpCategory, shapeCap * sizeof(uint64_t)); vb(shpMask, shapeCap * sizeof(uint64_t));
    vb(shpGroup, shapeCap * sizeof(int)); vb(shpFlags, shapeCap * sizeof(unsigned));
    vb(shpGeomOffset, shapeCap * sizeof(unsigned));

    vb(activeIdx, bodyCap * sizeof(int)); vb(activeFlags, bodyCap * sizeof(unsigned char));
    vb(activeCounter, sizeof(int));
    sizeCoarse(bodyCap);
    vb(touchCount, bodyCap * sizeof(int)); vb(touchHash, bodyCap * sizeof(unsigned int));
    vb(snapCount, bodyCap * sizeof(int)); vb(snapHash, bodyCap * sizeof(unsigned int));
    vb(wakeRequest, bodyCap * sizeof(unsigned char)); vb(wakeSeed, bodyCap * sizeof(unsigned char));
    vb(wakeAny, sizeof(int)); vb(wakeChanged, kIslandMaxRounds * sizeof(int));
    vb(floodChanged, kFloodMaxRounds * sizeof(int)); vb(islandSeed, bodyCap * sizeof(unsigned char));
    if (!wakeReadout.mapped())
        wakeReadout.init(dev, (VkDeviceSize)(1 + kFloodMaxRounds + kIslandMaxRounds) * sizeof(int32_t),
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT, MappedBuffer::HostAccess::Read);
    if (!stepFlagReadout.mapped())
        stepFlagReadout.init(dev, 2 * sizeof(int32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT, MappedBuffer::HostAccess::Read);
    primalIndirect.resize((VkDeviceSize)AVBD_MAX_COLORS_VK * 3 * sizeof(uint32_t), VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
    if (stepState.size() == 0)
        stepState.resize(16);
    vb(sleepCandidate, bodyCap * sizeof(unsigned char));
    vb(sleepBlock, bodyCap * sizeof(unsigned char));

    int jc = std::max(jointCap, 1);
    vb(jBodyA, jc * sizeof(int)); vb(jBodyB, jc * sizeof(int));
    vb(jRA, jc * sizeof(vec3)); vb(jRB, jc * sizeof(vec3));
    vb(jC0Lin, jc * sizeof(vec3)); vb(jC0Ang, jc * sizeof(vec3));
    vb(jPenaltyLin, jc * sizeof(vec3)); vb(jPenaltyAng, jc * sizeof(vec3));
    vb(jLambdaLin, jc * sizeof(vec3)); vb(jLambdaAng, jc * sizeof(vec3));
    vb(jStiffnessLin, jc * sizeof(float)); vb(jStiffnessAng, jc * sizeof(float));
    vb(jFracture, jc * sizeof(float)); vb(jFractureAng, jc * sizeof(float)); vb(jTorqueArm, jc * sizeof(float));
    vb(jRestAng, jc * sizeof(quat)); vb(jBroken, jc * sizeof(unsigned char));
    vb(jFrameA, jc * sizeof(quat)); vb(jFrameB, jc * sizeof(quat));
    vb(jAxisMode, jc * 6 * sizeof(unsigned char));
    vb(jAxisLo, jc * 6 * sizeof(float)); vb(jAxisHi, jc * 6 * sizeof(float));
    vb(jGeneric, jc * sizeof(unsigned char));

    int sc = std::max(springCap, 1);
    vb(sBodyA, sc * sizeof(int)); vb(sBodyB, sc * sizeof(int));
    vb(sRA, sc * sizeof(vec3)); vb(sRB, sc * sizeof(vec3));
    vb(sRest, sc * sizeof(float)); vb(sStiffness, sc * sizeof(float));

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
    // Same headroom constants as GpuSolver::stepLaunch()'s two overflow-retry sites
    // (broadphase pairs: +1024, narrowphase contacts: +4096) per VkWorld's capacity code.
    int newPairCap = std::max(m_pairCap, cappedGrow(m_pairCap, neededPairs, 1024, m_dbg.hardPairCap));
    int newContactCap = std::max(m_contactCap, cappedGrow(m_contactCap, neededContacts, 4096, m_dbg.hardContactCap));
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

// Per-body arrays for bodies [first, count), with the fresh-body state the capacity code's
// upload() (first == 0) and spawnGroup() (first > 0) give them.
void VkWorld::uploadBodyRange(const SceneDesc &scene, int first, int count)
{
    const int n = count >= 0 ? count : m_bodyCount - first;
    if (n <= 0)
        return;
    const size_t o3 = first * sizeof(vec3), o4 = first * sizeof(quat);
    const size_t of = first * sizeof(float), o1 = first * sizeof(unsigned char);

    pos.upload(&scene.pos[first], n * sizeof(vec3), o3);
    rot.upload(&scene.rot[first], n * sizeof(quat), o4);
    initialLin.upload(&scene.pos[first], n * sizeof(vec3), o3);
    initialAng.upload(&scene.rot[first], n * sizeof(quat), o4);
    inertialLin.upload(&scene.pos[first], n * sizeof(vec3), o3);
    inertialAng.upload(&scene.rot[first], n * sizeof(quat), o4);
    velLin.upload(&scene.velLin[first], n * sizeof(vec3), o3);
    velAng.upload(&scene.velAng[first], n * sizeof(vec3), o3);
    // prevVel starts at the initial velocity, as VkWorld's capacity code does, so the adaptive
    // warmstart sees zero acceleration on a moving body's first step.
    prevVelLin.upload(&scene.velLin[first], n * sizeof(vec3), o3);
    halfExtents.upload(&scene.halfExtents[first], n * sizeof(vec3), o3);
    shapeType.upload(&scene.shapeType[first], n * sizeof(unsigned char), o1);
    moment.upload(&scene.moment[first], n * sizeof(vec3), o3);
    mass.upload(&scene.mass[first], n * sizeof(float), of);
    friction.upload(&scene.friction[first], n * sizeof(float), of);

    std::vector<unsigned char> ones(n, 1), zeros1(n, 0);
    std::vector<int> zerosI(n, 0);
    awake.upload(ones.data(), n * sizeof(unsigned char), o1);
    sleepTimer.upload(zerosI.data(), n * sizeof(int), first * sizeof(int));
    restPos.upload(&scene.pos[first], n * sizeof(vec3), o3);
    restRot.upload(&scene.rot[first], n * sizeof(quat), o4);
    activeFlags.upload(zeros1.data(), n * sizeof(unsigned char), o1);

    // Gyration radius, |halfExtents|, exactly as VkWorld's capacity code derives it. The sleep rest
    // test turns rotation into distance through this; zero meant a spinning body could
    // fall asleep.
    std::vector<float> gyr(n);
    for (int i = 0; i < n; i++)
        gyr[i] = length(scene.halfExtents[first + i]);
    gyration.upload(gyr.data(), n * sizeof(float), of);

    // BodyProps: the authored entries, or defaults for a scene that has none.
    if (first == 0)
    {
        m_hostProps.clear();
        m_anyProps = false;
        m_anyRestitution = false;
        m_anyBullet = false;
    }
    m_hostProps.resize(m_bodyCount, BodyProps{});
    const bool authored = (int)scene.bodyProps.size() >= m_bodyCount;
    for (int i = first; i < m_bodyCount; i++)
    {
        if (authored)
            m_hostProps[i] = scene.bodyProps[i];
        if (!bodyPropsDefault(m_hostProps[i]))
            m_anyProps = true;
        if (m_hostProps[i].restitution > 0.0f)
            m_anyRestitution = true;
        if ((m_hostProps[i].flags & BODY_BULLET) != 0)
            m_anyBullet = true;
    }
    props.upload(&m_hostProps[first], n * sizeof(BodyProps), first * sizeof(BodyProps));
}

// The shape table for shapes [first, shapeCount). Shapes are only ever appended, so an
// earlier shape (and the contact cache entries keyed on it) never moves.
void VkWorld::uploadShapeRange(const SceneDesc &scene, int first, int count)
{
    const int n = count >= 0 ? count : m_shapeCount - first;
    if (n <= 0)
        return;
    const size_t o3 = first * sizeof(vec3), o4 = first * sizeof(quat);
    const size_t of = first * sizeof(float), o1 = first * sizeof(unsigned char), oi = first * sizeof(int);
    shpBody.upload(&scene.shapeBody[first], n * sizeof(int), oi);
    shpLocalPos.upload(&scene.shapeLocalPos[first], n * sizeof(vec3), o3);
    shpLocalRot.upload(&scene.shapeLocalRot[first], n * sizeof(quat), o4);
    shpHalf.upload(&scene.shapeHalf[first], n * sizeof(vec3), o3);
    shpType.upload(&scene.shapeKind[first], n * sizeof(unsigned char), o1);
    for (int i = 0; i < n; i++)
    {
        const int kind = scene.shapeKind[(size_t)(first + i)];
        if (kind == AVBD_SHAPE_HULL)
            m_hullShapes++;
        else if (kind == AVBD_SHAPE_MESH)
            m_meshShapes++;
    }
    shpFriction.upload(&scene.shapeFriction[first], n * sizeof(float), of);
    // Per-shape restitution and rolling resistance (a SceneBuilder scene has none: all zero).
    {
        std::vector<float> rest((size_t)n, 0.0f), roll((size_t)n, 0.0f);
        for (int i = 0; i < n; i++)
        {
            const size_t k = (size_t)(first + i);
            if (k < scene.shapeRestitution.size())
                rest[(size_t)i] = scene.shapeRestitution[k];
            if (k < scene.shapeRolling.size())
                roll[(size_t)i] = scene.shapeRolling[k];
            if (rest[(size_t)i] > 0.0f)
                m_anyShapeRestitution = true;
            if (roll[(size_t)i] > 0.0f)
                m_anyRolling = true;
        }
        shpRestitution.upload(rest.data(), n * sizeof(float), of);
        shpRolling.upload(roll.data(), n * sizeof(float), of);
    }
    shpCategory.upload(&scene.shapeCategory[first], n * sizeof(uint64_t), first * sizeof(uint64_t));
    shpMask.upload(&scene.shapeMask[first], n * sizeof(uint64_t), first * sizeof(uint64_t));
    shpGroup.upload(&scene.shapeGroup[first], n * sizeof(int), oi);
    shpFlags.upload(&scene.shapeFlags[first], n * sizeof(unsigned), oi);
    // geomOffset: the heap offset of the shape's blob (registerSceneGeometry has run), or none.
    std::vector<unsigned> geom((size_t)n, kNoGeometry);
    for (int i = 0; i < n; i++)
    {
        const size_t s = (size_t)(first + i);
        if (s < scene.shapeGeom.size() && scene.shapeGeom[s] >= 0 &&
            (size_t)scene.shapeGeom[s] < m_geomSceneOffset.size())
            geom[(size_t)i] = m_geomSceneOffset[(size_t)scene.shapeGeom[s]];
    }
    shpGeomOffset.upload(geom.data(), n * sizeof(unsigned), oi);
}

// Registers the scene's geometry blobs not yet in the heap (blobs are append-only, and
// addGeometry deduplicates by hash, so a grown scene costs only its new blobs).
void VkWorld::registerSceneGeometry(const SceneDesc &scene)
{
    for (size_t i = m_geomSceneOffset.size(); i < scene.geomBlob.size(); i++)
    {
        const std::vector<unsigned char> &blob = scene.geomBlob[i];
        const uint64_t hash = i < scene.geomHash.size() ? scene.geomHash[i] : 0;
        m_geomSceneOffset.push_back(addGeometry(blob.data(), (uint32_t)blob.size(), hash));
    }
}

uint32_t VkWorld::addGeometry(const void *blob, uint32_t bytes, uint64_t hash)
{
    assertIdle("VkWorld::addGeometry");
    if (blob == nullptr || bytes == 0)
        return kNoGeometry;
    const unsigned char *src = static_cast<const unsigned char *>(blob);
    if (hash == 0)
    {
        hash = 1469598103934665603ull; // FNV-1a
        for (uint32_t i = 0; i < bytes; i++)
            hash = (hash ^ src[i]) * 1099511628211ull;
        if (hash == 0)
            hash = 1;
    }
    const auto key = std::make_pair(hash, bytes);
    const auto it = m_geomIndex.find(key);
    if (it != m_geomIndex.end())
        return it->second;

    const uint32_t aligned = (bytes + 15u) & ~15u;
    const uint32_t offset = m_geomCursor;
    const uint64_t end = (uint64_t)offset + aligned;
    if (end >= kNoGeometry)
        return kNoGeometry; // 4 GiB of blobs: out of 32-bit offsets
    // Headroom: double, with a floor, so a run of small adds allocates once.
    if (end > geomHeap.size())
        geomHeap.resizePreserve(std::max<VkDeviceSize>(end, std::max<VkDeviceSize>(geomHeap.size() * 2, 1u << 16)));
    const uint32_t stageOffset = (uint32_t)m_geomPending.size();
    m_geomPending.resize((size_t)stageOffset + aligned, 0);
    memcpy(m_geomPending.data() + stageOffset, src, bytes);
    if (geomStage.size() < m_geomPending.size())
    {
        // Between steps, so nothing in flight reads the old stage.
        const VkDeviceSize cap = std::max<VkDeviceSize>(m_geomPending.size(), std::max<VkDeviceSize>(geomStage.size() * 2, 1u << 16));
        geomStage.destroy();
        geomStage.init(dev, cap, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MappedBuffer::HostAccess::Write);
    }
    m_geomCopies.push_back(GeomCopy{stageOffset, offset, aligned});
    m_geomCursor = (uint32_t)end;
    m_geomIndex[key] = offset;
    return offset;
}

// Authored joints, then the drag pool, as VkWorld's capacity code lays them out. Dual state is
// zeroed; append() restores it for joints that already existed.
void VkWorld::uploadJoints(const SceneDesc &scene)
{
    // jointInitialize() (gpu_forces) indexes bodyA/bodyB before checking broken, so
    // a drag slot's bodyB must be a real index even while nothing is held.
    if (m_jointCount > 0)
    {
        jBodyA.upload(scene.jointA.data(), m_jointCount * sizeof(int));
        jBodyB.upload(scene.jointB.data(), m_jointCount * sizeof(int));
        jRA.upload(scene.jointRA.data(), m_jointCount * sizeof(vec3));
        jRB.upload(scene.jointRB.data(), m_jointCount * sizeof(vec3));
        jStiffnessLin.upload(scene.jointKLin.data(), m_jointCount * sizeof(float));
        jStiffnessAng.upload(scene.jointKAng.data(), m_jointCount * sizeof(float));
        jFracture.upload(scene.jointFrac.data(), m_jointCount * sizeof(float));
        jFractureAng.upload(scene.jointFracAng.data(), m_jointCount * sizeof(float));
        jTorqueArm.upload(scene.jointArm.data(), m_jointCount * sizeof(float));
        jRestAng.upload(scene.jointRestAng.data(), m_jointCount * sizeof(quat));
        jFrameA.upload(scene.jointFrameA.data(), m_jointCount * sizeof(quat));
        jFrameB.upload(scene.jointFrameB.data(), m_jointCount * sizeof(quat));
        if (!scene.jointAxisMode.empty())
            jAxisMode.upload(scene.jointAxisMode.data(), scene.jointAxisMode.size() * sizeof(unsigned char));
        if (!scene.jointAxisLo.empty())
            jAxisLo.upload(scene.jointAxisLo.data(), scene.jointAxisLo.size() * sizeof(float));
        if (!scene.jointAxisHi.empty())
            jAxisHi.upload(scene.jointAxisHi.data(), scene.jointAxisHi.size() * sizeof(float));
        jGeneric.upload(scene.jointGeneric.data(), m_jointCount * sizeof(unsigned char));
    }
    jC0Lin.zero(m_jointSlots * sizeof(vec3));
    jC0Ang.zero(m_jointSlots * sizeof(vec3));
    jPenaltyLin.zero(m_jointSlots * sizeof(vec3));
    jPenaltyAng.zero(m_jointSlots * sizeof(vec3));
    jLambdaLin.zero(m_jointSlots * sizeof(vec3));
    jLambdaAng.zero(m_jointSlots * sizeof(vec3));

    // The drag pool, all broken (inert) until something is grabbed -- the capacity code's
    // values for a drag slot: world anchor, bodyB 0, infinite fracture.
    std::vector<int> dragA(kMaxDragJoints, -1), dragB(kMaxDragJoints, 0);
    std::vector<vec3> dragZeroV(kMaxDragJoints, vec3{0, 0, 0});
    std::vector<float> dragZeroF(kMaxDragJoints, 0.0f), dragInf(kMaxDragJoints, INFINITY), dragNeg(kMaxDragJoints, -1.0f);
    std::vector<quat> dragIdentQ(kMaxDragJoints, quat{0, 0, 0, 1});
    std::vector<unsigned char> dragGeneric(kMaxDragJoints, 0);
    const size_t j0 = m_jointCount;
    jBodyA.upload(dragA.data(), kMaxDragJoints * sizeof(int), j0 * sizeof(int));
    jBodyB.upload(dragB.data(), kMaxDragJoints * sizeof(int), j0 * sizeof(int));
    jRA.upload(dragZeroV.data(), kMaxDragJoints * sizeof(vec3), j0 * sizeof(vec3));
    jRB.upload(dragZeroV.data(), kMaxDragJoints * sizeof(vec3), j0 * sizeof(vec3));
    jStiffnessLin.upload(dragZeroF.data(), kMaxDragJoints * sizeof(float), j0 * sizeof(float));
    jStiffnessAng.upload(dragZeroF.data(), kMaxDragJoints * sizeof(float), j0 * sizeof(float));
    jFracture.upload(dragInf.data(), kMaxDragJoints * sizeof(float), j0 * sizeof(float));
    jFractureAng.upload(dragNeg.data(), kMaxDragJoints * sizeof(float), j0 * sizeof(float));
    jTorqueArm.upload(dragZeroF.data(), kMaxDragJoints * sizeof(float), j0 * sizeof(float));
    jRestAng.upload(dragIdentQ.data(), kMaxDragJoints * sizeof(quat), j0 * sizeof(quat));
    jFrameA.upload(dragIdentQ.data(), kMaxDragJoints * sizeof(quat), j0 * sizeof(quat));
    jFrameB.upload(dragIdentQ.data(), kMaxDragJoints * sizeof(quat), j0 * sizeof(quat));
    jGeneric.upload(dragGeneric.data(), kMaxDragJoints * sizeof(unsigned char), j0 * sizeof(unsigned char));
    std::vector<unsigned char> brokenAll(m_jointSlots, 0);
    for (int i = 0; i < kMaxDragJoints; i++)
        brokenAll[m_jointCount + i] = 1;
    jBroken.upload(brokenAll.data(), m_jointSlots * sizeof(unsigned char));
    m_brokenMirror = false;
    if (m_jointCount > 0 && brokenReadout.size() < (VkDeviceSize)m_jointCount)
    {
        brokenReadout.destroy();
        brokenReadout.init(dev, (VkDeviceSize)m_jointCount, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           MappedBuffer::HostAccess::Read);
    }
    m_dragActive = false;
    m_dragPending = false;
    m_dragCount = 0;
}

void VkWorld::refreshSlots(const SceneDesc &scene, const std::vector<int> &bodies, const std::vector<int> &shapes)
{
    assertIdle("VkWorld::refreshSlots");
    static const unsigned char zero1 = 0;
    for (int b : bodies)
    {
        if (b < 0 || b >= m_bodyCount)
            continue;
        uploadBodyRange(scene, b, 1);
        if (b < (int)m_hostMass.size() && b < (int)scene.mass.size())
            m_hostMass[b] = scene.mass[b]; // the drag stiffness scales with it
        wakeSeed.upload(&zero1, 1, (VkDeviceSize)b);
        if (b < (int)scene.asleep.size() && scene.asleep[b])
            awake.upload(&zero1, 1, (VkDeviceSize)b);
    }
    for (int s : shapes)
        if (s >= 0 && s < m_shapeCount)
            uploadShapeRange(scene, s, 1);
}

void VkWorld::refreshLists(const SceneDesc &scene)
{
    assertIdle("VkWorld::refreshLists");
    m_dynCount = (int)scene.dynamicIdx.size();
    m_staticCount = (int)scene.staticIdx.size();
    uploadIndexLists(scene);
    activeIdx.upload(scene.dynamicIdx.data(), m_dynCount * sizeof(int));
    m_activeCount = m_dynCount;
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
    a.pos = pos.address(); a.rot = rot.address(); a.velLin = velLin.address(); a.velAng = velAng.address();
    a.mass = mass.address(); a.moment = moment.address(); a.props = props.address();
    a.bodyA = cur.bodyA; a.bodyB = cur.bodyB; a.basis = cur.basis;
    a.contactOffset = cur.contactOffset; a.contactCount = cur.contactCount;
    a.rA = cur.rA; a.rB = cur.rB; a.lambda = cur.lambda;
    a.gKey = g.key; a.gIsA = g.isA; a.gStart = g.start; a.gEnd = g.end;
    a.dyn = dynamicIdx.address(); a.dynCount = (uint32_t)m_dynCount;
    a.out = restSlots.address();
    a.shapeA = cur.shapeA; a.shapeB = cur.shapeB; a.shpRestitution = shpRestitution.address();
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

void VkWorld::uploadSprings(const SceneDesc &scene)
{
    if (m_springCount <= 0)
        return;
    sBodyA.upload(scene.springA.data(), m_springCount * sizeof(int));
    sBodyB.upload(scene.springB.data(), m_springCount * sizeof(int));
    sRA.upload(scene.springRA.data(), m_springCount * sizeof(vec3));
    sRB.upload(scene.springRB.data(), m_springCount * sizeof(vec3));
    sRest.upload(scene.springRest.data(), m_springCount * sizeof(float));
    sStiffness.upload(scene.springK.data(), m_springCount * sizeof(float));
}

void VkWorld::uploadIndexLists(const SceneDesc &scene)
{
    dynamicIdx.upload(scene.dynamicIdx.data(), m_dynCount * sizeof(int));
    staticIdx.upload(scene.staticIdx.data(), m_staticCount * sizeof(int));

    // The body -> shapes CSR first, then the broadphase's shape lists: each body's shapes,
    // body by body in dynamicIdx / staticIdx order. A scene without addShape has shape i ==
    // body i, so these equal the body lists.
    {
        const int n = scene.count;
        const int ns = scene.shapeCount;
        m_bodyShapeStart.assign((size_t)n + 1, 0);
        for (int sh = 0; sh < ns; sh++)
            m_bodyShapeStart[scene.shapeBody[sh] + 1]++;
        for (int b = 0; b < n; b++)
            m_bodyShapeStart[b + 1] += m_bodyShapeStart[b];
        m_bodyShapeList.assign((size_t)ns, 0);
        std::vector<int> cursor(m_bodyShapeStart.begin(), m_bodyShapeStart.end() - 1);
        for (int sh = 0; sh < ns; sh++)
            m_bodyShapeList[cursor[scene.shapeBody[sh]]++] = sh;
    }
    std::vector<int> dynShapes, staticShapes;
    dynShapes.reserve(scene.shapeCount);
    staticShapes.reserve(scene.shapeCount);
    for (int b : scene.dynamicIdx)
        for (int k = m_bodyShapeStart[b]; k < m_bodyShapeStart[b + 1]; k++)
            dynShapes.push_back(m_bodyShapeList[k]);
    for (int b : scene.staticIdx)
        if (!((int)m_hostProps.size() > b && (m_hostProps[b].flags & BODY_KINEMATIC)))
            for (int k = m_bodyShapeStart[b]; k < m_bodyShapeStart[b + 1]; k++)
                staticShapes.push_back(m_bodyShapeList[k]);
    // Kinematic bodies (mass 0, BODY_KINEMATIC) move, so their shapes ride in the dynamic
    // list (the static table is rebuilt only when dirty) after the dynamics' own.
    for (int b : scene.staticIdx)
        if ((int)m_hostProps.size() > b && (m_hostProps[b].flags & BODY_KINEMATIC))
            for (int k = m_bodyShapeStart[b]; k < m_bodyShapeStart[b + 1]; k++)
                dynShapes.push_back(m_bodyShapeList[k]);
    m_dynShapeCount = (int)dynShapes.size();
    m_staticShapeCount = (int)staticShapes.size();
    dynShapeIdx.upload(dynShapes.data(), m_dynShapeCount * sizeof(int));
    staticShapeIdx.upload(staticShapes.data(), m_staticShapeCount * sizeof(int));
    broad.markStaticDirty();

    // Jointed neighbours and ignoreCollision pairs. Without these, every rope link and
    // bridge plank collides with the body it is jointed to.
    m_ignoreCount = (int)scene.ignorePairs.size();
    if (m_ignoreCount > 0)
    {
        ignorePairs.resize(m_ignoreCount * sizeof(unsigned long long));
        ignorePairs.upload(scene.ignorePairs.data(), m_ignoreCount * sizeof(unsigned long long));
    }
    m_hostMass = scene.mass;
    m_cellSize = scene.cellSize;
    splitOversize(scene, dynShapes);
}

// The uniform grid is exact only while no body's AABB is wider than a cell, so its cell
// used to be the largest body's diagonal -- one dropped 4 m cube made every cell of a
// pile of 1 m boxes about 7 m wide and the pair emit 9x slower for the same pairs.
// Instead: the cell is sized from the normal bodies (the largest diagonal within 2x the
// median), and the few bodies above that are "oversized", kept out of the grid and
// tested brute force (csEmitOversizePairs). The overlap set is geometric truth and the
// pair sort canonicalises its order, so the result is identical to the legacy rule. A
// uniform scene has no oversized body and gets the legacy cell exactly; more than
// kMaxOversizeBodies of them (brute force is O(L*N)) falls back to the legacy rule.
void VkWorld::splitOversize(const SceneDesc &scene, const std::vector<int> &dyn)
{
    m_gridCell = m_cellSize;
    m_gridCount = m_dynShapeCount;
    m_oversizeCount = 0;
    const int nDyn = (int)dyn.size();
    if (m_dbg.legacyBroadphase || nDyn <= 0)
        return;

    // Same formula SceneBuilder::finalize uses for the legacy cell.
    std::vector<float> diag(nDyn);
    for (int t = 0; t < nDyn; t++)
        diag[t] = 2.0f * length(scene.shapeHalf[dyn[t]]) + 2.0f * COLLISION_MARGIN;
    std::vector<float> sorted(diag);
    std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
    const float base = sorted[sorted.size() / 2];

    float cell = 0.0f;
    for (float d : diag)
        if (d <= 2.0f * base)
            cell = std::max(cell, d);
    cell = std::max(cell, 1e-3f);

    std::vector<int> grid, over;
    grid.reserve(nDyn);
    for (int t = 0; t < nDyn; t++)
        (diag[t] > cell ? over : grid).push_back(dyn[t]);
    if (over.empty() || grid.empty() || (int)over.size() > kMaxOversizeBodies)
        return;

    m_gridCell = cell;
    m_gridCount = (int)grid.size();
    m_oversizeCount = (int)over.size();
    gridIdx.upload(grid.data(), grid.size() * sizeof(int));
    oversizeIdx.upload(over.data(), over.size() * sizeof(int));
}

void VkWorld::uploadScene(const SceneDesc &scene)
{
    m_bodyCount = scene.count;
    m_shapeCount = scene.shapeCount;
    m_dynCount = (int)scene.dynamicIdx.size();
    m_staticCount = (int)scene.staticIdx.size();
    m_jointCount = scene.jointCount;
    m_springCount = scene.springCount;
    m_jointSlots = m_jointCount + kMaxDragJoints;

    int bodyCap = std::max(m_bodyCount, 1);
    int shapeCap = std::max(m_shapeCount, 1);
    // Pairs, contacts and manifolds are per shape pair: sized from the larger of the two
    // counts (equal for a scene without addShape).
    const int pairBase = std::max(bodyCap, shapeCap);
    int jointCap = m_jointSlots;
    int springCap = std::max(m_springCount, 1);
    int pairCap = std::max(pairBase * 8, 64);
    int contactCap = std::max(pairBase * 16, 128);
    if (m_dbg.initialPairCapPerBody > 0)
        pairCap = std::max(m_dbg.initialPairCapPerBody * pairBase, 1);
    if (m_dbg.initialContactCap > 0)
        contactCap = m_dbg.initialContactCap;
    if (m_dbg.hardPairCap > 0)
        pairCap = std::min(pairCap, m_dbg.hardPairCap);
    if (m_dbg.hardContactCap > 0)
        contactCap = std::min(contactCap, m_dbg.hardContactCap);
    int manifoldCap = std::max(pairBase * 8, 64);
    reserve(bodyCap, jointCap, springCap, pairCap, contactCap, manifoldCap, shapeCap);

    // A new scene starts a new heap: the allocation is kept (it never shrinks), the contents
    // and the dedup table are not.
    m_geomPending.clear(); m_geomCopies.clear(); m_geomIndex.clear(); m_geomSceneOffset.clear();
    m_geomCursor = 0;
    registerSceneGeometry(scene);
    m_anyShapeRestitution = false;
    m_anyRolling = false;
    uploadBodyRange(scene, 0);
    uploadShapeRange(scene, 0);
    uploadIndexLists(scene);
    // Seeds are read by the explicit flood, so they must start clean (the capacity code's
    // resizeZeroed); nothing queued against a previous scene survives it either.
    wakeSeed.zero();
    m_mutBatches.clear(); m_mutIdx.clear(); m_mutData.clear();
    m_jwRecs.clear();
    m_seedsPending = false;
    activeIdx.upload(scene.dynamicIdx.data(), m_dynCount * sizeof(int));
    m_activeCount = m_dynCount;
    uploadJoints(scene);
    uploadSprings(scene);
    computeRenderGroups(scene);
}

void VkWorld::append(const SceneDesc &scene)
{
    assertIdle("VkWorld::append");
    const int oldBodies = m_bodyCount;
    const int oldShapes = m_shapeCount;
    const int oldJoints = m_jointCount;
    if (scene.count < oldBodies || scene.shapeCount < oldShapes || scene.jointCount < oldJoints)
        return; // not a grown copy of the live scene; build() is the call for that

    // Solver state of the joints that already exist: rebuilding the joint block (the
    // drag pool has to stay after the authored joints) must not reset their warm start
    // or heal the ones that have broken.
    std::vector<vec3> c0l(oldJoints), c0a(oldJoints), pl(oldJoints), pa(oldJoints), ll(oldJoints), la(oldJoints);
    std::vector<unsigned char> brk(oldJoints);
    if (oldJoints > 0)
    {
        jC0Lin.readback(c0l.data(), oldJoints * sizeof(vec3));
        jC0Ang.readback(c0a.data(), oldJoints * sizeof(vec3));
        jPenaltyLin.readback(pl.data(), oldJoints * sizeof(vec3));
        jPenaltyAng.readback(pa.data(), oldJoints * sizeof(vec3));
        jLambdaLin.readback(ll.data(), oldJoints * sizeof(vec3));
        jLambdaAng.readback(la.data(), oldJoints * sizeof(vec3));
        jBroken.readback(brk.data(), oldJoints * sizeof(unsigned char));
    }
    releaseDrag();
    m_brokenMirror = false; // the joint block is rebuilt below

    m_bodyCount = scene.count;
    m_shapeCount = scene.shapeCount;
    m_dynCount = (int)scene.dynamicIdx.size();
    m_staticCount = (int)scene.staticIdx.size();
    m_jointCount = scene.jointCount;
    m_springCount = scene.springCount;
    m_jointSlots = m_jointCount + kMaxDragJoints;

    // Grow with the contents kept. Pair/contact capacity is left to the step's own
    // overflow-retry growth, which preserves the contact cache.
    if (m_bodyCount > m_bodyCap)
    {
        const int oldCap = std::max(m_bodyCap, 1);
        const int cap = std::max(m_bodyCount, oldCap + oldCap / 2);
        Buffer *bodyBufs[] = {&pos, &rot, &initialLin, &initialAng, &inertialLin, &inertialAng,
                              &velLin, &velAng, &prevVelLin, &halfExtents, &shapeType, &moment,
                              &mass, &friction, &awake, &sleepTimer, &restPos, &restRot, &gyration, &props,
                              &dynamicIdx, &staticIdx, &activeIdx, &activeFlags, &touchCount,
                              &touchHash, &snapCount, &snapHash, &wakeRequest, &wakeSeed,
                              &islandSeed, &sleepCandidate, &sleepBlock};
        for (Buffer *b : bodyBufs)
            b->resizePreserve((b->size() / oldCap + 1) * cap);
        constraints.reserve(cap, std::max(m_manifoldCap, 1), m_jointCap + m_springCap);
        sizeCoarse(cap);
        m_bodyCap = cap;
    }
    if (m_shapeCount > m_shapeCap)
    {
        const int oldCap = std::max(m_shapeCap, 1);
        const int cap = std::max(m_shapeCount, oldCap + oldCap / 2);
        Buffer *shapeBufs[] = {&shpBody, &shpLocalPos, &shpLocalRot, &shpHalf, &shpType, &shpFriction,
                               &shpRestitution, &shpRolling, &shpCategory, &shpMask, &shpGroup, &shpFlags, &shpGeomOffset, &dynShapeIdx,
                               &staticShapeIdx, &gridIdx};
        for (Buffer *b : shapeBufs)
            b->resizePreserve((b->size() / oldCap + 1) * cap);
        broad.reserve(cap, std::max(m_pairCap, 1));
        m_shapeCap = cap;
    }
    if (m_jointSlots > m_jointCap || m_springCount > m_springCap)
    {
        const int jc = std::max(m_jointSlots, m_jointCap);
        const int sc = std::max(std::max(m_springCount, m_springCap), 1);
        auto vb = [](Buffer &b, size_t bytes) { b.resize((VkDeviceSize)bytes); };
        vb(jBodyA, jc * sizeof(int)); vb(jBodyB, jc * sizeof(int));
        vb(jRA, jc * sizeof(vec3)); vb(jRB, jc * sizeof(vec3));
        vb(jC0Lin, jc * sizeof(vec3)); vb(jC0Ang, jc * sizeof(vec3));
        vb(jPenaltyLin, jc * sizeof(vec3)); vb(jPenaltyAng, jc * sizeof(vec3));
        vb(jLambdaLin, jc * sizeof(vec3)); vb(jLambdaAng, jc * sizeof(vec3));
        vb(jStiffnessLin, jc * sizeof(float)); vb(jStiffnessAng, jc * sizeof(float));
        vb(jFracture, jc * sizeof(float)); vb(jFractureAng, jc * sizeof(float)); vb(jTorqueArm, jc * sizeof(float));
        vb(jRestAng, jc * sizeof(quat)); vb(jBroken, jc * sizeof(unsigned char));
        vb(jFrameA, jc * sizeof(quat)); vb(jFrameB, jc * sizeof(quat));
        vb(jAxisMode, jc * 6 * sizeof(unsigned char));
        vb(jAxisLo, jc * 6 * sizeof(float)); vb(jAxisHi, jc * 6 * sizeof(float));
        vb(jGeneric, jc * sizeof(unsigned char));
        vb(sBodyA, sc * sizeof(int)); vb(sBodyB, sc * sizeof(int));
        vb(sRA, sc * sizeof(vec3)); vb(sRB, sc * sizeof(vec3));
        vb(sRest, sc * sizeof(float)); vb(sStiffness, sc * sizeof(float));
        constraints.reserve(m_bodyCap, std::max(m_manifoldCap, 1), jc + sc);
        m_jointCap = jc;
        m_springCap = sc;
    }

    registerSceneGeometry(scene);
    uploadBodyRange(scene, oldBodies);
    uploadShapeRange(scene, oldShapes);
    uploadIndexLists(scene);
    if (m_bodyCount > oldBodies)
    {
        // The appended bodies' seeds (the capacity code's zeroRange): a grown buffer's tail
        // is uninitialised, and the old bodies' pending seeds must be kept.
        const std::vector<unsigned char> clean((size_t)(m_bodyCount - oldBodies), 0);
        wakeSeed.upload(clean.data(), clean.size(), (VkDeviceSize)oldBodies);
    }
    uploadJoints(scene);
    uploadSprings(scene);
    if (oldJoints > 0)
    {
        jC0Lin.upload(c0l.data(), oldJoints * sizeof(vec3));
        jC0Ang.upload(c0a.data(), oldJoints * sizeof(vec3));
        jPenaltyLin.upload(pl.data(), oldJoints * sizeof(vec3));
        jPenaltyAng.upload(pa.data(), oldJoints * sizeof(vec3));
        jLambdaLin.upload(ll.data(), oldJoints * sizeof(vec3));
        jLambdaAng.upload(la.data(), oldJoints * sizeof(vec3));
        jBroken.upload(brk.data(), oldJoints * sizeof(unsigned char));
    }

    // Every dynamic body active for the next step; the sleep pass trims the list again.
    activeIdx.upload(scene.dynamicIdx.data(), m_dynCount * sizeof(int));
    m_activeCount = m_dynCount;
    computeRenderGroups(scene);
}

void VkWorld::build(const SceneDesc &scene, const SolverParams &params)
{
    assertIdle("VkWorld::build");
    m_params = params;
    if (!m_init)
        init();
    if (m_bodyCap > 0)
        reset();
    uploadScene(scene);
}

namespace
{
using Clock = std::chrono::steady_clock;
float msSince(Clock::time_point t0, Clock::time_point t1)
{
    return std::chrono::duration<float, std::milli>(t1 - t0).count();
}
} // namespace

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
constexpr int kMaxCapAttempts = 5;   // the pass after this many capacity reruns commits clamped
constexpr int kCapSlotsBase = 2048;  // phase 4 <= ~670 dispatches, phase 5 ~100 + JP rounds
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
    CommandList &cmd = m_stepCmd; HostRingBuffer &ring = m_stepRing;
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
        auto growStage = [&](MappedBuffer &stage, VkDeviceSize bytes) {
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
            ma.b = b; ma.type = batch.type;
            ma.idx = mutIdxBuf.address() + (VkDeviceAddress)batch.offset * sizeof(int);
            ma.data = mutDataBuf.address() + (VkDeviceAddress)batch.dataOffset;
            ma.count = batch.count; ma.dt = params.dt; ma.seed = wakeSeed.address();
            launchMutateApply(cmd, ring, mutatePl, ma, batch.count);
            cmd.barrier();
        }
        m_mutBatches.clear(); m_mutIdx.clear(); m_mutData.clear();
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
        JointInitArgsGpu ja{}; ja.b = b; ja.j = j; ja.p = p;
        launchJointInit(cmd, ring, solverPl, ja, m_jointSlots);
        cmd.barrier();
    }
    IntegrateBodySoAGpu ib{};
    ib.pos = pos.address(); ib.rot = rot.address();
    ib.initialLin = initialLin.address(); ib.initialAng = initialAng.address();
    ib.inertialLin = inertialLin.address(); ib.inertialAng = inertialAng.address();
    ib.velLin = velLin.address(); ib.velAng = velAng.address(); ib.prevVelLin = prevVelLin.address();
    ib.mass = mass.address(); ib.awake = b.awake;
    ib.count = (uint32_t)m_bodyCount;
    IntegrateArgsGpu ia{}; ia.b = ib; ia.p.dt = params.dt; ia.p.gravityX = params.gravity.x; ia.p.gravityY = params.gravity.y; ia.p.gravityZ = params.gravity.z;
    if (m_anyProps)
    {
        IntegratePropsArgsGpu pa{}; pa.b = ib; pa.p = ia.p; pa.props = props.address();
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
        m.key = cur.key; m.bodyA = cur.bodyA; m.bodyB = cur.bodyB; m.basis = cur.basis;
        m.shapeA = cur.shapeA; m.shapeB = cur.shapeB;
        m.friction = cur.friction; m.contactOffset = cur.contactOffset; m.contactCount = cur.contactCount;
        // count/contactTotal are host bookkeeping no shader reads (every consumer walks
        // `slots`, skipping contactCount <= 0): the record-time capacities stand in.
        m.count = slots; m.capacity = slots;
        m.feature = cur.feature; m.rA = cur.rA; m.rB = cur.rB; m.C0 = cur.C0;
        m.penalty = cur.penalty; m.lambda = cur.lambda; m.stick = cur.stick;
        m.contactTotal = m_contactCap; m.contactCapacity = m_contactCap;
        wd = WakeDetectCall{}; fc = WakeFloodContactsArgsGpu{}; fj = WakeFloodJointsArgsGpu{};
        fi = WakeFloodJointsArgsGpu{}; merge = WakeSeedMergeArgsGpu{}; mergeIsland = WakeSeedMergeArgsGpu{};
        isArgs = WakeIslandSeedArgsGpu{};
        floodDone = true; floodRec = islandRec = 0; floodBound = islandBound = 0;
        phase4Done = !sleepOn; p4Fresh = true;
    if (sleepOn)
    {
        wd.clear.dyn = dynamicIdx.address(); wd.clear.dynCount = m_dynCount;
        wd.clear.count = touchCount.address(); wd.clear.hash = touchHash.address();
        wd.accum.b = b; wd.accum.m = m; wd.accum.slots = slots;
        wd.accum.count = touchCount.address(); wd.accum.hash = touchHash.address();
        wd.accumJ.b = b; wd.accumJ.j = j; wd.accumJ.sp = sp;
        wd.accumJ.count = touchCount.address(); wd.accumJ.hash = touchHash.address();
        wd.request.b = b; wd.request.m = m; wd.request.slots = slots;
        wd.request.wakeLinVel = params.sleepLinVel; wd.request.wakeAngVel = params.sleepAngVel;
        wd.request.request = wakeRequest.address();
        wd.requestJ.b = b; wd.requestJ.j = j; wd.requestJ.sp = sp; wd.requestJ.request = wakeRequest.address();
        wd.lostSupport.b = b; wd.lostSupport.dyn = dynamicIdx.address(); wd.lostSupport.dynCount = m_dynCount;
        wd.lostSupport.count = touchCount.address(); wd.lostSupport.hash = touchHash.address();
        wd.lostSupport.snapCount = snapCount.address(); wd.lostSupport.snapHash = snapHash.address();
        wd.lostSupport.request = wakeRequest.address();
        wd.dynCount = m_dynCount; wd.slots = slots; wd.edgeCount = edgeCount;

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
        fc.b = b; fc.m = m; fc.slots = slots; fc.seed = wakeSeed.address();
        fj.b = b; fj.j = j; fj.sp = sp; fj.seed = wakeSeed.address();
        fi = fj;
        fi.seed = islandSeed.address();
        merge.dyn = dynamicIdx.address(); merge.dynCount = m_dynCount;
        merge.seed = wakeSeed.address(); merge.request = wakeRequest.address();
        mergeIsland = merge;
        mergeIsland.seed = islandSeed.address();
        isArgs.b = b; isArgs.dyn = dynamicIdx.address(); isArgs.dynCount = m_dynCount;
        isArgs.request = wakeRequest.address(); isArgs.seed = islandSeed.address(); isArgs.any = wakeAny.address();

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

    cmr.bodyA = m.bodyA; cmr.bodyB = m.bodyB; cmr.contactCount = m.contactCount; cmr.slots = slots;
    cjr.bodyA = j.bodyA; cjr.bodyB = j.bodyB; cjr.broken = j.broken; cjr.count = m_jointSlots;
    csr.bodyA = sp.bodyA; csr.bodyB = sp.bodyB; csr.count = m_springCount;
}

void VkWorld::recordAndSubmitPass()
{
    StepCtx &S = m_step;
    const SolverParams &params = m_params;
    CommandList &cmd = m_stepCmd; HostRingBuffer &ring = m_stepRing;
    auto &b = S.b;
    auto &j = S.j;
    auto &sp = S.sp;
    auto &p = S.p;
    auto &sleepOn = S.sleepOn;
    auto &npb = S.npb;
    auto &npp = S.npp;
    auto &edgeCount = S.edgeCount;
    auto &tsStart = S.tsStart;
    auto &tsEnd = S.tsEnd;
    auto &wd = S.wd;
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
    auto &jpRounds = S.jpRounds;
    auto &colorBound = S.colorBound;
    auto &recordBuildMs = S.recordBuildMs;
    auto &hostSubmitMs = S.hostSubmitMs;
    auto &hostRecordMs = S.hostRecordMs;
    auto &tSolve = S.tSolve;
    auto &frontPending = S.frontPending;
    auto &phase1Done = S.phase1Done;
    auto &capAttempt = S.capAttempt;
    auto &fc = S.fc;
    auto &fj = S.fj;
    auto &fi = S.fi;
    auto &merge = S.merge;
    auto &mergeIsland = S.mergeIsland;
    auto &isArgs = S.isArgs;
    auto &pass = S.pass;
    const Clock::time_point tPass = Clock::now();
    if (pass > 64)
    {
        fprintf(stderr, "[avbd_vk] step: %d reruns without converging\n", pass);
        abort();
    }
    tSolve = Clock::now();
    // Everything that can reallocate is grown before the first recorded command.
    // +25: the tail's fixed dispatches.
    const int coarseIters = std::min(std::max(params.coarseIterations, 0), params.iterations);
    const int gateSlots = params.iterations * (colorBound + 2) + 25
                          + 24 * coarseIters + 2; // coarse: reset, assign, 8 parities x 3
    // Reserved for the worst-case colour / JP-round bounds, not this pass's: a bound
    // that grows between steps must not allocate.
    m_gate.reserve(params.iterations * (AVBD_MAX_COLORS_VK + 2) + 24 + 128 + 24 * params.iterations + 2);
    m_gateCap.reserve(kCapSlotsBase + (int)VkConstraints::kMaxJpRounds);
    constraints.prepare(radix, m_dynCount);
    beginStepRecording();
    cmd.writeTimestamp(tsStart);
    cmd.fill(stepState, 0u);

    const bool front = frontPending;
    S.front = front;
    bool recordedP1 = false;
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
        bo.gridIdx = &gridIdx; bo.gridCount = m_gridCount;
        bo.oversizeIdx = &oversizeIdx; bo.oversizeCount = m_oversizeCount;
        broad.record(cmd, ring, bphPl, primPl, radix, pos.address(), rot.address(), broadphaseShapes(),
                     dynShapeIdx, m_dynShapeCount, staticShapeIdx, m_staticShapeCount, m_gridCell,
                     &ignorePairs, m_ignoreCount, &bo);
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

    const bool phase4ThisPass = !phase4Done;
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
            recordWakeFloodRounds(cmd, ring, mutatePl, fc, fj, slots, edgeCount, floodChanged.address(),
                                  floodRec, more);
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
                islandMore = p4First ? islandBound : std::max(islandBound, std::min(2 * islandRec, (int)kIslandMaxRounds));
            }
            else
            {
                islandFirst = islandRec;
                islandMore = std::min(islandRec, kIslandMaxRounds - islandRec);
            }
            recordWakeFloodRounds(cmd, ring, mutatePl, fc, fi, 0, edgeCount, wakeChanged.address(),
                                  islandFirst, islandMore);
            islandRec = islandFirst + islandMore;
            mergeIsland.gate = islandRec < kIslandMaxRounds
                                   ? wakeChanged.address() + (VkDeviceAddress)(islandRec - 1) * sizeof(int32_t)
                                   : 0;
            launchWakeSeedMerge(cmd, ring, mutatePl, mergeIsland, m_dynCount);
            cmd.barrier();
        }

        WakeApplyArgsGpu wa{};
        wa.b = b; wa.dyn = dynamicIdx.address(); wa.dynCount = m_dynCount; wa.request = wakeRequest.address();
        launchWakeApply(cmd, ring, wakePl, wa, m_dynCount);
        cmd.barrier();

        ActiveFlagsArgsGpu af{};
        af.b = b; af.dyn = dynamicIdx.address(); af.dynCount = m_dynCount; af.flags = activeFlags.address();
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
            cmd.copyToHost(floodChanged, wakeReadout, (VkDeviceSize)floodRec * sizeof(int32_t), 0,
                           sizeof(int32_t));
        if (edgeCount > 0)
            cmd.copyToHost(wakeChanged, wakeReadout, (VkDeviceSize)islandRec * sizeof(int32_t), 0,
                           (VkDeviceSize)(1 + kFloodMaxRounds) * sizeof(int32_t));
    }

    // --- Phase 5: constraint graph build, recorded into this submission ----------
    // m_dynCount is only the bound; the live count is read from activeCounter on device.
    const Clock::time_point tRec = Clock::now();
    const int recordedJp = constraints.recordBuild(cmd, ring, csPl, primPl, radix, cbr, cmr, cjr, csr, activeIdx,
                                                   m_dynCount, activeCounter.address(), jpRounds);
    if (m_dynCount > 0 && recordedJp > 0 && recordedJp < VkConstraints::kMaxJpRounds)
        launchStepFlagIfSet(cmd, ring, solverPl, constraints.remainingSlotAddr(recordedJp - 1),
                            stepState.address());
    if (colorBound < AVBD_MAX_COLORS_VK)
        launchStepColorBound(cmd, ring, solverPl, constraints.colorStartAddr(), constraints.colorEndAddr(),
                             stepState.address(), colorBound, AVBD_MAX_COLORS_VK);
    cmd.barrier();
    recordBuildMs = msSince(tRec, Clock::now());
    if (front)
        cmd.setGate(nullptr);

    VkConstraintGraph gRaw = constraints.graph();
    ConstraintGraphGpu g{};
    g.key = gRaw.key; g.isA = gRaw.isA; g.start = gRaw.start; g.end = gRaw.end;

    const bool restitutionOn = (m_anyRestitution || m_anyShapeRestitution) && slots > 0;
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

    // --- Phase 6: iteration loop --------------------------------------------------
    // GPU timestamps around each iteration's primal sweep and dual pass: slot 0 at
    // the start, 2it+1 after the primal colours, 2it+2 after the duals. Primal and
    // dual are summed over the iterations; iterMs is the whole loop.
    cmd.writeTimestamp(3);
    const bool withForces = edgeCount > 0;
    const bool warp4 = params.primalLanes > 1;
    // Small-world fusion (vk_small_world.h): the whole loop as one single-workgroup dispatch.
    // Decided on the host from counts known at record time. Anything the fused kernel does not
    // run (the coarse correction, thread-per-body primal) keeps the
    // multi-dispatch path below, as does a world past the workgroup's reach.
    const bool fusedIter = smallWorldEnabled() && warp4 && params.iterations > 0 && coarseIters == 0 &&
                           m_dynCount > 0 &&
                           m_dynCount <= kFusedIterMaxBodies && slots <= kFusedIterMaxManifoldSlots &&
                           m_jointSlots <= kFusedIterMaxJointSlots;
    m_lastFusedIter = fusedIter;
    if (fusedIter)
        m_fusedIterPasses++;
    if (!fusedIter)
        cmd.fill(primalIndirect, 0u);
    m_gate.begin(cmd, ring, solverPl.gateDispatchArgs, stepState.address(), gateSlots);
    cmd.setGate(&m_gate);

    // The colour ranges stay on device: the primal sweep reads its range and its
    // group count from them, so the only host input left is how many colours to
    // record. An empty colour dispatches zero groups.
    const VkDeviceAddress cStart = constraints.colorStartAddr();
    const VkDeviceAddress cEnd = constraints.colorEndAddr();
    if (!fusedIter)
        launchPrimalDispatchArgs(cmd, ring, solverPl, cStart, cEnd, primalIndirect, colorBound, warp4);
    if (!fusedIter)
        cmd.barrierIndirect();

    CoarseArgsGpu ca{};
    if (coarseIters > 0)
    {
        ca.pa.b = b; ca.pa.g = g; ca.pa.m = m; ca.pa.joints = j; ca.pa.springs = sp; ca.pa.p = p;
        ca.active = activeIdx.address(); ca.activeCount = activeCounter.address();
        ca.aggOf = coarseAggOf.address(); ca.keys = coarseKeys.address();
        ca.acc = coarseAcc.address(); ca.delta = coarseDelta.address(); ca.stats = coarseStats.address();
        ca.maxActive = m_dynCount; ca.bodyCount = m_bodyCount; ca.tableSize = m_coarseTable;
        ca.cell = params.coarseCell > 0.0f ? params.coarseCell : 4.0f * std::max(m_cellSize, 1e-3f);
        ca.damping = params.coarseDamping; ca.withForces = withForces ? 1 : 0;
        cmd.fill(coarseStats, 0u);
        launchCoarse(cmd, ring, solverPl.coarseReset, ca, std::max(m_coarseTable, m_bodyCount), 256);
        cmd.barrier();
        launchCoarse(cmd, ring, solverPl.coarseAssign, ca, m_dynCount, 256);
        cmd.barrier();
    }
    if (fusedIter)
    {
        PrimalArgsGpu pa{};
        pa.b = b; pa.g = g; pa.m = m; pa.joints = j; pa.springs = sp;
        pa.colorBodies = constraints.colorBodies().address();
        pa.p = p;
        pa.colorStart = cStart; pa.colorEnd = cEnd;
        pa.shapeRolling = m_anyRolling ? shpRolling.address() : 0;
        DualArgsGpu da{}; da.b = b; da.m = m; da.slots = slots; da.p = p;
        JointDualArgsGpu jd{}; jd.b = b; jd.j = j; jd.p = p;
        launchIterateSmall(cmd, ring, solverPl, pa, da, jd, params.iterations, colorBound, withForces,
                           m_jointSlots);
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
            pa.b = b; pa.g = g; pa.m = m; pa.joints = j; pa.springs = sp;
            pa.colorBodies = constraints.colorBodies().address();
            pa.p = p;
            pa.colorStart = cStart; pa.colorEnd = cEnd; pa.color = c;
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
        DualArgsGpu da{}; da.b = b; da.m = m; da.slots = slots; da.p = p;
        launchDual(cmd, ring, solverPl, da, slots);
        cmd.barrier();
        {
            JointDualArgsGpu jd{}; jd.b = b; jd.j = j; jd.p = p;
            launchJointDual(cmd, ring, solverPl, jd, m_jointSlots);
            cmd.barrier();
        }
        cmd.writeTimestamp((uint32_t)(3 + 2 * it + 2));
    }

    // --- Phase 7: velocity, explosion guard, kill-fallen -------------------------
    IntegrateBodySoAGpu ib{};
    ib.pos = pos.address(); ib.rot = rot.address();
    ib.initialLin = initialLin.address(); ib.initialAng = initialAng.address();
    ib.inertialLin = inertialLin.address(); ib.inertialAng = inertialAng.address();
    ib.velLin = velLin.address(); ib.velAng = velAng.address(); ib.prevVelLin = prevVelLin.address();
    ib.mass = mass.address(); ib.awake = awake.address();
    ib.count = (uint32_t)m_bodyCount;
    VelocityArgsGpu va{}; va.b = ib; va.p.dt = params.dt; va.p.gravityX = params.gravity.x; va.p.gravityY = params.gravity.y; va.p.gravityZ = params.gravity.z; va.p.spinExact = params.exactSpinVelocity ? 1u : 0u;
    if (m_anyProps)
    {
        VelocityPropsArgsGpu pa{}; pa.b = ib; pa.p = va.p; pa.props = props.address();
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
        ra.pos = b.pos; ra.rot = b.rot; ra.prevVelLin = b.prevVelLin;
        ra.bodyA = m.bodyA; ra.bodyB = m.bodyB; ra.shapeA = m.shapeA; ra.shapeB = m.shapeB;
        ra.basis = m.basis; ra.contactOffset = m.contactOffset; ra.contactCount = m.contactCount;
        ra.rA = m.rA; ra.rB = m.rB; ra.C0 = m.C0; ra.lambda = m.lambda;
        ra.shapeFlags = shpFlags.address();
        ra.counter = reportCounter.address(); ra.records = reportRecords.address();
        ra.capacity = (uint32_t)m_reportCap; ra.slots = (uint32_t)slots;
        launchContactReport(cmd, ring, reportPl, ra);
        cmd.barrier();
        // Nothing after this writes the report buffers, so the copies can ride here.
        cmd.copyToHost(reportCounter, reportCountReadout, sizeof(uint32_t));
        cmd.copyToHost(reportRecords, reportRecReadout, (VkDeviceSize)m_reportCap * sizeof(ContactReportRecordGpu));
    }

    if (m_dynCount > 0)
    {
        ExplosionGuardArgsGpu ega{};
        ega.b = ib; ega.dyn = dynamicIdx.address(); ega.dynCount = m_dynCount;
        ega.sleepTimer = sleepTimer.address(); ega.restPos = restPos.address(); ega.restRot = restRot.address();
        launchExplosionGuard(cmd, ring, integratePl, ega, m_dynCount);
        cmd.barrier();

        if (params.enableContinuous && m_dynShapeCount > 0)
        {
            // Continuous collision (ccd.slang): pose clamp to the earliest time of impact.
            CcdArgsGpu ca{};
            ca.pos = ib.pos; ca.rot = ib.rot; ca.velLin = ib.velLin; ca.initLin = ib.initialLin; ca.initAng = ib.initialAng;
            ca.mass = mass.address(); ca.awake = awake.address(); ca.sleepTimer = sleepTimer.address();
            ca.props = propsAddress();
            ca.shpBody = shpBody.address(); ca.shpLocalPos = shpLocalPos.address();
            ca.shpLocalRot = shpLocalRot.address(); ca.shpHalf = shpHalf.address();
            ca.shpType = shpType.address(); ca.shpFlags = shpFlags.address();
            ca.shpCategory = shpCategory.address(); ca.shpMask = shpMask.address();
            ca.shpGroup = shpGroup.address();
            ca.dynShape = dynShapeIdx.address(); ca.staticShape = staticShapeIdx.address();
            ca.ignore = ignorePairs.address(); ca.ignoreCount = (uint32_t)m_ignoreCount;
            ca.geomHeap = geomHeap.address(); ca.geomOffset = shpGeomOffset.address();
            ca.geomHeapBytes = geomHeap.size();
            ca.dynCount = (uint32_t)m_dynShapeCount; ca.staticCount = (uint32_t)m_staticShapeCount;
            ccd.record(cmd, ring, ca, m_bodyCap, m_anyBullet);
        }

        if (params.killFallenEnabled)
        {
            KillFallenArgsGpu kfa{};
            kfa.b = ib; kfa.dyn = dynamicIdx.address(); kfa.dynCount = m_dynCount; {
                // Kill height is measured along up = -normalize(gravity); zero gravity falls back to +Y.
                const float gl = std::sqrt(params.gravity.x * params.gravity.x + params.gravity.y * params.gravity.y + params.gravity.z * params.gravity.z);
                kfa.killHeight = params.killHeight;
                kfa.upX = gl > 0.0f ? -params.gravity.x / gl : 0.0f;
                kfa.upY = gl > 0.0f ? -params.gravity.y / gl : 1.0f;
                kfa.upZ = gl > 0.0f ? -params.gravity.z / gl : 0.0f;
            }
            kfa.sleepTimer = sleepTimer.address(); kfa.restPos = restPos.address(); kfa.restRot = restRot.address();
            launchKillFallen(cmd, ring, integratePl, kfa, m_dynCount);
            cmd.barrier();
        }
    }

    // --- Phase 8: sleep commit --------------------------------------------------
    if (sleepOn)
    {
        SleepCommitCall sc{};
        sc.fracture.b = b; sc.fracture.j = j; sc.fracture.ratio = params.sleepFracture;
        // csSleepBlockFracture stores block[body] for every joint loaded past sleepFracture of
        // its break force; a null block here lost the device the first time one was (a grab or
        // an impact on a jointed structure).
        sc.fracture.block = sleepBlock.address();
        sc.drag.dragIdx = 0; sc.drag.dragCount = 0; sc.drag.block = sleepBlock.address();
        sc.jointCount = m_jointSlots; sc.dragCount = 0; sc.dynCount = m_dynCount; sc.solveCount = m_dynCount;
        sc.commit.b = b; sc.commit.g = g; sc.commit.m = m; sc.commit.j = j; sc.commit.sp = sp;
        sc.commit.active = activeIdx.address(); sc.commit.activeCount = m_dynCount;
        sc.commit.activeCountPtr = activeCounter.address();
        sc.commit.candidate = sleepCandidate.address();
        sc.commit.touchCount = touchCount.address(); sc.commit.touchHash = touchHash.address();
        sc.commit.snapCount = snapCount.address(); sc.commit.snapHash = snapHash.address();
        sc.mark.b = b; sc.mark.dyn = dynamicIdx.address(); sc.mark.dynCount = m_dynCount;
        sc.mark.candidate = sleepCandidate.address(); sc.mark.frames = params.sleepFrames;
        sc.mark.linVel = params.sleepLinVel; sc.mark.angVel = params.sleepAngVel; sc.mark.disp = params.sleepDisp;
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
            S.pass++;
            return false;
        }
        narrow.finish(broad.pairCount());
        frontPending = false;
        if (broad.overflowed())
        {
            result.ok = false; result.truncated = true;
            result.failPhase = "broadphase"; result.failResource = "pairs";
            result.required = broad.producedPairs(); result.capacity = m_pairCap;
        }
        else if (narrow.overflowed())
        {
            result.ok = false; result.truncated = true;
            result.failPhase = "narrowphase"; result.failResource = "contacts";
            result.required = narrow.requiredContacts(); result.capacity = m_contactCap;
        }
        else if (narrow.hullRecorded() && narrow.hullPairs() > narrow.hullCapacity())
        {
            result.ok = false; result.truncated = true;
            result.failPhase = "narrowphase"; result.failResource = "hullPairs";
            result.required = narrow.hullPairs(); result.capacity = narrow.hullCapacity();
        }
        else if (narrow.meshRecorded() && narrow.childOverflowed())
        {
            result.ok = false; result.truncated = true;
            result.failPhase = "narrowphase"; result.failResource = "meshChildren";
            result.required = narrow.childRaw(); result.capacity = narrow.childCapacity();
        }
    }

    if (phase4ThisPass)
    {
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
    constraints.finishRecorded();
    m_lastColors = constraints.colorCount();

    if (!stepFlag[0])
    {
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
        return true;
    }

    // Flagged: grow whichever bound was too small and go again.
    const bool jpShort = !constraints.jpConverged() && jpRounds < VkConstraints::kMaxJpRounds;
    const bool colorShort = colorBound < AVBD_MAX_COLORS_VK && constraints.colorCount() > colorBound;
    if (phase4Done && !jpShort && !colorShort)
    {
        fprintf(stderr, "[avbd_vk] step: flag set but phase 4, JP and colours are all within bounds\n");
        abort();
    }
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
        jpRounds = std::min(2 * jpRounds, (int)VkConstraints::kMaxJpRounds);
    if (colorShort)
        colorBound = std::min(constraints.colorCount() + 2, (int)AVBD_MAX_COLORS_VK);
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
    b.pos = pos.address(); b.rot = rot.address();
    b.initialLin = initialLin.address(); b.initialAng = initialAng.address();
    b.inertialLin = inertialLin.address(); b.inertialAng = inertialAng.address();
    b.velLin = velLin.address(); b.velAng = velAng.address(); b.prevVelLin = prevVelLin.address();
    b.halfExtents = halfExtents.address(); b.shapeType = shapeType.address();
    b.moment = moment.address(); b.mass = mass.address(); b.friction = friction.address();
    b.awake = awake.address(); b.sleepTimer = sleepTimer.address();
    b.restPos = restPos.address(); b.restRot = restRot.address();
    b.gyration = gyration.address();
    b.props = propsAddress();
    b.count = (uint32_t)m_bodyCount;

    FullJointSoAGpu &j = S.j;
    j.bodyA = jBodyA.address(); j.bodyB = jBodyB.address();
    j.rA = jRA.address(); j.rB = jRB.address();
    j.C0Lin = jC0Lin.address(); j.C0Ang = jC0Ang.address();
    j.penaltyLin = jPenaltyLin.address(); j.penaltyAng = jPenaltyAng.address();
    j.lambdaLin = jLambdaLin.address(); j.lambdaAng = jLambdaAng.address();
    j.stiffnessLin = jStiffnessLin.address(); j.stiffnessAng = jStiffnessAng.address();
    j.fracture = jFracture.address(); j.fractureAng = jFractureAng.address(); j.torqueArm = jTorqueArm.address();
    j.restAng = jRestAng.address(); j.broken = jBroken.address();
    j.frameA = jFrameA.address(); j.frameB = jFrameB.address();
    j.axisMode = jAxisMode.address(); j.axisLo = jAxisLo.address(); j.axisHi = jAxisHi.address();
    j.generic = jGeneric.address();
    j.count = (uint32_t)m_jointSlots;

    S.sp = springView();

    S.sleepOn = params.sleepEnabled && params.sleepFrames > 0 && m_dynCount > 0;

    // Phases 2+3 inputs (see recordAndSubmitPass for the capacity-rerun contract).
    S.npb.pos = pos.address(); S.npb.rot = rot.address();
    S.npb.shapeBody = shpBody.address(); S.npb.localPos = shpLocalPos.address();
    S.npb.localRot = shpLocalRot.address(); S.npb.halfExtents = shpHalf.address();
    S.npb.shapeType = shpType.address(); S.npb.friction = shpFriction.address();
    S.npb.flags = shpFlags.address();
    S.npb.mass = mass.address(); S.npb.moment = moment.address();
    S.npb.velLin = velLin.address(); S.npb.velAng = velAng.address();
    S.npb.props = propsAddress();
    S.npb.category = shpCategory.address(); S.npb.mask = shpMask.address(); S.npb.group = shpGroup.address();
    S.hasHulls = m_hullShapes > 0 && geomHeap.size() > 0;
    S.nph.geomHeap = geomHeap.address(); S.nph.geomOffset = shpGeomOffset.address();
    S.nph.geomHeapBytes = geomHeap.size();
    S.hasMeshes = m_meshShapes > 0 && geomHeap.size() > 0;
    S.nmh.geom = S.nph;
    S.nmh.prim = &primPl;
    S.nmh.radix = &radix;
    S.npp.dt = params.dt; S.npp.alpha = params.alpha; S.npp.gamma = params.gamma;
    S.npp.autoKStart = params.autoKStart; S.npp.dualDamping = params.dualDamping; S.npp.dualDampVel = params.dualDampVel;
    S.edgeCount = m_jointSlots + m_springCount;

    // Timestamp layout: see recordAndSubmitPass.
    S.iters = std::max(params.iterations, 0);
    S.tsStart = (uint32_t)(2 * S.iters + 4); S.tsEnd = (uint32_t)(2 * S.iters + 5);
    if (S.tsEnd + 1 > m_stepCmd.maxTimestamps())
    {
        m_stepCmd.destroy();
        m_stepCmd.init(dev, S.tsEnd + 1);
    }
    S.frontMs = -1.0f;

    S.npp.contactCap = effectiveContactCap(params.contactCap);

    S.cbr.mass = mass.address(); S.cbr.awake = b.awake;
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

void VkWorld::zeroDerivedState() {}

void VkWorld::computeRenderGroups(const SceneDesc &scene)
{
    // solver_impl's recomputeShapeRegions(): groups 0..2 are the shape classes and
    // always exist; a body's authored group (>= 0) overrides its shape. Every group's
    // region is sized for every shape that could land in it, so the kernel's atomic can
    // never run past its region and culling only ever compacts within one. Instances are
    // per shape, so the group is per shape: the owning body's authored group, else the
    // shape's own class.
    const int n = scene.shapeCount;
    std::vector<int> eff(n);
    int groups = 3;
    for (int i = 0; i < n; i++)
    {
        const int body = scene.shapeBody[i];
        const int authored = body < (int)scene.renderGroup.size() ? scene.renderGroup[body] : -1;
        const unsigned char s = scene.shapeKind[i];
        eff[i] = authored >= 0 ? authored : (s < 3 ? (int)s : 0);
        groups = std::max(groups, eff[i] + 1);
    }
    std::vector<int> count(groups, 0);
    for (int i = 0; i < n; i++)
        count[eff[i]]++;
    m_groupBase.assign(groups, 0);
    for (int k = 1; k < groups; k++)
        m_groupBase[k] = m_groupBase[k - 1] + count[k - 1];
    m_groupVisible.assign(groups, 0);

    if (groupIdx.size() < (VkDeviceSize)std::max(n, 1) * sizeof(int))
        groupIdx.resize((VkDeviceSize)std::max(n, 1) * sizeof(int));
    if (n > 0)
        groupIdx.upload(eff.data(), n * sizeof(int));
    if (shapeBaseBuf.size() < (VkDeviceSize)groups * sizeof(int32_t))
    {
        shapeBaseBuf.resize(groups * sizeof(int32_t));
        shapeVisBuf.resize(groups * sizeof(int32_t));
    }
    shapeBaseBuf.upload(m_groupBase.data(), groups * sizeof(int32_t));
}

FullSpringSoAGpu VkWorld::springView() const
{
    FullSpringSoAGpu sp{};
    sp.bodyA = sBodyA.address(); sp.bodyB = sBodyB.address();
    sp.rA = sRA.address(); sp.rB = sRB.address();
    sp.rest = sRest.address(); sp.stiffness = sStiffness.address();
    sp.count = (uint32_t)m_springCount;
    return sp;
}

FullManifoldSoAGpu VkWorld::manifoldView() const
{
    FullManifoldSoAGpu m{};
    const VkManifoldSet &cur = narrow.manifolds();
    m.key = cur.key; m.bodyA = cur.bodyA; m.bodyB = cur.bodyB; m.basis = cur.basis;
    m.shapeA = cur.shapeA; m.shapeB = cur.shapeB;
    m.friction = cur.friction; m.contactOffset = cur.contactOffset; m.contactCount = cur.contactCount;
    m.count = narrow.activeManifolds(); m.capacity = narrow.manifoldSlots();
    m.feature = cur.feature; m.rA = cur.rA; m.rB = cur.rB; m.C0 = cur.C0;
    m.penalty = cur.penalty; m.lambda = cur.lambda; m.stick = cur.stick;
    m.contactTotal = narrow.contactTotal(); m.contactCapacity = m_contactCap;
    return m;
}

void VkWorld::writeInstances(const RenderInstancesParams &params)
{
    assertIdle("VkWorld::writeInstances");
    VkDeviceSize need = (VkDeviceSize)std::max(m_shapeCount, 1) * sizeof(BodyInstanceGpu);
    if (instances.size() < need)
        instances.resize(need, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    std::fill(m_groupVisible.begin(), m_groupVisible.end(), 0);
    if (m_bodyCount == 0)
    {
        m_instanceCount = 0;
        return;
    }

    const FullBodySoAGpu b = bodyView();
    WriteInstancesArgsGpu args{};
    args.b = b;
    args.out = instances.address();
    args.maxInstances = m_shapeCount;
    args.s = shapeView();
    args.ov = params.overlay;

    // Overlay inputs that live in this VkWorld's own buffers -- the demo cannot pass
    // them, the same way the instance writer gathers them solver-internally on the reference: graphColor
    // is the constraint graph's per-body Jones-Plassmann color, and the contact degree is
    // the sleep logic's touch fingerprint, only maintained while the rest test is running.
    // render.slang null-guards every one, so a mode whose input is absent falls through
    // to plain instance colour.
    const int mode = params.overlay.mode;
    args.ov.colorCount = constraints.colorCount();
    args.ov.graphColor = mode == OVERLAY_GRAPH_COLOR ? constraints.bodyColor().address() : 0;
    args.ov.degree = mode == OVERLAY_DEGREE && params.overlay.sleepFrames > 0.0f ? touchCount.address() : 0;
    args.ov.load = 0;
    args.ov.push = args.ov.pull = args.ov.stress = args.ov.cone = args.ov.penetration = args.ov.jointDegree = 0;

    const VkDeviceSize perBody = (VkDeviceSize)m_bodyCount * sizeof(float);
    auto ensure = [perBody](Buffer &buf) {
        if (buf.size() < perBody)
            buf.resize(perBody);
        return buf.address();
    };

    CommandList cmd; HostRingBuffer ring;
    cmd.init(dev); ring.init(dev, 1 << 16);
    cmd.begin();

    const int slots = narrow.manifoldSlots();
    if (mode == OVERLAY_LOAD)
    {
        OverlayLoadClearArgsGpu lc{};
        lc.load = ensure(overlayLoad);
        lc.n = m_bodyCount;
        launchOverlayLoadClear(cmd, ring, renderPl, lc, m_bodyCount);
        cmd.barrier();
        if (slots > 0)
        {
            OverlayLoadAccumArgsGpu la{};
            la.b = b; la.m = manifoldView(); la.slots = slots; la.load = lc.load;
            launchOverlayLoadAccum(cmd, ring, renderPl, la, slots);
            cmd.barrier();
        }
        args.ov.load = lc.load;
    }

    // Structural overlays: a reduction from the constraint arrays onto the bodies, run
    // only for the overlay selected and only over the buffers it reads (the instance writer).
    const bool wantPush = mode == OVERLAY_JOINT_LOAD || mode == OVERLAY_AXIAL_LOAD;
    const bool wantStress = mode == OVERLAY_JOINT_STRESS;
    const bool wantCone = mode == OVERLAY_FRICTION;
    const bool wantPen = mode == OVERLAY_PENETRATION;
    const bool wantJointDeg = mode == OVERLAY_JOINT_DEGREE;
    if (wantPush || wantStress || wantCone || wantPen || wantJointDeg)
    {
        OverlayStatsGpu st{};
        if (wantPush)
        {
            st.push = ensure(overlayPush);
            st.pull = ensure(overlayPull);
        }
        if (wantStress)
            st.stress = ensure(overlayStress);
        if (wantCone)
            st.cone = ensure(overlayCone);
        if (wantPen)
            st.penetration = ensure(overlayPen);
        if (wantJointDeg)
            st.jointDegree = ensure(overlayJointDeg);

        OverlayStatsClearArgsGpu sc{};
        sc.s = st; sc.n = m_bodyCount;
        launchOverlayStatsClear(cmd, ring, renderPl, sc, m_bodyCount);
        cmd.barrier();

        // Contacts feed push, cone and penetration; joints feed push, pull, stress and
        // degree. Each launch is skipped when nothing it produces was asked for.
        if (slots > 0 && (st.push || st.cone || st.penetration))
        {
            OverlayStatsContactsArgsGpu ca{};
            ca.m = manifoldView(); ca.slots = slots; ca.s = st;
            launchOverlayStatsContacts(cmd, ring, renderPl, ca, slots);
            cmd.barrier();
        }
        // Only authored joints: the drag slots are a UI affordance, and counting them
        // would put a permanent +1 on whatever the mouse is holding.
        const int forceCount = m_jointCount + m_springCount;
        if (forceCount > 0 && (st.push || st.pull || st.stress || st.jointDegree))
        {
            OverlayStatsForcesArgsGpu fa{};
            fa.b = b; fa.j = jointView(); fa.sp = springView(); fa.jointsAuthored = m_jointCount; fa.s = st;
            launchOverlayStatsForces(cmd, ring, renderPl, fa, forceCount);
            cmd.barrier();
        }
        args.ov.push = st.push; args.ov.pull = st.pull; args.ov.stress = st.stress;
        args.ov.cone = st.cone; args.ov.penetration = st.penetration; args.ov.jointDegree = st.jointDegree;
    }

    if (params.frustum)
    {
        frustumBuf.upload(params.frustum, 24 * sizeof(float));
        args.frustum = frustumBuf.address();
        visibleCounterBuf.zero(sizeof(int32_t));
        args.visible = visibleCounterBuf.address();
    }

    const int groups = (int)m_groupBase.size();
    if (params.segmentByShape)
    {
        shapeVisBuf.zero(groups * sizeof(int32_t));
        args.groupOf = groupIdx.address();
        args.shapeBase = shapeBaseBuf.address();
        args.shapeVisible = shapeVisBuf.address();
    }

    launchWriteInstances(cmd, ring, renderPl, args, m_shapeCount);
    cmd.submit(); cmd.wait();

    if (params.segmentByShape)
    {
        shapeVisBuf.readback(m_groupVisible.data(), groups * sizeof(int32_t));
        int total = 0;
        for (int k = 0; k < groups; k++)
            total += m_groupVisible[k];
        m_instanceCount = std::min(total, m_shapeCount);
    }
    else if (params.frustum)
    {
        int32_t vis = 0;
        visibleCounterBuf.readback(&vis, sizeof(int32_t));
        m_instanceCount = std::min((int)vis, m_shapeCount);
    }
    else
    {
        m_instanceCount = m_shapeCount;
    }

    cmd.destroy(); ring.destroy();
}

void VkWorld::writeDebugInstances(const DebugDrawViewGpu &view, int maxMarkers)
{
    assertIdle("VkWorld::writeDebugInstances");
    m_debugInstanceCount = 0;
    m_debugRequested = 0;
    const bool any = view.contactPoints || view.contactNormals || view.manifoldLinks;
    const int slots = narrow.manifoldSlots();
    if (!any || slots <= 0 || maxMarkers <= 0)
        return;

    // GpuSolver::debugInstanceBudget(): what the enabled markers could want -- a point
    // and a normal per contact, a link per manifold -- capped by maxMarkers. Sizing to the
    // cap alone would grow the buffer to 120k instances for a five-block tower.
    long long want = 0;
    const long long contacts = narrow.contactTotal();
    if (view.contactPoints)
        want += contacts;
    if (view.contactNormals)
        want += contacts;
    if (view.manifoldLinks)
        want += slots;
    const int limit = (int)std::min<long long>(want, maxMarkers);
    if (limit <= 0)
        return;
    VkDeviceSize need = (VkDeviceSize)limit * sizeof(BodyInstanceGpu);
    if (debugInstances.size() < need)
        debugInstances.resize(need, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    WriteDebugInstancesArgsGpu args{};
    args.b = bodyView();
    args.m = manifoldView();
    args.slots = slots;
    args.out = debugInstances.address();
    args.maxInstances = limit;
    debugCounterBuf.zero(sizeof(int32_t));
    args.counter = debugCounterBuf.address();
    args.d = view;

    CommandList cmd; HostRingBuffer ring;
    cmd.init(dev); ring.init(dev, 1 << 16);
    cmd.begin();
    launchWriteDebugInstances(cmd, ring, renderPl, args, slots);
    cmd.submit(); cmd.wait();

    // The counter overshoots `limit` by however many markers did not fit, which is
    // exactly what the UI needs in order to say the view is incomplete.
    int32_t requested = 0;
    debugCounterBuf.readback(&requested, sizeof(int32_t));
    m_debugRequested = requested;
    m_debugInstanceCount = std::min((int)requested, limit);
    cmd.destroy(); ring.destroy();
}

FullBodySoAGpu VkWorld::bodyView() const
{
    FullBodySoAGpu b{};
    b.pos = pos.address(); b.rot = rot.address();
    b.initialLin = initialLin.address(); b.initialAng = initialAng.address();
    b.inertialLin = inertialLin.address(); b.inertialAng = inertialAng.address();
    b.velLin = velLin.address(); b.velAng = velAng.address(); b.prevVelLin = prevVelLin.address();
    b.halfExtents = halfExtents.address(); b.shapeType = shapeType.address();
    b.moment = moment.address(); b.mass = mass.address(); b.friction = friction.address();
    b.awake = awake.address(); b.sleepTimer = sleepTimer.address();
    b.restPos = restPos.address(); b.restRot = restRot.address();
    b.gyration = gyration.address();
    b.props = propsAddress();
    b.count = (uint32_t)m_bodyCount;
    return b;
}

ShapeSoAGpu VkWorld::shapeView() const
{
    ShapeSoAGpu v{};
    v.body = shpBody.address(); v.localPos = shpLocalPos.address(); v.localRot = shpLocalRot.address();
    v.halfExtents = shpHalf.address(); v.shapeType = shpType.address(); v.friction = shpFriction.address();
    v.category = shpCategory.address(); v.mask = shpMask.address(); v.group = shpGroup.address();
    v.flags = shpFlags.address(); v.geomOffset = shpGeomOffset.address();
    v.count = (uint32_t)m_shapeCount;
    return v;
}

BroadphaseShapes VkWorld::broadphaseShapes() const
{
    BroadphaseShapes v;
    v.shapeBody = shpBody.address(); v.localPos = shpLocalPos.address(); v.localRot = shpLocalRot.address();
    v.halfExtents = shpHalf.address(); v.flags = shpFlags.address();
    return v;
}

FullJointSoAGpu VkWorld::jointView() const
{
    FullJointSoAGpu j{};
    j.bodyA = jBodyA.address(); j.bodyB = jBodyB.address();
    j.rA = jRA.address(); j.rB = jRB.address();
    j.C0Lin = jC0Lin.address(); j.C0Ang = jC0Ang.address();
    j.penaltyLin = jPenaltyLin.address(); j.penaltyAng = jPenaltyAng.address();
    j.lambdaLin = jLambdaLin.address(); j.lambdaAng = jLambdaAng.address();
    j.stiffnessLin = jStiffnessLin.address(); j.stiffnessAng = jStiffnessAng.address();
    j.fracture = jFracture.address(); j.fractureAng = jFractureAng.address(); j.torqueArm = jTorqueArm.address();
    j.restAng = jRestAng.address(); j.broken = jBroken.address();
    j.frameA = jFrameA.address(); j.frameB = jFrameB.address();
    j.axisMode = jAxisMode.address(); j.axisLo = jAxisLo.address(); j.axisHi = jAxisHi.address();
    j.generic = jGeneric.address();
    j.count = (uint32_t)m_jointSlots;
    return j;
}

void VkWorld::releaseDragSlots(int first, int count)
{
    if (count <= 0)
        return;
    DragReleaseArgsGpu args{};
    args.j = jointView();
    args.slot0 = m_jointCount + first;
    args.count = count;

    CommandList cmd; HostRingBuffer ring;
    cmd.init(dev); ring.init(dev, 1 << 14);
    cmd.begin();
    launchDragRelease(cmd, ring, interactPl, args, count);
    cmd.submit(); cmd.wait();
    cmd.destroy(); ring.destroy();
}

// the drag path's GpuSolver::dragBodies, step for step: the same per-body stiffness
// (5000 * mass * grabStrength * weight), the same "fresh grab" rule (the group changed
// size, or nothing was held), and the same release of slots the group no longer uses.
void VkWorld::dragBodies(const int *bodies, const vec3 *targets, const vec3 *locals, int count,
                         const float *weights, float grabStrength)
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
    args.bodies = dragBodies_.address(); args.targets = dragTargets.address();
    args.locals = dragLocals.address(); args.stiffness = dragStiffness.address();
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

    CommandList cmd; HostRingBuffer ring;
    cmd.init(dev); ring.init(dev, 1 << 14);
    cmd.begin();
    launchWakeAll(cmd, ring, wakePl, args, m_dynCount);
    cmd.submit(); cmd.wait();
    cmd.destroy(); ring.destroy();
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
    args.b = bodyView(); args.type = (int)type;
    args.idx = readIdxBuf.address(); args.out = readDataBuf.address(); args.count = count;

    VkWorld &self = const_cast<VkWorld &>(*this); // the pipelines are not const
    CommandList cmd; HostRingBuffer ring;
    cmd.init(self.dev); ring.init(self.dev, 1 << 14);
    cmd.begin();
    launchBodyGather(cmd, ring, self.mutatePl, args, count);
    cmd.submit(); cmd.wait();
    cmd.destroy(); ring.destroy();
    readDataBuf.readback(data, outBytes);
}

void VkWorld::setTransform(int body, vec3 p, quat r)
{
    BodyTransform x{p, r};
    setBodyData(&body, &x, BodyWrite::Transform, 1);
}

void VkWorld::setLinearVelocity(int body, vec3 v) { setBodyData(&body, &v, BodyWrite::LinearVelocity, 1); }

void VkWorld::setAngularVelocity(int body, vec3 w) { setBodyData(&body, &w, BodyWrite::AngularVelocity, 1); }

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

void VkWorld::applyTorque(int body, vec3 torque) { setBodyData(&body, &torque, BodyWrite::Torque, 1); }

void VkWorld::setAwake(int body, bool on)
{
    unsigned char v = on ? 1 : 0;
    setBodyData(&body, &v, BodyWrite::Awake, 1);
}

void VkWorld::setBodyProps(int body, const BodyProps &p) { setBodyData(&body, &p, BodyWrite::Props, 1); }

void VkWorld::setTargetTransform(int body, vec3 p, quat r, float dt)
{
    BodyTarget t{p, r, dt};
    setBodyData(&body, &t, BodyWrite::Target, 1);
}

BodyProps VkWorld::bodyProps(int body) const
{
    if (body < 0 || body >= (int)m_hostProps.size())
        return BodyProps{};
    return m_hostProps[body];
}

void VkWorld::wakeRegion(vec3 aabbMin, vec3 aabbMax)
{
    assertIdle("VkWorld::wakeRegion");
    if (m_dynCount == 0)
        return;
    // Seeded straight into the device buffer rather than queued: a wake is idempotent
    // and the flood that consumes it runs after every batch has landed (the mutation path).
    WakeSeedRegionArgsGpu args{};
    args.b = bodyView(); args.dyn = dynamicIdx.address(); args.dynCount = m_dynCount;
    args.loX = aabbMin.x; args.loY = aabbMin.y; args.loZ = aabbMin.z;
    args.hiX = aabbMax.x; args.hiY = aabbMax.y; args.hiZ = aabbMax.z;
    args.seed = wakeSeed.address(); args.changed = floodChanged.address();
    floodChanged.zero(sizeof(int));

    CommandList cmd; HostRingBuffer ring;
    cmd.init(dev); ring.init(dev, 1 << 14);
    cmd.begin();
    launchWakeSeedRegion(cmd, ring, mutatePl, args, m_dynCount);
    cmd.submit(); cmd.wait();
    cmd.destroy(); ring.destroy();
    m_seedsPending = true;
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

uint64_t VkWorld::deviceBytesAllocated() const
{
    const Buffer *all[] = {
        &pos, &rot, &initialLin, &initialAng, &inertialLin, &inertialAng,
        &velLin, &velAng, &prevVelLin, &halfExtents, &shapeType, &moment, &mass, &friction,
        &awake, &sleepTimer, &restPos, &restRot, &gyration, &props, &dynamicIdx, &staticIdx, &gridIdx, &oversizeIdx,
        &shpBody, &shpLocalPos, &shpLocalRot, &shpHalf, &shpType, &shpFriction, &shpRestitution, &shpRolling, &shpCategory, &shpMask,
        &shpGroup, &shpFlags, &shpGeomOffset, &geomHeap, &dynShapeIdx, &staticShapeIdx,
        &jBodyA, &jBodyB, &jRA, &jRB, &jC0Lin, &jC0Ang, &jPenaltyLin, &jPenaltyAng,
        &jLambdaLin, &jLambdaAng, &jStiffnessLin, &jStiffnessAng, &jFracture, &jFractureAng, &jTorqueArm,
        &jRestAng, &jBroken, &jFrameA, &jFrameB, &jAxisMode, &jAxisLo, &jAxisHi, &jGeneric,
        &sBodyA, &sBodyB, &sRA, &sRB, &sRest, &sStiffness,
        &activeIdx, &activeFlags, &activeCounter, &touchCount, &touchHash, &snapCount, &snapHash,
        &wakeRequest, &wakeSeed, &wakeAny, &wakeChanged, &sleepCandidate, &sleepBlock,
        &ignorePairs, &floodChanged, &islandSeed, &primalIndirect, &stepState,
        &mutIdxBuf, &mutDataBuf, &readIdxBuf, &readDataBuf,
        &instances, &debugInstances, &groupIdx,
        &overlayLoad, &overlayPush, &overlayPull, &overlayStress, &overlayCone, &overlayPen, &overlayJointDeg,
        &frustumBuf, &shapeBaseBuf, &shapeVisBuf, &visibleCounterBuf, &debugCounterBuf,
        &dragBodies_, &dragTargets, &dragLocals, &dragStiffness,
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
