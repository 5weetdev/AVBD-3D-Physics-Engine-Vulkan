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
        const VkDeviceSize cap =
            std::max<VkDeviceSize>(m_geomPending.size(), std::max<VkDeviceSize>(geomStage.size() * 2, 1u << 16));
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
    std::vector<float> dragZeroF(kMaxDragJoints, 0.0f), dragInf(kMaxDragJoints, INFINITY),
        dragNeg(kMaxDragJoints, -1.0f);
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
    m_geomPending.clear();
    m_geomCopies.clear();
    m_geomIndex.clear();
    m_geomSceneOffset.clear();
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
    m_mutBatches.clear();
    m_mutIdx.clear();
    m_mutData.clear();
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
        Buffer *bodyBufs[] = {
            &pos,         &rot,        &initialLin,  &initialAng,     &inertialLin, &inertialAng, &velLin,
            &velAng,      &prevVelLin, &halfExtents, &shapeType,      &moment,      &mass,        &friction,
            &awake,       &sleepTimer, &restPos,     &restRot,        &gyration,    &props,       &dynamicIdx,
            &staticIdx,   &activeIdx,  &activeFlags, &touchCount,     &touchHash,   &snapCount,   &snapHash,
            &wakeRequest, &wakeSeed,   &islandSeed,  &sleepCandidate, &sleepBlock};
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
        Buffer *shapeBufs[] = {&shpBody,        &shpLocalPos, &shpLocalRot,    &shpHalf, &shpType,  &shpFriction,
                               &shpRestitution, &shpRolling,  &shpCategory,    &shpMask, &shpGroup, &shpFlags,
                               &shpGeomOffset,  &dynShapeIdx, &staticShapeIdx, &gridIdx};
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
        vb(jAxisMode, (size_t)jc * 6 * sizeof(unsigned char));
        vb(jAxisLo, (size_t)jc * 6 * sizeof(float));
        vb(jAxisHi, (size_t)jc * 6 * sizeof(float));
        vb(jGeneric, jc * sizeof(unsigned char));
        vb(sBodyA, sc * sizeof(int));
        vb(sBodyB, sc * sizeof(int));
        vb(sRA, sc * sizeof(vec3));
        vb(sRB, sc * sizeof(vec3));
        vb(sRest, sc * sizeof(float));
        vb(sStiffness, sc * sizeof(float));
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

} // namespace avbdvk
