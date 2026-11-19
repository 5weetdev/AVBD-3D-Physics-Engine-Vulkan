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

#include "vk_narrowphase.h"
#include "vk_args.h"

#include <algorithm>

namespace avbdvk
{

void NarrowphasePipelines::init(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t spirvWords)
{
    narrowphase.init(dev, layout, spirv, spirvWords, "csNarrowphase", "csNarrowphase");
}

void NarrowphasePipelines::initHull(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t spirvWords)
{
    hullArgs.init(dev, layout, spirv, spirvWords, "csNarrowphaseHullArgs", "csNarrowphaseHullArgs");
    hull.init(dev, layout, spirv, spirvWords, "csNarrowphaseHull", "csNarrowphaseHull");
    hullInit = true;
}

void NarrowphasePipelines::initMesh(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t spirvWords)
{
    meshArgs.init(dev, layout, spirv, spirvWords, "csMeshArgs", "csMeshArgs");
    meshChildCount.init(dev, layout, spirv, spirvWords, "csMeshChildCount", "csMeshChildCount");
    meshMidphase.init(dev, layout, spirv, spirvWords, "csMeshMidphase", "csMeshMidphase");
    meshChild.init(dev, layout, spirv, spirvWords, "csMeshChild", "csMeshChild");
    meshChildFilter.init(dev, layout, spirv, spirvWords, "csMeshChildFilter", "csMeshChildFilter");
    meshInit = true;
}

void NarrowphasePipelines::destroy()
{
    narrowphase.destroy();
    if (hullInit)
    {
        hullArgs.destroy();
        hull.destroy();
        hullInit = false;
    }
    if (meshInit)
    {
        meshArgs.destroy();
        meshChildCount.destroy();
        meshMidphase.destroy();
        meshChild.destroy();
        meshChildFilter.destroy();
        meshInit = false;
    }
}

// Device-side argument layout now lives in vk_args.h (NpBodySoAGpu, NpManifoldSoAGpu,
// SolverParamsNpGpu, NarrowphaseArgsGpu), shared with the layout-gate probe
// (tools/test_vk_layout.cpp + src/shaders/layout_probe.slang) so the struct the
// probe verifies is the exact same one dispatched here -- see vk_args.h's
// NarrowphaseArgsGpu comment for the padding bug this consolidation guards against.

static NpManifoldSoAGpu toGpu(const VkManifoldSet &s)
{
    NpManifoldSoAGpu g;
    g.key = s.key;
    g.bodyA = s.bodyA;
    g.bodyB = s.bodyB;
    g.shapeA = s.shapeA;
    g.shapeB = s.shapeB;
    g.basis = s.basis;
    g.friction = s.friction;
    g.contactOffset = s.contactOffset;
    g.contactCount = s.contactCount;
    g.feature = s.feature;
    g.rA = s.rA;
    g.rB = s.rB;
    g.C0 = s.C0;
    g.penalty = s.penalty;
    g.lambda = s.lambda;
    g.stick = s.stick;
    return g;
}

void VkNarrowphase::init(Device &dev)
{
    m_dev = &dev;
    ring.init(dev, 1 << 22);
    cmd.init(dev, 2);

    keyA.init(dev);
    keyB.init(dev);
    basisA.init(dev);
    basisB.init(dev);
    frictionA.init(dev);
    frictionB.init(dev);
    offsetA.init(dev);
    offsetB.init(dev);
    countA.init(dev);
    countB.init(dev);
    bodyAA.init(dev);
    bodyAB.init(dev);
    bodyBA.init(dev);
    bodyBB.init(dev);
    shapeAA.init(dev);
    shapeAB.init(dev);
    shapeBA.init(dev);
    shapeBB.init(dev);
    featA.init(dev);
    featB.init(dev);
    rAA.init(dev);
    rAB.init(dev);
    rBA.init(dev);
    rBB.init(dev);
    c0A.init(dev);
    c0B.init(dev);
    penA.init(dev);
    penB.init(dev);
    lamA.init(dev);
    lamB.init(dev);
    stickA.init(dev);
    stickB.init(dev);
    counter.init(dev);
    activeCount.init(dev);
    hullList.init(dev);
    hullCounter.init(dev);
    hullIndirect.init(dev);
    hullScratch.init(dev);
    satSepA.init(dev);
    satSepB.init(dev);
    satBitsA.init(dev);
    satBitsB.init(dev);
    meshList.init(dev);
    meshCounter.init(dev);
    meshIndirect.init(dev);
    childKeys.init(dev);
    childSorted.init(dev);
    childCounter.init(dev);
    childCount.init(dev);
    meshScratch.init(dev);
    if (!m_readout.mapped())
        m_readout.init(dev, 4 * sizeof(int32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT, MappedBuffer::HostAccess::Read);
}

void VkNarrowphase::destroy()
{
    release();
    m_readout.destroy();
    cmd.destroy();
    ring.destroy();
}

void VkNarrowphase::release()
{
    keyA.destroy();
    keyB.destroy();
    basisA.destroy();
    basisB.destroy();
    frictionA.destroy();
    frictionB.destroy();
    offsetA.destroy();
    offsetB.destroy();
    countA.destroy();
    countB.destroy();
    bodyAA.destroy();
    bodyAB.destroy();
    bodyBA.destroy();
    bodyBB.destroy();
    shapeAA.destroy();
    shapeAB.destroy();
    shapeBA.destroy();
    shapeBB.destroy();
    featA.destroy();
    featB.destroy();
    rAA.destroy();
    rAB.destroy();
    rBA.destroy();
    rBB.destroy();
    c0A.destroy();
    c0B.destroy();
    penA.destroy();
    penB.destroy();
    lamA.destroy();
    lamB.destroy();
    stickA.destroy();
    stickB.destroy();
    counter.destroy();
    activeCount.destroy();
    hullList.destroy();
    hullCounter.destroy();
    hullIndirect.destroy();
    hullScratch.destroy();
    satSepA.destroy();
    satSepB.destroy();
    satBitsA.destroy();
    satBitsB.destroy();
    meshList.destroy();
    meshCounter.destroy();
    meshIndirect.destroy();
    childKeys.destroy();
    childSorted.destroy();
    childCounter.destroy();
    childCount.destroy();
    meshScratch.destroy();
    hullCap = 0;
    m_hullPairs = 0;
    m_hullRecorded = false;
    childCap = 0;
    m_childBase = 0;
    m_childRaw = 0;
    m_prevChildBase = 0;
    m_prevChildCount = 0;
    m_meshRecorded = false;
    m_childOverflow = false;
}

void VkNarrowphase::reserve(int maxPairs_, int maxContacts_)
{
    maxPairs = maxPairs_;
    maxContacts = maxContacts_;

    // Per-slot arrays hold the pair slots and the child slots (Phase 1.5a): child slot i is
    // manifold slot childBase + i, so the array must reach past maxPairs by childCap.
    const VkDeviceSize slotN = (VkDeviceSize)slotCapacity();
    keyA.resize(slotN * sizeof(uint64_t));
    keyB.resize(slotN * sizeof(uint64_t));
    basisA.resize(slotN * sizeof(float) * 9);
    basisB.resize(slotN * sizeof(float) * 9);
    frictionA.resize(slotN * sizeof(float));
    frictionB.resize(slotN * sizeof(float));
    offsetA.resize(slotN * sizeof(int32_t));
    offsetB.resize(slotN * sizeof(int32_t));
    countA.resize(slotN * sizeof(int32_t));
    countB.resize(slotN * sizeof(int32_t));
    countA.zero();
    countB.zero(); // matches the reference resizeZeroed on countA/countB
    bodyAA.resize(slotN * sizeof(int32_t));
    bodyAB.resize(slotN * sizeof(int32_t));
    bodyBA.resize(slotN * sizeof(int32_t));
    bodyBB.resize(slotN * sizeof(int32_t));
    shapeAA.resize(slotN * sizeof(int32_t));
    shapeAB.resize(slotN * sizeof(int32_t));
    shapeBA.resize(slotN * sizeof(int32_t));
    shapeBB.resize(slotN * sizeof(int32_t));

    featA.resize((VkDeviceSize)maxContacts * sizeof(int32_t));
    featB.resize((VkDeviceSize)maxContacts * sizeof(int32_t));
    rAA.resize((VkDeviceSize)maxContacts * sizeof(float) * 3);
    rAB.resize((VkDeviceSize)maxContacts * sizeof(float) * 3);
    rBA.resize((VkDeviceSize)maxContacts * sizeof(float) * 3);
    rBB.resize((VkDeviceSize)maxContacts * sizeof(float) * 3);
    c0A.resize((VkDeviceSize)maxContacts * sizeof(float) * 3);
    c0B.resize((VkDeviceSize)maxContacts * sizeof(float) * 3);
    penA.resize((VkDeviceSize)maxContacts * sizeof(float) * 3);
    penB.resize((VkDeviceSize)maxContacts * sizeof(float) * 3);
    lamA.resize((VkDeviceSize)maxContacts * sizeof(float) * 3);
    lamB.resize((VkDeviceSize)maxContacts * sizeof(float) * 3);
    stickA.resize((VkDeviceSize)maxContacts * sizeof(uint8_t));
    stickB.resize((VkDeviceSize)maxContacts * sizeof(uint8_t));

    counter.resize(sizeof(int32_t));
    activeCount.resize(sizeof(int32_t));
    // A fresh reserve drops the cache; rebuild the hull arrays at the new size (lazily if never used).
    if (satSepA.size() > 0)
    {
        satSepA.destroy();
        satSepB.destroy();
        satBitsA.destroy();
        satBitsB.destroy();
        ensureHull();
    }
    // The mesh pair list is per-record scratch sized by the pair capacity; rebuilt lazily otherwise.
    if (meshCounter.size() > 0)
        meshList.resize((VkDeviceSize)std::max(maxPairs, 1) * sizeof(int32_t));

    reset();
}

// NOTE: Buffer::resize() does not preserve contents (see vk_buffer.h) -- unlike the
// the reference class's reserveKeep, a grow() here loses the warm-start cache for one step, the
// same way a fresh reserve() would. That is a known simplification versus the reference
// path documented in the narrowphase's grow() comment; acceptable because the gates only
// exercise small, non-growing scenes. TODO: a content-preserving resize (readback+upload or a
// copy command) once a growth-under-load parity test exists.
void VkNarrowphase::grow(int maxPairs_, int maxContacts_)
{
    // Contact-cache-preserving grow, mirroring the capacity code's
    // reserveKeep(): both A and B halves carry live warm-start state (whichever one
    // is "prev" next build() call), so both must survive -- unlike reserve(), this
    // must NOT reset() or drop slots/contacts/active/prevSlots/flip/curWasB, and it
    // must not zero countA/countB (their live per-pair contact counts are exactly
    // the state the next build() reads as "previous manifold" via prevKeysAddr).
    maxPairs = maxPairs_;
    maxContacts = maxContacts_;

    // Per-slot arrays (pair slots then child slots) keep their contents across the resize, so the
    // previous step's child region survives a growth rerun the same way the pair region does.
    auto pb = [&](Buffer &buf, VkDeviceSize elemBytes)
    { buf.resizePreserve((VkDeviceSize)slotCapacity() * elemBytes); };
    auto cb = [&](Buffer &buf, VkDeviceSize elemBytes) { buf.resizePreserve((VkDeviceSize)maxContacts * elemBytes); };

    pb(keyA, sizeof(uint64_t));
    pb(keyB, sizeof(uint64_t));
    pb(basisA, sizeof(float) * 9);
    pb(basisB, sizeof(float) * 9);
    pb(frictionA, sizeof(float));
    pb(frictionB, sizeof(float));
    pb(offsetA, sizeof(int32_t));
    pb(offsetB, sizeof(int32_t));
    pb(countA, sizeof(int32_t));
    pb(countB, sizeof(int32_t));
    pb(bodyAA, sizeof(int32_t));
    pb(bodyAB, sizeof(int32_t));
    pb(bodyBA, sizeof(int32_t));
    pb(bodyBB, sizeof(int32_t));
    pb(shapeAA, sizeof(int32_t));
    pb(shapeAB, sizeof(int32_t));
    pb(shapeBA, sizeof(int32_t));
    pb(shapeBB, sizeof(int32_t));

    cb(featA, sizeof(int32_t));
    cb(featB, sizeof(int32_t));
    cb(rAA, sizeof(float) * 3);
    cb(rAB, sizeof(float) * 3);
    cb(rBA, sizeof(float) * 3);
    cb(rBB, sizeof(float) * 3);
    cb(c0A, sizeof(float) * 3);
    cb(c0B, sizeof(float) * 3);
    cb(penA, sizeof(float) * 3);
    cb(penB, sizeof(float) * 3);
    cb(lamA, sizeof(float) * 3);
    cb(lamB, sizeof(float) * 3);
    cb(stickA, sizeof(uint8_t));
    cb(stickB, sizeof(uint8_t));

    // The b3SATCache rides the same A/B ping-pong as the manifolds, so it survives a growth
    // rerun with them (allocated only once a hull shape has been recorded).
    if (satSepA.size() > 0)
    {
        pb(satSepA, sizeof(float));
        pb(satSepB, sizeof(float));
        pb(satBitsA, sizeof(uint32_t));
        pb(satBitsB, sizeof(uint32_t));
        hullList.resize((VkDeviceSize)maxPairs * sizeof(int32_t));
    }
    if (meshCounter.size() > 0)
        meshList.resize((VkDeviceSize)maxPairs * sizeof(int32_t));

    counter.resize(sizeof(int32_t));
    activeCount.resize(sizeof(int32_t));
    // Note: counter/activeCount are per-build scratch outputs, not persistent
    // warm-start state -- an ordinary resize() (not preserving) is correct here,
    // build() overwrites them unconditionally every call.
}

// Allocated on the first record() with a hull shape (warmup), grown only between passes.
void VkNarrowphase::ensureHull()
{
    if (satSepA.size() == 0)
    {
        const VkDeviceSize n = (VkDeviceSize)std::max(maxPairs, 1);
        satSepA.resize(n * sizeof(float));
        satSepB.resize(n * sizeof(float));
        satBitsA.resize(n * sizeof(uint32_t));
        satBitsB.resize(n * sizeof(uint32_t));
        satSepA.zero();
        satSepB.zero();
        satBitsA.zero();
        satBitsB.zero();
        hullCounter.resize(sizeof(int32_t));
        hullIndirect.resize(3 * sizeof(uint32_t));
        hullList.resize(n * sizeof(int32_t));
    }
    if (hullCap == 0)
        growHull(64);
}

// Scratch slots for `neededPairs` simultaneous hull lanes, with headroom; never shrinks.
void VkNarrowphase::growHull(int neededPairs)
{
    if (neededPairs <= hullCap)
        return;
    const int target = std::max(neededPairs + neededPairs / 4 + 16, hullCap * 2);
    hullScratch.resize((VkDeviceSize)target * kHullScratchStride);
    hullCap = target;
}

// Allocated on the first record() with a mesh shape (warmup). The child capacity starts at 64 and
// grows only between passes (growChild); the per-lane triangle scratch is fixed-size.
void VkNarrowphase::ensureMesh()
{
    if (meshCounter.size() == 0)
    {
        meshList.resize((VkDeviceSize)std::max(maxPairs, 1) * sizeof(int32_t));
        meshCounter.resize(sizeof(int32_t));
        meshIndirect.resize(3 * sizeof(uint32_t));
        childCounter.resize(sizeof(int32_t));
        childCount.resize(sizeof(int32_t));
        meshScratch.resize((VkDeviceSize)kMeshWorkers * kMeshScratchStride);
    }
    if (childCap == 0)
        growChild(64);
}

// Child slots for `neededChildren` raw child keys, with headroom; never shrinks. The per-slot manifold
// arrays are regrown with contents preserved (grow()), so the previous step's child region survives.
void VkNarrowphase::growChild(int neededChildren)
{
    if (neededChildren <= childCap)
        return;
    const int target = std::max(neededChildren + neededChildren / 4 + 16, childCap * 2);
    childCap = target;
    childKeys.resize((VkDeviceSize)target * sizeof(uint64_t));
    childSorted.resize((VkDeviceSize)target * sizeof(uint64_t));
    grow(maxPairs, maxContacts);
}

void VkNarrowphase::reset()
{
    slots = contacts = active = prevSlots = 0;
    requiredContactCount = 0;
    overflow = false;
    flip = false;
    curWasB = false;
    m_prevChildBase = 0;
    m_prevChildCount = 0;
}

void VkNarrowphase::sets(VkManifoldSet &curSet, VkManifoldSet &prevSet) const
{
    VkManifoldSet a{}, bSet{};
    a.key = keyA.address();
    a.basis = basisA.address();
    a.friction = frictionA.address();
    a.contactOffset = offsetA.address();
    a.contactCount = countA.address();
    a.feature = featA.address();
    a.rA = rAA.address();
    a.rB = rBA.address();
    a.C0 = c0A.address();
    a.penalty = penA.address();
    a.lambda = lamA.address();
    a.stick = stickA.address();
    a.bodyA = bodyAA.address();
    a.bodyB = bodyBA.address();
    a.shapeA = shapeAA.address();
    a.shapeB = shapeBA.address();

    bSet.key = keyB.address();
    bSet.basis = basisB.address();
    bSet.friction = frictionB.address();
    bSet.contactOffset = offsetB.address();
    bSet.contactCount = countB.address();
    bSet.feature = featB.address();
    bSet.rA = rAB.address();
    bSet.rB = rBB.address();
    bSet.C0 = c0B.address();
    bSet.penalty = penB.address();
    bSet.lambda = lamB.address();
    bSet.stick = stickB.address();
    bSet.bodyA = bodyAB.address();
    bSet.bodyB = bodyBB.address();
    bSet.shapeA = shapeAB.address();
    bSet.shapeB = shapeBB.address();

    curSet = flip ? bSet : a;
    prevSet = flip ? a : bSet;
}

void VkNarrowphase::build(NarrowphasePipelines &pl, const VkNpBodyRefs &bodies, const Buffer &pairs, int pairCount,
                          const VkNarrowphaseParams &params)
{
    ring.reset();
    cmd.begin();
    record(cmd, ring, pl, bodies, pairs, pairCount, nullptr, params);
    cmd.submit();
    cmd.wait();
    finish(pairCount);
}

void VkNarrowphase::record(CommandList &cmd, HostRingBuffer &ring, NarrowphasePipelines &pl, const VkNpBodyRefs &bodies,
                           const Buffer &pairs, int pairCapacity, const Buffer *livePairs,
                           const VkNarrowphaseParams &params, const VkNpHullRefs *hulls, const VkNpMeshRefs *meshes)
{
    m_readoutRecorded = false;
    m_hullRecorded = false;
    m_meshRecorded = false;
    m_childRaw = 0;
    m_childOverflow = false;
    m_pendingSlots = 0;
    cmd.fill(counter, 0u, sizeof(int32_t));
    cmd.fill(activeCount, 0u, sizeof(int32_t));
    // Every per-pair array holds maxPairs slots; a larger broadphase capacity must not
    // index past them (VkWorld keeps the two equal, this only makes it impossible).
    pairCapacity = std::min(pairCapacity, maxPairs);
    m_pendingSlots = pairCapacity;
    if (pairCapacity == 0)
        return;

    // The mesh path needs its geometry, its radix sort and its pipelines; without them a mesh pair
    // never reaches csNarrowphase's list (the divert is keyed on meshList != 0 below).
    const bool meshPath = meshes != nullptr && pl.meshInit && meshes->geom.geomHeap != 0 &&
                          meshes->geom.geomOffset != 0 && meshes->prim != nullptr && meshes->radix != nullptr;
    if (meshPath)
    {
        ensureMesh();
        cmd.fill(meshCounter, 0u, sizeof(int32_t));
        cmd.fill(childCounter, 0u, sizeof(int32_t));
        m_meshRecorded = true;
        m_childBase = pairCapacity;
        // The record-time slot count covers the child region too: the constraint graph and the
        // dual / emit dispatches walk pendingSlots(), and a child slot they miss never solves.
        m_pendingSlots = m_childBase + childCap;
        reserveRadix(*meshes->radix, (uint32_t)childCap, sizeof(uint64_t), 0);
    }

    VkManifoldSet curSet{}, prevSet{};
    sets(curSet, prevSet);

    NpBodySoAGpu b;
    b.pos = bodies.pos;
    b.rot = bodies.rot;
    b.shapeBody = bodies.shapeBody;
    b.localPos = bodies.localPos;
    b.localRot = bodies.localRot;
    b.halfExtents = bodies.halfExtents;
    b.shapeType = bodies.shapeType;
    b.flags = bodies.flags;
    b.friction = bodies.friction;
    b.mass = bodies.mass;
    b.moment = bodies.moment;
    b.velLin = bodies.velLin;
    b.velAng = bodies.velAng;
    b.props = bodies.props;
    b.category = bodies.category;
    b.mask = bodies.mask;
    b.group = bodies.group;

    NarrowphaseArgsGpu args;
    args.b = b;
    args.pairs = pairs.address();
    args.pairCount = (uint32_t)pairCapacity;
    args.cur = toGpu(curSet);
    args.prev = toGpu(prevSet);
    args.prevKeys = prevSet.key;
    args.prevCount = (uint32_t)prevSlots;
    args.contactCounter = counter.address();
    args.activeCounter = activeCount.address();
    args.contactCapacity = maxContacts;
    args.p.dt = params.dt;
    args.p.alpha = params.alpha;
    args.p.gamma = params.gamma;
    args.p.autoKStart = params.autoKStart ? 1u : 0u;
    args.p.dualDamping = params.dualDamping;
    args.p.dualDampVel = params.dualDampVel;
    args.livePairs = livePairs ? livePairs->address() : 0;
    args.contactCap = params.contactCap;

    const bool hullPath = hulls != nullptr && pl.hullInit && hulls->geomHeap != 0 && hulls->geomOffset != 0;
    if (hullPath)
    {
        ensureHull();
        cmd.fill(hullCounter, 0u, sizeof(int32_t));
        args.hullList = hullList.address();
        args.hullCounter = hullCounter.address();
    }
    if (meshPath)
    {
        args.meshList = meshList.address();
        args.meshCounter = meshCounter.address();
    }

    VkDeviceAddress addr = ring.write(&args, sizeof(args));

    const uint32_t block = 128;
    cmd.dispatch(pl.narrowphase, ((uint32_t)pairCapacity + block - 1) / block, addr);
    cmd.barrier();

    if (hullPath)
    {
        m_hullRecorded = true;
        NarrowphaseHullArgsGpu h;
        h.np = args;
        h.geomHeap = hulls->geomHeap;
        h.geomOffset = hulls->geomOffset;
        h.geomHeapBytes = hulls->geomHeapBytes;
        h.scratch = hullScratch.address();
        h.satSepCur = flip ? satSepB.address() : satSepA.address();
        h.satBitsCur = flip ? satBitsB.address() : satBitsA.address();
        h.satSepPrev = flip ? satSepA.address() : satSepB.address();
        h.satBitsPrev = flip ? satBitsA.address() : satBitsB.address();
        h.hullCapacity = (uint32_t)hullCap;
        h.scratchStride = kHullScratchStride;
        const VkDeviceAddress hAddr = ring.write(&h, sizeof(h));

        NarrowphaseHullDispatchArgsGpu d;
        d.counter = hullCounter.address();
        d.indirect = hullIndirect.address();
        d.capacity = (uint32_t)maxPairs;
        d.groupSize = 32;
        const VkDeviceAddress dAddr = ring.write(&d, sizeof(d));
        cmd.dispatch(pl.hullArgs, 1, dAddr);
        cmd.barrierIndirect();
        cmd.dispatchIndirect(pl.hull, hullIndirect, 0, hAddr);
        cmd.barrier();
    }

    if (meshPath)
    {
        const VkNpHullRefs &g = meshes->geom;

        // 1. Indirect groups for the midphase, from the mesh pair count csNarrowphase appended.
        NarrowphaseMeshDispatchArgsGpu da;
        da.counter = meshCounter.address();
        da.indirect = meshIndirect.address();
        da.capacity = (uint32_t)pairCapacity;
        da.groupSize = 32;
        cmd.dispatch(pl.meshArgs, 1, ring.write(&da, sizeof(da)));
        cmd.barrierIndirect();

        // 2. Midphase: one lane per listed mesh pair, child keys appended to childKeys.
        NarrowphaseMeshArgsGpu m;
        m.np = args;
        m.geomHeap = g.geomHeap;
        m.geomOffset = g.geomOffset;
        m.geomHeapBytes = g.geomHeapBytes;
        m.childKeys = childKeys.address();
        m.childCounter = childCounter.address();
        m.childCapacity = (uint32_t)childCap;
        m.meshCount = (uint32_t)pairCapacity;
        cmd.dispatchIndirect(pl.meshMidphase, meshIndirect, 0, ring.write(&m, sizeof(m)));
        cmd.barrier();

        // 3. Clamped child count: childCount = min(raw, childCap). It is the sort's count and the
        //    child kernel's loop bound.
        NarrowphaseMeshDispatchArgsGpu dc;
        dc.counter = childCounter.address();
        dc.indirect = childCount.address();
        dc.capacity = (uint32_t)childCap;
        dc.groupSize = 1;
        cmd.dispatch(pl.meshChildCount, 1, ring.write(&dc, sizeof(dc)));
        cmd.barrier();

        // 4. Sort the child keys: sorted child i is then manifold slot childBase + i.
        radixSortKeysU64(cmd, ring, *meshes->prim, *meshes->radix, childKeys, childSorted, (uint32_t)childCap,
                         childCount.address());
        cmd.barrier();

        // 5. Per-child triangle narrowphase, kMeshWorkers lanes striding over the sorted list.
        NarrowphaseMeshChildArgsGpu c;
        c.np = args;
        c.geomHeap = g.geomHeap;
        c.geomOffset = g.geomOffset;
        c.scratch = meshScratch.address();
        c.childSorted = childSorted.address();
        c.childCounter = childCount.address();
        c.childInfo = childKeys.address(); // the unsorted keys are dead after the sort
        c.geomHeapBytes = g.geomHeapBytes;
        c.childCapacity = (uint32_t)childCap;
        c.childBase = (uint32_t)m_childBase;
        c.prevChildBase = (uint32_t)m_prevChildBase;
        c.prevChildCount = (uint32_t)m_prevChildCount;
        c.workers = kMeshWorkers;
        c.scratchStride = kMeshScratchStride;
        const VkDeviceAddress cAddr = ring.write(&c, sizeof(c));
        cmd.dispatch(pl.meshChild, kMeshWorkers / 32, cAddr);
        cmd.barrier();

        // 6. Box3D's ghost-collision filter across a pair's triangles (mesh_contact.c): drops edge and
        //    vertex manifolds that an accepted neighbouring triangle already covers.
        cmd.dispatch(pl.meshChildFilter, kMeshWorkers / 32, cAddr);
        cmd.barrier();
    }
}

void VkNarrowphase::recordReadout(CommandList &cmd)
{
    cmd.copyToHost(counter, m_readout, sizeof(int32_t), 0, 0);
    cmd.copyToHost(activeCount, m_readout, sizeof(int32_t), 0, sizeof(int32_t));
    if (m_hullRecorded)
        cmd.copyToHost(hullCounter, m_readout, sizeof(int32_t), 0, 2 * sizeof(int32_t));
    if (m_meshRecorded)
        cmd.copyToHost(childCounter, m_readout, sizeof(int32_t), 0, 3 * sizeof(int32_t));
    m_readoutRecorded = true;
}

// Reads the raw child-key count (the atomic append total, which can exceed childCap). Overflow of the
// child capacity is reported through childOverflowed(); the caller reruns after growChild().
void VkNarrowphase::readChildRaw()
{
    if (!m_meshRecorded)
    {
        m_childRaw = 0;
        m_childOverflow = false;
        return;
    }
    int32_t raw = 0;
    if (m_readoutRecorded)
        raw = m_readout.mappedAs<int32_t>()[3];
    else
        childCounter.readback(&raw, sizeof(int32_t));
    m_childRaw = raw;
    m_childOverflow = raw > childCap;
}

void VkNarrowphase::finish(int pairCount, bool commit)
{
    if (!commit)
    {
        int32_t contactsOut = 0;
        if (pairCount > 0)
        {
            if (m_readoutRecorded)
                contactsOut = m_readout.mappedAs<int32_t>()[0];
            else
                counter.readback(&contactsOut, sizeof(int32_t));
        }
        if (m_hullRecorded && m_readoutRecorded)
            m_hullPairs = m_readout.mappedAs<int32_t>()[2];
        readChildRaw();
        m_readoutRecorded = false;
        requiredContactCount = contactsOut;
        overflow = contactsOut > maxContacts;
        return;
    }
    VkManifoldSet curSet{}, prevSet{};
    sets(curSet, prevSet);
    curWasB = flip;
    cur = curSet;
    prev = prevSet;

    slots = pairCount;
    contacts = 0;
    active = 0;
    overflow = false;

    if (pairCount == 0)
    {
        prevSlots = 0;
        m_prevChildCount = 0;
        return;
    }
    // A mesh record owns the child region [childBase, childBase + childCap) after the pair slots.
    if (m_meshRecorded)
        slots = m_childBase + childCap;

    int32_t contactsOut = 0, activeOut = 0;
    if (m_readoutRecorded)
    {
        const int32_t *ro = m_readout.mappedAs<int32_t>();
        contactsOut = ro[0];
        activeOut = ro[1];
    }
    else
    {
        counter.readback(&contactsOut, sizeof(int32_t));
        activeCount.readback(&activeOut, sizeof(int32_t));
    }
    if (m_hullRecorded && m_readoutRecorded)
        m_hullPairs = m_readout.mappedAs<int32_t>()[2];
    readChildRaw();
    m_readoutRecorded = false;
    contacts = contactsOut;
    active = activeOut;

    requiredContactCount = contacts;
    overflow = contacts > maxContacts;
    if (overflow)
        contacts = maxContacts;

    if (!overflow && !m_childOverflow)
    {
        prevSlots = pairCount;
        // The child region commits with the flip: the next step's warm start reads it from `prev`.
        m_prevChildBase = m_childBase;
        m_prevChildCount = m_meshRecorded ? std::min(m_childRaw, childCap) : 0;
        flip = !flip;
    }
}

} // namespace avbdvk
