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

#pragma once

// Vulkan host orchestration for src/shaders/narrowphase.slang, mirroring
// engine/collide/gpu_narrowphase.cuh's Narrowphase class: double-buffered manifold
// storage (flip swaps which half is "current" vs "previous" each build so the warm
// start always reads last step's dual state), reserve()/grow() with the same
// reserve-vs-reserveKeep growth split, reset(), and the same overflow contract
// (requiredContacts() larger than contactTotal() on overflow, prevSlots only advances
// on a non-overflowed build).

#include "vk_buffer.h"
#include "vk_cmd.h"
#include "vk_device.h"
#include "vk_pipeline.h"
#include "vk_primitives.h"

#include <cstdint>

namespace avbdvk
{

struct NarrowphasePipelines
{
    Pipeline narrowphase;
    // Phase 1.3: hull pairs. Both stay empty (hullReady() false) until initHull runs.
    Pipeline hullArgs, hull;
    bool hullInit = false;
    // Phase 1.5a: mesh pairs (narrowphase_mesh.slang). Empty until initMesh runs.
    Pipeline meshArgs, meshChildCount, meshMidphase, meshChild, meshChildFilter;
    bool meshInit = false;

    void init(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t spirvWords);
    void initHull(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t spirvWords);
    void initMesh(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t spirvWords);
    void destroy();
};

// Device addresses for one manifold set (cur or prev), field for field matching
// narrowphase.slang's NpManifoldSoA / types.slang's ManifoldSoA.
struct VkManifoldSet
{
    VkDeviceAddress key = 0;
    VkDeviceAddress bodyA = 0, bodyB = 0;
    VkDeviceAddress shapeA = 0, shapeB = 0; // the shape pair (the cache key); bodyA/B are its bodies
    VkDeviceAddress basis = 0;
    VkDeviceAddress friction = 0;
    VkDeviceAddress contactOffset = 0, contactCount = 0;
    VkDeviceAddress feature = 0;
    VkDeviceAddress rA = 0, rB = 0;
    VkDeviceAddress C0 = 0;
    VkDeviceAddress penalty = 0, lambda = 0;
    VkDeviceAddress stick = 0;
};

// Narrowphase solver-params subset narrowphase.slang's SolverParamsNp expects.
struct VkNarrowphaseParams
{
    float dt = 0.0f;
    float alpha = 0.97f;
    float gamma = 0.99f;
    bool autoKStart = false;
    float dualDamping = 0.0f;
    float dualDampVel = 0.0f;
    int contactCap = 0; // 0 = no cap
};

// Device addresses for the body and shape SoA fields narrowphase.slang reads (NpBodySoA).
// pos/rot/mass/moment/vel are per body; the rest are per SHAPE (the pair list holds shape
// indices). shapeType may be left 0 (nullptr) for an all-box scene; flags must not be 0.
struct VkNpBodyRefs
{
    VkDeviceAddress pos = 0, rot = 0;
    VkDeviceAddress shapeBody = 0, localPos = 0, localRot = 0;
    VkDeviceAddress halfExtents = 0, shapeType = 0, friction = 0, flags = 0;
    VkDeviceAddress mass = 0, moment = 0;
    VkDeviceAddress velLin = 0, velAng = 0;
    VkDeviceAddress props = 0;                         // may be 0: no body has a non-default BodyProps
    VkDeviceAddress category = 0, mask = 0, group = 0; // per shape collision filter, may be 0
};

// The geometry heap as the hull kernel reads it (VkWorld owns the buffers). A scene without a
// hull shape passes nullptr to record() and the hull path costs nothing.
struct VkNpHullRefs
{
    VkDeviceAddress geomHeap = 0, geomOffset = 0; // heap base; per-shape blob offsets
    uint64_t geomHeapBytes = 0;
};

// The mesh path (Phase 1.5a) reads the same geometry heap as the hull path, and sorts its child keys
// with the broadphase's radix pipelines and scratch (VkWorld owns both). A scene without a mesh shape
// passes nullptr to record() and the mesh path costs nothing.
struct VkNpMeshRefs
{
    VkNpHullRefs geom;
    PrimitivesPipelines *prim = nullptr;
    RadixScratch *radix = nullptr;
};

class VkNarrowphase
{
public:
    void init(Device &dev);
    void destroy();
    // Frees the capacity-sized buffers and keeps the command pool and ring, so the
    // next reserve() rebuilds from nothing. VkWorld::reset() calls this: a scene
    // switch must not hold the old scene's memory until the new one is built.
    void release();

    void reserve(int maxPairs, int maxContacts);
    void grow(int maxPairs, int maxContacts);
    void reset();

    void build(NarrowphasePipelines &pl, const VkNpBodyRefs &bodies, const Buffer &pairs, int pairCount,
               const VkNarrowphaseParams &params);

    // record() appends the build to a caller's command buffer. With livePairs set,
    // pairCapacity is a capacity: the dispatch covers it, the live count is
    // min(*livePairs, pairCapacity) read on device, and every slot past it gets
    // contactCount 0, so a consumer sized by capacity skips it. finish(pairCount) runs
    // after the submission completes: reads the counters back and commits the
    // ping-pong flip -- skipping it (a failed attempt to be rerun) leaves the warm-start
    // cache exactly as it was. build() is record + submit + finish.
    void record(CommandList &cmd, HostRingBuffer &ring, NarrowphasePipelines &pl, const VkNpBodyRefs &bodies,
                const Buffer &pairs, int pairCapacity, const Buffer *livePairs, const VkNarrowphaseParams &params,
                const VkNpHullRefs *hulls = nullptr, const VkNpMeshRefs *meshes = nullptr);
    // commit == false reads only the counters (contactTotal/requiredContacts/overflowed): a
    // failed attempt that will be rerun, which must not flip the ping-pong or advance
    // prevSlots. The set a pending record() writes is pendingOutput(), valid before finish().
    void finish(int pairCount, bool commit = true);
    // Optional: records copies of both counters into a host-mapped readout after
    // record(), so finish() reads mapped memory instead of two one-shot readback
    // submits. finish() falls back to readback() when this was not recorded.
    void recordReadout(CommandList &cmd);

    const VkManifoldSet &manifolds() const { return cur; }
    // The set the pending (recorded, not yet finished) record() writes -- the one manifolds()
    // returns once finish() has committed it -- and the slot count that record covers.
    VkManifoldSet pendingOutput() const
    {
        VkManifoldSet c{}, p{};
        sets(c, p);
        return c;
    }
    int pendingSlots() const { return m_pendingSlots; }
    VkDeviceAddress counterAddress() const { return counter.address(); }
    int contactCapacity() const { return maxContacts; }
    int manifoldSlots() const { return slots; }
    int contactTotal() const { return contacts; }
    int activeManifolds() const { return active; }
    bool overflowed() const { return overflow; }
    int requiredContacts() const { return requiredContactCount; }

    // Hull pairs (Phase 1.3). hullCounterAddress() counts the hull pairs the last record() found
    // (0 when it recorded no hull path); hullCapacity() is the number of per-lane scratch slots.
    // The world raises the step flag when the count exceeds the capacity and calls growHull()
    // from the rerun, like the pair and contact capacities. hullPairs() is the count read back by
    // finish() (valid for a failed attempt as well).
    VkDeviceAddress hullCounterAddress() const { return hullCounter.address(); }
    bool hullRecorded() const { return m_hullRecorded; }
    int hullCapacity() const { return hullCap; }
    int hullPairs() const { return m_hullPairs; }
    void growHull(int neededPairs);
    const Buffer &bufSatSep(bool useB) const { return useB ? satSepB : satSepA; }

    // Mesh children (Phase 1.5a). Child keys are appended by csMeshMidphase; childCounterAddress()
    // is the raw append count, which can exceed childCapacity(). The sorted child i is manifold slot
    // childBase (= the pair capacity of the record) + i. When the raw count exceeds the capacity the
    // pass is discarded: the world calls growChild(childRaw()) and reruns, like the hull scratch.
    // Manifold slots are pair slots then child slots, so manifoldSlots() includes the child region
    // once a mesh path has been recorded.
    bool meshRecorded() const { return m_meshRecorded; }
    int childCapacity() const { return childCap; }
    int childRaw() const { return m_childRaw; }
    bool childOverflowed() const { return m_childOverflow; }
    int childBase() const { return m_childBase; }
    VkDeviceAddress childCounterAddress() const { return childCounter.address(); }
    void growChild(int neededChildren);
    // Largest manifold slot count the mesh path can address: pair capacity plus child capacity.
    int slotCapacity() const { return maxPairs + childCap; }
    const Buffer &bufSatBits(bool useB) const { return useB ? satBitsB : satBitsA; }

    // Raw buffers, exposed for the test harness's readback / comparison against the reference.
    const Buffer &bufKey(bool useB) const { return useB ? keyB : keyA; }
    const Buffer &bufBasis(bool useB) const { return useB ? basisB : basisA; }
    const Buffer &bufFriction(bool useB) const { return useB ? frictionB : frictionA; }
    const Buffer &bufOffset(bool useB) const { return useB ? offsetB : offsetA; }
    const Buffer &bufCount(bool useB) const { return useB ? countB : countA; }
    const Buffer &bufBodyA(bool useB) const { return useB ? bodyAB : bodyAA; }
    const Buffer &bufBodyB(bool useB) const { return useB ? bodyBB : bodyBA; }
    const Buffer &bufShapeA(bool useB) const { return useB ? shapeAB : shapeAA; }
    const Buffer &bufShapeB(bool useB) const { return useB ? shapeBB : shapeBA; }
    const Buffer &bufFeature(bool useB) const { return useB ? featB : featA; }
    const Buffer &bufRA(bool useB) const { return useB ? rAB : rAA; }
    const Buffer &bufRB(bool useB) const { return useB ? rBB : rBA; }
    const Buffer &bufC0(bool useB) const { return useB ? c0B : c0A; }
    const Buffer &bufPenalty(bool useB) const { return useB ? penB : penA; }
    const Buffer &bufLambda(bool useB) const { return useB ? lamB : lamA; }
    const Buffer &bufStick(bool useB) const { return useB ? stickB : stickA; }
    bool curIsB() const { return curWasB; }

    // Transitive byte total (point 3): both A/B halves plus the per-build scratch
    // (counter/activeCount), for VkWorld::deviceBytesAllocated()'s memory gate.
    uint64_t bytesAllocated() const
    {
        const Buffer *all[] = {
            &keyA,         &keyB,        &basisA,   &basisB,      &frictionA,    &frictionB,   &offsetA,   &offsetB,
            &countA,       &countB,      &bodyAA,   &bodyAB,      &bodyBA,       &bodyBB,      &shapeAA,   &shapeAB,
            &shapeBA,      &shapeBB,     &featA,    &featB,       &rAA,          &rAB,         &rBA,       &rBB,
            &c0A,          &c0B,         &penA,     &penB,        &lamA,         &lamB,        &stickA,    &stickB,
            &counter,      &activeCount, &hullList, &hullCounter, &hullIndirect, &hullScratch, &satSepA,   &satSepB,
            &satBitsA,     &satBitsB,    &meshList, &meshCounter, &meshIndirect, &childCount,  &childKeys, &childSorted,
            &childCounter, &meshScratch,
        };
        uint64_t total = 0;
        for (const Buffer *b : all)
            total += (uint64_t)b->size();
        return total;
    }

private:
    Device *m_dev = nullptr;
    HostRingBuffer ring;
    CommandList cmd;

    // Double-buffered manifold storage, A/B halves swapped each build via `flip`,
    // exactly like keyA/keyB etc in the reference class.
    Buffer keyA, keyB;
    Buffer basisA, basisB;
    Buffer frictionA, frictionB;
    Buffer offsetA, offsetB, countA, countB;
    Buffer bodyAA, bodyAB, bodyBA, bodyBB;
    Buffer shapeAA, shapeAB, shapeBA, shapeBB;
    Buffer featA, featB;
    Buffer rAA, rAB, rBA, rBB, c0A, c0B, penA, penB, lamA, lamB;
    Buffer stickA, stickB;

    Buffer counter, activeCount;
    // Hull pairs: the compacted pair-slot list, its counter, the indirect dispatch, the per-lane
    // b3 scratch, and the persisted b3SATCache (one record per manifold slot, ping-ponged with the
    // manifold halves). Allocated by the first record() that has a hull shape, then only grown.
    Buffer hullList, hullCounter, hullIndirect, hullScratch;
    Buffer satSepA, satSepB, satBitsA, satBitsB;
    int hullCap = 0;
    int m_hullPairs = 0;
    bool m_hullRecorded = false;
    void ensureHull();

    // Mesh pairs (Phase 1.5a): the pair-slot list csNarrowphase appends mesh-vs-convex pairs to, its
    // counter and indirect dispatch, the raw child-key append list and its sorted copy, the clamped
    // child count (also the sort's count), and the per-lane scratch the Box3D triangle collision uses.
    Buffer meshList, meshCounter, meshIndirect;
    Buffer childKeys, childSorted, childCounter, childCount;
    Buffer meshScratch;
    int childCap = 0;
    int m_childBase = 0;
    int m_childRaw = 0;
    int m_prevChildBase = 0;
    int m_prevChildCount = 0;
    bool m_meshRecorded = false;
    bool m_childOverflow = false;
    void ensureMesh();
    void readChildRaw();

    MappedBuffer
        m_readout; // [0] = contact counter, [1] = active manifold count, [2] = hull pairs, [3] = raw child count
    bool m_readoutRecorded = false;
    int m_pendingSlots = 0;

    VkManifoldSet cur{}, prev{};

    int maxPairs = 0, maxContacts = 0;
    int slots = 0, contacts = 0, active = 0;
    int requiredContactCount = 0;
    int prevSlots = 0;
    bool overflow = false;
    bool flip = false;
    bool curWasB = false;

    void sets(VkManifoldSet &curSet, VkManifoldSet &prevSet) const;
};

} // namespace avbdvk
