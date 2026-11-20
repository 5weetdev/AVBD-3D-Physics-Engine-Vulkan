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

// Vulkan host orchestration for src/shaders/broadphase.slang, mirroring
// engine/collide/gpu_broadphase.cuh's Broadphase class phase for phase: AABB+bin,
// bucket sort (radixSortPairsU32), cell ranges, pair emission, final key sort
// (radixSortKeysU64).
//
// record() appends the whole build to a caller's command buffer and never submits: the
// final key sort is sized by capacity and reads the live pair count from counter() on
// device, so a consumer (the narrowphase) can be recorded straight after it. finish()
// reads the count back once the caller's submission has completed. build() is the
// standalone form -- its own command buffer, one submit -- that the gates use.

#include "vk_buffer.h"
#include "vk_cmd.h"
#include "vk_device.h"
#include "vk_pipeline.h"
#include "vk_primitives.h"

#include <cstdint>

namespace avbdvk
{

struct BroadphasePipelines
{
    Pipeline aabbAndBin, aabbStatic, aabbBinStatic, resetRanges, cellRanges, emitPairs, emitCrossPairs,
        emitOversizePairs;

    void init(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t spirvWords);
    void destroy();
};

// The shape table the broadphase keys on. Every index list (dynamic, static, grid, oversize)
// and every pair the broadphase emits is in SHAPE indices; shapeBody maps a shape to its body
// (same-body shapes never pair; the ignore list is body-pair based) and bodyPos/bodyRot are
// per body. A shape whose flags have AVBD_SHAPE_FLAG_IDENTITY (bit 0) uses the body pose
// as is.
struct BroadphaseShapes
{
    VkDeviceAddress shapeBody = 0;
    VkDeviceAddress localPos = 0, localRot = 0;
    VkDeviceAddress halfExtents = 0;
    VkDeviceAddress flags = 0;
};

// Optional split of the dynamic shapes for a mixed-size scene: `gridIdx` are the bodies
// small enough for the uniform grid, `oversizeIdx` the few too wide for it, which are
// tested brute force against everything. Together they are exactly the dynamic set.
struct BroadphaseOversize
{
    const Buffer *gridIdx = nullptr;
    int gridCount = 0;
    const Buffer *oversizeIdx = nullptr;
    int oversizeCount = 0;
};

class VkBroadphase
{
public:
    void init(Device &dev);
    void destroy();
    // Frees the capacity-sized buffers and keeps the command pool and ring, so the
    // next reserve() rebuilds from nothing. VkWorld::reset() calls this: a scene
    // switch must not hold the old scene's memory until the new one is built.
    void release();

    void reserve(int maxShapes, int maxPairs); // capacities in SHAPES

    // The static table (statics binned into their own sorted cell table) is rebuilt in
    // the next record() only when it is dirty: a static moved, the static list or the
    // cell size changed, or reserve() reallocated the body-indexed AABB arrays.
    void markStaticDirty() { m_staticDirty = true; }
    bool staticDirty() const { return m_staticDirty; }

    // bodyPos/bodyRot: device addresses of the per-body vec3*/quat* arrays; shapes: the shape
    // table (see BroadphaseShapes). dynamicIdx/staticIdx list SHAPE indices -- see
    // broadphase.slang's BphBodySoA. ignore/ignoreCount: optional sorted PairKey (uint64) suppression
    // list, same contract as the reference class.
    void build(BroadphasePipelines &pl, PrimitivesPipelines &prim, RadixScratch &radix, VkDeviceAddress bodyPos,
               VkDeviceAddress bodyRot, const BroadphaseShapes &shapes, const Buffer &dynamicIdx, int dynamicCount,
               const Buffer &staticIdx, int staticCount, float cellSize, const Buffer *ignore = nullptr,
               int ignoreCount = 0, const BroadphaseOversize *oversize = nullptr);

    void record(CommandList &cmd, HostRingBuffer &ring, BroadphasePipelines &pl, PrimitivesPipelines &prim,
                RadixScratch &radix, VkDeviceAddress bodyPos, VkDeviceAddress bodyRot, const BroadphaseShapes &shapes,
                const Buffer &dynamicIdx, int dynamicCount, const Buffer &staticIdx, int staticCount, float cellSize,
                const Buffer *ignore = nullptr, int ignoreCount = 0, const BroadphaseOversize *oversize = nullptr);
    // After the recording submission has completed: reads the produced count back and
    // sets pairCount()/overflowed()/producedPairs().
    void finish();
    // Optional: records a copy of the produced count into a host-mapped readout, so
    // finish() reads mapped memory instead of a one-shot readback submit. Call after
    // record(), before the submission. finish() falls back to readback() without it.
    void recordReadout(CommandList &cmd);

    const Buffer &pairs() const { return sortedPairs; }
    // Pairs emitted this build, on device (may exceed capacity(); consumers clamp).
    const Buffer &counter() const { return m_counter; }
    int pairCount() const { return numPairs; }
    bool overflowed() const { return overflow; }
    int capacity() const { return pairCapacity; }
    int producedPairs() const { return producedPairCount; }

    // Debug/test-only accessors to the intermediate buffers, for isolating a mismatch
    // to a specific stage (AABB, cell id, hash bucket, sort, range) rather than only
    // ever seeing the final pair set.
    const Buffer &dbgAabbMin() const { return aabbMin; }
    const Buffer &dbgAabbMax() const { return aabbMax; }
    const Buffer &dbgCellId() const { return cellId; }
    const Buffer &dbgHash() const { return hash; }
    const Buffer &dbgHashSorted() const { return hashSorted; }
    const Buffer &dbgBinBody() const { return binBody; }
    const Buffer &dbgBinBodySorted() const { return binBodySorted; }
    const Buffer &dbgCellStart() const { return cellStart; }
    const Buffer &dbgCellEnd() const { return cellEnd; }
    int dbgTableSize() const { return tableSize; }

    double lastBuildMs() const { return m_lastBuildMs; }

    // Transitive byte total (point 3): every device buffer this class owns, for
    // VkWorld::deviceBytesAllocated()'s memory gate.
    uint64_t bytesAllocated() const
    {
        const Buffer *all[] = {&aabbMin,       &aabbMax,     &cellId,   &hash,           &hashSorted,  &binBody,
                               &binBodySorted, &cellStart,   &cellEnd,  &rawPairs,       &sortedPairs, &m_counter,
                               &sHash,         &sHashSorted, &sBinBody, &sBinBodySorted, &sCellStart,  &sCellEnd};
        uint64_t total = 0;
        for (const Buffer *b : all)
            total += (uint64_t)b->size();
        return total;
    }

private:
    Device *m_dev = nullptr;
    HostRingBuffer m_ring;
    CommandList m_cmd;

    Buffer aabbMin, aabbMax;
    Buffer cellId;
    Buffer hash, hashSorted;
    Buffer binBody, binBodySorted;
    Buffer cellStart, cellEnd;
    Buffer rawPairs, sortedPairs;
    Buffer m_counter;
    // Static table: list-indexed hash/body, ranges over sTableSize + 1 buckets (the extra
    // one holds the statics too wide for a cell, walked brute force).
    Buffer sHash, sHashSorted, sBinBody, sBinBodySorted, sCellStart, sCellEnd;
    int sTableSize = 0;
    bool m_staticDirty = true;
    float m_staticCell = 0.0f;
    int m_staticBuiltCount = -1;
    MappedBuffer m_readout;
    bool m_readoutRecorded = false;

    int pairCapacity = 0;
    int tableSize = 0;
    int numPairs = 0;
    int producedPairCount = 0;
    bool m_recorded = false;
    bool overflow = false;
    double m_lastBuildMs = 0.0;
};

} // namespace avbdvk
