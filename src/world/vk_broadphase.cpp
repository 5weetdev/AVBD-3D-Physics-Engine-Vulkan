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

#include "vk_broadphase.h"
#include "vk_args.h"

namespace avbdvk
{

void BroadphasePipelines::init(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t spirvWords)
{
    aabbAndBin.init(dev, layout, spirv, spirvWords, "csAabbAndBin", "csAabbAndBin");
    aabbStatic.init(dev, layout, spirv, spirvWords, "csAabbStatic", "csAabbStatic");
    aabbBinStatic.init(dev, layout, spirv, spirvWords, "csAabbBinStatic", "csAabbBinStatic");
    resetRanges.init(dev, layout, spirv, spirvWords, "csResetRanges", "csResetRanges");
    cellRanges.init(dev, layout, spirv, spirvWords, "csCellRanges", "csCellRanges");
    emitPairs.init(dev, layout, spirv, spirvWords, "csEmitPairs", "csEmitPairs");
    emitCrossPairs.init(dev, layout, spirv, spirvWords, "csEmitCrossPairs", "csEmitCrossPairs");
    emitOversizePairs.init(dev, layout, spirv, spirvWords, "csEmitOversizePairs", "csEmitOversizePairs");
}

void BroadphasePipelines::destroy()
{
    aabbAndBin.destroy();
    aabbStatic.destroy();
    aabbBinStatic.destroy();
    resetRanges.destroy();
    cellRanges.destroy();
    emitPairs.destroy();
    emitCrossPairs.destroy();
    emitOversizePairs.destroy();
}

// Device-side argument layouts now live in vk_args.h, shared with the layout-gate probe
// (tools/test_vk_layout.cpp + src/shaders/layout_probe.slang) so the struct the
// probe verifies is the exact same one dispatched here.

static int nextPow2i(int v)
{
    int p = 1;
    while (p < v)
        p <<= 1;
    return p;
}

void VkBroadphase::init(Device &dev)
{
    m_dev = &dev;
    ring.init(dev, 1 << 22);
    cmd.init(dev, 8);

    aabbMin.init(dev);
    aabbMax.init(dev);
    cellId.init(dev);
    hash.init(dev);
    hashSorted.init(dev);
    binBody.init(dev);
    binBodySorted.init(dev);
    cellStart.init(dev);
    cellEnd.init(dev);
    rawPairs.init(dev);
    sortedPairs.init(dev);
    m_counter.init(dev);
    sHash.init(dev);
    sHashSorted.init(dev);
    sBinBody.init(dev);
    sBinBodySorted.init(dev);
    sCellStart.init(dev);
    sCellEnd.init(dev);
    if (!m_readout.mapped())
        m_readout.init(dev, sizeof(int32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT, MappedBuffer::HostAccess::Read);
}

void VkBroadphase::destroy()
{
    release();
    m_readout.destroy();
    cmd.destroy();
    ring.destroy();
}

void VkBroadphase::release()
{
    aabbMin.destroy();
    aabbMax.destroy();
    cellId.destroy();
    hash.destroy();
    hashSorted.destroy();
    binBody.destroy();
    binBodySorted.destroy();
    cellStart.destroy();
    cellEnd.destroy();
    rawPairs.destroy();
    sortedPairs.destroy();
    m_counter.destroy();
    sHash.destroy();
    sHashSorted.destroy();
    sBinBody.destroy();
    sBinBodySorted.destroy();
    sCellStart.destroy();
    sCellEnd.destroy();
    m_staticDirty = true;
    m_staticBuiltCount = -1;
}

void VkBroadphase::reserve(int maxBodies /* shapes */, int maxPairs)
{
    tableSize = nextPow2i(maxBodies * 2 < 1024 ? 1024 : maxBodies * 2);
    pairCapacity = maxPairs;

    aabbMin.resize((VkDeviceSize)maxBodies * sizeof(float) * 3);
    aabbMax.resize((VkDeviceSize)maxBodies * sizeof(float) * 3);
    cellId.resize((VkDeviceSize)maxBodies * sizeof(uint64_t));
    hash.resize((VkDeviceSize)maxBodies * sizeof(uint32_t));
    hashSorted.resize((VkDeviceSize)maxBodies * sizeof(uint32_t));
    binBody.resize((VkDeviceSize)maxBodies * sizeof(int32_t));
    binBodySorted.resize((VkDeviceSize)maxBodies * sizeof(int32_t));
    cellStart.resize((VkDeviceSize)tableSize * sizeof(int32_t));
    cellEnd.resize((VkDeviceSize)tableSize * sizeof(int32_t));
    rawPairs.resize((VkDeviceSize)maxPairs * sizeof(uint64_t));
    sortedPairs.resize((VkDeviceSize)maxPairs * sizeof(uint64_t));
    m_counter.resize(sizeof(int32_t));

    sTableSize = tableSize;
    sHash.resize((VkDeviceSize)maxBodies * sizeof(uint32_t));
    sHashSorted.resize((VkDeviceSize)maxBodies * sizeof(uint32_t));
    sBinBody.resize((VkDeviceSize)maxBodies * sizeof(int32_t));
    sBinBodySorted.resize((VkDeviceSize)maxBodies * sizeof(int32_t));
    sCellStart.resize((VkDeviceSize)(sTableSize + 1) * sizeof(int32_t));
    sCellEnd.resize((VkDeviceSize)(sTableSize + 1) * sizeof(int32_t));
    // aabbMin/aabbMax/cellId may have been reallocated: the statics' entries are gone.
    m_staticDirty = true;
}

void VkBroadphase::build(BroadphasePipelines &pl, PrimitivesPipelines &prim, RadixScratch &radix,
                         VkDeviceAddress bodyPos, VkDeviceAddress bodyRot, const BroadphaseShapes &shapes,
                         const Buffer &dynamicIdx, int dynamicCount, const Buffer &staticIdx, int staticCount,
                         float cellSize, const Buffer *ignore, int ignoreCount, const BroadphaseOversize *oversize)
{
    // Standalone (gate) form: the caller's statics may have changed between calls.
    m_staticDirty = true;
    ring.reset();
    cmd.begin();
    cmd.writeTimestamp(0);
    record(cmd, ring, pl, prim, radix, bodyPos, bodyRot, shapes, dynamicIdx, dynamicCount, staticIdx, staticCount,
           cellSize, ignore, ignoreCount, oversize);
    cmd.writeTimestamp(1);
    cmd.submit();
    cmd.wait();
    finish();

    double ms0 = cmd.timestampDeltaNs(0, 1) / 1.0e6;
    m_lastBuildMs = ms0 > 0 ? ms0 : 0.0;
}

void VkBroadphase::finish()
{
    numPairs = 0;
    producedPairCount = 0;
    overflow = false;
    if (!m_recorded)
        return;
    int32_t produced = 0;
    if (m_readoutRecorded)
        produced = *m_readout.mappedAs<int32_t>();
    else
        m_counter.readback(&produced, sizeof(int32_t));
    m_readoutRecorded = false;
    producedPairCount = produced;
    overflow = produced > pairCapacity;
    numPairs = overflow ? pairCapacity : produced;
}

void VkBroadphase::recordReadout(CommandList &cmd)
{
    if (!m_recorded)
        return;
    cmd.copyToHost(m_counter, m_readout, sizeof(int32_t));
    m_readoutRecorded = true;
}

void VkBroadphase::record(CommandList &cmd, HostRingBuffer &ring, BroadphasePipelines &pl, PrimitivesPipelines &prim,
                          RadixScratch &radix, VkDeviceAddress bodyPos, VkDeviceAddress bodyRot,
                          const BroadphaseShapes &shapes, const Buffer &dynamicIdx, int dynamicCount,
                          const Buffer &staticIdx, int staticCount, float cellSize, const Buffer *ignore,
                          int ignoreCount, const BroadphaseOversize *oversize)
{
    // The counter is zeroed even with nothing to do: a consumer recorded after this
    // reads it on device, and must see zero pairs rather than last build's count.
    m_readoutRecorded = false;
    cmd.fill(m_counter, 0u, sizeof(int32_t));
    m_recorded = dynamicCount > 0;
    if (dynamicCount == 0)
        return;

    const uint32_t mask = (uint32_t)(tableSize - 1);
    const float invCellSize = 1.0f / cellSize;
    const uint32_t block = 256;

    // Both sorts below share `radix` inside one submission; grow it for the larger of
    // each buffer up front so neither frees scratch the other's dispatches address.
    reserveRadix(radix, (uint32_t)dynamicCount, sizeof(uint32_t), sizeof(int32_t));
    reserveRadix(radix, (uint32_t)pairCapacity, sizeof(uint64_t), 0);
    if (staticCount > 0)
        reserveRadix(radix, (uint32_t)staticCount, sizeof(uint32_t), sizeof(int32_t));

    BphBodySoAGpu b{
        bodyPos, bodyRot, shapes.shapeBody, shapes.localPos, shapes.localRot, shapes.halfExtents, shapes.flags, 0u, 0u};

    // The grid only bins the non-oversized bodies; oversized ones (too wide for the
    // 27-cell neighbourhood to be exact at this cell size) are tested brute force.
    const bool split = oversize && oversize->oversizeCount > 0 && oversize->gridCount > 0 && oversize->gridIdx &&
                       oversize->oversizeIdx;
    const bool useGrid = split;
    const Buffer &gridIdx = useGrid ? *oversize->gridIdx : dynamicIdx;
    const int gridCount = useGrid ? oversize->gridCount : dynamicCount;
    reserveRadix(radix, (uint32_t)gridCount, sizeof(uint32_t), sizeof(int32_t));

    AabbBinArgsGpu abArgs{b,
                          gridIdx.address(),
                          (uint32_t)gridCount,
                          invCellSize,
                          aabbMin.address(),
                          aabbMax.address(),
                          cellId.address(),
                          hash.address(),
                          binBody.address(),
                          mask,
                          0u};
    VkDeviceAddress abAddr = ring.write(&abArgs, sizeof(abArgs));
    cmd.dispatch(pl.aabbAndBin, (gridCount + block - 1) / block, abAddr);

    if (split)
    {
        // aabbStatic is just "AABB of every body in this list" -- exactly what the
        // oversized bodies need, since they skip aabbAndBin.
        AabbStaticArgsGpu aoArgs{b,
                                 oversize->oversizeIdx->address(),
                                 (uint32_t)oversize->oversizeCount,
                                 0u,
                                 aabbMin.address(),
                                 aabbMax.address()};
        VkDeviceAddress aoAddr = ring.write(&aoArgs, sizeof(aoArgs));
        cmd.dispatch(pl.aabbStatic, (oversize->oversizeCount + block - 1) / block, aoAddr);
    }

    cmd.barrier();

    // Static table (A1), rebuilt only when dirty. Statics never move on their own, so
    // their AABBs, cell ids and sorted table stay valid across steps. Its sort runs
    // before the dynamic one and shares `radix` with it, separated by barriers.
    const uint32_t sMask = (uint32_t)(sTableSize - 1);
    if (staticCount > 0 && (m_staticDirty || m_staticCell != cellSize || m_staticBuiltCount != staticCount))
    {
        StaticBinArgsGpu sbArgs{b,
                                staticIdx.address(),
                                (uint32_t)staticCount,
                                invCellSize,
                                aabbMin.address(),
                                aabbMax.address(),
                                cellId.address(),
                                sHash.address(),
                                sBinBody.address(),
                                sMask,
                                cellSize};
        VkDeviceAddress sbAddr = ring.write(&sbArgs, sizeof(sbArgs));
        cmd.dispatch(pl.aabbBinStatic, (staticCount + block - 1) / block, sbAddr);
        cmd.barrier();
        radixSortPairsU32(cmd, ring, prim, radix, sHash, sHashSorted, sBinBody, sBinBodySorted, (uint32_t)staticCount);
        cmd.barrier();
        ResetRangesArgsGpu srArgs{sCellStart.address(), sCellEnd.address(), (uint32_t)sTableSize + 1u, 0u};
        VkDeviceAddress srAddr = ring.write(&srArgs, sizeof(srArgs));
        cmd.dispatch(pl.resetRanges, (sTableSize + 1 + block - 1) / block, srAddr);
        cmd.barrier();
        CellRangesArgsGpu scArgs{sHashSorted.address(), (uint32_t)staticCount, 0u, sCellStart.address(),
                                 sCellEnd.address()};
        VkDeviceAddress scAddr = ring.write(&scArgs, sizeof(scArgs));
        cmd.dispatch(pl.cellRanges, (staticCount + block - 1) / block, scAddr);
        cmd.barrier();
        m_staticDirty = false;
        m_staticCell = cellSize;
        m_staticBuiltCount = staticCount;
    }

    // Sort dynamic bodies by bucket hash (uint32 key, int32 body-index value) --
    // same radixSortPairsU32 call the broadphase's radixSortPairs<unsigned,int>
    // wrapper performs. (This used to need a submit+wait split here because
    // radixSortPairsU32 raced its own skip-mask reduce pass against this cmd's still-
    // unsubmitted aabbAndBin write -- fixed at the source: the primitives' skip-mask
    // pass is now record-only, appended to this same `cmd` like everything else, so a
    // single recording + one submit is correct again.)
    radixSortPairsU32(cmd, ring, prim, radix, hash, hashSorted, binBody, binBodySorted, (uint32_t)gridCount);
    cmd.barrier();

    ResetRangesArgsGpu rrArgs{cellStart.address(), cellEnd.address(), (uint32_t)tableSize, 0u};
    VkDeviceAddress rrAddr = ring.write(&rrArgs, sizeof(rrArgs));
    cmd.dispatch(pl.resetRanges, (tableSize + block - 1) / block, rrAddr);
    cmd.barrier();

    CellRangesArgsGpu crArgs{hashSorted.address(), (uint32_t)gridCount, 0u, cellStart.address(), cellEnd.address()};
    VkDeviceAddress crAddr = ring.write(&crArgs, sizeof(crArgs));
    cmd.dispatch(pl.cellRanges, (gridCount + block - 1) / block, crAddr);
    cmd.barrier();

    const VkDeviceAddress ignoreAddr = (ignore && ignoreCount > 0) ? ignore->address() : 0;

    EmitPairsArgsGpu epArgs{binBodySorted.address(),
                            (uint32_t)gridCount,
                            0u,
                            cellId.address(),
                            aabbMin.address(),
                            aabbMax.address(),
                            cellStart.address(),
                            cellEnd.address(),
                            invCellSize,
                            mask,
                            rawPairs.address(),
                            m_counter.address(),
                            pairCapacity,
                            0,
                            ignoreAddr,
                            ignoreCount,
                            0,
                            shapes.shapeBody,
                            staticCount > 0 ? sBinBodySorted.address() : 0,
                            staticCount > 0 ? sCellStart.address() : 0,
                            staticCount > 0 ? sCellEnd.address() : 0,
                            sMask,
                            0u};
    VkDeviceAddress epAddr = ring.write(&epArgs, sizeof(epArgs));
    cmd.dispatch(pl.emitPairs, (gridCount + block - 1) / block, epAddr);

    if (staticCount > 0)
    {
        // Gridded dynamics x the statics too wide for a cell (the bucket past the
        // probed range); oversized dynamics x every static.
        const VkDeviceAddress bigStart = sCellStart.address() + (VkDeviceSize)(sMask + 1u) * sizeof(int32_t);
        const VkDeviceAddress bigEnd = sCellEnd.address() + (VkDeviceSize)(sMask + 1u) * sizeof(int32_t);
        EmitCrossPairsArgsGpu ecArgs{gridIdx.address(),
                                     (uint32_t)gridCount,
                                     0u,
                                     sBinBodySorted.address(),
                                     (uint32_t)staticCount,
                                     0u,
                                     bigStart,
                                     bigEnd,
                                     aabbMin.address(),
                                     aabbMax.address(),
                                     rawPairs.address(),
                                     m_counter.address(),
                                     pairCapacity,
                                     0,
                                     ignoreAddr,
                                     ignoreCount,
                                     0,
                                     shapes.shapeBody};
        VkDeviceAddress ecAddr = ring.write(&ecArgs, sizeof(ecArgs));
        cmd.dispatch(pl.emitCrossPairs, (gridCount + block - 1) / block, ecAddr);
        if (split)
        {
            EmitCrossPairsArgsGpu eosArgs{oversize->oversizeIdx->address(),
                                          (uint32_t)oversize->oversizeCount,
                                          0u,
                                          staticIdx.address(),
                                          (uint32_t)staticCount,
                                          0u,
                                          0,
                                          0,
                                          aabbMin.address(),
                                          aabbMax.address(),
                                          rawPairs.address(),
                                          m_counter.address(),
                                          pairCapacity,
                                          0,
                                          ignoreAddr,
                                          ignoreCount,
                                          0,
                                          shapes.shapeBody};
            VkDeviceAddress eosAddr = ring.write(&eosArgs, sizeof(eosArgs));
            cmd.dispatch(pl.emitCrossPairs, (oversize->oversizeCount + block - 1) / block, eosAddr);
        }
    }
    if (split)
    {
        cmd.barrier();
        EmitOversizePairsArgsGpu eoArgs{gridIdx.address(),
                                        (uint32_t)gridCount,
                                        0u,
                                        oversize->oversizeIdx->address(),
                                        (uint32_t)oversize->oversizeCount,
                                        0u,
                                        aabbMin.address(),
                                        aabbMax.address(),
                                        rawPairs.address(),
                                        m_counter.address(),
                                        pairCapacity,
                                        0,
                                        ignoreAddr,
                                        ignoreCount,
                                        0,
                                        shapes.shapeBody};
        VkDeviceAddress eoAddr = ring.write(&eoArgs, sizeof(eoArgs));
        const uint32_t threads = (uint32_t)gridCount + (uint32_t)oversize->oversizeCount;
        cmd.dispatch(pl.emitOversizePairs, (threads + block - 1) / block, eoAddr);
    }
    cmd.barrier();

    // Final canonicalising key sort -- same second CUB call (radixSortKeys<PairKey>)
    // as the reference build(). Recorded in the same submission: the sort is sized by
    // pairCapacity and reads the live pair count from `counter` on device (clamped to
    // the capacity, which is also what the emit kernels' guard wrote up to), so the
    // host no longer needs the count before recording it.
    if (pairCapacity > 0)
        radixSortKeysU64(cmd, ring, prim, radix, rawPairs, sortedPairs, (uint32_t)pairCapacity, m_counter.address());
}

} // namespace avbdvk
