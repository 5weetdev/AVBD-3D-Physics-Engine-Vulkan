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

// Broadphase argument structs (broadphase.slang). Mirrors of the Slang structs: field order and padding are byte-for-byte, and
// avbd_test_vk_layout checks them against the compiler's offsets.

#include "vk_buffer.h"

#include <cstdint>

namespace avbdvk
{

// ===========================================================================
// broadphase.slang
// ===========================================================================

struct BphBodySoAGpu
{
    VkDeviceAddress pos, rot;  // per body
    VkDeviceAddress shapeBody; // per shape from here down
    VkDeviceAddress localPos, localRot, halfExtents, flags;
    uint32_t count, _pad0;
};
static_assert(sizeof(BphBodySoAGpu) % 8 == 0, "layout");

struct AabbBinArgsGpu
{
    BphBodySoAGpu b;
    VkDeviceAddress idx;
    uint32_t n;
    float invCellSize;
    VkDeviceAddress aabbMin, aabbMax, cellId, hash;
    VkDeviceAddress binBody;
    uint32_t mask;
    uint32_t _pad0;
};
static_assert(sizeof(AabbBinArgsGpu) % 8 == 0, "layout");

struct AabbStaticArgsGpu
{
    BphBodySoAGpu b;
    VkDeviceAddress idx;
    uint32_t n, _pad0;
    VkDeviceAddress aabbMin, aabbMax;
};
static_assert(sizeof(AabbStaticArgsGpu) % 8 == 0, "layout");

struct ResetRangesArgsGpu
{
    VkDeviceAddress cellStart, cellEnd;
    uint32_t tableSize, _pad0;
};
static_assert(sizeof(ResetRangesArgsGpu) % 8 == 0, "layout");

struct CellRangesArgsGpu
{
    VkDeviceAddress hashSorted;
    uint32_t n, _pad0;
    VkDeviceAddress cellStart, cellEnd;
};
static_assert(sizeof(CellRangesArgsGpu) % 8 == 0, "layout");

struct EmitPairsArgsGpu
{
    VkDeviceAddress binBodySorted;
    uint32_t n, _pad0;
    VkDeviceAddress cellId, aabbMin, aabbMax, cellStart, cellEnd;
    float invCellSize;
    uint32_t mask;
    VkDeviceAddress pairs, counter;
    int32_t capacity, _pad1;
    VkDeviceAddress ignore;
    int32_t ignoreCount, _pad2;
    VkDeviceAddress shapeBody;
    VkDeviceAddress sBinBodySorted; // may be null
    VkDeviceAddress sCellStart;     // may be null: no static table
    VkDeviceAddress sCellEnd;       // may be null
    uint32_t sMask, _pad3;
};
static_assert(sizeof(EmitPairsArgsGpu) % 8 == 0, "layout");

struct StaticBinArgsGpu
{
    BphBodySoAGpu b;
    VkDeviceAddress idx;
    uint32_t n;
    float invCellSize;
    VkDeviceAddress aabbMin, aabbMax, cellId, hash;
    VkDeviceAddress binBody;
    uint32_t mask;
    float cellSize;
};
static_assert(sizeof(StaticBinArgsGpu) % 8 == 0, "layout");

struct EmitCrossPairsArgsGpu
{
    VkDeviceAddress aIdx;
    uint32_t nA, _pad0;
    VkDeviceAddress bIdx;
    uint32_t nB, _pad1;
    VkDeviceAddress bStart; // may be null
    VkDeviceAddress bEnd;   // may be null
    VkDeviceAddress aabbMin, aabbMax, pairs, counter;
    int32_t capacity, _pad2;
    VkDeviceAddress ignore;
    int32_t ignoreCount, _pad3;
    VkDeviceAddress shapeBody;
};
static_assert(sizeof(EmitCrossPairsArgsGpu) % 8 == 0, "layout");

struct EmitOversizePairsArgsGpu
{
    VkDeviceAddress gridIdx;
    uint32_t nGrid, _pad0;
    VkDeviceAddress oversizeIdx;
    uint32_t nOver, _pad1;
    VkDeviceAddress aabbMin, aabbMax, pairs, counter;
    int32_t capacity, _pad2;
    VkDeviceAddress ignore;
    int32_t ignoreCount, _pad3;
    VkDeviceAddress shapeBody;
};
static_assert(sizeof(EmitOversizePairsArgsGpu) % 8 == 0, "layout");

} // namespace avbdvk
