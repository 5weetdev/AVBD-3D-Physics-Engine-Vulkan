#pragma once

// Render instance, pose and overlay argument structs (render.slang). Mirrors of the Slang structs: field order and padding are byte-for-byte, and
// avbd_test_vk_layout checks them against the compiler's offsets.

#include "vk_buffer.h"
#include "solver_args.h"

#include <cstdint>

namespace avbdvk
{

// ===========================================================================
// render.slang -- reuses FullBodySoAGpu (defined above) for the body
// fields both csWriteInstances and csPackPose read. Field order transcribed from
// render.slang at the time this was written.
// ===========================================================================

// Mirrors BodyInstance byte for byte (44 bytes, no
// padding needed since every field is already 4-byte aligned and the struct ends on
// a 4-byte boundary).
struct BodyInstanceGpu
{
    float pos[3];
    float rot[4];
    float half[3];
    uint32_t rgba;
};
static_assert(sizeof(BodyInstanceGpu) == 44, "must match BodyInstance/AvbdInstance");

// Mirrors render.slang's OverlayView / render_types's OverlayView, field for
// field. _pad0 sits where the compiler needs 4 bytes to bring the pointer fields that
// follow (jointDegree onward) back onto 8-byte alignment after the three trailing
// floats (jointLoadMax/degreeJointMax/penetrationMax) -- verified against the real
// Slang-computed offsets by OverlayViewProbe in layout_probe.slang.
struct OverlayViewGpu
{
    int32_t mode = 0;
    float speedMax = 0, loadMax = 0, degreeMax = 0, invCellSize = 0, sleepFrames = 0;
    int32_t colorCount = 0, colorStatics = 0;
    VkDeviceAddress graphColor = 0, load = 0, degree = 0;
    float jointLoadMax = 0, degreeJointMax = 0, penetrationMax = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress jointDegree = 0, push = 0, pull = 0, stress = 0, cone = 0, penetration = 0;
};
static_assert(sizeof(OverlayViewGpu) % 8 == 0, "layout");

// Full k_write_instances port: frustum culling, every OverlayMode, and per-shape/group
// segmentation, matching the reference parameter order (b, out, maxInstances, frustum,
// visible, ov, groupOf, shapeBase, shapeVisible) exactly.
struct WriteInstancesArgsGpu
{
    FullBodySoAGpu b;
    VkDeviceAddress out = 0;
    int32_t maxInstances = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress frustum = 0; // const float* (24 floats), or 0
    VkDeviceAddress visible = 0; // int* atomic counter, or 0
    OverlayViewGpu ov;
    VkDeviceAddress groupOf = 0;      // const int*, or 0
    VkDeviceAddress shapeBase = 0;    // const int*, or 0
    VkDeviceAddress shapeVisible = 0; // int*, or 0
    ShapeSoAGpu s;                    // one instance per shape; groupOf is per shape
};
static_assert(sizeof(WriteInstancesArgsGpu) % 8 == 0, "layout");

// the overlay reductions' OverlayStats: 6 int* reduction targets.
struct OverlayStatsGpu
{
    VkDeviceAddress push = 0, pull = 0, stress = 0, cone = 0, penetration = 0, jointDegree = 0;
};
static_assert(sizeof(OverlayStatsGpu) % 8 == 0, "layout");

struct OverlayStatsClearArgsGpu
{
    OverlayStatsGpu s;
    int32_t n = 0;
    int32_t _pad0 = 0;
};
static_assert(sizeof(OverlayStatsClearArgsGpu) % 8 == 0, "layout");

struct OverlayStatsContactsArgsGpu
{
    FullManifoldSoAGpu m;
    int32_t slots = 0;
    int32_t _pad0 = 0;
    OverlayStatsGpu s;
};
static_assert(sizeof(OverlayStatsContactsArgsGpu) % 8 == 0, "layout");

struct OverlayStatsForcesArgsGpu
{
    FullBodySoAGpu b;
    FullJointSoAGpu j;
    FullSpringSoAGpu sp;
    int32_t jointsAuthored = 0;
    int32_t _pad0 = 0;
    OverlayStatsGpu s;
};
static_assert(sizeof(OverlayStatsForcesArgsGpu) % 8 == 0, "layout");

struct OverlayLoadClearArgsGpu
{
    VkDeviceAddress load = 0;
    int32_t n = 0;
    int32_t _pad0 = 0;
};
static_assert(sizeof(OverlayLoadClearArgsGpu) % 8 == 0, "layout");

struct OverlayLoadAccumArgsGpu
{
    FullBodySoAGpu b;
    FullManifoldSoAGpu m;
    int32_t slots = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress load = 0;
};
static_assert(sizeof(OverlayLoadAccumArgsGpu) % 8 == 0, "layout");

// render_types's DebugDrawView.
struct DebugDrawViewGpu
{
    int32_t contactPoints = 0, contactNormals = 0, manifoldLinks = 0, tint = 0;
    float pointSize = 0, normalLength = 0, forceMax = 0;
    int32_t _pad0 = 0;
};
static_assert(sizeof(DebugDrawViewGpu) % 8 == 0, "layout");

struct WriteDebugInstancesArgsGpu
{
    FullBodySoAGpu b;
    FullManifoldSoAGpu m;
    int32_t slots = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress out = 0;
    int32_t maxInstances = 0;
    int32_t _pad1 = 0;
    VkDeviceAddress counter = 0;
    DebugDrawViewGpu d;
};
static_assert(sizeof(WriteDebugInstancesArgsGpu) % 8 == 0, "layout");

} // namespace avbdvk
