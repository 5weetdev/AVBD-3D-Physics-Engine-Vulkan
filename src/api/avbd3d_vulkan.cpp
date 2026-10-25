// SPDX-FileCopyrightText: 2026 avbd3d contributors
// SPDX-License-Identifier: MIT

// avbd3d_vulkan.h: the shared GPU and the render data (docs/SOLVER_3D_ROADMAP.md A.7).

#include "avbd3d_internal.h"

#include "avbd3d_world.h"
#include "scene_types.h"
#include "vk_device.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace avbd3d_impl;

struct avbd3d_Gpu
{
    avbdvk::Device device;
};

namespace
{
// Equal to kInternalWorld in avbd3d_api.cpp. Every def struct carries it in internalValue.
constexpr int kInternalWorld = 1152023;

static_assert(sizeof(avbd3d_RenderInstance) == sizeof(BodyInstance), "avbd3d_RenderInstance mirrors BodyInstance");
static_assert(offsetof(avbd3d_RenderInstance, rgba) == offsetof(BodyInstance, rgba), "layout");
static_assert((int)avbd3d_overlayCount == (int)OVERLAY_MODE_COUNT, "avbd3d_Overlay mirrors OverlayMode");
static_assert((int)avbd3d_contactTintFlat == (int)CONTACT_TINT_FLAT, "avbd3d_ContactTint mirrors ContactTint");

// Vulkan handles as the opaque values the header uses. Non-dispatchable handles are pointers on
// 64-bit targets and uint64_t on 32-bit ones; memcpy covers both.
template <class H> uint64_t toU64(H h)
{
    static_assert(sizeof(H) <= sizeof(uint64_t), "handle size");
    uint64_t v = 0;
    std::memcpy(&v, &h, sizeof h);
    return v;
}
template <class H> H fromU64(uint64_t v)
{
    H h;
    std::memcpy(&h, &v, sizeof h);
    return h;
}
} // namespace

// ---- GPU ----------------------------------------------------------------------------------

avbd3d_GpuDef avbd3d_DefaultGpuDef(void)
{
    avbd3d_GpuDef d{};
    d.internalValue = kInternalWorld;
    return d;
}

avbd3d_Gpu *avbd3d_CreateGpu(const avbd3d_GpuDef *def, avbd3d_Error *outErr)
{
    *outErr = avbd3d_errorNone;
    if (!def || def->internalValue != kInternalWorld)
        return nullptr;
    avbd3d_Gpu *g = new avbd3d_Gpu;
    try
    {
        if (def->enableValidation)
            avbdvk::setValidationOverride(true);
        if (def->instanceExtensionCount > 0 && def->instanceExtensions)
            g->device.requireInstanceExtensions(std::vector<const char *>(
                def->instanceExtensions, def->instanceExtensions + def->instanceExtensionCount));
        if (def->makeSurface)
        {
            avbd3d_MakeSurfaceFcn *make = def->makeSurface;
            void *ctx = def->surfaceContext;
            g->device.enablePresentation([make, ctx](VkInstance instance) {
                return fromU64<VkSurfaceKHR>(make((void *)instance, ctx));
            });
        }
        g->device.setVerbose(def->verbose);
        g->device.init();
    }
    catch (...)
    {
        std::string text;
        *outErr = errorOfCurrentException(text);
        fprintf(stderr, "[avbd3d] avbd3d_CreateGpu failed: %s\n", text.c_str());
        delete g;
        return nullptr;
    }
    return g;
}

void avbd3d_DestroyGpu(avbd3d_Gpu *gpu)
{
    if (!gpu)
        return;
    destroyWorldsOn(gpu);
    delete gpu; // Device's destructor shuts it down
}

avbd3d_WorldId avbd3d_CreateWorldOnGpu(const avbd3d_WorldDef *def, avbd3d_Gpu *gpu, avbd3d_Error *outErr)
{
    if (!gpu)
        return avbd3d_WorldId{0, 0};
    return createWorldOn(def, &gpu->device, gpu, *outErr);
}

avbd3d_VulkanContext avbd3d_Gpu_GetVulkanContext(const avbd3d_Gpu *gpu)
{
    avbd3d_VulkanContext c{};
    if (!gpu)
        return c;
    const avbdvk::Device &d = gpu->device;
    c.instance = (void *)d.instance();
    c.physicalDevice = (void *)d.physicalDevice();
    c.device = (void *)d.device();
    c.queue = (void *)d.queue();
    c.queueFamily = d.queueFamily();
    c.apiVersion = VK_API_VERSION_1_3; // the instance's
    c.commandPool = toU64(d.commandPool());
    c.pipelineCache = toU64(d.pipelineCache());
    c.surface = toU64(d.surface());
    c.subgroupSize = d.subgroupSize();
    c.deviceName = d.deviceName();
    return c;
}

uint64_t avbd3d_Gpu_GetAllocatedBytes(const avbd3d_Gpu *gpu) { return gpu ? gpu->device.allocatedBytes() : 0; }

int avbd3d_GetValidationMessageCount(void) { return (int)avbdvk::Device::validationMessageCount(); }

// ---- render data --------------------------------------------------------------------------

avbd3d_RenderDef avbd3d_DefaultRenderDef(void)
{
    const OverlayOptions o;
    avbd3d_RenderDef d{};
    d.overlay = avbd3d_overlayNone;
    d.speedMax = o.speedMax;
    d.loadMax = o.loadMax;
    d.jointLoadMax = o.jointLoadMax;
    d.penetrationMax = o.penetrationMax;
    d.degreeMax = o.degreeMax;
    d.jointDegreeMax = o.jointDegreeMax;
    d.colorStatics = o.colorStatics;
    d.internalValue = kInternalWorld;
    return d;
}

void avbd3d_World_WriteInstances(avbd3d_WorldId id, const avbd3d_RenderDef *def)
{
    World *w = settledDevice(id);
    if (!w || !def || def->internalValue != kInternalWorld || w->error() != avbd3d_errorNone)
        return;
    try
    {
        w->syncRenderGroups();
        OverlayOptions o;
        o.mode = (OverlayMode)(def->overlay >= 0 && def->overlay < avbd3d_overlayCount ? def->overlay : 0);
        o.speedMax = def->speedMax;
        o.loadMax = def->loadMax;
        o.degreeMax = def->degreeMax;
        o.jointLoadMax = def->jointLoadMax;
        o.jointDegreeMax = def->jointDegreeMax;
        o.penetrationMax = def->penetrationMax;
        o.colorStatics = def->colorStatics;
        avbdvk::VkWorld &vk = w->vk();
        vk.writeInstances({def->frustumPlanes, vk.overlayView(o, w->params), /*segmentByShape*/ true});
    }
    catch (...)
    {
        w->noteException();
    }
}

uint64_t avbd3d_World_GetInstanceBuffer(avbd3d_WorldId id)
{
    World *w = settledDevice(id);
    return w ? toU64(w->vk().instanceBuffer()) : 0;
}

int avbd3d_World_GetRenderGroupCount(avbd3d_WorldId id)
{
    World *w = settledDevice(id);
    return w ? w->vk().renderGroupCount() : 0;
}

avbd3d_RenderRange avbd3d_World_GetRenderGroupRange(avbd3d_WorldId id, int group)
{
    avbd3d_RenderRange r{0, 0};
    if (World *w = settledDevice(id))
    {
        r.first = w->vk().groupSegmentBase(group);
        r.count = w->vk().groupSegmentCount(group);
    }
    return r;
}

void avbd3d_Body_SetRenderGroup(avbd3d_BodyId id, int group)
{
    int b = 0;
    World *w = worldOfBody(id, b);
    if (!w)
        return;
    const int g = group < 0 ? -1 : group;
    if (w->desc.renderGroup[b] == g)
        return;
    w->desc.renderGroup[b] = g;
    w->renderGroupsDirty = true;
}

int avbd3d_Body_GetRenderGroup(avbd3d_BodyId id)
{
    int b = 0;
    World *w = worldOfBody(id, b);
    return w ? w->desc.renderGroup[b] : -1;
}

avbd3d_ContactMarkerDef avbd3d_DefaultContactMarkerDef(void)
{
    const DebugDrawOptions o;
    avbd3d_ContactMarkerDef d{};
    d.points = true;
    d.tint = (avbd3d_ContactTint)o.contactTint;
    d.pointSize = o.contactPointSize;
    d.normalLength = o.normalLength;
    d.forceMax = o.forceMax;
    d.maxMarkers = o.maxMarkers;
    d.internalValue = kInternalWorld;
    return d;
}

int avbd3d_World_WriteContactMarkers(avbd3d_WorldId id, const avbd3d_ContactMarkerDef *def)
{
    World *w = settledDevice(id);
    if (!w || !def || def->internalValue != kInternalWorld || w->error() != avbd3d_errorNone)
        return 0;
    if (!def->points && !def->normals && !def->manifoldLinks)
        return 0;
    try
    {
        avbdvk::DebugDrawViewGpu v{};
        v.contactPoints = def->points ? 1 : 0;
        v.contactNormals = def->normals ? 1 : 0;
        v.manifoldLinks = def->manifoldLinks ? 1 : 0;
        v.tint = (int32_t)def->tint;
        v.pointSize = def->pointSize;
        v.normalLength = def->normalLength;
        v.forceMax = def->forceMax;
        w->vk().writeDebugInstances(v, def->maxMarkers);
        return w->vk().debugInstanceCount();
    }
    catch (...)
    {
        w->noteException();
        return 0;
    }
}

uint64_t avbd3d_World_GetContactMarkerBuffer(avbd3d_WorldId id)
{
    World *w = settledDevice(id);
    return w ? toU64(w->vk().debugInstanceBuffer()) : 0;
}

int avbd3d_World_GetContactMarkersRequested(avbd3d_WorldId id)
{
    World *w = settledDevice(id);
    return w ? w->vk().debugInstanceRequested() : 0;
}
