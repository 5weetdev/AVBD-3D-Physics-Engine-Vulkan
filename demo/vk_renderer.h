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

// Instanced box/sphere/capsule renderer for the Vulkan demo: the Vulkan side of
// demos/sdl_gl/gpu_render.h's Renderer, feature for feature -- flat or per-instance
// colour, fwidth outlines, unlit, a 2048^2 PCF shadow map, planar ground shadows (the
// stencil keeps overlapping casters from double-darkening), 8-tap SSAO and the two
// background themes -- all driven by the shared RenderOptions (demo/
// render_options.h). One Slang module (shaders/scene.slang) holds every pass.
//
// It reads VkWorld::instanceBuffer() (and the rod / marker buffers) directly as the
// per-instance vertex buffer: no host readback, no host-visible copy. VkWorld's
// writeInstances() is self-contained (begins/submits/waits its own dispatch), so the
// compute->vertex hazard is already resolved on the shared queue by draw time.
//
// Boxes and spheres share the pos+normal vertex format: a sphere is a unit-sphere mesh
// fed through the same iPos + qrot(iRot, aPos*iHalf) transform, which is only correct
// because addSphere() gives it a uniform half-extent (r,r,r). Capsules cannot use that
// trick: addCapsule()'s half-extent is {r,r,r+halfSegment}, a non-uniform scale that
// would turn a unit sphere's poles into an ellipsoid. Capsules get their own mesh
// (aDir/aSign) and vertex entry, computing localPos = aDir*radius +
// float3(0,0,aSign*halfSegment) with radius = iHalf.x and halfSegment = iHalf.z -
// iHalf.x, per docs/VK_DEMO_INTERFACE.md.
//
// A frame is two calls. prepare() runs OUTSIDE the main render pass, after the frame's
// fence wait: it writes that frame's parameters and records the shadow-map and SSAO
// depth prepasses into their own depth-only passes. drawScene() runs inside the main
// pass: the lit solid pass, then the planar shadows. The main pass's depth attachment
// must carry stencil (planar shadows count casters per pixel in it) and be cleared to 0.

#include <vulkan/vulkan.h>

#include <cstdint>

#include "render_options.h"
#include "vk_context.h" // demovk::Device, demovk::MappedBuffer

// AVBD_SHAPE_BOX/SPHERE/CAPSULE (src/core/collision.h) == 0/1/2.
enum VkRendererShape
{
    kShapeBox = 0,
    kShapeSphere = 1,
    kShapeCapsule = 2,
};

// One instanced draw: `count` BodyInstances of `shape` starting at `first` in `buf`.
// `bodies` is true for solver bodies and false for joint rods / debug markers, which
// use the same alpha byte for other things (see scene.slang) and never cast shadows.
struct VkDrawItem
{
    int shape = kShapeBox;
    VkBuffer buf = VK_NULL_HANDLE;
    uint32_t first = 0;
    uint32_t count = 0;
    bool bodies = true;
};

class VkRenderer
{
public:
    static constexpr int kMaxFramesInFlight = 4;

    // mainPass: the render pass drawScene() records into (colour + depth/stencil).
    // width/height: its extent, which the SSAO depth prepass matches.
    void init(demovk::Device &dev, VkRenderPass mainPass, uint32_t width, uint32_t height);
    // Swapchain resize. Waits for the device to go idle, then rebuilds the SSAO depth target.
    void resize(uint32_t width, uint32_t height);
    void destroy();

    // Outside any render pass, once the fence for frameSlot (0..kMaxFramesInFlight-1)
    // has signalled: this overwrites that slot's parameters.
    void prepare(VkCommandBuffer cmd, int frameSlot, const VkDrawItem *items, int itemCount,
                 const float viewProj[16], const RenderOptions &opts);
    // Inside mainPass, with the viewport and scissor set to the full extent.
    void drawScene(VkCommandBuffer cmd, const VkDrawItem *items, int itemCount);

private:
    enum Pass
    {
        PASS_MAIN = 0,
        PASS_LIGHT_DEPTH = 1,
        PASS_CAMERA_DEPTH = 2,
        PASS_PLANAR = 3,
        PASS_COUNT = 4
    };

    struct Mesh
    {
        VkBuffer vbo = VK_NULL_HANDLE, ibo = VK_NULL_HANDLE;
        VkDeviceMemory vboMem = VK_NULL_HANDLE, iboMem = VK_NULL_HANDLE;
        uint32_t indexCount = 0;
        VkIndexType indexType = VK_INDEX_TYPE_UINT16;
    };

    // A sampled depth image with its own framebuffer on m_depthPass.
    struct DepthTarget
    {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkFramebuffer fb = VK_NULL_HANDLE;
        uint32_t width = 0, height = 0;
    };

    void createMeshBuffers();
    void createDepthPass();
    void createDescriptors();
    void createPipelines(VkRenderPass mainPass);
    void createDepthTarget(DepthTarget &t, uint32_t w, uint32_t h);
    void destroyDepthTarget(DepthTarget &t);
    void writeDescriptors();
    void drawItems(VkCommandBuffer cmd, Pass pass, const VkDrawItem *items, int itemCount);
    void beginDepthPass(VkCommandBuffer cmd, const DepthTarget &t);

    demovk::Device *m_dev = nullptr;

    Mesh m_box, m_sphere, m_capsule;

    VkShaderModule m_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_descPool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
    VkSampler m_shadowSampler = VK_NULL_HANDLE, m_pointSampler = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    // [0] pos+normal meshes, [1] capsule; one pipeline per Pass.
    VkPipeline m_pipelines[2][PASS_COUNT]{};

    VkRenderPass m_depthPass = VK_NULL_HANDLE;
    VkFormat m_depthFormat = VK_FORMAT_D32_SFLOAT;
    DepthTarget m_shadowMap;   // light's view, kShadowMapSize^2
    DepthTarget m_cameraDepth; // camera's view, swapchain-sized, for SSAO

    // One FrameParams slot per frame in flight, read by the shaders through its address.
    demovk::MappedBuffer m_frameParams;
    int m_slot = 0;
    bool m_drawPlanar = false;
};
