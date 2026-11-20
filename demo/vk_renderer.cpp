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

#include "vk_renderer.h"

#include "scene_types.h" // BodyInstance layout

#include "demo_app.h"    // DemoApp::matMul, kNear / kFar
#include "scene_spirv.h" // generated: scene_spirv / scene_spirv_words

#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace demovk;

namespace
{

// Unit cube, vertices at +/-1.
const float kCubeVerts[] = {-1, -1, 1,  0, 0,  1,  1, -1, 1, 0, 0,  1, 1, 1,  1, 0,  0,  1, -1, 1,  1,  0,  0,  1,  -1,
                            -1, -1, 0,  0, -1, -1, 1, -1, 0, 0, -1, 1, 1, -1, 0, 0,  -1, 1, -1, -1, 0,  0,  -1, -1, 1,
                            -1, 0,  1,  0, -1, 1,  1, 0,  1, 0, 1,  1, 1, 0,  1, 0,  1,  1, -1, 0,  1,  0,  -1, -1, -1,
                            0,  -1, 0,  1, -1, -1, 0, -1, 0, 1, -1, 1, 0, -1, 0, -1, -1, 1, 0,  -1, 0,  1,  -1, -1, 1,
                            0,  0,  1,  1, -1, 1,  0, 0,  1, 1, 1,  1, 0, 0,  1, -1, 1,  1, 0,  0,  -1, -1, -1, -1, 0,
                            0,  -1, -1, 1, -1, 0,  0, -1, 1, 1, -1, 0, 0, -1, 1, -1, -1, 0, 0};

const uint16_t kCubeIndices[] = {0,  1,  2,  0,  2,  3,  4,  5,  6,  4,  6,  7,  8,  9,  10, 8,  10, 11,
                                 12, 13, 14, 12, 14, 15, 16, 17, 18, 16, 18, 19, 20, 21, 22, 20, 22, 23};

// Unit UV-sphere (pos == normal, radius 1), drawn through scene.slang's vsMesh for
// rendering addSphere() bodies (uniform half-extent -> elementwise scale is exact).
void buildSphereMesh(std::vector<float> &verts, std::vector<uint32_t> &indices, int stacks = 10, int slices = 16)
{
    for (int i = 0; i <= stacks; ++i)
    {
        float phi = (float)M_PI * (0.5f - (float)i / stacks); // +pi/2 .. -pi/2
        float y = sinf(phi), r = cosf(phi);
        for (int j = 0; j <= slices; ++j)
        {
            float theta = 2.0f * (float)M_PI * (float)j / slices;
            float x = r * cosf(theta), z = r * sinf(theta);
            // pos
            verts.push_back(x);
            verts.push_back(z);
            verts.push_back(y);
            // normal (== pos on a unit sphere)
            verts.push_back(x);
            verts.push_back(z);
            verts.push_back(y);
        }
    }
    int stride = slices + 1;
    for (int i = 0; i < stacks; ++i)
    {
        for (int j = 0; j < slices; ++j)
        {
            uint32_t a = i * stride + j, b = a + 1;
            uint32_t c = (i + 1) * stride + j, d = c + 1;
            indices.push_back(a);
            indices.push_back(c);
            indices.push_back(b);
            indices.push_back(b);
            indices.push_back(c);
            indices.push_back(d);
        }
    }
}

// Capsule mesh: aDir (unit vector) + aSign (+1 top hemisphere/cylinder-top ring, -1
// bottom). See scene.slang's vsCapsule / vk_renderer.h for how the vertex shader turns this into
// a real capsule via localPos = aDir*radius + (0,0,aSign*halfSegment). Built as a UV
// sphere with an extra duplicated equator ring (sign flips there) to open up the
// cylindrical band.
void buildCapsuleMesh(std::vector<float> &verts, std::vector<uint32_t> &indices, int halfStacks = 5, int slices = 16)
{
    int stacks = halfStacks * 2; // even, so i == stacks/2 lands exactly on the equator
    int stride = slices + 1;
    std::vector<int> ringStart;

    auto pushRing = [&](float phi, float sign)
    {
        ringStart.push_back((int)(verts.size() / 4));
        float y = sinf(phi), r = cosf(phi);
        for (int j = 0; j <= slices; ++j)
        {
            float theta = 2.0f * (float)M_PI * (float)j / slices;
            float x = r * cosf(theta), z = r * sinf(theta);
            verts.push_back(x);
            verts.push_back(z);
            verts.push_back(y);
            verts.push_back(sign);
        }
    };

    for (int i = 0; i <= stacks; ++i)
    {
        float phi = (float)M_PI * (0.5f - (float)i / stacks); // +pi/2 .. -pi/2
        if (i == stacks / 2)
        {
            // Equator: two coincident-direction rings, split by sign, so the shader
            // pushes them apart by +/-halfSegment to form the cylinder's two ends.
            pushRing(0.0f, +1.0f);
            pushRing(0.0f, -1.0f);
        }
        else
        {
            float sign = (i < stacks / 2) ? +1.0f : -1.0f;
            pushRing(phi, sign);
        }
    }

    // Connect consecutive rings (skip across the duplicated equator pair, which is
    // itself the two indices around the cylinder band).
    for (size_t r = 0; r + 1 < ringStart.size(); ++r)
    {
        int a0 = ringStart[r], b0 = ringStart[r + 1];
        for (int j = 0; j < slices; ++j)
        {
            uint32_t a = a0 + j, b = a + 1;
            uint32_t c = b0 + j, d = c + 1;
            indices.push_back(a);
            indices.push_back(c);
            indices.push_back(b);
            indices.push_back(b);
            indices.push_back(c);
            indices.push_back(d);
        }
    }
    (void)stride;
}

VkBuffer createDeviceLocalBuffer(Device &dev, VkDeviceSize bytes, VkBufferUsageFlags usage, VkDeviceMemory &outMem,
                                 const void *initialData)
{
    VkBuffer buf = VK_NULL_HANDLE;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(dev.device(), &bci, nullptr, &buf));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev.device(), buf, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = findMemoryType(dev.physicalDevice(), req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(dev.device(), &mai, nullptr, &outMem));
    VK_CHECK(vkBindBufferMemory(dev.device(), buf, outMem, 0));

    if (initialData)
    {
        // One-shot staging upload via a temporary host-visible buffer + copy, exactly
        // Buffer::upload()'s approach in src/world/vk_buffer.cpp.
        VkBuffer staging = VK_NULL_HANDLE;
        VkDeviceMemory stagingMem = VK_NULL_HANDLE;
        VkBufferCreateInfo sbci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        sbci.size = bytes;
        sbci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        sbci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(dev.device(), &sbci, nullptr, &staging));

        VkMemoryRequirements sreq;
        vkGetBufferMemoryRequirements(dev.device(), staging, &sreq);
        VkMemoryAllocateInfo smai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        smai.allocationSize = sreq.size;
        smai.memoryTypeIndex =
            findMemoryType(dev.physicalDevice(), sreq.memoryTypeBits,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VK_CHECK(vkAllocateMemory(dev.device(), &smai, nullptr, &stagingMem));
        VK_CHECK(vkBindBufferMemory(dev.device(), staging, stagingMem, 0));

        void *mapped = nullptr;
        VK_CHECK(vkMapMemory(dev.device(), stagingMem, 0, bytes, 0, &mapped));
        memcpy(mapped, initialData, (size_t)bytes);
        vkUnmapMemory(dev.device(), stagingMem);

        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = dev.commandPool();
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VK_CHECK(vkAllocateCommandBuffers(dev.device(), &cbai, &cmd));

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        VkBufferCopy copy{0, 0, bytes};
        vkCmdCopyBuffer(cmd, staging, buf, 1, &copy);
        VK_CHECK(vkEndCommandBuffer(cmd));

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        VK_CHECK(vkQueueSubmit(dev.queue(), 1, &si, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(dev.queue()));

        vkFreeCommandBuffers(dev.device(), dev.commandPool(), 1, &cmd);
        vkDestroyBuffer(dev.device(), staging, nullptr);
        vkFreeMemory(dev.device(), stagingMem, nullptr);
    }
    return buf;
}

VkShaderModule createShaderModule(Device &dev, const uint32_t *spirv, size_t words)
{
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = words * sizeof(uint32_t);
    ci.pCode = spirv;
    VkShaderModule mod = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(dev.device(), &ci, nullptr, &mod));
    return mod;
}

} // namespace

namespace
{

constexpr uint32_t kShadowMapSize = 2048;

// scene.slang's FrameParams, scalar layout. The asserts pin the offsets spirv-dis
// reports for the shader's struct, so a field added on one side only fails to compile.
struct FrameParams
{
    float viewProj[16];
    float lightViewProj[16];
    float planarViewProj[16];
    float lightDir[3];
    uint32_t shadowMode;
    float blockBaseColor[3];
    uint32_t useBlockColoring;
    float outlineColor[3];
    float outlineThickness;
    uint32_t blockOutlines;
    uint32_t unlit;
    float shadowStrength;
    uint32_t ssaoEnabled;
    float ssaoRadius;
    float ssaoBias;
    float ssaoStrength;
    float zNear;
    float zFar;
    float screenSize[2];
    float shadowTexel;
};
static_assert(offsetof(FrameParams, lightDir) == 192, "FrameParams drifted from scene.slang");
static_assert(offsetof(FrameParams, blockOutlines) == 240, "FrameParams drifted from scene.slang");
static_assert(offsetof(FrameParams, screenSize) == 276, "FrameParams drifted from scene.slang");
static_assert(sizeof(FrameParams) == 288, "FrameParams drifted from scene.slang");

// Slots are 256-byte aligned so each frame in flight owns its own cache lines.
constexpr VkDeviceSize kFrameSlotBytes = (sizeof(FrameParams) + 255) & ~VkDeviceSize(255);

struct PushConstants
{
    uint64_t frameAddr;
    uint32_t pass;
    uint32_t bodies;
};
static_assert(sizeof(PushConstants) == 16, "PushConstants drifted from scene.slang");

void copy3(float out[3], vec3 v)
{
    out[0] = v.x;
    out[1] = v.y;
    out[2] = v.z;
}

// One-shot command buffer on the device's pool, submitted and waited on.
template <typename F> void submitNow(Device &dev, F &&record)
{
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = dev.commandPool();
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(dev.device(), &cbai, &cmd));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    record(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(dev.queue(), 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(dev.queue()));
    vkFreeCommandBuffers(dev.device(), dev.commandPool(), 1, &cmd);
}

enum class PipeKind
{
    Main,       // lit solid pass: LESS, depth write, back-face cull
    LightDepth, // shadow map: front-face cull against self-shadow acne
    CamDepth,   // SSAO prepass: back-face cull
    Planar      // flattened casters: blend, LEQUAL, no depth write, stencil EQUAL 0 / INCR
};

VkPipeline createPipeline(Device &dev, VkRenderPass pass, VkPipelineLayout layout, VkShaderModule mod,
                          const char *vsName, const VkVertexInputBindingDescription *bindings,
                          const VkVertexInputAttributeDescription *attrs, uint32_t attrCount, PipeKind kind)
{
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = mod;
    stages[0].pName = vsName;
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = mod;
    stages[1].pName = kind == PipeKind::Planar ? "fsPlanar" : "fsMain";
    const bool depthOnly = kind == PipeKind::LightDepth || kind == PipeKind::CamDepth;

    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 2;
    vi.pVertexBindingDescriptions = bindings;
    vi.vertexAttributeDescriptionCount = attrCount;
    vi.pVertexAttributeDescriptions = attrs;

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = kind == PipeKind::LightDepth ? VK_CULL_MODE_FRONT_BIT
                  : kind == PipeKind::Planar   ? VK_CULL_MODE_NONE
                                               : VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = kind == PipeKind::Planar ? VK_FALSE : VK_TRUE;
    ds.depthCompareOp = kind == PipeKind::Planar ? VK_COMPARE_OP_LESS_OR_EQUAL : VK_COMPARE_OP_LESS;
    if (kind == PipeKind::Planar)
    {
        // Each pixel takes the first caster that reaches it and counts itself: without
        // this, two overlapping flattened boxes blend twice and the overlap goes darker.
        ds.stencilTestEnable = VK_TRUE;
        ds.front.failOp = VK_STENCIL_OP_KEEP;
        ds.front.passOp = VK_STENCIL_OP_INCREMENT_AND_CLAMP;
        ds.front.depthFailOp = VK_STENCIL_OP_KEEP;
        ds.front.compareOp = VK_COMPARE_OP_EQUAL;
        ds.front.compareMask = 0xFF;
        ds.front.writeMask = 0xFF;
        ds.front.reference = 0;
        ds.back = ds.front;
    }

    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    if (kind == PipeKind::Planar)
    {
        cba.blendEnable = VK_TRUE;
        cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        cba.colorBlendOp = VK_BLEND_OP_ADD;
        cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        cba.alphaBlendOp = VK_BLEND_OP_ADD;
    }
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = depthOnly ? 0 : 1;
    cb.pAttachments = depthOnly ? nullptr : &cba;

    VkDynamicState dynStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates = dynStates;

    VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    // Depth-only passes have no colour attachment and need no fragment stage at all.
    pci.stageCount = depthOnly ? 1 : 2;
    pci.pStages = stages;
    pci.pVertexInputState = &vi;
    pci.pInputAssemblyState = &ia;
    pci.pViewportState = &vp;
    pci.pRasterizationState = &rs;
    pci.pMultisampleState = &ms;
    pci.pDepthStencilState = &ds;
    pci.pColorBlendState = &cb;
    pci.pDynamicState = &dyn;
    pci.layout = layout;
    pci.renderPass = pass;
    pci.subpass = 0;

    VkPipeline p = VK_NULL_HANDLE;
    VK_CHECK(vkCreateGraphicsPipelines(dev.device(), dev.pipelineCache(), 1, &pci, nullptr, &p));
    return p;
}

} // namespace

void VkRenderer::init(Device &dev, VkRenderPass mainPass, uint32_t width, uint32_t height)
{
    m_dev = &dev;
    createMeshBuffers();
    createDepthPass();
    createDepthTarget(m_shadowMap, kShadowMapSize, kShadowMapSize);
    createDepthTarget(m_cameraDepth, width, height);
    createDescriptors();
    writeDescriptors();
    createPipelines(mainPass);
    m_frameParams.init(dev, kFrameSlotBytes * kMaxFramesInFlight);
    memset(m_frameParams.mapped(), 0, (size_t)m_frameParams.size());
}

void VkRenderer::createMeshBuffers()
{
    m_box.vbo = createDeviceLocalBuffer(*m_dev, sizeof(kCubeVerts), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, m_box.vboMem,
                                        kCubeVerts);
    m_box.ibo = createDeviceLocalBuffer(*m_dev, sizeof(kCubeIndices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, m_box.iboMem,
                                        kCubeIndices);
    m_box.indexCount = (uint32_t)(sizeof(kCubeIndices) / sizeof(kCubeIndices[0]));
    m_box.indexType = VK_INDEX_TYPE_UINT16;

    std::vector<float> sv;
    std::vector<uint32_t> si;
    buildSphereMesh(sv, si);
    m_sphere.vbo = createDeviceLocalBuffer(*m_dev, sv.size() * sizeof(float), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                           m_sphere.vboMem, sv.data());
    m_sphere.ibo = createDeviceLocalBuffer(*m_dev, si.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                           m_sphere.iboMem, si.data());
    m_sphere.indexCount = (uint32_t)si.size();
    m_sphere.indexType = VK_INDEX_TYPE_UINT32;

    std::vector<float> cv;
    std::vector<uint32_t> ci;
    buildCapsuleMesh(cv, ci);
    m_capsule.vbo = createDeviceLocalBuffer(*m_dev, cv.size() * sizeof(float), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                            m_capsule.vboMem, cv.data());
    m_capsule.ibo = createDeviceLocalBuffer(*m_dev, ci.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                            m_capsule.iboMem, ci.data());
    m_capsule.indexCount = (uint32_t)ci.size();
    m_capsule.indexType = VK_INDEX_TYPE_UINT32;
}

void VkRenderer::createDepthPass()
{
    // Clear, store, and leave the image ready to sample. D32_SFLOAT is a required
    // depth-attachment and sampled format, so it needs no fallback.
    VkAttachmentDescription att{};
    att.format = m_depthFormat;
    att.samples = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.pDepthStencilAttachment = &ref;

    // In: the previous frame's main pass may still be sampling this image (the targets
    // are shared by the frames in flight), so the clear waits for those reads. Out: the
    // depth writes are visible to the main pass's fragment shader.
    VkSubpassDependency deps[2]{};
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass = 0;
    deps[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    deps[0].srcAccessMask = 0;
    deps[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[1].srcSubpass = 0;
    deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpci.attachmentCount = 1;
    rpci.pAttachments = &att;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &sub;
    rpci.dependencyCount = 2;
    rpci.pDependencies = deps;
    VK_CHECK(vkCreateRenderPass(m_dev->device(), &rpci, nullptr, &m_depthPass));
}

void VkRenderer::createDepthTarget(DepthTarget &t, uint32_t w, uint32_t h)
{
    VkDevice d = m_dev->device();
    t.width = w > 0 ? w : 1;
    t.height = h > 0 ? h : 1;

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = m_depthFormat;
    ici.extent = {t.width, t.height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(d, &ici, nullptr, &t.image));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(d, t.image, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex =
        findMemoryType(m_dev->physicalDevice(), req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(d, &mai, nullptr, &t.mem));
    VK_CHECK(vkBindImageMemory(d, t.image, t.mem, 0));

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = t.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = m_depthFormat;
    vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(d, &vci, nullptr, &t.view));

    VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fci.renderPass = m_depthPass;
    fci.attachmentCount = 1;
    fci.pAttachments = &t.view;
    fci.width = t.width;
    fci.height = t.height;
    fci.layers = 1;
    VK_CHECK(vkCreateFramebuffer(d, &fci, nullptr, &t.fb));

    // Clear it once now, so the image is in SHADER_READ_ONLY with defined contents from
    // the first frame: the descriptor set always points at both targets, whether or not
    // the options that fill them are on.
    submitNow(*m_dev,
              [&](VkCommandBuffer cmd)
              {
                  beginDepthPass(cmd, t);
                  vkCmdEndRenderPass(cmd);
              });
}

void VkRenderer::destroyDepthTarget(DepthTarget &t)
{
    VkDevice d = m_dev->device();
    if (t.fb)
        vkDestroyFramebuffer(d, t.fb, nullptr);
    if (t.view)
        vkDestroyImageView(d, t.view, nullptr);
    if (t.image)
        vkDestroyImage(d, t.image, nullptr);
    if (t.mem)
        vkFreeMemory(d, t.mem, nullptr);
    t = DepthTarget{};
}

void VkRenderer::beginDepthPass(VkCommandBuffer cmd, const DepthTarget &t)
{
    VkClearValue clear{};
    clear.depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo rpbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpbi.renderPass = m_depthPass;
    rpbi.framebuffer = t.fb;
    rpbi.renderArea = {{0, 0}, {t.width, t.height}};
    rpbi.clearValueCount = 1;
    rpbi.pClearValues = &clear;
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
}

void VkRenderer::createDescriptors()
{
    VkDevice d = m_dev->device();

    // Filter the shadow map linearly (comparing the filtered depth)
    // where the device can filter D32_SFLOAT, which is optional in Vulkan.
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(m_dev->physicalDevice(), m_depthFormat, &fp);
    const bool linear = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;

    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod = 0.0f;
    VK_CHECK(vkCreateSampler(d, &sci, nullptr, &m_shadowSampler));
    sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
    VK_CHECK(vkCreateSampler(d, &sci, nullptr, &m_pointSampler));

    VkDescriptorSetLayoutBinding b[4]{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, &m_shadowSampler};
    b[2] = {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[3] = {3, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, &m_pointSampler};
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = 4;
    lci.pBindings = b;
    VK_CHECK(vkCreateDescriptorSetLayout(d, &lci, nullptr, &m_setLayout));

    VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2}, {VK_DESCRIPTOR_TYPE_SAMPLER, 2}};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1;
    pci.poolSizeCount = 2;
    pci.pPoolSizes = sizes;
    VK_CHECK(vkCreateDescriptorPool(d, &pci, nullptr, &m_descPool));

    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = m_descPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &m_setLayout;
    VK_CHECK(vkAllocateDescriptorSets(d, &ai, &m_set));
}

void VkRenderer::writeDescriptors()
{
    VkDescriptorImageInfo img[2]{};
    img[0] = {VK_NULL_HANDLE, m_shadowMap.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    img[1] = {VK_NULL_HANDLE, m_cameraDepth.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet w[2]{};
    for (int i = 0; i < 2; i++)
    {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = m_set;
        w[i].dstBinding = (uint32_t)(i * 2); // 0 shadow map, 2 camera depth
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w[i].pImageInfo = &img[i];
    }
    vkUpdateDescriptorSets(m_dev->device(), 2, w, 0, nullptr);
}

void VkRenderer::createPipelines(VkRenderPass mainPass)
{
    m_module = createShaderModule(*m_dev, scene_spirv, scene_spirv_words);

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &m_setLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcRange;
    VK_CHECK(vkCreatePipelineLayout(m_dev->device(), &plci, nullptr, &m_pipelineLayout));

    // Binding 0 per-vertex mesh data, binding 1 the per-instance BodyInstance, for both
    // formats: [0] pos + normal (24 bytes), [1] capsule aDir + aSign (16 bytes).
    VkVertexInputBindingDescription bindings[2][2]{};
    bindings[0][0] = {0, 6 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX};
    bindings[1][0] = {0, 4 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX};
    bindings[0][1] = bindings[1][1] = {1, sizeof(BodyInstance), VK_VERTEX_INPUT_RATE_INSTANCE};

    VkVertexInputAttributeDescription attrs[2][6]{};
    attrs[0][0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0};
    attrs[0][1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 3 * sizeof(float)};
    attrs[1][0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0};
    attrs[1][1] = {1, 0, VK_FORMAT_R32_SFLOAT, 3 * sizeof(float)};
    for (int f = 0; f < 2; f++)
    {
        attrs[f][2] = {2, 1, VK_FORMAT_R32G32B32_SFLOAT, (uint32_t)offsetof(BodyInstance, pos)};
        attrs[f][3] = {3, 1, VK_FORMAT_R32G32B32A32_SFLOAT, (uint32_t)offsetof(BodyInstance, rot)};
        attrs[f][4] = {4, 1, VK_FORMAT_R32G32B32_SFLOAT, (uint32_t)offsetof(BodyInstance, half)};
        attrs[f][5] = {5, 1, VK_FORMAT_R8G8B8A8_UNORM, (uint32_t)offsetof(BodyInstance, rgba)};
    }

    const char *vsNames[2] = {"vsMesh", "vsCapsule"};
    for (int f = 0; f < 2; f++)
    {
        auto make = [&](VkRenderPass pass, PipeKind kind)
        {
            return createPipeline(*m_dev, pass, m_pipelineLayout, m_module, vsNames[f], bindings[f], attrs[f], 6, kind);
        };
        m_pipelines[f][PASS_MAIN] = make(mainPass, PipeKind::Main);
        m_pipelines[f][PASS_LIGHT_DEPTH] = make(m_depthPass, PipeKind::LightDepth);
        m_pipelines[f][PASS_CAMERA_DEPTH] = make(m_depthPass, PipeKind::CamDepth);
        m_pipelines[f][PASS_PLANAR] = make(mainPass, PipeKind::Planar);
    }
}

void VkRenderer::resize(uint32_t width, uint32_t height)
{
    if (!m_dev || (width == m_cameraDepth.width && height == m_cameraDepth.height))
        return;
    // The descriptor set and the old target may still be in use by frames in flight.
    vkDeviceWaitIdle(m_dev->device());
    destroyDepthTarget(m_cameraDepth);
    createDepthTarget(m_cameraDepth, width, height);
    writeDescriptors();
}

void VkRenderer::drawItems(VkCommandBuffer cmd, Pass pass, const VkDrawItem *items, int itemCount)
{
    int bound = -1;
    for (int i = 0; i < itemCount; i++)
    {
        const VkDrawItem &it = items[i];
        if (it.count == 0 || it.buf == VK_NULL_HANDLE)
            continue;
        // Only bodies cast shadows or occlude for SSAO; skip draws that would all clip.
        if (pass != PASS_MAIN && !it.bodies)
            continue;

        const int fmt = it.shape == kShapeCapsule ? 1 : 0;
        if (fmt != bound)
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelines[fmt][pass]);
            bound = fmt;
        }

        PushConstants pc{};
        pc.frameAddr = m_frameParams.address() + (VkDeviceAddress)m_slot * kFrameSlotBytes;
        pc.pass = (uint32_t)pass;
        pc.bodies = it.bodies ? 1u : 0u;
        vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(pc), &pc);

        const Mesh *mesh = it.shape == kShapeSphere ? &m_sphere : it.shape == kShapeCapsule ? &m_capsule : &m_box;
        // Binding 1 (per-instance) is offset by first*sizeof(BodyInstance) rather than
        // passing firstInstance to the draw: offsetting the binding is the portable way
        // to select a shape's segment of the buffer.
        VkBuffer vbs[2] = {mesh->vbo, it.buf};
        VkDeviceSize offs[2] = {0, (VkDeviceSize)it.first * sizeof(BodyInstance)};
        vkCmdBindVertexBuffers(cmd, 0, 2, vbs, offs);
        vkCmdBindIndexBuffer(cmd, mesh->ibo, 0, mesh->indexType);
        vkCmdDrawIndexed(cmd, mesh->indexCount, it.count, 0, 0, 0);
    }
}

void VkRenderer::prepare(VkCommandBuffer cmd, int frameSlot, const VkDrawItem *items, int itemCount,
                         const float viewProj[16], const RenderOptions &opts)
{
    m_slot = frameSlot % kMaxFramesInFlight;

    FrameParams fp{};
    memcpy(fp.viewProj, viewProj, sizeof(fp.viewProj));
    shadowLightViewProj(opts, /*vulkanClip*/ true, fp.lightViewProj);
    float planar[16];
    planarShadowMatrix(opts, planar);
    DemoApp::matMul(fp.planarViewProj, viewProj, planar);
    copy3(fp.lightDir, opts.lightDir);
    fp.shadowMode = (uint32_t)opts.shadowMode;
    copy3(fp.blockBaseColor, opts.blockBaseColor);
    fp.useBlockColoring = (opts.blockColoring || opts.forceInstanceColor) ? 1u : 0u;
    copy3(fp.outlineColor, opts.outlineColor);
    fp.outlineThickness = opts.outlineThickness;
    fp.blockOutlines = opts.blockOutlines ? 1u : 0u;
    fp.unlit = opts.unlit ? 1u : 0u;
    fp.shadowStrength = opts.shadowStrength;
    fp.ssaoEnabled = opts.ssaoEnabled ? 1u : 0u;
    fp.ssaoRadius = opts.ssaoRadius;
    fp.ssaoBias = opts.ssaoBias;
    fp.ssaoStrength = opts.ssaoStrength;
    fp.zNear = DemoApp::kNear;
    fp.zFar = DemoApp::kFar;
    fp.screenSize[0] = (float)m_cameraDepth.width;
    fp.screenSize[1] = (float)m_cameraDepth.height;
    fp.shadowTexel = 1.0f / (float)kShadowMapSize;
    memcpy((uint8_t *)m_frameParams.mapped() + (size_t)m_slot * kFrameSlotBytes, &fp, sizeof(fp));

    m_drawPlanar = opts.shadowMode == SHADOW_GROUND_PLANAR && opts.shadowStrength > 0.001f;

    auto depthPass = [&](const DepthTarget &t, Pass pass)
    {
        beginDepthPass(cmd, t);
        VkViewport viewport{0, 0, (float)t.width, (float)t.height, 0.0f, 1.0f};
        VkRect2D scissor{{0, 0}, {t.width, t.height}};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        drawItems(cmd, pass, items, itemCount);
        vkCmdEndRenderPass(cmd);
    };
    if (opts.shadowMode == SHADOW_MAP)
        depthPass(m_shadowMap, PASS_LIGHT_DEPTH);
    if (opts.ssaoEnabled)
        depthPass(m_cameraDepth, PASS_CAMERA_DEPTH);
}

void VkRenderer::drawScene(VkCommandBuffer cmd, const VkDrawItem *items, int itemCount)
{
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_set, 0, nullptr);
    drawItems(cmd, PASS_MAIN, items, itemCount);
    // Planar shadows go over the solid ground.
    if (m_drawPlanar)
        drawItems(cmd, PASS_PLANAR, items, itemCount);
}

void VkRenderer::destroy()
{
    if (!m_dev)
        return;
    VkDevice d = m_dev->device();
    if (d)
    {
        vkDeviceWaitIdle(d);
        for (auto &row : m_pipelines)
            for (VkPipeline &p : row)
            {
                if (p)
                    vkDestroyPipeline(d, p, nullptr);
                p = VK_NULL_HANDLE;
            }
        if (m_pipelineLayout)
            vkDestroyPipelineLayout(d, m_pipelineLayout, nullptr);
        if (m_module)
            vkDestroyShaderModule(d, m_module, nullptr);
        if (m_descPool)
            vkDestroyDescriptorPool(d, m_descPool, nullptr);
        if (m_setLayout)
            vkDestroyDescriptorSetLayout(d, m_setLayout, nullptr);
        if (m_shadowSampler)
            vkDestroySampler(d, m_shadowSampler, nullptr);
        if (m_pointSampler)
            vkDestroySampler(d, m_pointSampler, nullptr);
        destroyDepthTarget(m_shadowMap);
        destroyDepthTarget(m_cameraDepth);
        if (m_depthPass)
            vkDestroyRenderPass(d, m_depthPass, nullptr);
        m_frameParams.destroy();
        for (Mesh *m : {&m_box, &m_sphere, &m_capsule})
        {
            if (m->vbo)
                vkDestroyBuffer(d, m->vbo, nullptr);
            if (m->vboMem)
                vkFreeMemory(d, m->vboMem, nullptr);
            if (m->ibo)
                vkDestroyBuffer(d, m->ibo, nullptr);
            if (m->iboMem)
                vkFreeMemory(d, m->iboMem, nullptr);
        }
    }
    m_pipelineLayout = VK_NULL_HANDLE;
    m_module = VK_NULL_HANDLE;
    m_descPool = VK_NULL_HANDLE;
    m_set = VK_NULL_HANDLE;
    m_setLayout = VK_NULL_HANDLE;
    m_shadowSampler = m_pointSampler = VK_NULL_HANDLE;
    m_depthPass = VK_NULL_HANDLE;
    m_box = Mesh{};
    m_sphere = Mesh{};
    m_capsule = Mesh{};
    m_dev = nullptr;
}
