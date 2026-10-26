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

// Vulkan Physics Engine demo, Vulkan solver + Vulkan renderer.
//
// Everything that is not the window or the drawing -- scenes, the panel, the camera,
// grabbing and spawning -- is demo/demo_app. This file owns the swapchain, the
// render pass and its "Graphics" panel section, and renders straight from the world's own
// instance buffer.
//
// DemoApp owns the simulation: a SceneBuilder authors scenes and holds the parameters,
// and an avbd3d world (app.world()) simulates. This file only talks to the solver through
// the public avbd3d_* API: it creates the GPU, shares its device and draws what the
// World_Write* calls produce.

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#include <backends/imgui_impl_sdl2.h>
#include <backends/imgui_impl_vulkan.h>
#include <imgui.h>

#include <vulkan/vulkan.h>

#include "vk_renderer.h"

#include "demo_app.h"
#include "demo_log.h"
#include "maths.h"
#include "vk_context.h"

#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#define WinWidth 1280
#define WinHeight 720
#define kFramesInFlight 2

static SDL_Window *window = nullptr;

static DemoApp app;
static VkRenderer g_renderer;

// The swapchain and renderer share the device of the solver's GPU rather than creating a
// second one: Vulkan objects are device-scoped, so the world's instance buffer can only be
// bound in command buffers of the same device.
static Avbd3dGpu *g_gpu = nullptr;
static demovk::Device g_device;

static VkSurfaceKHR g_surface = VK_NULL_HANDLE;
static VkSwapchainKHR g_swapchain = VK_NULL_HANDLE;
static VkFormat g_swapFormat = VK_FORMAT_B8G8R8A8_UNORM;
static VkExtent2D g_swapExtent{WinWidth, WinHeight};
static std::vector<VkImage> g_swapImages;
static std::vector<VkImageView> g_swapViews;
static std::vector<VkFramebuffer> g_framebuffers;
// One "render finished" semaphore PER SWAPCHAIN IMAGE, not per frame-in-flight: the
// index passed to vkQueuePresentKHR is the acquired image index, which does not cycle
// in lockstep with g_frameIdx (kFramesInFlight=2 vs. however many swapchain images
// there are), so reusing a frame-indexed semaphore as the present wait can signal one
// that a previous present of a DIFFERENT image is still consuming (validation:
// VUID-vkQueueSubmit-pSignalSemaphores-00067).
static std::vector<VkSemaphore> g_renderDonePerImage;
static VkRenderPass g_renderPass = VK_NULL_HANDLE;

static VkImage g_depthImage = VK_NULL_HANDLE;
static VkDeviceMemory g_depthMem = VK_NULL_HANDLE;
static VkImageView g_depthView = VK_NULL_HANDLE;
// Depth + stencil: the planar ground shadows count casters per pixel in the stencil
// (see VkRenderer). Picked at startup by chooseDepthFormat().
static VkFormat g_depthFormat = VK_FORMAT_D32_SFLOAT_S8_UINT;

static VkCommandPool g_cmdPool = VK_NULL_HANDLE;
static VkCommandBuffer g_cmdBufs[kFramesInFlight]{};
static VkSemaphore g_imageAvail[kFramesInFlight]{};
static VkFence g_inFlight[kFramesInFlight]{};
static int g_frameIdx = 0;

// The swapchain is built with FIFO (always supported). The panel's VSync checkbox
// only changes between FIFO and MAILBOX -- and MAILBOX is not guaranteed to exist, so
// if the device lacks it the checkbox honestly says it is ignored.
static bool g_wantVsync = true;
static bool g_mailboxAvailable = false;
static bool g_swapDirty = false; // recreate on the next frame boundary

// Everything the "Graphics" panel drives -- shadows, SSAO, outlines, unlit, block colour,
// background theme -- lives in demo/render_options, which is renderer-agnostic.
static RenderOptions g_renderOpts;

// ---------------------------------------------------------------------------
// Gribb-Hartmann frustum extraction (standard plane extraction), for
// avbd3d_world_write_instances's frustum culling.
// ---------------------------------------------------------------------------
static void extractFrustum(const float m[16], float planes[24])
{
    const float rows[4][4] = {
        {m[0], m[4], m[8], m[12]},
        {m[1], m[5], m[9], m[13]},
        {m[2], m[6], m[10], m[14]},
        {m[3], m[7], m[11], m[15]},
    };

    const int src[6] = {0, 0, 1, 1, 2, 2};
    const float sign[6] = {1, -1, 1, -1, 1, -1};

    for (int i = 0; i < 6; i++)
    {
        float p[4];
        for (int k = 0; k < 4; k++)
            p[k] = rows[3][k] + sign[i] * rows[src[i]][k];

        const float len = sqrtf(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        const float inv = len > 0.0f ? 1.0f / len : 0.0f;
        for (int k = 0; k < 4; k++)
            planes[i * 4 + k] = p[k] * inv;
    }
}

// ---------------------------------------------------------------------------
// Swapchain / render pass / framebuffers
// ---------------------------------------------------------------------------
static VkSurfaceFormatKHR chooseSurfaceFormat()
{
    uint32_t n = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(g_device.physicalDevice(), g_surface, &n, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(g_device.physicalDevice(), g_surface, &n, formats.data());
    for (auto &f : formats)
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return f;
    return formats[0];
}

// Neither depth-stencil format is mandatory on its own; one of the two always is.
static VkFormat chooseDepthFormat()
{
    const VkFormat candidates[2] = {VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};
    for (VkFormat f : candidates)
    {
        VkFormatProperties fp;
        vkGetPhysicalDeviceFormatProperties(g_device.physicalDevice(), f, &fp);
        if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
            return f;
    }
    fprintf(stderr, "avbd_vk: no depth-stencil attachment format\n");
    exit(1);
}

static void createDepthResources()
{
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = g_depthFormat;
    ici.extent = {g_swapExtent.width, g_swapExtent.height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    VK_CHECK(vkCreateImage(g_device.device(), &ici, nullptr, &g_depthImage));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(g_device.device(), g_depthImage, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = demovk::findMemoryType(g_device.physicalDevice(), req.memoryTypeBits,
                                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(g_device.device(), &mai, nullptr, &g_depthMem));
    VK_CHECK(vkBindImageMemory(g_device.device(), g_depthImage, g_depthMem, 0));

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = g_depthImage;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = g_depthFormat;
    vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(g_device.device(), &vci, nullptr, &g_depthView));
}

static void destroySwapchainObjects()
{
    for (auto fb : g_framebuffers) if (fb) vkDestroyFramebuffer(g_device.device(), fb, nullptr);
    g_framebuffers.clear();
    for (auto v : g_swapViews) if (v) vkDestroyImageView(g_device.device(), v, nullptr);
    g_swapViews.clear();
    for (auto s : g_renderDonePerImage) if (s) vkDestroySemaphore(g_device.device(), s, nullptr);
    g_renderDonePerImage.clear();
    if (g_depthView) { vkDestroyImageView(g_device.device(), g_depthView, nullptr); g_depthView = VK_NULL_HANDLE; }
    if (g_depthImage) { vkDestroyImage(g_device.device(), g_depthImage, nullptr); g_depthImage = VK_NULL_HANDLE; }
    if (g_depthMem) { vkFreeMemory(g_device.device(), g_depthMem, nullptr); g_depthMem = VK_NULL_HANDLE; }
}

static bool createSwapchain()
{
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_device.physicalDevice(), g_surface, &caps);

    int w = 0, h = 0;
    SDL_Vulkan_GetDrawableSize(window, &w, &h);
    if (w == 0 || h == 0)
        return false; // minimized

    g_swapExtent.width = (uint32_t)clamp((float)w, (float)caps.minImageExtent.width, (float)caps.maxImageExtent.width);
    g_swapExtent.height = (uint32_t)clamp((float)h, (float)caps.minImageExtent.height, (float)caps.maxImageExtent.height);

    VkSurfaceFormatKHR fmt = chooseSurfaceFormat();
    g_swapFormat = fmt.format;

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
        imageCount = caps.maxImageCount;

    // MAILBOX when it exists (unlocked fps for the benchmark), else FIFO (always
    // supported). IMMEDIATE exists only on non-shared machines, which this is not.
    g_mailboxAvailable = false;
    {
        uint32_t n = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(g_device.physicalDevice(), g_surface, &n, nullptr);
        std::vector<VkPresentModeKHR> modes(n);
        vkGetPhysicalDeviceSurfacePresentModesKHR(g_device.physicalDevice(), g_surface, &n, modes.data());
        for (auto pm : modes)
            if (pm == VK_PRESENT_MODE_MAILBOX_KHR)
                g_mailboxAvailable = true;
    }

    VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface = g_surface;
    sci.minImageCount = imageCount;
    sci.imageFormat = fmt.format;
    sci.imageColorSpace = fmt.colorSpace;
    sci.imageExtent = g_swapExtent;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = (g_wantVsync || !g_mailboxAvailable) ? VK_PRESENT_MODE_FIFO_KHR
                                                           : VK_PRESENT_MODE_MAILBOX_KHR;
    sci.clipped = VK_TRUE;
    sci.oldSwapchain = g_swapchain;
    VkSwapchainKHR newSwapchain = VK_NULL_HANDLE;
    VK_CHECK(vkCreateSwapchainKHR(g_device.device(), &sci, nullptr, &newSwapchain));

    if (g_swapchain)
    {
        destroySwapchainObjects();
        vkDestroySwapchainKHR(g_device.device(), g_swapchain, nullptr);
    }
    g_swapchain = newSwapchain;

    uint32_t n = 0;
    vkGetSwapchainImagesKHR(g_device.device(), g_swapchain, &n, nullptr);
    g_swapImages.resize(n);
    vkGetSwapchainImagesKHR(g_device.device(), g_swapchain, &n, g_swapImages.data());

    g_swapViews.resize(n);
    for (uint32_t i = 0; i < n; i++)
    {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = g_swapImages[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = g_swapFormat;
        vci.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                          VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(g_device.device(), &vci, nullptr, &g_swapViews[i]));
    }

    createDepthResources();

    g_framebuffers.resize(n);
    for (uint32_t i = 0; i < n; i++)
    {
        VkImageView atts[2] = {g_swapViews[i], g_depthView};
        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass = g_renderPass;
        fci.attachmentCount = 2;
        fci.pAttachments = atts;
        fci.width = g_swapExtent.width;
        fci.height = g_swapExtent.height;
        fci.layers = 1;
        VK_CHECK(vkCreateFramebuffer(g_device.device(), &fci, nullptr, &g_framebuffers[i]));
    }

    g_renderDonePerImage.resize(n);
    for (uint32_t i = 0; i < n; i++)
    {
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(g_device.device(), &sci, nullptr, &g_renderDonePerImage[i]));
    }
    return true;
}

static void createRenderPass()
{
    VkAttachmentDescription atts[2]{};
    atts[0].format = g_swapFormat;
    atts[0].samples = VK_SAMPLE_COUNT_1_BIT;
    atts[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    atts[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    atts[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    atts[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    atts[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    atts[1].format = g_depthFormat;
    atts[1].samples = VK_SAMPLE_COUNT_1_BIT;
    atts[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    atts[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // planar shadows count from 0
    atts[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    atts[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &colorRef;
    sub.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = 0;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpci.attachmentCount = 2;
    rpci.pAttachments = atts;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &sub;
    rpci.dependencyCount = 1;
    rpci.pDependencies = &dep;
    VK_CHECK(vkCreateRenderPass(g_device.device(), &rpci, nullptr, &g_renderPass));
}

// ---------------------------------------------------------------------------
// Screenshot: reads back the current swapchain image via a linear-tiled staging
// image, so the demo can be verified headlessly (--frames N --screenshot out.bmp).
// Written as an uncompressed BMP -- no PNG encoder in this codebase and one is not
// worth adding for a verification artifact the Read tool can open either way.
// ---------------------------------------------------------------------------
static void writeBmp(const char *path, const uint8_t *bgra, int w, int h)
{
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "avbd_vk: could not open %s for write\n", path); return; }
    const int rowBytes = w * 3;
    const int rowPad = (4 - (rowBytes % 4)) % 4;
    const int imgSize = (rowBytes + rowPad) * h;
    const int fileSize = 54 + imgSize;
    uint8_t header[54] = {0};
    header[0] = 'B'; header[1] = 'M';
    *(int32_t *)&header[2] = fileSize;
    *(int32_t *)&header[10] = 54;
    *(int32_t *)&header[14] = 40;
    *(int32_t *)&header[18] = w;
    *(int32_t *)&header[22] = h;
    *(int16_t *)&header[26] = 1;
    *(int16_t *)&header[28] = 24;
    *(int32_t *)&header[34] = imgSize;
    fwrite(header, 1, 54, f);
    std::vector<uint8_t> row(rowBytes + rowPad, 0);
    for (int y = h - 1; y >= 0; y--)
    {
        const uint8_t *src = bgra + (size_t)y * w * 4;
        for (int x = 0; x < w; x++)
        {
            row[x * 3 + 0] = src[x * 4 + 0]; // B
            row[x * 3 + 1] = src[x * 4 + 1]; // G
            row[x * 3 + 2] = src[x * 4 + 2]; // R
        }
        fwrite(row.data(), 1, rowBytes + rowPad, f);
    }
    fclose(f);
    fprintf(stdout, "avbd_vk: wrote screenshot %s (%dx%d)\n", path, w, h);
}

static void screenshotImage(VkImage srcImage, int w, int h, const char *path)
{
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_B8G8R8A8_UNORM;
    ici.extent = {(uint32_t)w, (uint32_t)h, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_LINEAR;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage dst = VK_NULL_HANDLE;
    VK_CHECK(vkCreateImage(g_device.device(), &ici, nullptr, &dst));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(g_device.device(), dst, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = demovk::findMemoryType(g_device.physicalDevice(), req.memoryTypeBits,
                                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateMemory(g_device.device(), &mai, nullptr, &mem));
    VK_CHECK(vkBindImageMemory(g_device.device(), dst, mem, 0));

    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = g_cmdPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(g_device.device(), &cbai, &cmd));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

    auto barrier = [&](VkImage img, VkImageLayout oldL, VkImageLayout newL, VkAccessFlags srcA,
                       VkAccessFlags dstA, VkPipelineStageFlags srcS, VkPipelineStageFlags dstS)
    {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = oldL; b.newLayout = newL;
        b.srcAccessMask = srcA; b.dstAccessMask = dstA;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, srcS, dstS, 0, 0, nullptr, 0, nullptr, 1, &b);
    };

    barrier(dst, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
           VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    barrier(srcImage, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
           VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT,
           VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkImageCopy region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.extent = {(uint32_t)w, (uint32_t)h, 1};
    vkCmdCopyImage(cmd, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    barrier(dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT,
           VK_ACCESS_HOST_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
    barrier(srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
           VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_MEMORY_READ_BIT,
           VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

    VK_CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(g_device.queue(), 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(g_device.queue()));
    vkFreeCommandBuffers(g_device.device(), g_cmdPool, 1, &cmd);

    VkImageSubresource sub{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
    VkSubresourceLayout layout;
    vkGetImageSubresourceLayout(g_device.device(), dst, &sub, &layout);

    void *mapped = nullptr;
    VK_CHECK(vkMapMemory(g_device.device(), mem, 0, VK_WHOLE_SIZE, 0, &mapped));
    std::vector<uint8_t> pixels((size_t)w * h * 4);
    for (int y = 0; y < h; y++)
        memcpy(&pixels[(size_t)y * w * 4], (uint8_t *)mapped + layout.offset + (size_t)y * layout.rowPitch, (size_t)w * 4);
    vkUnmapMemory(g_device.device(), mem);
    writeBmp(path, pixels.data(), w, h);

    vkDestroyImage(g_device.device(), dst, nullptr);
    vkFreeMemory(g_device.device(), mem, nullptr);
}

// ---------------------------------------------------------------------------
// The renderer's "Graphics" section of the panel (demo/render_options).
// ---------------------------------------------------------------------------
static void graphicsUI()
{
    renderOptionsUI(g_renderOpts);
    ImGui::Text("swapchain %s", g_mailboxAvailable ? "FIFO / MAILBOX" : "FIFO");
}

static Uint64 g_perfFreq = 0; // SDL_GetPerformanceFrequency, cached in main()

static void printUsage(FILE *f)
{
    fprintf(f,
            "usage: avbd3d_demo [options]\n"
            "  --scene <index|name>   start in this scene (name is case-insensitive)\n"
            "  --frames N             benchmark: vsync off, alternating cull blocks, print timings,\n"
            "                         quit after N frames\n"
            "  --screenshot file.bmp  write the last frame of --frames / --smoke to a BMP\n"
            "  --look modern|original rendering preset\n"
            "  --camdist D            camera distance from the look point\n"
            "  --no-sleep             start with sleeping off (\"Never sleep\"), for benchmarks\n"
            "  --verbose              print the selected device and whether validation is on\n"
            "  --log                  start with the session log on (panel: Logging)\n"
            "  --smoke                unattended run of every scene with validation on; the window is\n"
            "                         created hidden and never shown. Exit code is nonzero on a\n"
            "                         step failure or a validation message\n"
            "  --help                 this text\n"
            "environment: AVBD_VK_VALIDATION=1 turns the Vulkan validation layers on.\n");
}

// ---------------------------------------------------------------------------
// --smoke: every scene, scripted. Per scene: load (a scene switch right after the
// previous frame was submitted), spawn an object at the screen centre, grab the body at
// the centre and drag it a few frames, release. The window is resized (swapchain
// recreated) and restored during two of the scenes. All of it goes through the DemoApp
// hooks the mouse and the panel use.
// ---------------------------------------------------------------------------
static const int kSmokeFramesPerScene = 20;
static int g_smokeFrame = 0;     // frames completed in the run
static int g_smokeActed = -1;    // the g_smokeFrame the script last acted on
static bool g_smokeFinal = false; // the frame about to be drawn is the last

static void smokeTick()
{
    const int sceneCount = app.sceneCount();
    const int scene = g_smokeFrame / kSmokeFramesPerScene;
    const int f = g_smokeFrame % kSmokeFramesPerScene;
    g_smokeFinal = scene == sceneCount - 1 && f == kSmokeFramesPerScene - 1;
    if (g_smokeFrame == g_smokeActed)
        return; // a frame that was skipped (out-of-date swapchain) is redrawn, not rescripted
    g_smokeActed = g_smokeFrame;
    if (scene >= sceneCount)
    {
        app.quit();
        return;
    }

    int w = 0, h = 0;
    SDL_Vulkan_GetDrawableSize(window, &w, &h);
    const bool resizeScene = scene == 1 % sceneCount || scene == sceneCount - 1;

    switch (f)
    {
    case 0: app.loadSceneIndex(scene); break;
    case 3: app.spawnAtPixel(w / 2, h / 2); break;
    case 6: app.grabAtPixel(w / 2, h / 2); break;
    case 7: case 8: case 9: case 10: app.moveGrab(w / 2 + (f - 6) * 24, h / 2 - (f - 6) * 12); break;
    case 11: app.releaseGrab(); break;
    case 13: if (resizeScene) SDL_SetWindowSize(window, 960, 540); break;
    case 16: if (resizeScene) SDL_SetWindowSize(window, WinWidth, WinHeight); break;
    default: break;
    }
}

// A fatal engine error (VK_CHECK, a device loss) prints its cause to stderr and calls
// abort(), which would close the window and its console with nothing on screen. In an
// interactive run, hold the process on a message box so the console line can be read.
// With the session log open, the abort is also recorded there (with a backtrace) first.
static bool g_abortBox = false;
static void onAbort(int)
{
    fflush(stderr);
    demolog::fatal("abort()");
    if (!g_abortBox)
        return;
    char msg[1400];
    if (demolog::active())
        snprintf(msg, sizeof(msg),
                 "The demo hit a fatal error and must close.\n"
                 "The cause is printed in the console window and written to the log:\n%s",
                 demolog::path());
    else
        snprintf(msg, sizeof(msg), "The demo hit a fatal error and must close.\n"
                                   "The cause is printed in the console window.");
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "avbd3d_demo: fatal error", msg, nullptr);
}

int main(int argc, char **argv)
{
    DemoLaunch launch;
    const char *screenshotPath = nullptr;
    bool verbose = false, smoke = false;
    for (int i = 1; i < argc; i++)
    {
        const char *a = argv[i];
        const bool hasValue = i + 1 < argc;
        if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0)
        {
            printUsage(stdout);
            return 0;
        }
        else if (strcmp(a, "--verbose") == 0) verbose = true;
        else if (strcmp(a, "--smoke") == 0) smoke = true;
        else if (strcmp(a, "--no-sleep") == 0) launch.noSleep = true;
        else if (strcmp(a, "--log") == 0) launch.log = true;
        else if (strcmp(a, "--frames") == 0 && hasValue) launch.benchFrames = atoi(argv[++i]);
        else if (strcmp(a, "--screenshot") == 0 && hasValue) screenshotPath = argv[++i];
        else if (strcmp(a, "--scene") == 0 && hasValue) launch.sceneName = argv[++i];
        else if (strcmp(a, "--camdist") == 0 && hasValue) launch.camDist = (float)atof(argv[++i]);
        else if (strcmp(a, "--look") == 0 && hasValue)
        {
            // The panel's two preset buttons, for a headless screenshot of either look.
            const char *look = argv[++i];
            if (strcmp(look, "modern") == 0) applyModernLook(g_renderOpts);
            else if (strcmp(look, "original") == 0) applyOriginalLook(g_renderOpts);
            else
            {
                fprintf(stderr, "avbd3d_demo: unknown --look %s (modern|original)\n", look);
                printUsage(stderr);
                return 2;
            }
        }
        else
        {
            fprintf(stderr, "avbd3d_demo: %s '%s'\n", hasValue ? "unknown option" : "unknown option or missing value:", a);
            printUsage(stderr);
            return 2;
        }
    }
    const int benchFrames = launch.benchFrames;
    launch.unattended = smoke || benchFrames > 0;
    // The handlers are installed in every mode; they only write anything once the log is
    // open, and only an interactive run holds the process on a message box.
    g_abortBox = !launch.unattended;
    signal(SIGABRT, onAbort);
    demolog::installCrashHandler();
    std::string commandLine;
    for (int i = 0; i < argc; i++)
        commandLine += (i ? " " : "") + std::string(argv[i]);
    launch.commandLine = commandLine.c_str();

    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    // --smoke never shows its window: it is created hidden and nothing calls SDL_ShowWindow
    // or SDL_RaiseWindow on it. The full render path (swapchain, scene, ImGui) still runs.
    window = SDL_CreateWindow("Vulkan Physics Engine", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                              WinWidth, WinHeight,
                              SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | (smoke ? SDL_WINDOW_HIDDEN : 0));
    if (!window)
    {
        fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        return 1;
    }
    if (smoke)
        printf("smoke: window hidden: %s\n", (SDL_GetWindowFlags(window) & SDL_WINDOW_HIDDEN) ? "yes" : "no");

    // One presentable GPU for the solver and the swapchain/renderer (see g_device).
    unsigned int extCount = 0;
    SDL_Vulkan_GetInstanceExtensions(window, &extCount, nullptr);
    std::vector<const char *> instExts(extCount);
    SDL_Vulkan_GetInstanceExtensions(window, &extCount, instExts.data());

    Avbd3dGpuDef gpuDef = avbd3d_default_gpu_def();
    gpuDef.instanceExtensions = instExts.data();
    gpuDef.instanceExtensionCount = (int)instExts.size();
    gpuDef.makeSurface = [](void *instance, void *) -> uint64_t {
        VkSurfaceKHR surf = VK_NULL_HANDLE;
        if (!SDL_Vulkan_CreateSurface(window, (VkInstance)instance, &surf))
        {
            fprintf(stderr, "SDL_Vulkan_CreateSurface failed: %s\n", SDL_GetError());
            return 0;
        }
        return (uint64_t)surf;
    };
    gpuDef.enableValidation = smoke; // the point of the run
    gpuDef.verbose = verbose;
    if (avbd3d_create_gpu(&gpuDef, &g_gpu) != AVBD3D_OK || !g_gpu)
    {
        fprintf(stderr, "avbd3d_demo: no Vulkan device could be created\n");
        return 1;
    }
    g_device.init(g_gpu);
    app.setGpu(g_gpu);
    g_surface = g_device.surface();

    g_depthFormat = chooseDepthFormat();
    createRenderPass();
    if (!createSwapchain())
    {
        fprintf(stderr, "avbd_vk: initial swapchain creation failed (window minimized?)\n");
        return 1;
    }
    g_renderer.init(g_device, g_renderPass, g_swapExtent.width, g_swapExtent.height);

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = g_device.queueFamily();
    VK_CHECK(vkCreateCommandPool(g_device.device(), &poolInfo, nullptr, &g_cmdPool));

    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = g_cmdPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = kFramesInFlight;
    VK_CHECK(vkAllocateCommandBuffers(g_device.device(), &cbai, g_cmdBufs));

    for (int i = 0; i < kFramesInFlight; i++)
    {
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(g_device.device(), &sci, nullptr, &g_imageAvail[i]));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VK_CHECK(vkCreateFence(g_device.device(), &fci, nullptr, &g_inFlight[i]));
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplSDL2_InitForVulkan(window);

    ImGui_ImplVulkan_InitInfo vii{};
    vii.Instance = g_device.instance();
    vii.PhysicalDevice = g_device.physicalDevice();
    vii.Device = g_device.device();
    vii.QueueFamily = g_device.queueFamily();
    vii.Queue = g_device.queue();
    vii.RenderPass = g_renderPass;
    vii.MinImageCount = 2;
    vii.ImageCount = (uint32_t)g_swapImages.size();
    vii.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    vii.PipelineCache = g_device.pipelineCache();
    vii.DescriptorPoolSize = 8;
    ImGui_ImplVulkan_Init(&vii);

    if (verbose)
        fprintf(stdout, "avbd3d_demo: validation layers %s\n",
                (smoke || getenv("AVBD_VK_VALIDATION")) ? "ON" : "off");

    // Wire the demo together.
    DemoHost host;
    host.panelTitle = "Physics Engine";
    host.drawableSize = [](int &w, int &h) { SDL_Vulkan_GetDrawableSize(window, &w, &h); };
    host.windowSize = [](int &w, int &h) { SDL_GetWindowSize(window, &w, &h); };
    host.graphicsUI = graphicsUI;
    host.setVsync = [](bool on) {
        if (g_wantVsync != on)
        {
            g_wantVsync = on;
            g_swapDirty = true;
        }
    };
    static const char *kExtraSceneNames[] = {"Spheres & Capsules"};
    host.extraSceneNames.assign(std::begin(kExtraSceneNames), std::end(kExtraSceneNames));

    app.init(host);

    if (!app.loadInitialScene(launch))
        return 2;

    g_perfFreq = SDL_GetPerformanceFrequency();
    Uint64 prev = SDL_GetPerformanceCounter();
    int frameNo = 0;

    while (app.running())
    {
        SDL_Event e;
        while (SDL_PollEvent(&e))
        {
            ImGui_ImplSDL2_ProcessEvent(&e);
            app.handleEvent(e);
            if (e.type == SDL_WINDOWEVENT &&
                (e.window.event == SDL_WINDOWEVENT_RESIZED || e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED))
            {
                g_swapDirty = true;
            }
        }

        if (smoke)
            smokeTick();
        app.update();
        demolog::setActivity("render");

        if (g_swapDirty && g_swapchain)
        {
            vkDeviceWaitIdle(g_device.device());
            if (createSwapchain())
                g_renderer.resize(g_swapExtent.width, g_swapExtent.height);
            g_swapDirty = false;
        }

        // Render straight from the world's own instance buffer: each World_Write* call
        // records, submits and waits for its own dispatch, so the buffers are ready to bind
        // for this frame's draw with no additional compute->vertex barrier on this shared queue.
        int w = 0, h = 0;
        SDL_Vulkan_GetDrawableSize(window, &w, &h);

        float proj[16], view[16], viewProj[16];
        DemoApp::perspective(proj, h > 0 ? (float)w / (float)h : 1.0f, /*vulkanClip*/ true);
        app.viewMatrix(view);
        DemoApp::matMul(viewProj, proj, view);

        Avbd3dWorld *world = app.world();
        const OverlayOptions &ov = app.overlay();
        Avbd3dRenderDef rd = avbd3d_default_render_def();
        rd.overlay = ov.mode;
        rd.speedMax = ov.speedMax;
        rd.loadMax = ov.loadMax;
        rd.jointLoadMax = ov.jointLoadMax;
        rd.penetrationMax = ov.penetrationMax;
        rd.degreeMax = ov.degreeMax;
        rd.jointDegreeMax = ov.jointDegreeMax;
        rd.colorStatics = ov.colorStatics;

        // CPU-side wall clock for the panel's render/ui lines: there are no timestamp
        // queries wired on this path, so what is measured is the encode+submit cost,
        // the part of the render this process performs.
        const Uint64 tDraw0 = SDL_GetPerformanceCounter();
        float planes[24];
        if (app.cullEnabled() && h > 0)
        {
            extractFrustum(viewProj, planes);
            rd.frustumPlanes = planes;
        }
        if (avbd3d_world_write_instances(world, &rd) != AVBD3D_OK)
            fprintf(stderr, "avbd3d_demo: world_write_instances failed\n");

        const DebugDrawOptions &dd = app.debugDraw();
        int markers = 0;
        if (dd.any())
        {
            Avbd3dContactMarkerDef md = avbd3d_default_contact_marker_def();
            md.points = dd.contactPoints;
            md.normals = dd.contactNormals;
            md.manifoldLinks = dd.manifoldLinks;
            md.tint = dd.contactTint;
            md.pointSize = dd.contactPointSize;
            md.normalLength = dd.normalLength;
            md.forceMax = dd.forceMax;
            md.maxMarkers = dd.maxMarkers;
            if (avbd3d_world_write_contact_markers(world, &md, &markers) != AVBD3D_OK)
                markers = 0;
        }
        // Joint and spring rods are deliberately not drawn: the demo shows rigid bodies only.

        vkWaitForFences(g_device.device(), 1, &g_inFlight[g_frameIdx], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex = 0;
        VkResult acq = vkAcquireNextImageKHR(g_device.device(), g_swapchain, UINT64_MAX,
                                             g_imageAvail[g_frameIdx], VK_NULL_HANDLE, &imageIndex);
        if (acq == VK_ERROR_OUT_OF_DATE_KHR)
        {
            // The old swapchain's images may still be in flight.
            vkDeviceWaitIdle(g_device.device());
            if (createSwapchain())
                g_renderer.resize(g_swapExtent.width, g_swapExtent.height);
            continue;
        }

        vkResetFences(g_device.device(), 1, &g_inFlight[g_frameIdx]);

        VkCommandBuffer cmd = g_cmdBufs[g_frameIdx];
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

        app.findShadowPlane(g_renderOpts.groundPlanePoint, g_renderOpts.groundPlaneNormal);
        // An overlay writes its result into the instance colour, so the flat block
        // colour steps aside for it without disturbing the setting.
        g_renderOpts.forceInstanceColor = app.overlay().mode != OVERLAY_NONE;
        g_renderOpts.cull = app.cullEnabled();
        g_renderOpts.camTarget = app.camTarget();
        g_renderOpts.camDistance = app.camDistance();
        g_renderOpts.lightDir = demoLightDir();

        // One draw per render group. Groups 0..2 are the shape classes; a group >= 3 is
        // an authored mesh+material slot (a Unity concern), which this demo has no mesh
        // for, so it draws those bodies as boxes. Rods and debug markers are
        // BodyInstances too (see render.slang), drawn as stretched boxes.
        std::vector<VkDrawItem> items;
        int drawn = 0;
        // Buffers are re-fetched after every Write: a grow reallocates them.
        uint64_t instAddr = 0;
        avbd3d_world_get_instance_buffer(world, &instAddr);
        VkBuffer instBuf = (VkBuffer)(uintptr_t)instAddr;
        int32_t groupCount = 0;
        avbd3d_world_get_render_group_count(world, &groupCount);
        for (int group = 0; group < groupCount; group++)
        {
            Avbd3dRenderRange r{};
            avbd3d_world_get_render_group_range(world, group, &r);
            if (r.count <= 0)
                continue;
            items.push_back({group < 3 ? group : kShapeBox, instBuf, (uint32_t)r.first, (uint32_t)r.count, true});
            drawn += r.count;
        }
        if (markers > 0)
            {
            uint64_t markerAddr = 0;
            avbd3d_world_get_contact_marker_buffer(world, &markerAddr);
            items.push_back({kShapeBox, (VkBuffer)(uintptr_t)markerAddr, 0, (uint32_t)markers, false});
        }

        // Shadow-map and SSAO depth prepasses: their own render passes, before the main one.
        g_renderer.prepare(cmd, g_frameIdx, items.data(), (int)items.size(), viewProj, g_renderOpts);

        float clearColor[4];
        backgroundColor(g_renderOpts.background, clearColor);
        VkClearValue clears[2]{};
        clears[0].color = {{clearColor[0], clearColor[1], clearColor[2], clearColor[3]}};
        clears[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rpbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rpbi.renderPass = g_renderPass;
        rpbi.framebuffer = g_framebuffers[imageIndex];
        rpbi.renderArea = {{0, 0}, g_swapExtent};
        rpbi.clearValueCount = 2;
        rpbi.pClearValues = clears;
        vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport{0, 0, (float)g_swapExtent.width, (float)g_swapExtent.height, 0.0f, 1.0f};
        VkRect2D scissor{{0, 0}, g_swapExtent};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        g_renderer.drawScene(cmd, items.data(), (int)items.size());
        app.setDrawnCount(drawn);

        const Uint64 tUi0 = SDL_GetPerformanceCounter();
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
        app.drawUI();
        ImGui::Render();
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
        const Uint64 tUi1 = SDL_GetPerformanceCounter();

        vkCmdEndRenderPass(cmd);
        VK_CHECK(vkEndCommandBuffer(cmd));

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &g_imageAvail[g_frameIdx];
        si.pWaitDstStageMask = &waitStage;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &g_renderDonePerImage[imageIndex];
        VK_CHECK(vkQueueSubmit(g_device.queue(), 1, &si, g_inFlight[g_frameIdx]));

        const Uint64 tDraw1 = SDL_GetPerformanceCounter();
        app.setRenderTimings((float)((tDraw1 - tDraw0) * 1000.0 / g_perfFreq),
                             (float)((tUi1 - tUi0) * 1000.0 / g_perfFreq));

        // Screenshot must be captured before vkQueuePresentKHR: once an image is handed
        // to the presentation engine, touching it again (even just to read it back)
        // without re-acquiring it is a validation error. Wait for the render to finish,
        // then copy while we still own the image.
        if (screenshotPath && ((benchFrames > 0 && frameNo == benchFrames - 1) || (smoke && g_smokeFinal)))
        {
            vkQueueWaitIdle(g_device.queue());
            screenshotImage(g_swapImages[imageIndex], (int)g_swapExtent.width,
                            (int)g_swapExtent.height, screenshotPath);
        }

        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &g_renderDonePerImage[imageIndex];
        pi.swapchainCount = 1;
        pi.pSwapchains = &g_swapchain;
        pi.pImageIndices = &imageIndex;
        VkResult pres = vkQueuePresentKHR(g_device.queue(), &pi);
        if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR)
        {
            vkDeviceWaitIdle(g_device.device());
            createSwapchain();
        }

        g_frameIdx = (g_frameIdx + 1) % kFramesInFlight;

        Uint64 now = SDL_GetPerformanceCounter();
        float frameMs = (float)((now - prev) * 1000.0 / g_perfFreq);
        prev = now;
        app.endFrame(frameMs);

        frameNo++;
        g_smokeFrame++;
        if (benchFrames > 0 && frameNo >= benchFrames)
            break;
        if (smoke && g_smokeFinal)
            break;
    }

    int exitCode = 0;
    if (app.stepFailures() > 0)
        exitCode = 1;
    if (smoke)
    {
        int32_t vmsgs = 0;
        avbd3d_get_validation_message_count(&vmsgs);
        printf("smoke: scenes %d frames %d step-failures %d validation-messages %d\n", app.sceneCount(),
               frameNo, app.stepFailures(), vmsgs);
        if (vmsgs > 0)
            exitCode = 1;
    }

    vkDeviceWaitIdle(g_device.device());

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    g_renderer.destroy();
    for (int i = 0; i < kFramesInFlight; i++)
    {
        vkDestroySemaphore(g_device.device(), g_imageAvail[i], nullptr);
        vkDestroyFence(g_device.device(), g_inFlight[i], nullptr);
    }
    vkDestroyCommandPool(g_device.device(), g_cmdPool, nullptr);
    destroySwapchainObjects();
    vkDestroySwapchainKHR(g_device.device(), g_swapchain, nullptr);
    vkDestroyRenderPass(g_device.device(), g_renderPass, nullptr);

    // The renderer's objects first, then the world, then the GPU (which destroys the
    // surface it made).
    g_device.destroy();
    app.shutdown();
    avbd3d_destroy_gpu(g_gpu);
    demolog::line("exit code %d", exitCode);
    demolog::close();

    SDL_DestroyWindow(window);
    SDL_Quit();
    return exitCode;
}
