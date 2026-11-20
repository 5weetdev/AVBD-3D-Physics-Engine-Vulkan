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

// The renderer's view of the solver's GPU. The demo draws on the device avbd3d_create_gpu made
// (Vulkan objects are device-scoped, so the instance buffer the world writes can only be bound
// on that device); this wraps Avbd3dVulkanContext in typed accessors and adds the few helpers
// the renderer needs: a check macro, a memory-type search, its own command pool and a
// persistently mapped buffer with a device address. Nothing here reaches into src/world.

#include "avbd3d/avbd3d_vulkan.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Prints and aborts: a demo has nothing to fall back to, and the abort keeps the failing
// stack for the device-lost recipe in docs/TESTING.md.
#define VK_CHECK(expr)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        VkResult demo_vk_res_ = (expr);                                                                                \
        if (demo_vk_res_ != VK_SUCCESS)                                                                                \
        {                                                                                                              \
            fprintf(stderr, "Vulkan error %d at %s:%d\n    %s\n", (int)demo_vk_res_, __FILE__, __LINE__, #expr);       \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

namespace demovk
{

inline bool tryFindMemoryType(VkPhysicalDevice phys, uint32_t typeBits, VkMemoryPropertyFlags props, uint32_t &out)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
        {
            out = i;
            return true;
        }
    return false;
}

inline uint32_t findMemoryType(VkPhysicalDevice phys, uint32_t typeBits, VkMemoryPropertyFlags props)
{
    uint32_t i = 0;
    if (!tryFindMemoryType(phys, typeBits, props, i))
    {
        fprintf(stderr, "avbd3d_demo: no suitable memory type\n");
        abort();
    }
    return i;
}

class Device
{
public:
    void init(const Avbd3dGpu *gpu)
    {
        if (avbd3d_gpu_get_vulkan_context(gpu, &m_ctx) != AVBD3D_OK)
            m_ctx = {};
        // The solver's pool is reset only by the solver; the renderer allocates from its own.
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = m_ctx.queueFamily;
        VK_CHECK(vkCreateCommandPool(device(), &pci, nullptr, &m_pool));
    }
    void destroy()
    {
        if (m_pool)
            vkDestroyCommandPool(device(), m_pool, nullptr);
        m_pool = VK_NULL_HANDLE;
    }

    VkInstance instance() const { return (VkInstance)m_ctx.instance; }
    VkPhysicalDevice physicalDevice() const { return (VkPhysicalDevice)m_ctx.physicalDevice; }
    VkDevice device() const { return (VkDevice)m_ctx.device; }
    VkQueue queue() const { return (VkQueue)m_ctx.queue; }
    uint32_t queueFamily() const { return m_ctx.queueFamily; }
    VkPipelineCache pipelineCache() const { return (VkPipelineCache)(uintptr_t)m_ctx.pipelineCache; }
    VkSurfaceKHR surface() const { return (VkSurfaceKHR)(uintptr_t)m_ctx.surface; }
    VkCommandPool commandPool() const { return m_pool; }
    const char *deviceName() const { return m_ctx.deviceName; }

private:
    Avbd3dVulkanContext m_ctx = {};
    VkCommandPool m_pool = VK_NULL_HANDLE;
};

// A host-written buffer the shaders read through its device address. Prefers device-local
// host-visible memory (ReBAR), falling back to plain host-coherent memory.
class MappedBuffer
{
public:
    MappedBuffer() = default;
    ~MappedBuffer() { destroy(); }
    MappedBuffer(const MappedBuffer &) = delete;
    MappedBuffer &operator=(const MappedBuffer &) = delete;

    void init(Device &dev, VkDeviceSize bytes)
    {
        m_device = dev.device();
        m_size = bytes;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = bytes;
        bi.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(m_device, &bi, nullptr, &m_buffer));

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(m_device, m_buffer, &req);
        const VkMemoryPropertyFlags coherent =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        uint32_t type = 0;
        if (!tryFindMemoryType(dev.physicalDevice(), req.memoryTypeBits, coherent | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                               type))
            type = findMemoryType(dev.physicalDevice(), req.memoryTypeBits, coherent);

        VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.pNext = &flags;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = type;
        VK_CHECK(vkAllocateMemory(m_device, &ai, nullptr, &m_memory));
        VK_CHECK(vkBindBufferMemory(m_device, m_buffer, m_memory, 0));
        VK_CHECK(vkMapMemory(m_device, m_memory, 0, bytes, 0, &m_mapped));

        VkBufferDeviceAddressInfo dai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        dai.buffer = m_buffer;
        m_address = vkGetBufferDeviceAddress(m_device, &dai);
    }

    void destroy()
    {
        if (!m_device)
            return;
        if (m_mapped)
            vkUnmapMemory(m_device, m_memory);
        if (m_buffer)
            vkDestroyBuffer(m_device, m_buffer, nullptr);
        if (m_memory)
            vkFreeMemory(m_device, m_memory, nullptr);
        m_device = VK_NULL_HANDLE;
        m_buffer = VK_NULL_HANDLE;
        m_memory = VK_NULL_HANDLE;
        m_mapped = nullptr;
        m_address = 0;
        m_size = 0;
    }

    VkBuffer handle() const { return m_buffer; }
    VkDeviceAddress address() const { return m_address; }
    VkDeviceSize size() const { return m_size; }
    void *mapped() const { return m_mapped; }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkBuffer m_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_memory = VK_NULL_HANDLE;
    VkDeviceAddress m_address = 0;
    void *m_mapped = nullptr;
    VkDeviceSize m_size = 0;
};

} // namespace demovk
