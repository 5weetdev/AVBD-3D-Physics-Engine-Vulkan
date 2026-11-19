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

#include "vk_render.h"

namespace avbdvk
{

// Every entry point of this module, named once; init and destroy walk the same table.
static std::array<PipelineEntry, 7> entries(RenderPipelines &p)
{
    return {{
        {&p.writeInstances, "csWriteInstances"},
        {&p.overlayStatsClear, "csOverlayStatsClear"},
        {&p.overlayStatsContacts, "csOverlayStatsContacts"},
        {&p.overlayStatsForces, "csOverlayStatsForces"},
        {&p.overlayLoadClear, "csOverlayLoadClear"},
        {&p.overlayLoadAccum, "csOverlayLoadAccum"},
        {&p.writeDebugInstances, "csWriteDebugInstances"},
    }};
}

void RenderPipelines::init(Device &dev, PipelineLayout &layout, const RenderModules &m)
{
    const std::array<PipelineEntry, 7> all = entries(*this);
    // entries() lists the kernels grouped by module: writeInstances | overlay x5 | debug x1.
    initPipelines(dev, layout, m.render, m.renderWords, all.data(), 1);
    initPipelines(dev, layout, m.overlay, m.overlayWords, all.data() + 1, 5);
    initPipelines(dev, layout, m.debug, m.debugWords, all.data() + 6, 1);
}

void RenderPipelines::destroy()
{
    destroyPipelines(entries(*this));
}

void launchWriteInstances(CommandList &cmd, HostRingBuffer &ring, RenderPipelines &pl,
                          const WriteInstancesArgsGpu &args, int shapeCount)
{
    if (shapeCount <= 0)
        return;
    VkDeviceAddress addr = ring.write(&args, sizeof(args));
    const uint32_t block = 256;
    cmd.dispatch(pl.writeInstances, ((uint32_t)shapeCount + block - 1) / block, addr);
}

void launchOverlayStatsClear(CommandList &cmd, HostRingBuffer &ring, RenderPipelines &pl,
                             const OverlayStatsClearArgsGpu &args, int n)
{
    if (n <= 0)
        return;
    VkDeviceAddress addr = ring.write(&args, sizeof(args));
    const uint32_t block = 256;
    cmd.dispatch(pl.overlayStatsClear, ((uint32_t)n + block - 1) / block, addr);
}

void launchOverlayStatsContacts(CommandList &cmd, HostRingBuffer &ring, RenderPipelines &pl,
                                const OverlayStatsContactsArgsGpu &args, int slots)
{
    if (slots <= 0)
        return;
    VkDeviceAddress addr = ring.write(&args, sizeof(args));
    const uint32_t block = 256;
    cmd.dispatch(pl.overlayStatsContacts, ((uint32_t)slots + block - 1) / block, addr);
}

void launchOverlayStatsForces(CommandList &cmd, HostRingBuffer &ring, RenderPipelines &pl,
                              const OverlayStatsForcesArgsGpu &args, int jointsAndSprings)
{
    if (jointsAndSprings <= 0)
        return;
    VkDeviceAddress addr = ring.write(&args, sizeof(args));
    const uint32_t block = 256;
    cmd.dispatch(pl.overlayStatsForces, ((uint32_t)jointsAndSprings + block - 1) / block, addr);
}

void launchOverlayLoadClear(CommandList &cmd, HostRingBuffer &ring, RenderPipelines &pl,
                            const OverlayLoadClearArgsGpu &args, int n)
{
    if (n <= 0)
        return;
    VkDeviceAddress addr = ring.write(&args, sizeof(args));
    const uint32_t block = 256;
    cmd.dispatch(pl.overlayLoadClear, ((uint32_t)n + block - 1) / block, addr);
}

void launchOverlayLoadAccum(CommandList &cmd, HostRingBuffer &ring, RenderPipelines &pl,
                            const OverlayLoadAccumArgsGpu &args, int slots)
{
    if (slots <= 0)
        return;
    VkDeviceAddress addr = ring.write(&args, sizeof(args));
    const uint32_t block = 256;
    cmd.dispatch(pl.overlayLoadAccum, ((uint32_t)slots + block - 1) / block, addr);
}

void launchWriteDebugInstances(CommandList &cmd, HostRingBuffer &ring, RenderPipelines &pl,
                               const WriteDebugInstancesArgsGpu &args, int slots)
{
    if (slots <= 0)
        return;
    VkDeviceAddress addr = ring.write(&args, sizeof(args));
    const uint32_t block = 256;
    cmd.dispatch(pl.writeDebugInstances, ((uint32_t)slots + block - 1) / block, addr);
}

} // namespace avbdvk
