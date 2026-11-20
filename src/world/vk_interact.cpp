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

#include "vk_interact.h"

namespace avbdvk
{

void InteractPipelines::init(Device &dev, PipelineLayout &layout, const uint32_t *dragSpirv, size_t dragWords)
{
    dragUpdate.init(dev, layout, dragSpirv, dragWords, "csDragUpdate", "csDragUpdate");
    dragRelease.init(dev, layout, dragSpirv, dragWords, "csDragRelease", "csDragRelease");
}

void InteractPipelines::destroy()
{
    dragUpdate.destroy();
    dragRelease.destroy();
}

void launchDragUpdate(CommandList &cmd, HostRingBuffer &ring, InteractPipelines &pl, const DragUpdateArgsGpu &args,
                      int count)
{
    if (count <= 0)
        return;
    VkDeviceAddress addr = ring.write(&args, sizeof(args));
    const uint32_t block = 128;
    cmd.dispatch(pl.dragUpdate, ((uint32_t)count + block - 1) / block, addr);
}

void launchDragRelease(CommandList &cmd, HostRingBuffer &ring, InteractPipelines &pl, const DragReleaseArgsGpu &args,
                       int count)
{
    if (count <= 0)
        return;
    VkDeviceAddress addr = ring.write(&args, sizeof(args));
    const uint32_t block = 256;
    cmd.dispatch(pl.dragRelease, ((uint32_t)count + block - 1) / block, addr);
}

} // namespace avbdvk
