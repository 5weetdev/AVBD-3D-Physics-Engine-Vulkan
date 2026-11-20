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
