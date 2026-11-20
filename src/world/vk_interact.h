#pragma once

// Vulkan host orchestration for src/shaders/drag.slang: the drag update and release
// dispatches. The grab's host-side graph walks and buffer bookkeeping are the caller's
// job; this file only wraps the two device dispatches.

#include "vk_args.h"
#include "vk_buffer.h"
#include "vk_cmd.h"
#include "vk_device.h"
#include "vk_pipeline.h"

#include <cstdint>

namespace avbdvk
{

struct InteractPipelines
{
    Pipeline dragUpdate;
    Pipeline dragRelease;

    // dragSpirv/dragWords: src/shaders/drag.slang's embedded SPIR-V.
    void init(Device &dev, PipelineLayout &layout, const uint32_t *dragSpirv, size_t dragWords);
    void destroy();
};

void launchDragUpdate(CommandList &cmd, HostRingBuffer &ring, InteractPipelines &pl, const DragUpdateArgsGpu &args,
                      int count);

void launchDragRelease(CommandList &cmd, HostRingBuffer &ring, InteractPipelines &pl, const DragReleaseArgsGpu &args,
                       int count);

} // namespace avbdvk
