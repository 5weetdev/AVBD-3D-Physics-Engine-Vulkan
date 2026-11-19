// vk_ccd.h -- host side of ccd.slang: the post-solve continuous collision clamp.
// Owned by VkWorld, which fills the CcdArgsGpu body/shape pointers; this class owns the pipelines
// and the buffers only the CCD uses (flagged list, counter, per-body fraction, TOI lane scratch).
#pragma once

#include "vk_args.h"
#include "vk_buffer.h"
#include "vk_cmd.h"
#include "vk_device.h"
#include "vk_pipeline.h"

namespace avbdvk
{

class VkCcd
{
public:
    void init(Device &dev, PipelineLayout &layout);
    void destroy();
    // Drops the scene-sized buffers (scene reset); pipelines stay.
    void release();

    // Records both passes (non-bullet, then bullet). `a` carries every pointer except list,
    // counter, frac and scratch, which this fills. bullets: some body has the bullet flag.
    void record(CommandList &cmd, HostRingBuffer &ring, CcdArgsGpu a, int bodyCap, bool bullets);

private:
    Pipeline flag, toi, apply;
    Buffer list, counter, frac, scratch;
    VkDeviceSize m_fracBytes = 0;
    bool m_init = false;
};

} // namespace avbdvk
