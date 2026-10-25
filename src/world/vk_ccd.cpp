// vk_ccd.cpp -- host side of ccd.slang.

#include "vk_ccd.h"

#include "ccd_spirv.h"

#include <algorithm>

namespace avbdvk
{

static const uint32_t kCcdLanes = 512; // worker lanes of csCcdToi (one scratch slot each)
// Per lane: B3Scratch plus two 128-point proxies (hull points / mesh triangle); ccd.slang's guard
// returns silently when a slot is too small.
static const uint32_t kCcdScratchStride = 32768;

void VkCcd::init(Device &dev, PipelineLayout &layout)
{
    flag.init(dev, layout, ccd_spirv, ccd_spirv_words, "csCcdFlag", "csCcdFlag");
    toi.init(dev, layout, ccd_spirv, ccd_spirv_words, "csCcdToi", "csCcdToi");
    apply.init(dev, layout, ccd_spirv, ccd_spirv_words, "csCcdApply", "csCcdApply");
    list.init(dev); counter.init(dev); frac.init(dev); scratch.init(dev);
    m_init = true;
}

void VkCcd::release()
{
    list.destroy(); counter.destroy(); frac.destroy(); scratch.destroy();
    m_fracBytes = 0;
}

void VkCcd::destroy()
{
    if (!m_init)
        return;
    release();
    flag.destroy(); toi.destroy(); apply.destroy();
    m_init = false;
}

void VkCcd::record(CommandList &cmd, HostRingBuffer &ring, CcdArgsGpu a, int bodyCap, bool bullets)
{
    if (a.dynCount == 0)
        return;
    // Grow-only; a grown fraction buffer is refilled with all-ones (no hit) (the between-passes invariant).
    const VkDeviceSize listBytes = (VkDeviceSize)a.dynCount * sizeof(int);
    if (list.size() < listBytes)
        list.resize(listBytes + listBytes / 2);
    if (counter.size() < sizeof(int))
        counter.resize(sizeof(int));
    const VkDeviceSize fracBytes = (VkDeviceSize)std::max(bodyCap, 1) * sizeof(uint32_t);
    if (m_fracBytes < fracBytes)
    {
        frac.resize(fracBytes);
        m_fracBytes = fracBytes;
        cmd.fill(frac, 0xFFFFFFFFu, fracBytes);
        cmd.barrier();
    }
    a.scratchStride = (uint32_t)kCcdScratchStride;
    a.lanes = kCcdLanes;
    const VkDeviceSize scratchBytes = (VkDeviceSize)kCcdLanes * a.scratchStride;
    if (scratch.size() < scratchBytes)
        scratch.resize(scratchBytes);
    a.list = list.address();
    a.counter = counter.address();
    a.frac = frac.address();
    a.scratch = scratch.address();

    const uint32_t passes = bullets ? 2u : 1u;
    for (uint32_t pass = 0; pass < passes; pass++)
    {
        a.pass = pass;
        cmd.fill(counter, 0u, sizeof(int));
        cmd.barrier();
        const VkDeviceAddress args = ring.write(&a, sizeof(a));
        cmd.dispatch(flag, (a.dynCount + 255u) / 256u, args);
        cmd.barrier();
        cmd.dispatch(toi, kCcdLanes / 64u, args);
        cmd.barrier();
        cmd.dispatch(apply, (a.dynCount + 63u) / 64u, args);
        cmd.barrier();
    }
}

} // namespace avbdvk
