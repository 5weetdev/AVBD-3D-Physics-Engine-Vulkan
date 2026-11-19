// vk_contact_report.cpp -- host side of contact_report.slang.

#include "vk_contact_report.h"
#include "vk_util.h"

namespace avbdvk
{

// Every entry point of this module, named once; init and destroy walk the same table.
static std::array<PipelineEntry, 1> entries(ContactReportPipelines &p)
{
    return {{
        {&p.report, "csContactReport"},
    }};
}

void ContactReportPipelines::init(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t spirvWords)
{
    initPipelines(dev, layout, spirv, spirvWords, entries(*this));
}

void ContactReportPipelines::destroy()
{
    destroyPipelines(entries(*this));
}

void launchContactReport(CommandList &cmd, HostRingBuffer &ring, ContactReportPipelines &pl,
                         const ContactReportArgsGpu &args)
{
    uint32_t groups = ceilDiv((int)args.slots, 128);
    if (groups == 0)
        return;
    cmd.dispatchArgs(pl.report, groups, ring, args);
}

// Every entry point of this module, named once; init and destroy walk the same table.
static std::array<PipelineEntry, 3> entries(RestitutionPipelines &p)
{
    return {{
        {&p.pre, "csRestitutionPre"},
        {&p.impulse, "csRestitutionImpulse"},
        {&p.apply, "csRestitutionApply"},
    }};
}

void RestitutionPipelines::init(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t spirvWords)
{
    initPipelines(dev, layout, spirv, spirvWords, entries(*this));
}

void RestitutionPipelines::destroy()
{
    destroyPipelines(entries(*this));
}

void launchRestitutionPre(CommandList &cmd, HostRingBuffer &ring, RestitutionPipelines &pl,
                          const RestitutionArgsGpu &args)
{
    uint32_t groups = ceilDiv((int)args.slots, 128);
    if (groups > 0)
        cmd.dispatchArgs(pl.pre, groups, ring, args);
}

void launchRestitutionImpulse(CommandList &cmd, HostRingBuffer &ring, RestitutionPipelines &pl,
                              const RestitutionArgsGpu &args)
{
    uint32_t groups = ceilDiv((int)args.slots, 128);
    if (groups > 0)
        cmd.dispatchArgs(pl.impulse, groups, ring, args);
}

void launchRestitutionApply(CommandList &cmd, HostRingBuffer &ring, RestitutionPipelines &pl,
                            const RestitutionArgsGpu &args)
{
    uint32_t groups = ceilDiv((int)args.dynCount, 128);
    if (groups > 0)
        cmd.dispatchArgs(pl.apply, groups, ring, args);
}

} // namespace avbdvk
