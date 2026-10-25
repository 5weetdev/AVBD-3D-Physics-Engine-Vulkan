// vk_contact_report.cpp -- host side of contact_report.slang.

#include "vk_contact_report.h"

namespace avbdvk
{

static uint32_t ceilDiv(int n, int d)
{
    return (uint32_t)((n + d - 1) / d);
}

void ContactReportPipelines::init(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t spirvWords)
{
    report.init(dev, layout, spirv, spirvWords, "csContactReport", "csContactReport");
}

void ContactReportPipelines::destroy()
{
    report.destroy();
}

void launchContactReport(CommandList &cmd, HostRingBuffer &ring, ContactReportPipelines &pl,
                         const ContactReportArgsGpu &args)
{
    uint32_t groups = ceilDiv((int)args.slots, 128);
    if (groups == 0)
        return;
    cmd.dispatch(pl.report, groups, ring.write(&args, sizeof(args)));
}

void RestitutionPipelines::init(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t spirvWords)
{
    pre.init(dev, layout, spirv, spirvWords, "csRestitutionPre", "csRestitutionPre");
    impulse.init(dev, layout, spirv, spirvWords, "csRestitutionImpulse", "csRestitutionImpulse");
    apply.init(dev, layout, spirv, spirvWords, "csRestitutionApply", "csRestitutionApply");
}

void RestitutionPipelines::destroy()
{
    pre.destroy();
    impulse.destroy();
    apply.destroy();
}

void launchRestitutionPre(CommandList &cmd, HostRingBuffer &ring, RestitutionPipelines &pl,
                          const RestitutionArgsGpu &args)
{
    uint32_t groups = ceilDiv((int)args.slots, 128);
    if (groups > 0)
        cmd.dispatch(pl.pre, groups, ring.write(&args, sizeof(args)));
}

void launchRestitutionImpulse(CommandList &cmd, HostRingBuffer &ring, RestitutionPipelines &pl,
                              const RestitutionArgsGpu &args)
{
    uint32_t groups = ceilDiv((int)args.slots, 128);
    if (groups > 0)
        cmd.dispatch(pl.impulse, groups, ring.write(&args, sizeof(args)));
}

void launchRestitutionApply(CommandList &cmd, HostRingBuffer &ring, RestitutionPipelines &pl,
                            const RestitutionArgsGpu &args)
{
    uint32_t groups = ceilDiv((int)args.dynCount, 128);
    if (groups > 0)
        cmd.dispatch(pl.apply, groups, ring.write(&args, sizeof(args)));
}

} // namespace avbdvk
