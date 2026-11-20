#include "vk_integrate.h"
#include "vk_util.h"

namespace avbdvk
{

// Every entry point of this module, named once; init and destroy walk the same table.
static std::array<PipelineEntry, 6> entries(IntegratePipelines &p)
{
    return {{
        {&p.integrate, "csIntegrate"},
        {&p.velocity, "csVelocity"},
        {&p.killFallen, "csKillFallen"},
        {&p.explosionGuard, "csExplosionGuard"},
        {&p.integrateProps, "csIntegrateProps"},
        {&p.velocityProps, "csVelocityProps"},
    }};
}

void IntegratePipelines::init(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t words)
{
    initPipelines(dev, layout, spirv, words, entries(*this));
}

void IntegratePipelines::destroy()
{
    destroyPipelines(entries(*this));
}

void launchIntegrate(CommandList &cmd, HostRingBuffer &ring, IntegratePipelines &pl, const IntegrateArgsGpu &args,
                     int bodyCount)
{
    uint32_t groups = ceilDiv(bodyCount, 256);
    if (groups == 0)
        return;
    cmd.dispatchArgs(pl.integrate, groups, ring, args);
}

void launchVelocity(CommandList &cmd, HostRingBuffer &ring, IntegratePipelines &pl, const VelocityArgsGpu &args,
                    int bodyCount)
{
    uint32_t groups = ceilDiv(bodyCount, 256);
    if (groups == 0)
        return;
    cmd.dispatchArgs(pl.velocity, groups, ring, args);
}

void launchIntegrateProps(CommandList &cmd, HostRingBuffer &ring, IntegratePipelines &pl,
                          const IntegratePropsArgsGpu &args, int bodyCount)
{
    uint32_t groups = ceilDiv(bodyCount, 256);
    if (groups == 0)
        return;
    cmd.dispatchArgs(pl.integrateProps, groups, ring, args);
}

void launchVelocityProps(CommandList &cmd, HostRingBuffer &ring, IntegratePipelines &pl,
                         const VelocityPropsArgsGpu &args, int bodyCount)
{
    uint32_t groups = ceilDiv(bodyCount, 256);
    if (groups == 0)
        return;
    cmd.dispatchArgs(pl.velocityProps, groups, ring, args);
}

void launchKillFallen(CommandList &cmd, HostRingBuffer &ring, IntegratePipelines &pl, const KillFallenArgsGpu &args,
                      int dynCount)
{
    uint32_t groups = ceilDiv(dynCount, 256);
    if (groups == 0)
        return;
    cmd.dispatchArgs(pl.killFallen, groups, ring, args);
}

void launchExplosionGuard(CommandList &cmd, HostRingBuffer &ring, IntegratePipelines &pl,
                          const ExplosionGuardArgsGpu &args, int dynCount)
{
    uint32_t groups = ceilDiv(dynCount, 256);
    if (groups == 0)
        return;
    cmd.dispatchArgs(pl.explosionGuard, groups, ring, args);
}

} // namespace avbdvk
