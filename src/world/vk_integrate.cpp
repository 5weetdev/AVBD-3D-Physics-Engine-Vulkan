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
    VkDeviceAddress argsAddr = ring.write(&args, sizeof(args));
    cmd.dispatch(pl.integrate, groups, argsAddr);
}

void launchVelocity(CommandList &cmd, HostRingBuffer &ring, IntegratePipelines &pl, const VelocityArgsGpu &args,
                    int bodyCount)
{
    uint32_t groups = ceilDiv(bodyCount, 256);
    if (groups == 0)
        return;
    VkDeviceAddress argsAddr = ring.write(&args, sizeof(args));
    cmd.dispatch(pl.velocity, groups, argsAddr);
}

void launchIntegrateProps(CommandList &cmd, HostRingBuffer &ring, IntegratePipelines &pl,
                          const IntegratePropsArgsGpu &args, int bodyCount)
{
    uint32_t groups = ceilDiv(bodyCount, 256);
    if (groups == 0)
        return;
    VkDeviceAddress argsAddr = ring.write(&args, sizeof(args));
    cmd.dispatch(pl.integrateProps, groups, argsAddr);
}

void launchVelocityProps(CommandList &cmd, HostRingBuffer &ring, IntegratePipelines &pl,
                         const VelocityPropsArgsGpu &args, int bodyCount)
{
    uint32_t groups = ceilDiv(bodyCount, 256);
    if (groups == 0)
        return;
    VkDeviceAddress argsAddr = ring.write(&args, sizeof(args));
    cmd.dispatch(pl.velocityProps, groups, argsAddr);
}

void launchKillFallen(CommandList &cmd, HostRingBuffer &ring, IntegratePipelines &pl, const KillFallenArgsGpu &args,
                      int dynCount)
{
    uint32_t groups = ceilDiv(dynCount, 256);
    if (groups == 0)
        return;
    VkDeviceAddress argsAddr = ring.write(&args, sizeof(args));
    cmd.dispatch(pl.killFallen, groups, argsAddr);
}

void launchExplosionGuard(CommandList &cmd, HostRingBuffer &ring, IntegratePipelines &pl,
                          const ExplosionGuardArgsGpu &args, int dynCount)
{
    uint32_t groups = ceilDiv(dynCount, 256);
    if (groups == 0)
        return;
    VkDeviceAddress argsAddr = ring.write(&args, sizeof(args));
    cmd.dispatch(pl.explosionGuard, groups, argsAddr);
}

} // namespace avbdvk
