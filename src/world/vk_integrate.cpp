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

namespace avbdvk
{

static uint32_t ceilDiv(int n, int d)
{
    if (n <= 0)
        return 0;
    return (uint32_t)((n + d - 1) / d);
}

void IntegratePipelines::init(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t words)
{
    integrate.init(dev, layout, spirv, words, "csIntegrate", "csIntegrate");
    velocity.init(dev, layout, spirv, words, "csVelocity", "csVelocity");
    killFallen.init(dev, layout, spirv, words, "csKillFallen", "csKillFallen");
    explosionGuard.init(dev, layout, spirv, words, "csExplosionGuard", "csExplosionGuard");
    integrateProps.init(dev, layout, spirv, words, "csIntegrateProps", "csIntegrateProps");
    velocityProps.init(dev, layout, spirv, words, "csVelocityProps", "csVelocityProps");
}

void IntegratePipelines::destroy()
{
    integrate.destroy();
    velocity.destroy();
    killFallen.destroy();
    explosionGuard.destroy();
    integrateProps.destroy();
    velocityProps.destroy();
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
