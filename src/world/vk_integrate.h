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

#pragma once

// Vulkan host orchestration for src/shaders/integrate.slang: csIntegrate (inertial
// predictor + adaptive warmstart), csVelocity (BDF1 reconstruction), csKillFallen and
// csExplosionGuard (the two end-of-step circuit breakers). Mirrors the reference launch
// seam in the integrate stage / the launch seam.
//
// integrate.slang's BodySoA/IntegrateArgs/VelocityArgs/KillFallenArgs/ExplosionGuardArgs
// are private to that file (not `public`, no importable surface) -- same situation as
// primal.slang's forces args before vk_args.h existed for those. Rather than duplicating
// integrate.slang's *whole* BodySoA subset into vk_args.h (it would be the fourth copy of
// the same 11 fields, one per consumer), this header re-declares the args structs
// byte-for-byte, following the exact precedent already established by
// src/shaders/layout_probe.slang's IntegrateArgsProbe / tools/test_vk_layout.cpp's
// IntegrateArgsReal: an independent transcription in the same field order, verified by
// a GPU probe dispatch rather than by #include.

#include "vk_buffer.h"
#include "vk_cmd.h"
#include "vk_device.h"
#include "vk_pipeline.h"

#include <cstdint>

namespace avbdvk
{

// Mirrors integrate.slang's BodySoA. VkDeviceAddress in place of each typed pointer.
struct IntegrateBodySoAGpu
{
    VkDeviceAddress pos = 0;
    VkDeviceAddress rot = 0;
    VkDeviceAddress initialLin = 0;
    VkDeviceAddress initialAng = 0;
    VkDeviceAddress inertialLin = 0;
    VkDeviceAddress inertialAng = 0;
    VkDeviceAddress velLin = 0;
    VkDeviceAddress velAng = 0;
    VkDeviceAddress prevVelLin = 0;
    VkDeviceAddress mass = 0;
    VkDeviceAddress awake = 0;
    uint32_t count = 0;
    uint32_t _pad0 = 0;
};
static_assert(sizeof(IntegrateBodySoAGpu) % 8 == 0, "IntegrateBodySoAGpu must stay 8-byte aligned");

struct IntegrateSolverParamsGpu
{
    float dt = 0.0f;
    float gravityX = 0.0f;
    float gravityY = 0.0f;
    float gravityZ = 0.0f;
    uint32_t spinExact = 1; // SolverParams.exactSpinVelocity (velocity kernels only)
    uint32_t _padP = 0;
};

struct IntegrateArgsGpu
{
    IntegrateBodySoAGpu b;
    IntegrateSolverParamsGpu p;
};

struct VelocityArgsGpu
{
    IntegrateBodySoAGpu b;
    IntegrateSolverParamsGpu p;
};

// Per-body settings variants (integrate.slang csIntegrateProps / csVelocityProps): the same
// args plus the BodyProps array, dispatched instead of the originals only when some body has a
// non-default BodyProps.
struct IntegratePropsArgsGpu
{
    IntegrateBodySoAGpu b;
    IntegrateSolverParamsGpu p;
    VkDeviceAddress props = 0;
};
static_assert(sizeof(IntegratePropsArgsGpu) % 8 == 0, "layout");
using VelocityPropsArgsGpu = IntegratePropsArgsGpu;

// Mirrors integrate.slang's KillFallenArgs. The rest/sleep fields travel as their own
// pointers rather than growing IntegrateBodySoAGpu (see integrate.slang's own comment on
// why -- that struct is byte-pinned to tools/test_vk_integrate.cpp's BodySoAAddr).
struct KillFallenArgsGpu
{
    IntegrateBodySoAGpu b;
    VkDeviceAddress dyn = 0;
    int32_t dynCount = 0;
    float killHeight = 0.0f; // along (upX, upY, upZ), the unit up axis
    float upX = 0.0f;
    float upY = 1.0f;
    float upZ = 0.0f;
    float _padK = 0.0f;
    VkDeviceAddress sleepTimer = 0;
    VkDeviceAddress restPos = 0;
    VkDeviceAddress restRot = 0;
};
static_assert(sizeof(KillFallenArgsGpu) % 8 == 0, "KillFallenArgsGpu must stay 8-byte aligned");

struct ExplosionGuardArgsGpu
{
    IntegrateBodySoAGpu b;
    VkDeviceAddress dyn = 0;
    int32_t dynCount = 0;
    int32_t _pad0 = 0;
    VkDeviceAddress sleepTimer = 0;
    VkDeviceAddress restPos = 0;
    VkDeviceAddress restRot = 0;
};
static_assert(sizeof(ExplosionGuardArgsGpu) % 8 == 0, "ExplosionGuardArgsGpu must stay 8-byte aligned");

struct IntegratePipelines
{
    Pipeline integrate, velocity, killFallen, explosionGuard;
    Pipeline integrateProps, velocityProps;

    // spirv/words: src/shaders/integrate.slang's embedded SPIR-V.
    void init(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t words);
    void destroy();
};

// All four kernels are numthreads(256,1,1), one thread per list entry (body index for
// integrate/velocity, dynamic-list slot for killFallen/explosionGuard).
void launchIntegrate(CommandList &cmd, HostRingBuffer &ring, IntegratePipelines &pl,
                     const IntegrateArgsGpu &args, int bodyCount);

void launchVelocity(CommandList &cmd, HostRingBuffer &ring, IntegratePipelines &pl,
                    const VelocityArgsGpu &args, int bodyCount);

void launchIntegrateProps(CommandList &cmd, HostRingBuffer &ring, IntegratePipelines &pl,
                          const IntegratePropsArgsGpu &args, int bodyCount);

void launchVelocityProps(CommandList &cmd, HostRingBuffer &ring, IntegratePipelines &pl,
                         const VelocityPropsArgsGpu &args, int bodyCount);

void launchKillFallen(CommandList &cmd, HostRingBuffer &ring, IntegratePipelines &pl,
                      const KillFallenArgsGpu &args, int dynCount);

void launchExplosionGuard(CommandList &cmd, HostRingBuffer &ring, IntegratePipelines &pl,
                          const ExplosionGuardArgsGpu &args, int dynCount);

} // namespace avbdvk
