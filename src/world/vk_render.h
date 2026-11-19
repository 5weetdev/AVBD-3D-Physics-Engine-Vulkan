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

// Vulkan host orchestration for src/shaders/render.slang -- the CORE instance/pose
// write path only (see render.slang's header comment for the disclosed scope
// reduction versus the instance writer, the overlay reductions, the debug-draw writer).
// Mirrors GpuSolver::writeInstances()'s core branch.

#include "vk_args.h"
#include "vk_buffer.h"
#include "vk_cmd.h"
#include "vk_device.h"
#include "vk_pipeline.h"

#include <cstdint>

namespace avbdvk
{

// The embedded SPIR-V of the three render modules (render, render_overlay, render_debug).
struct RenderModules
{
    const uint32_t *render;
    size_t renderWords;
    const uint32_t *overlay;
    size_t overlayWords;
    const uint32_t *debug;
    size_t debugWords;
};

struct RenderPipelines
{
    Pipeline writeInstances;

    // Phase 5C-cont: the overlay reductions + the debug-draw writer ports (see render.slang).
    Pipeline overlayStatsClear;
    Pipeline overlayStatsContacts;
    Pipeline overlayStatsForces;
    Pipeline overlayLoadClear;
    Pipeline overlayLoadAccum;
    Pipeline writeDebugInstances;

    void init(Device &dev, PipelineLayout &layout, const RenderModules &modules);
    void destroy();
};

// Writes one BodyInstanceGpu per SHAPE (args.s) into `out` (a device-local buffer sized for at
// least min(shapeCount, maxInstances) entries). Full port: frustum culling (args.frustum
// != 0), overlay colouring (args.ov.mode != OVERLAY_NONE), and per-shape/group
// segmentation (args.shapeBase != 0) all match the instance writer's
// k_write_instances exactly -- see render.slang. `shapeCount` sizes the dispatch.
void launchWriteInstances(CommandList &cmd, HostRingBuffer &ring, RenderPipelines &pl,
                          const WriteInstancesArgsGpu &args, int shapeCount);

// the overlay reductions ports: three-pass per-frame overlay stat reduction
// (clear -> per-manifold contact pass -> per-joint/spring force pass), then two-pass
// peak-contact-load-per-body reduction (clear -> accumulate). Dispatch counts mirror
// the reference launch bounds (n bodies / slots contacts / joints+springs).
void launchOverlayStatsClear(CommandList &cmd, HostRingBuffer &ring, RenderPipelines &pl,
                             const OverlayStatsClearArgsGpu &args, int n);
void launchOverlayStatsContacts(CommandList &cmd, HostRingBuffer &ring, RenderPipelines &pl,
                                const OverlayStatsContactsArgsGpu &args, int slots);
void launchOverlayStatsForces(CommandList &cmd, HostRingBuffer &ring, RenderPipelines &pl,
                              const OverlayStatsForcesArgsGpu &args, int jointsAndSprings);
void launchOverlayLoadClear(CommandList &cmd, HostRingBuffer &ring, RenderPipelines &pl,
                            const OverlayLoadClearArgsGpu &args, int n);
void launchOverlayLoadAccum(CommandList &cmd, HostRingBuffer &ring, RenderPipelines &pl,
                            const OverlayLoadAccumArgsGpu &args, int slots);

// the debug-draw writer ports: contact point/normal/manifold-link markers and
// joint/spring force rods, each compacted via an atomic `counter`.
void launchWriteDebugInstances(CommandList &cmd, HostRingBuffer &ring, RenderPipelines &pl,
                               const WriteDebugInstancesArgsGpu &args, int slots);

} // namespace avbdvk
