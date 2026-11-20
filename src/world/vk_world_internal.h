#pragma once

// Constants and helpers shared by the VkWorld translation units (vk_world*.cpp). Not a public header.

#include "vk_world.h"

#include <chrono>

namespace avbdvk
{

// Drag slots beyond the authored joints, always present -- pool size: a multi-body grab holds up to this many bodies.
inline constexpr int kMaxDragJoints = 256;
// dragStage layout: bodies, targets, locals, stiffness, each kMaxDragJoints long.
inline constexpr VkDeviceSize kDragBodiesOff = 0;
inline constexpr VkDeviceSize kDragTargetsOff = kDragBodiesOff + kMaxDragJoints * sizeof(int);
inline constexpr VkDeviceSize kDragLocalsOff = kDragTargetsOff + kMaxDragJoints * sizeof(vec3);
inline constexpr VkDeviceSize kDragStiffOff = kDragLocalsOff + kMaxDragJoints * sizeof(vec3);
inline constexpr VkDeviceSize kDragStageBytes = kDragStiffOff + kMaxDragJoints * sizeof(float);
// Above this many oversized bodies the broadphase falls back to the legacy cell (brute force is O(L*N)).
inline constexpr int kMaxOversizeBodies = 64;


// A contact cap below 4 is raised to 4. Measured on the settled 10k pile
// (bench/vk_lod_2026-10-01.md): the reduction seeds on the deepest point, which moves
// between corners as a box rocks, so a cap of 2 or 3 drops a different corner of a face
// each step, the returning one starts cold, and the pile never settles (cap 3: 3x the
// manifolds, 1.3x the step; cap 1 4.5x). Four keeps a box face's whole quad.
inline int effectiveContactCap(int cap)
{
    return cap > 0 ? std::max(cap, 4) : 0;
}

inline constexpr int kMaxCapAttempts = 5;  // the pass after this many capacity reruns commits clamped
inline constexpr int kCapSlotsBase = 2048; // phase 4 <= ~670 dispatches, phase 5 ~100 + JP rounds
// First zero slot + 1 (the convergence round), or 0 if every slot changed.
inline int convergedAt(const int32_t *flags, int n)
{
    for (int r = 0; r < n; r++)
        if (flags[r] == 0)
            return r + 1;
    return 0;
}

using Clock = std::chrono::steady_clock;
inline float msSince(Clock::time_point t0, Clock::time_point t1)
{
    return std::chrono::duration<float, std::milli>(t1 - t0).count();
}

} // namespace avbdvk
