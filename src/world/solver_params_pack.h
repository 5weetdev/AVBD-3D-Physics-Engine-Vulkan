/*
 * Host-side packing of the real SolverParams (src/core/solver_params.h) into the
 * GPU-safe layout the Vulkan backend actually uploads and src/shaders/types.slang's
 * SolverParamsGpu reads.
 *
 * SolverParams cannot be byte-mirrored: it has `bool sleepEnabled`, `bool autoKStart`,
 * `bool killFallenEnabled` mixed in with floats and ints. C++ `bool` is 1 byte with up
 * to 3 bytes of UNINITIALIZED padding before the next 4-byte-aligned field; HLSL/Slang
 * `bool` is a 4-byte type. A raw memcpy of SolverParams would let the GPU read that
 * padding as meaningful bits. So every field is repacked explicitly here, each bool
 * widened to a full uint (0 or 1), with no gaps -- this is the one real conversion in
 * the pipeline, not just a cast.
 *
 * Field order matches SolverParamsGpu in src/shaders/types.slang exactly; keep the
 * two declarations in lockstep by hand (this is exactly what tools/test_vk_layout.cpp's
 * csProbeSolverParams gate exists to catch if they ever drift).
 */
#pragma once

#include "solver_params.h"

#include <cstdint>

namespace avbdvk
{

// Mirrors src/shaders/types.slang's SolverParamsGpu field-for-field: 20 x 4-byte
// scalars, no padding, no pointers.
struct SolverParamsGpu
{
    float dt;
    float worldFrameInertia; // 0 or 1; was a vestigial gravity copy
    int32_t iterations;
    int32_t primalLanes;
    float alpha;
    float betaLin;
    float betaAng;
    float gamma;
    uint32_t sleepEnabled;
    int32_t sleepFrames;
    float sleepLinVel;
    float sleepAngVel;
    float sleepDisp;
    float sleepFracture;
    uint32_t autoKStart;
    float dualDamping;
    float dualDampVel;
    uint32_t killFallenEnabled;
    float killHeight;
    float grabStrength;
};

inline SolverParamsGpu packSolverParams(const SolverParams &p)
{
    SolverParamsGpu g{};
    g.dt = p.dt;
    g.worldFrameInertia = p.worldFrameInertia ? 1.0f : 0.0f;
    g.iterations = p.iterations;
    g.primalLanes = p.primalLanes;
    g.alpha = p.alpha;
    g.betaLin = p.betaLin;
    g.betaAng = p.betaAng;
    g.gamma = p.gamma;
    g.sleepEnabled = p.sleepEnabled ? 1u : 0u;
    g.sleepFrames = p.sleepFrames;
    g.sleepLinVel = p.sleepLinVel;
    g.sleepAngVel = p.sleepAngVel;
    g.sleepDisp = p.sleepDisp;
    g.sleepFracture = p.sleepFracture;
    g.autoKStart = p.autoKStart ? 1u : 0u;
    g.dualDamping = p.dualDamping;
    g.dualDampVel = p.dualDampVel;
    g.killFallenEnabled = p.killFallenEnabled ? 1u : 0u;
    g.killHeight = p.killHeight;
    g.grabStrength = p.grabStrength;
    return g;
}

} // namespace avbdvk
