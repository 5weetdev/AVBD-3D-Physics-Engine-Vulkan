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

#include "scene_common.h"

static SceneView sceneEmptyGpu(SceneBuilder *s)
{
    s->clear();
    return {50.0f, 5.0f};
}

static SceneView sceneGroundGpu(SceneBuilder *s)
{
    s->clear();
    s->addBody({10000.0f, 10000.0f, 1.0f}, 0.0f, 0.5f, {0, 0, 0});
    s->addBody({1, 1, 1}, 1.0f, 0.5f, {0, 0, 4});
    return {20.0f, 2.0f};
}

static SceneView sceneDynamicFrictionGpu(SceneBuilder *s)
{
    s->clear();
    s->addBody({10000.0f, 10000.0f, 1.0f}, 0.0f, 0.5f, {0, 0, 0});
    for (int x = 0; x < 10; x++)
        s->addBody({1, 1, 0.5f}, 1.0f, 5.0f - (x / 10.0f * 5.0f),
                   {0, -30.0f + x * 2.0f, 0.75f}, {10.0f, 0, 0});
    return {50.0f, 2.0f};
}

static SceneView sceneStaticFrictionGpu(SceneBuilder *s)
{
    s->clear();
    s->addBody({10000.0f, 10000.0f, 1.0f}, 0.0f, 0.5f, {0, 0, 0});

    const float angle = rad(30.0f);
    const quat rampRot = {0, sinf(angle * 0.5f), 0, cosf(angle * 0.5f)};
    const vec3 rampPos = {0, 0, 3};
    s->addBody({40, 24, 1}, 0.0f, 1.0f, rampPos, {0, 0, 0}, {0, 0, 0}, rampRot);

    vec3 rampTangent = normalize(rotate(rampRot, vec3{1, 0, 0}));
    vec3 rampNormal = normalize(rotate(rampRot, vec3{0, 0, 1}));

    for (int i = 0; i <= 10; i++)
    {
        float friction = i / 10.0f * 0.25f + 0.25f;
        float y = -10.0f + i * 2.0f;
        vec3 pos = rampPos + rampTangent * -12.0f + vec3{0, y, 0} + rampNormal * 1.05f;
        s->addBody({1, 1, 1}, 1.0f, friction, pos);
    }
    return {50.0f, 5.0f};
}

static SceneView scenePyramidGpu(SceneBuilder *s)
{
    const int SIZE = 16;
    s->clear();
    s->addBody({10000.0f, 10000.0f, 1.0f}, 0.0f, 0.5f, {0.0f, 0.0f, -0.5f});
    for (int y = 0; y < SIZE; y++)
        for (int x = 0; x < SIZE - y; x++)
            s->addBody({1, 0.5f, 0.5f}, 1.0f, 0.5f,
                       {x * 1.01f + y * 0.5f - SIZE / 2.0f, 0.0f, y * 0.85f + 0.5f});
    return {40.0f, 6.0f};
}

static SceneView sceneStackGpu(SceneBuilder *s)
{
    s->clear();
    s->addBody({10000.0f, 10000.0f, 1.0f}, 0.0f, 0.5f, {0, 0, 0});
    for (int i = 0; i < 10; i++)
        s->addBody({1, 1, 1}, 1.0f, 0.5f, {0, 0, i * 1.5f + 1.0f});
    return {30.0f, 7.0f};
}


// Fix 2 needs contacts that have no warm start to inherit, and every settled scene is
// the opposite of that: gpu_narrowphase matches contacts by feature key and carries
// their penalty forward, so after a few steps almost every contact is warm started and
// the initial stiffness is overwritten before it can matter. That is exactly why AVBD
// reports k_start as having no visible effect, and why measuring Fix 2 on a pile would
// measure nothing.
//
// This scene keeps first impacts happening. Bodies start spread over a wide range of
// heights, so they reach the ground in a continuous stream rather than all at once,
// and the contact set keeps turning over for the whole measurement window.
static SceneView sceneFreshContactsGpu(SceneBuilder *s)
{
    const int W = 32;
    const float SIZE = 1.0f;
    const float SPACING = 2.5f;

    s->clear();
    s->addBody({10000.0f, 10000.0f, 1.0f}, 0.0f, 0.5f, {0, 0, -0.5f});

    // A fixed integer hash rather than rand(), so the scene is identical on every run
    // and across builds -- a benchmark whose geometry moves between A and B is not a
    // benchmark.
    for (int x = 0; x < W; x++)
        for (int y = 0; y < W; y++)
        {
            unsigned int h = (unsigned int)(x * 73856093) ^ (unsigned int)(y * 19349663);
            h ^= h >> 13;
            h *= 1274126177u;
            h ^= h >> 16;
            float t = (float)(h & 0xFFFF) / 65535.0f;

            float px = (x - (W - 1) * 0.5f) * SPACING;
            float py = (y - (W - 1) * 0.5f) * SPACING;
            float pz = 3.0f + t * 120.0f;
            s->addBody({SIZE, SIZE, SIZE}, 1.0f, 0.5f, {px, py, pz});
        }
    return {70.0f, 20.0f};
}

// A tall single column of equal boxes: the adversarial case for penalty initialization.
//
// Every contact in this stack has an identical effective mass w = J M^-1 J^T, so the
// autoKStart estimate k = 1/(w dt^2) hands all 40 of them the same stiffness. But the
// load they carry is not the same at all: the bottom contact supports 39 boxes and the
// top one supports one, a 39:1 spread in the force the penalty has to reach. The
// effective-mass estimate is a purely local quantity and cannot see that, so whatever
// headroom is left above it here has to come from somewhere non-local -- which is
// exactly the question of whether a per-contact predictor is enough or whether load
// path has to be propagated along the column.
//
// Read the per-contact penalty spread and the residual-vs-iterations curve here before
// building anything: if the analytic estimate is already near-optimal on this column,
// there is nothing for a local model to learn.
static SceneView sceneStackRatioDeepGpu(SceneBuilder *s)
{
    const int HEIGHT = 40;
    const float SIZE = 1.0f;

    s->clear();
    const float groundThickness = 1.0f;
    s->addBody({10000.0f, 10000.0f, groundThickness}, 0.0f, 0.5f, {0, 0, 0});

    // Stacked face to face with no gap. A seam here would let the column gain speed
    // before it makes contact, and the first few steps would then be measuring an
    // impact rather than a static load path.
    float topZ = groundThickness * 0.5f;
    for (int i = 0; i < HEIGHT; i++)
    {
        const float half = SIZE * 0.5f;
        const float centerZ = topZ + half;
        s->addBody({SIZE, SIZE, SIZE}, 1.0f, 0.5f, {0, 0, centerZ});
        topZ = centerZ + half;
    }
    return {60.0f, 20.0f};
}

static SceneView sceneStackRatioGpu(SceneBuilder *s)
{
    s->clear();
    const float groundThickness = 1.0f;
    s->addBody({10000.0f, 10000.0f, groundThickness}, 0.0f, 0.5f, {0, 0, 0});

    float topZ = groundThickness * 0.5f;
    float size = 1.0f;
    for (int i = 0; i < 4; i++)
    {
        float half = size * 0.5f;
        float centerZ = topZ + half;
        s->addBody({size, size, size}, 1.0f, 0.5f, {0, 0, centerZ});
        topZ = centerZ + half;
        size *= 2.0f;
    }
    return {40.0f, 8.0f};
}
