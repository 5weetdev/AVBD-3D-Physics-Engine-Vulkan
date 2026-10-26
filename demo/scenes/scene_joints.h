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

static SceneView sceneRopeGpu(SceneBuilder *s)
{
    s->clear();
    s->addBody({10000.0f, 10000.0f, 1.0f}, 0.0f, 0.5f, {0, 0, -20});

    int prev = -1;
    for (int i = 0; i < 20; i++)
    {
        int curr = s->addBody({1, 0.5f, 0.5f}, i == 0 ? 0.0f : 1.0f, 0.5f, {(float)i, 0.0f, 10.0f});
        if (prev >= 0)
            s->addJoint(prev, curr, {0.5f, 0, 0}, {-0.5f, 0, 0});
        prev = curr;
    }
    return {40.0f, 0.0f};
}

static SceneView sceneHeavyRopeGpu(SceneBuilder *s)
{
    const int N = 20;
    const float SIZE = 5;
    s->clear();
    s->addBody({10000.0f, 10000.0f, 1.0f}, 0.0f, 0.5f, {0, 0, -20});

    int prev = -1;
    for (int i = 0; i < N; i++)
    {
        int curr = s->addBody(i == N - 1 ? vec3{SIZE, SIZE, SIZE} : vec3{1, 0.5f, 0.5f},
                              i == 0 ? 0.0f : 1.0f, 0.5f,
                              {(float)i + (i == N - 1 ? SIZE / 2 : 0), 0.0f, 10.0f});
        if (prev >= 0)
            s->addJoint(prev, curr, {0.5f, 0, 0}, i == N - 1 ? vec3{-SIZE / 2, 0, 0} : vec3{-0.5f, 0, 0});
        prev = curr;
    }
    return {50.0f, 0.0f};
}


// AVBD Figure 7a, reproduced so our parameter study can be compared against the
// published one. 50 rigid links form a chain holding a heavy ball, initialized in the
// vertical pose and left quasi-static; the ball's height over time is the signal.
//
// This is the case a single global k_start provably cannot serve: the top link carries
// the whole ball plus the chain, the bottom link carries almost nothing, so the
// constraints along it want stiffnesses orders of magnitude apart.
//
// Note that addBody takes FULL extents, so a link of length LEN has its endpoints at
// +/- LEN/2 and that is where the joints must go. The ball's density is derived from
// the target mass ratio rather than typed in, so changing the link size or the ratio
// keeps the scene consistent instead of silently producing a different experiment.
static SceneView sceneChainRatioGpu(SceneBuilder *s)
{
    const int N = 50;
    const float LEN = 1.0f;    // link length along z (full extent)
    const float THICK = 0.25f; // link cross-section (full extent)
    const float BALL = 2.0f;   // ball size (full extent)
    const float TOP = 60.0f;   // anchor height, chosen so the chain hangs clear
    const float RATIO = 50000.0f;

    const float linkMass = THICK * THICK * LEN;             // density 1
    const float ballDensity = linkMass * RATIO / (BALL * BALL * BALL);

    s->clear();
    s->addBody({10000.0f, 10000.0f, 1.0f}, 0.0f, 0.5f, {0, 0, -20});

    int prev = -1;
    for (int i = 0; i < N; i++)
    {
        // Link i spans [TOP - LEN*(i+1), TOP - LEN*i], so consecutive centres are one
        // full link apart and the shared joint lands on both links' faces.
        float cz = TOP - LEN * (i + 0.5f);
        int curr = s->addBody({THICK, THICK, LEN}, i == 0 ? 0.0f : 1.0f, 0.5f, {0, 0, cz});
        if (prev >= 0)
            s->addJoint(prev, curr, {0, 0, -LEN * 0.5f}, {0, 0, LEN * 0.5f});
        prev = curr;
    }

    // The heavy ball, hung off the last link. Its centre sits half a ball below the
    // chain's lower end so the joint lands exactly on its top face, leaving zero
    // initial constraint error -- the profiler checks for that and would report it.
    float ballZ = TOP - LEN * N - BALL * 0.5f;
    s->addJoint(prev, s->addBody({BALL, BALL, BALL}, ballDensity, 0.5f, {0, 0, ballZ}),
                {0, 0, -LEN * 0.5f}, {0, 0, BALL * 0.5f});

    return {90.0f, 20.0f};
}

static SceneView sceneSpringGpu(SceneBuilder *s)
{
    s->clear();
    s->addBody({10000.0f, 10000.0f, 1.0f}, 0.0f, 0.5f, {0, 0, 0});

    int anchor = s->addBody({1, 1, 1}, 0.0f, 0.5f, {0, 0, 14.0f});
    int block = s->addBody({2, 2, 2}, 1.0f, 0.5f, {0, 0, 8.0f});
    s->addSpring(anchor, block, {0, 0, 0}, {0, 0, 0}, 100.0f, 4.0f);
    return {25.0f, 8.0f};
}

static SceneView sceneSpringsRatioGpu(SceneBuilder *s)
{
    const int N = 8;
    s->clear();
    s->addBody({10000.0f, 10000.0f, 1.0f}, 0.0f, 0.5f, {0, 0, -10});

    int prev = -1;
    for (int i = 0; i < N; i++)
    {
        float x = (i - (N - 1) * 0.5f) * 3.0f;
        int curr = s->addBody({1, 0.75f, 0.75f}, i == 0 || i == N - 1 ? 0.0f : 1.0f, 0.5f, {x, 0.0f, 12.0f});
        if (prev >= 0)
            s->addSpring(prev, curr, {0.5f, 0, 0}, {-0.5f, 0, 0}, i % 2 == 0 ? 10.0f : 10000.0f, 3.0f);
        prev = curr;
    }
    return {35.0f, 8.0f};
}

static SceneView sceneSoftBodyGpu(SceneBuilder *s)
{
    s->clear();
    s->addBody({10000.0f, 10000.0f, 1.0f}, 0.0f, 0.5f, {0, 0, 0});

    const float Klin = 1000.0f;
    const float Kang = 250.0f;
    const int W = 4;
    const int D = 4;
    const int H = 4;
    const int N = 3;
    const float size = 0.8f;
    const float half = size * 0.5f;
    const float baseZ = 8.0f;
    const float stackGap = 2.0f;

    for (int i = 0; i < N; i++)
    {
        int grid[W][D][H];
        float stackZ = i * (H * size + stackGap);

        for (int x = 0; x < W; x++)
            for (int y = 0; y < D; y++)
                for (int z = 0; z < H; z++)
                {
                    float px = (x - (W - 1) * 0.5f) * size;
                    float py = (y - (D - 1) * 0.5f) * size;
                    float pz = baseZ + stackZ + z * size;
                    grid[x][y][z] = s->addBody({size, size, size}, 1.0f, 0.5f, {px, py, pz});
                }

        for (int x = 1; x < W; x++)
            for (int y = 0; y < D; y++)
                for (int z = 0; z < H; z++)
                    s->addJoint(grid[x - 1][y][z], grid[x][y][z], {half, 0, 0}, {-half, 0, 0}, Klin, Kang);

        for (int x = 0; x < W; x++)
            for (int y = 1; y < D; y++)
                for (int z = 0; z < H; z++)
                    s->addJoint(grid[x][y - 1][z], grid[x][y][z], {0, half, 0}, {0, -half, 0}, Klin, Kang);

        for (int x = 0; x < W; x++)
            for (int y = 0; y < D; y++)
                for (int z = 1; z < H; z++)
                    s->addJoint(grid[x][y][z - 1], grid[x][y][z], {0, 0, half}, {0, 0, -half}, Klin, Kang);

        // Diagonal neighbours are not jointed, but they do interpenetrate as the
        // lattice deforms, so collision between them is suppressed explicitly.
        for (int x = 1; x < W; x++)
            for (int y = 0; y < D; y++)
                for (int z = 1; z < H; z++)
                {
                    s->ignoreCollision(grid[x - 1][y][z - 1], grid[x][y][z]);
                    s->ignoreCollision(grid[x][y][z - 1], grid[x - 1][y][z]);
                }

        for (int x = 0; x < W; x++)
            for (int y = 1; y < D; y++)
                for (int z = 1; z < H; z++)
                {
                    s->ignoreCollision(grid[x][y - 1][z - 1], grid[x][y][z]);
                    s->ignoreCollision(grid[x][y][z - 1], grid[x][y - 1][z]);
                }

        for (int x = 1; x < W; x++)
            for (int y = 1; y < D; y++)
                for (int z = 0; z < H; z++)
                {
                    s->ignoreCollision(grid[x - 1][y - 1][z], grid[x][y][z]);
                    s->ignoreCollision(grid[x - 1][y][z], grid[x][y - 1][z]);
                }
    }
    return {25.0f, 8.0f};
}

static SceneView sceneBridgeGpu(SceneBuilder *s)
{
    const int N = 40;
    const float plankLength = 1.0f;
    const float plankWidth = 4.0f;
    const float plankHeight = 0.5f;
    const float halfLength = plankLength * 0.5f;
    const float halfWidth = plankWidth * 0.5f;

    s->clear();
    s->addBody({10000.0f, 10000.0f, 1.0f}, 0.0f, 0.5f, {0, 0, 0});

    int prev = -1;
    for (int i = 0; i < N; i++)
    {
        int curr = s->addBody({plankLength, plankWidth, plankHeight},
                              i == 0 || i == N - 1 ? 0.0f : 1.0f, 0.5f,
                              {(float)i - N / 2.0f, 0.0f, 10.0f});
        if (prev >= 0)
        {
            s->addJoint(prev, curr, {halfLength, halfWidth, 0}, {-halfLength, halfWidth, 0}, INFINITY, INFINITY);
            s->addJoint(prev, curr, {halfLength, -halfWidth, 0}, {-halfLength, -halfWidth, 0}, INFINITY, INFINITY);
        }
        prev = curr;
    }

    for (int x = 0; x < N / 4; x++)
        for (int y = 0; y < N / 8; y++)
            s->addBody({1, 1, 1}, 1.0f, 0.5f, {(float)x - N / 8.0f, 0.0f, (float)y + 12.0f});

    return {60.0f, 8.0f};
}

