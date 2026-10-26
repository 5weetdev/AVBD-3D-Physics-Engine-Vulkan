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

// Deterministic jitter. A perfect lattice settles into degenerate face-face contacts
// and an unrepresentative color count, which would make the whole pile easier than a
// real one -- so every box is nudged.
static unsigned int sceneRng = 1234567u;
static float sceneRand(float lo, float hi)
{
    sceneRng = sceneRng * 1664525u + 1013904223u;
    return lo + (hi - lo) * ((sceneRng >> 8) & 0xFFFFFF) / (float)0xFFFFFF;
}

static SceneView buildPile(SceneBuilder *s, int layers, float groundSize, float camDist)
{
    s->clear();
    sceneRng = 1234567u;
    s->addBody({10000.0f, 10000.0f, 1.0f}, 0.0f, 0.5f, {0, 0, -0.5f});

    const float pitch = 1.05f;
    for (int z = 0; z < layers; z++)
        for (int y = 0; y < layers; y++)
            for (int x = 0; x < layers; x++)
                s->addBody({1, 1, 1}, 1.0f, 0.5f,
                           {(x - layers * 0.5f) * pitch + sceneRand(-0.02f, 0.02f),
                            (y - layers * 0.5f) * pitch + sceneRand(-0.02f, 0.02f),
                            z * pitch + 0.6f});

    return {camDist, layers * pitch * 0.4f};
}

static SceneView sceneMassive10kGpu(SceneBuilder *s) { return buildPile(s, 22, 120.0f, 70.0f); }

// 47^3 + 1 ground = 103,824 bodies. The target scene for the port.
static SceneView sceneMassive100kGpu(SceneBuilder *s) { return buildPile(s, 47, 200.0f, 150.0f); }

// 59^3 + 1 ground = 205,380 bodies.
static SceneView sceneMassive200kGpu(SceneBuilder *s) { return buildPile(s, 59, 250.0f, 200.0f); }

// 74^3 + 1 ground = 405,225 bodies.
static SceneView sceneMassive400kGpu(SceneBuilder *s) { return buildPile(s, 74, 320.0f, 260.0f); }

// 84^3 + 1 ground = 592,705 bodies.
static SceneView sceneMassive600kGpu(SceneBuilder *s) { return buildPile(s, 84, 380.0f, 320.0f); }

static void addPyramidWall(SceneBuilder *s, vec3 origin, int size, bool rotate90)
{
    // Double-thickness pyramid (2 rows of bricks in depth = 2 * size*(size+1)/2 blocks).
    // Bricks are tightly packed: 0.50m layer height matches the 0.5m box height,
    // and bottom layer center is at z = 0.25m so the bottom face rests flush at z = 0 (ground).
    for (int y = 0; y < size; y++)
    {
        const float z = y * 0.50f + 0.25f;
        for (int x = 0; x < size - y; x++)
        {
            const float offsetAlongBase = (x + y * 0.5f - (size - 1) * 0.5f) * 1.0f;
            for (int d = 0; d < 2; d++)
            {
                const float offsetPerp = (d == 0 ? -0.25f : 0.25f);
                if (!rotate90)
                {
                    // Oriented along X axis, depth along Y
                    s->addBody({1.0f, 0.5f, 0.5f}, 1.0f, 0.5f,
                               {origin.x + offsetAlongBase, origin.y + offsetPerp, z});
                }
                else
                {
                    // Oriented along Y axis, depth along X
                    s->addBody({0.5f, 1.0f, 0.5f}, 1.0f, 0.5f,
                               {origin.x + offsetPerp, origin.y + offsetAlongBase, z});
                }
            }
        }
    }
}

static SceneView buildPyramidField(SceneBuilder *s, int gridX, int gridY, int size, float pitch, float groundMargin, float camDist)
{
    s->clear();
    const float extentX = (gridX - 1) * pitch + size * 1.05f;
    const float extentY = (gridY - 1) * pitch + size * 1.05f;
    const float maxExtent = fmaxf(extentX, extentY);
    const float groundSize = maxExtent + groundMargin;

    s->addBody({10000.0f, 10000.0f, 1.0f}, 0.0f, 0.5f, {0.0f, 0.0f, -0.5f});

    for (int gy = 0; gy < gridY; gy++)
    {
        for (int gx = 0; gx < gridX; gx++)
        {
            const float ox = (gx - (gridX - 1) * 0.5f) * pitch;
            const float oy = (gy - (gridY - 1) * 0.5f) * pitch;
            const bool rotate90 = ((gx + gy) % 2) != 0;
            addPyramidWall(s, {ox, oy, 0.0f}, size, rotate90);
        }
    }

    return {camDist, size * 0.85f * 0.45f};
}

// 5x5 grid of SIZE=20 double-thickness pyramids (420 blocks each) = 10,500 blocks + 1 ground = 10,501 bodies.
static SceneView scenePyramids10kGpu(SceneBuilder *s)
{
    return buildPyramidField(s, 5, 5, 20, 26.0f, 50.0f, 140.0f);
}

// 15x16 grid of SIZE=20 double-thickness pyramids = 100,800 blocks + 1 ground = 100,801 bodies.
static SceneView scenePyramids100kGpu(SceneBuilder *s)
{
    return buildPyramidField(s, 15, 16, 20, 26.0f, 80.0f, 420.0f);
}

// 22x22 grid of SIZE=20 double-thickness pyramids = 203,280 blocks + 1 ground = 203,281 bodies.
static SceneView scenePyramids200kGpu(SceneBuilder *s)
{
    return buildPyramidField(s, 22, 22, 20, 26.0f, 100.0f, 600.0f);
}

// 31x31 grid of SIZE=20 double-thickness pyramids = 403,620 blocks + 1 ground = 403,621 bodies.
static SceneView scenePyramids400kGpu(SceneBuilder *s)
{
    return buildPyramidField(s, 31, 31, 20, 26.0f, 120.0f, 840.0f);
}
