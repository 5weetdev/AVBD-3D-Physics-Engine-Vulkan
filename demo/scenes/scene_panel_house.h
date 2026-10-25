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

// Large Prefabricated Panel Complex (~5,000 blocks)
static SceneView scenePanelHouseGpu(SceneBuilder *s)
{
    s->clear();

    // Large static ground plane
    int ground = s->addBody({10000.0f, 10000.0f, 1.0f}, 0.0f, 0.5f, {0.0f, 0.0f, -0.5f});

    const float breakForce = 3200.0f;
    const float breakForceMinor = 1400.0f;
    const float panelAngStiffness = 1000000.0f; // Increased for more rigidity (less jello)

    const int floors = 18;
    const float wallHeight = 2.6f;
    const float wallThick = 0.2f;
    const float slabThick = 0.2f;
    const float floorPitch = wallHeight + slabThick; // 2.8m

    const int baysX = 8;
    const int baysY = 3;
    const float baySize = 3.0f;

    // Twin residential towers: Tower A at X = -16m, Tower B at X = +16m
    const float towerOffsetX[2] = {-16.0f, +16.0f};
    int towerInnerSlabs[2][18]; // [tower 0=A, 1=B][floor] inner edge slab at by = 1

    for (int t = 0; t < 2; t++)
    {
        const float tox = towerOffsetX[t];
        int prevFloorSlabs[8][3];
        for (int bx = 0; bx < baysX; bx++)
            for (int by = 0; by < baysY; by++)
                prevFloorSlabs[bx][by] = -1;

        for (int k = 0; k < floors; k++)
        {
            const float zBase = k * floorPitch;
            const float zw = zBase + wallHeight * 0.5f;
            const float zs = zBase + wallHeight + slabThick * 0.5f;

            int curFloorSlabs[8][3];

            // 1. Floor slabs (8 x 3 = 24 per floor)
            for (int bx = 0; bx < baysX; bx++)
            {
                for (int by = 0; by < baysY; by++)
                {
                    const float cx = tox + (bx - (baysX - 1) * 0.5f) * baySize;
                    const float cy = (by - (baysY - 1) * 0.5f) * baySize;
                    curFloorSlabs[bx][by] = s->addBody({baySize, baySize, slabThick}, 1.0f, 0.6f, {cx, cy, zs});
                    if (t == 0 && bx == baysX - 1 && by == 1)
                        towerInnerSlabs[0][k] = curFloorSlabs[bx][by];
                    if (t == 1 && bx == 0 && by == 1)
                        towerInnerSlabs[1][k] = curFloorSlabs[bx][by];

                    // Tie adjacent slabs in X
                    if (bx > 0)
                        s->addJointWorld(curFloorSlabs[bx - 1][by], curFloorSlabs[bx][by],
                                         {cx - baySize * 0.5f, cy, zs},
                                         INFINITY, INFINITY, breakForce);
                    // Tie adjacent slabs in Y
                    if (by > 0)
                        s->addJointWorld(curFloorSlabs[bx][by - 1], curFloorSlabs[bx][by],
                                         {cx, cy - baySize * 0.5f, zs},
                                         INFINITY, INFINITY, breakForce);
                }
            }

            // 2. Transverse interior & exterior wall panels (9 x 3 = 27 per floor)
            for (int bx = 0; bx <= baysX; bx++)
            {
                const float wx = tox + (bx - baysX * 0.5f) * baySize;
                for (int by = 0; by < baysY; by++)
                {
                    const float wy = (by - (baysY - 1) * 0.5f) * baySize;
                    int wall = s->addBody({wallThick, baySize * 0.95f, wallHeight}, 1.0f, 0.6f, {wx, wy, zw});

                    // Anchor ground floor walls to foundation ground
                    if (k == 0)
                        s->addJointWorld(ground, wall, {wx, wy, 0.0f}, INFINITY, INFINITY, breakForce);

                    // Joint to slab resting on top
                    if (bx < baysX)
                        s->addJointWorld(wall, curFloorSlabs[bx][by], {wx, wy, zBase + wallHeight},
                                         INFINITY, panelAngStiffness, breakForce);
                    if (bx > 0)
                        s->addJointWorld(wall, curFloorSlabs[bx - 1][by], {wx, wy, zBase + wallHeight},
                                         INFINITY, panelAngStiffness, breakForce);

                    // Joint to lower slab below
                    if (prevFloorSlabs[0][0] >= 0)
                    {
                        if (bx < baysX)
                            s->addJointWorld(prevFloorSlabs[bx][by], wall, {wx, wy, zBase},
                                             INFINITY, panelAngStiffness, breakForce);
                        if (bx > 0)
                            s->addJointWorld(prevFloorSlabs[bx - 1][by], wall, {wx, wy, zBase},
                                             INFINITY, panelAngStiffness, breakForce);
                    }
                }
            }

            // 3. Longitudinal facade & corridor wall panels (8 x 4 = 32 per floor)
            for (int bx = 0; bx < baysX; bx++)
            {
                const float wx = tox + (bx - (baysX - 1) * 0.5f) * baySize;
                for (int by = 0; by <= baysY; by++)
                {
                    const float wy = (by - baysY * 0.5f) * baySize;
                    int wall = s->addBody({baySize * 0.95f, wallThick, wallHeight}, 1.0f, 0.6f, {wx, wy, zw});

                    // Anchor ground floor longitudinal walls to foundation ground
                    if (k == 0)
                        s->addJointWorld(ground, wall, {wx, wy, 0.0f}, INFINITY, INFINITY, breakForce);

                    // Joint to slab above
                    if (by < baysY)
                        s->addJointWorld(wall, curFloorSlabs[bx][by], {wx, wy, zBase + wallHeight},
                                         INFINITY, panelAngStiffness, breakForce);
                    if (by > 0)
                        s->addJointWorld(wall, curFloorSlabs[bx][by - 1], {wx, wy, zBase + wallHeight},
                                         INFINITY, panelAngStiffness, breakForce);

                    // Joint to lower slab below
                    if (prevFloorSlabs[0][0] >= 0)
                    {
                        if (by < baysY)
                            s->addJointWorld(prevFloorSlabs[bx][by], wall, {wx, wy, zBase},
                                             INFINITY, panelAngStiffness, breakForce);
                        if (by > 0)
                            s->addJointWorld(prevFloorSlabs[bx][by - 1], wall, {wx, wy, zBase},
                                             INFINITY, panelAngStiffness, breakForce);
                    }
                }
            }

            // 4. Balcony slabs & parapet railings (8 x 2 = 16 balconies, 16 railings = 32 per floor)
            for (int bx = 0; bx < baysX; bx++)
            {
                const float bxPos = tox + (bx - (baysX - 1) * 0.5f) * baySize;
                for (int side = 0; side < 2; side++)
                {
                    const float sgn = (side == 0) ? -1.0f : +1.0f;
                    const float faceY = sgn * (baysY * 0.5f * baySize);
                    const float balY = faceY + sgn * 0.6f;
                    int balSlab = s->addBody({baySize * 0.95f, 1.2f, 0.15f}, 0.8f, 0.6f, {bxPos, balY, zs});
                    int balRail = s->addBody({baySize * 0.95f, 0.15f, 0.9f}, 0.5f, 0.6f, {bxPos, balY + 0.5f * sgn, zs + 0.525f});

                    int attachedSlab = (side == 0) ? curFloorSlabs[bx][0] : curFloorSlabs[bx][baysY - 1];
                    s->addJointWorld(attachedSlab, balSlab, {bxPos, faceY, zs},
                                     INFINITY, panelAngStiffness, breakForceMinor);
                    s->addJointWorld(balSlab, balRail, {bxPos, balY + 0.5f * sgn, zs + 0.075f},
                                     INFINITY, panelAngStiffness, breakForceMinor);
                }
            }

            // 5. Interior apartment partition walls (8 x 3 = 24 per floor)
            for (int bx = 0; bx < baysX; bx++)
            {
                for (int by = 0; by < baysY; by++)
                {
                    const float px = tox + (bx - (baysX - 1) * 0.5f) * baySize + 0.4f;
                    const float py = (by - (baysY - 1) * 0.5f) * baySize;
                    const float partH = wallHeight * 0.95f;
                    const float zw_part = (zBase + wallHeight) - partH * 0.5f;
                    int part = s->addBody({0.15f, 1.6f, partH}, 0.6f, 0.5f, {px, py, zw_part});
                    s->addJointWorld(curFloorSlabs[bx][by], part, {px, py, zBase + wallHeight},
                                     INFINITY, panelAngStiffness, breakForceMinor);
                }
            }

            for (int bx = 0; bx < baysX; bx++)
                for (int by = 0; by < baysY; by++)
                    prevFloorSlabs[bx][by] = curFloorSlabs[bx][by];
        }
    }

    // Skyway bridges connecting the twin towers at floors 5, 10, 15 (3 floors x 12 blocks = 36 blocks)
    const int skyFloors[3] = {5, 10, 15};
    for (int sf = 0; sf < 3; sf++)
    {
        const float zs = skyFloors[sf] * floorPitch + wallHeight + slabThick * 0.5f;
        int prevBridge = -1;
        for (int i = 0; i < 4; i++)
        {
            const float bx = -3.0f + i * 2.0f;
            int bSlab = s->addBody({1.98f, 3.0f, 0.2f}, 1.0f, 0.6f, {bx, 0.0f, zs});
            int bRailL = s->addBody({1.98f, 0.15f, 1.0f}, 0.5f, 0.6f, {bx, -1.4f, zs + 0.6f});
            int bRailR = s->addBody({1.98f, 0.15f, 1.0f}, 0.5f, 0.6f, {bx, +1.4f, zs + 0.6f});
            s->addJointWorld(bSlab, bRailL, {bx, -1.4f, zs + 0.1f}, INFINITY, INFINITY, breakForceMinor);
            s->addJointWorld(bSlab, bRailR, {bx, +1.4f, zs + 0.1f}, INFINITY, INFINITY, breakForceMinor);

            if (i == 0)
            {
                // Connect west end of skyway to Tower A's inner edge slab (at X = -4.0m)
                s->addJointWorld(towerInnerSlabs[0][skyFloors[sf]], bSlab, {-4.0f, 0.0f, zs}, INFINITY, INFINITY, breakForce);
            }

            if (prevBridge >= 0)
                s->addJointWorld(prevBridge, bSlab, {bx - 1.0f, 0.0f, zs}, INFINITY, INFINITY, breakForce);
            prevBridge = bSlab;

            if (i == 3)
            {
                // Connect east end of skyway to Tower B's inner edge slab (at X = +4.0m)
                s->addJointWorld(bSlab, towerInnerSlabs[1][skyFloors[sf]], {+4.0f, 0.0f, zs}, INFINITY, INFINITY, breakForce);
            }
        }
    }

    return {115.0f, 26.0f};
}
