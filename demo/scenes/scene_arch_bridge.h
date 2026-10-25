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

// Monumental 132m Triple Masonry Arch Bridge (Roman style, ~1800 blocks)
static SceneView sceneArchBridgeGpu(SceneBuilder *s)
{
    s->clear();

    const float archBreakForce = 500000.0f;
    const float deckBreakForce = 500000.0f;
    const float minorBreakForce = 150000.0f;
    const float PI_F = 3.14159265359f;

    // Ground and river bed
    int ground = s->addBody({10000.0f, 10000.0f, 1.0f}, 0.0f, 0.8f, {0.0f, 0.0f, -0.5f});

    // Monumental static abutments to take the outward thrust
    int abutment1 = s->addBody({16.0f, 8.0f, 31.0f}, 0.0f, 0.8f, {-74.0f, 0.0f, 15.5f});
    int abutment2 = s->addBody({16.0f, 8.0f, 31.0f}, 0.0f, 0.8f, {74.0f, 0.0f, 15.5f});

    // Piers
    int pierTopLeft[2][4];
    int pierTopMiddle[2][4];
    int pierTopRight[2][4];
    int upperPierTop[2][4];

    for (int p = 0; p < 2; p++)
    {
        float cx_pier = (p == 0) ? -23.0f : 23.0f;
        int pierBlocks[3][4];
        int prevPierBlocks[3][4];

        for (int z_layer = 0; z_layer < 30; z_layer++)
        {
            float zC = z_layer * 1.0f + 0.5f;
            if (z_layer < 10)
            {
                // Lower pier (water level): width 6m to support arches
                for (int i = 0; i < 3; i++)
                {
                    float x = cx_pier - 2.0f + i * 2.0f;
                    for (int r = 0; r < 4; r++)
                    {
                        float y = -3.0f + r * 2.0f;
                        int b = s->addBody({2.0f, 2.0f, 1.0f}, 2.0f, 0.8f, {x, y, zC});
                        
                        if (z_layer > 0)
                        {
                            s->addJointWorld(prevPierBlocks[i][r], b, {x, y, zC - 0.5f}, INFINITY, INFINITY, archBreakForce);
                        }
                        if (r > 0)
                        {
                            s->addJointWorld(pierBlocks[i][r - 1], b, {x, y - 1.0f, zC}, INFINITY, INFINITY, archBreakForce);
                            s->ignoreCollision(pierBlocks[i][r - 1], b);
                        }
                        if (i > 0)
                        {
                            s->addJointWorld(pierBlocks[i - 1][r], b, {x - 1.0f, y, zC}, INFINITY, INFINITY, archBreakForce);
                            s->ignoreCollision(pierBlocks[i - 1][r], b);
                        }

                        pierBlocks[i][r] = b;
                        if (z_layer == 9)
                        {
                            if (i == 0) pierTopLeft[p][r] = b;
                            if (i == 1) pierTopMiddle[p][r] = b;
                            if (i == 2) pierTopRight[p][r] = b;
                        }
                    }
                }
                for (int i = 0; i < 3; i++)
                    for (int r = 0; r < 4; r++)
                        prevPierBlocks[i][r] = pierBlocks[i][r];
            }
            else
            {
                // Upper pier (spandrel level): narrower (1.5m) to stay clear of the arch rings
                for (int r = 0; r < 4; r++)
                {
                    float y = -3.0f + r * 2.0f;
                    int b = s->addBody({1.5f, 2.0f, 1.0f}, 2.0f, 0.8f, {cx_pier, y, zC});
                    
                    if (z_layer == 10)
                    {
                        s->addJointWorld(pierTopMiddle[p][r], b, {cx_pier, y, 10.0f}, INFINITY, INFINITY, archBreakForce);
                    }
                    else
                    {
                        s->addJointWorld(prevPierBlocks[1][r], b, {cx_pier, y, zC - 0.5f}, INFINITY, INFINITY, archBreakForce);
                    }
                    if (r > 0)
                    {
                        s->addJointWorld(pierBlocks[1][r - 1], b, {cx_pier, y - 1.0f, zC}, INFINITY, INFINITY, archBreakForce);
                        s->ignoreCollision(pierBlocks[1][r - 1], b);
                    }
                    
                    pierBlocks[1][r] = b;
                    if (z_layer == 29)
                    {
                        upperPierTop[p][r] = b;
                    }
                }
                for (int r = 0; r < 4; r++)
                    prevPierBlocks[1][r] = pierBlocks[1][r];
            }
        }
    }

    // Arches (3 spans, perfectly matched voussoir blocks rotated along the tangent)
    const int N = 40;
    const float R = 20.8f;
    const float T = 1.6f;
    const float L = PI_F * R / (float)N;

    int archBlocks[3][4][40];

    for (int k = 0; k < 3; k++)
    {
        float cx = -46.0f + k * 46.0f;
        int prevRingBlock[4];
        for (int r = 0; r < 4; r++) prevRingBlock[r] = -1;

        for (int i = 0; i < N; i++)
        {
            float theta = (i + 0.5f) * (PI_F / (float)N);
            float phi = theta - PI_F / 2.0f;
            quat rot = {0.0f, sinf(phi * 0.5f), 0.0f, cosf(phi * 0.5f)};

            for (int r = 0; r < 4; r++)
            {
                float y = -3.0f + r * 2.0f;
                // Arch rests at springing line z=10.
                vec3 pos = {cx - R * cosf(theta), y, 10.0f + R * sinf(theta)};
                int b = s->addBody({L, 2.0f, T}, 2.0f, 0.8f, pos, {0, 0, 0}, {0, 0, 0}, rot);
                archBlocks[k][r][i] = b;

                if (i > 0)
                {
                    float faceTheta = i * (PI_F / (float)N);
                    vec3 facePos = {cx - R * cosf(faceTheta), y, 10.0f + R * sinf(faceTheta)};
                    s->addJointWorld(prevRingBlock[r], b, facePos, INFINITY, INFINITY, archBreakForce);
                    s->ignoreCollision(prevRingBlock[r], b);
                }
                else
                {
                    vec3 facePos = {cx - R, y, 10.0f};
                    int support = (k == 0) ? abutment1 : pierTopRight[k - 1][r];
                    s->addJointWorld(support, b, facePos, INFINITY, INFINITY, archBreakForce);
                    s->ignoreCollision(support, b);
                }

                if (r > 0)
                {
                    vec3 latPos = {pos.x, y - 1.0f, pos.z};
                    s->addJointWorld(archBlocks[k][r - 1][i], b, latPos, INFINITY, INFINITY, archBreakForce);
                    s->ignoreCollision(archBlocks[k][r - 1][i], b);
                }

                prevRingBlock[r] = b;
            }
        }

        // Connect the rightmost block to the right abutment/pier
        for (int r = 0; r < 4; r++)
        {
            vec3 facePos = {cx + R, -3.0f + r * 2.0f, 10.0f};
            int support = (k == 2) ? abutment2 : pierTopLeft[k][r];
            s->addJointWorld(support, prevRingBlock[r], facePos, INFINITY, INFINITY, archBreakForce);
            s->ignoreCollision(support, prevRingBlock[r]);
        }
    }

    // Deck
    int deckBlocks[4][66];
    int prevDeck[4] = {-1, -1, -1, -1};

    for (int i = 0; i < 66; i++)
    {
        float x = -65.0f + i * 2.0f;
        for (int r = 0; r < 4; r++)
        {
            float y = -3.0f + r * 2.0f;
            int b = s->addBody({2.0f, 2.0f, 1.0f}, 2.0f, 0.8f, {x, y, 30.5f});
            deckBlocks[r][i] = b;

            if (i > 0)
            {
                s->addJointWorld(prevDeck[r], b, {x - 1.0f, y, 30.5f}, INFINITY, INFINITY, deckBreakForce);
                s->ignoreCollision(prevDeck[r], b);
            }
            else
            {
                s->addJointWorld(abutment1, b, {x - 1.0f, y, 30.5f}, INFINITY, INFINITY, deckBreakForce);
            }

            if (i == 65)
            {
                s->addJointWorld(abutment2, b, {x + 1.0f, y, 30.5f}, INFINITY, INFINITY, deckBreakForce);
            }

            if (r > 0)
            {
                s->addJointWorld(deckBlocks[r - 1][i], b, {x, y - 1.0f, 30.5f}, INFINITY, INFINITY, deckBreakForce);
                s->ignoreCollision(deckBlocks[r - 1][i], b);
            }
            
            // Connect to upper piers
            if (fabsf(x - (-23.0f)) < 0.1f)
            {
                s->addJointWorld(upperPierTop[0][r], b, {x, y, 30.0f}, INFINITY, INFINITY, deckBreakForce);
            }
            if (fabsf(x - 23.0f) < 0.1f)
            {
                s->addJointWorld(upperPierTop[1][r], b, {x, y, 30.0f}, INFINITY, INFINITY, deckBreakForce);
            }

            prevDeck[r] = b;
        }
    }

    // Spandrel columns to support deck over the arches
    int spandrelCols[3][4][40];
    for (int k = 0; k < 3; k++)
        for (int r = 0; r < 4; r++)
            for (int i = 0; i < 40; i++)
                spandrelCols[k][r][i] = -1;

    for (int k = 0; k < 3; k++)
    {
        float cx = -46.0f + k * 46.0f;
        for (int i = 0; i < N; i++)
        {
            float theta = (i + 0.5f) * (PI_F / (float)N);
            float x_i = cx - R * cosf(theta);
            // Measure exact height of the arch block's extrados (top face)
            float z_extrados = 10.0f + (R + T * 0.5f) * sinf(theta);

            if (z_extrados < 29.5f)
            {
                float H = 30.0f - z_extrados;
                float colZ = z_extrados + H * 0.5f;

                int i_deck = (int)roundf((x_i + 65.0f) / 2.0f);
                i_deck = i_deck < 0 ? 0 : (i_deck > 65 ? 65 : i_deck);

                int prevColR = -1;
                for (int r = 0; r < 4; r++)
                {
                    float y = -3.0f + r * 2.0f;
                    int col = s->addBody({1.0f, 2.0f, H}, 1.0f, 0.8f, {x_i, y, colZ});
                    spandrelCols[k][r][i] = col;

                    vec3 anchorBottom = {x_i, y, z_extrados};
                    s->addJointWorld(archBlocks[k][r][i], col, anchorBottom, INFINITY, INFINITY, minorBreakForce);

                    vec3 anchorTop = {x_i, y, 30.0f};
                    s->addJointWorld(deckBlocks[r][i_deck], col, anchorTop, INFINITY, INFINITY, minorBreakForce);

                    if (i > 0 && spandrelCols[k][r][i - 1] >= 0)
                    {
                        s->ignoreCollision(spandrelCols[k][r][i - 1], col);
                    }
                    if (r > 0 && prevColR >= 0)
                    {
                        s->addJointWorld(prevColR, col, {x_i, y - 1.0f, colZ}, INFINITY, INFINITY, minorBreakForce);
                        s->ignoreCollision(prevColR, col);
                    }
                    prevColR = col;
                }
            }
        }
    }

    // Parapets
    for (int i = 0; i < 80; i++)
    {
        float x = -79.0f + i * 2.0f;
        int pl = s->addBody({2.0f, 0.5f, 1.0f}, 1.0f, 0.8f, {x, -3.75f, 31.5f});
        int pr = s->addBody({2.0f, 0.5f, 1.0f}, 1.0f, 0.8f, {x, 3.75f, 31.5f});
        
        if (x < -66.0f)
        {
            s->addJointWorld(abutment1, pl, {x, -3.75f, 31.0f}, INFINITY, INFINITY, minorBreakForce);
            s->addJointWorld(abutment1, pr, {x, 3.75f, 31.0f}, INFINITY, INFINITY, minorBreakForce);
        }
        else if (x > 66.0f)
        {
            s->addJointWorld(abutment2, pl, {x, -3.75f, 31.0f}, INFINITY, INFINITY, minorBreakForce);
            s->addJointWorld(abutment2, pr, {x, 3.75f, 31.0f}, INFINITY, INFINITY, minorBreakForce);
        }
        else
        {
            int i_deck = (int)roundf((x + 65.0f) / 2.0f);
            s->addJointWorld(deckBlocks[0][i_deck], pl, {x, -3.75f, 31.0f}, INFINITY, INFINITY, minorBreakForce);
            s->addJointWorld(deckBlocks[3][i_deck], pr, {x, 3.75f, 31.0f}, INFINITY, INFINITY, minorBreakForce);
        }
    }

    return {130.0f, 16.0f};
}
