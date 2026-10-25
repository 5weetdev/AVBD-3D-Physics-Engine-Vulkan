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

// Grand 180m Suspension Bridge (~10,000 blocks)
static SceneView sceneSuspensionBridgeGpu(SceneBuilder *s)
{
    s->clear();

    const float cableBreakForce = 250000.0f;
    const float hangerBreakForce = 60000.0f;
    const float deckBreakForce = 25000.0f;
    const float minorBreakForce = 10000.0f;

    // 1. Canyon water, shore cable anchorages and road abutments
    int waterGround = s->addBody({10000.0f, 10000.0f, 1.0f}, 0.0f, 0.5f, {0.0f, 0.0f, -0.5f});
    // Left & right cable anchor blocks (placed outside the roadway span, meeting cable ends at X = +-90)
    int leftAnchorL = s->addBody({8.0f, 2.0f, 12.0f}, 0.0f, 0.6f, {-94.0f, -3.6f, 6.0f});
    int leftAnchorR = s->addBody({8.0f, 2.0f, 12.0f}, 0.0f, 0.6f, {-94.0f, +3.6f, 6.0f});
    int rightAnchorL = s->addBody({8.0f, 2.0f, 12.0f}, 0.0f, 0.6f, {+94.0f, -3.6f, 6.0f});
    int rightAnchorR = s->addBody({8.0f, 2.0f, 12.0f}, 0.0f, 0.6f, {+94.0f, +3.6f, 6.0f});

    // Left & right roadway approach abutments (sitting under the deck at Z <= 10.5m)
    int leftRoadAbutment = s->addBody({8.0f, 7.6f, 10.5f}, 0.0f, 0.6f, {-94.0f, 0.0f, 5.25f});
    int rightRoadAbutment = s->addBody({8.0f, 7.6f, 10.5f}, 0.0f, 0.6f, {+94.0f, 0.0f, 5.25f});

    // Towers at X = -45m and X = +45m rising to Z = 45m
    const float towerX[2] = {-45.0f, +45.0f};
    int towerSaddles[2][2]; // [tower][side 0=South, 1=North]
    int prevLeg1[2][2], prevLeg2[2][2];

    for (int t = 0; t < 2; t++)
    {
        const float tx = towerX[t];
        for (int tr = 0; tr < 15; tr++)
        {
            const float zc = tr * 3.0f + 1.5f;
            const float zTop = (tr + 1) * 3.0f;

            for (int side = 0; side < 2; side++)
            {
                const float ty = (side == 0) ? -3.6f : +3.6f;
                int leg1 = s->addBody({0.8f, 0.8f, 2.95f}, 1.5f, 0.6f, {tx - 0.82f, ty, zc});
                int leg2 = s->addBody({0.8f, 0.8f, 2.95f}, 1.5f, 0.6f, {tx + 0.82f, ty, zc});

                // Joint bricks together to form a solid wider tower
                s->addJointWorld(leg1, leg2, {tx, ty, zc}, INFINITY, INFINITY, 500000.0f);

                // Anchor tower base to riverbed bedrock
                if (tr == 0)
                {
                    s->addJointWorld(waterGround, leg1, {tx - 0.82f, ty, 0.025f}, INFINITY, INFINITY, 500000.0f);
                    s->addJointWorld(waterGround, leg2, {tx + 0.82f, ty, 0.025f}, INFINITY, INFINITY, 500000.0f);
                }
                else
                {
                    s->addJointWorld(prevLeg1[t][side], leg1, {tx - 0.82f, ty, zc - 1.5f}, INFINITY, INFINITY, 500000.0f);
                    s->addJointWorld(prevLeg2[t][side], leg2, {tx + 0.82f, ty, zc - 1.5f}, INFINITY, INFINITY, 500000.0f);
                }

                prevLeg1[t][side] = leg1;
                prevLeg2[t][side] = leg2;

                if (tr == 14)
                {
                    towerSaddles[t][side] = s->addBody({1.6f, 1.2f, 0.8f}, 2.0f, 0.6f, {tx, ty, 45.4f});
                    s->addJointWorld(leg1, towerSaddles[t][side], {tx - 0.82f, ty, 44.5f}, INFINITY, INFINITY, 500000.0f);
                    s->addJointWorld(leg2, towerSaddles[t][side], {tx + 0.82f, ty, 44.5f}, INFINITY, INFINITY, 500000.0f);
                }
            }

            // Cross struts between tower legs
            int strut = s->addBody({0.5f, 6.4f, 0.4f}, 1.0f, 0.6f, {tx, 0.0f, zTop});
            s->addJointWorld(prevLeg1[t][0], strut, {tx - 0.5f, -3.6f, zTop}, INFINITY, INFINITY, 500000.0f);
            s->addJointWorld(prevLeg1[t][1], strut, {tx - 0.5f, +3.6f, zTop}, INFINITY, INFINITY, 500000.0f);
            s->addJointWorld(prevLeg2[t][0], strut, {tx + 0.5f, -3.6f, zTop}, INFINITY, INFINITY, 500000.0f);
            s->addJointWorld(prevLeg2[t][1], strut, {tx + 0.5f, +3.6f, zTop}, INFINITY, INFINITY, 500000.0f);
        }
    }

    // Main Suspension Cables (2 cables x 180 blocks = 360 blocks)
    // Pre-compute node positions along the catenary, then place rotated blocks.
    auto cableZ = [](float x) -> float {
        if (fabsf(x) <= 45.0f)
        {
            const float t = x / 45.0f;
            return 14.0f + 31.4f * (t * t);
        }
        else
        {
            const float d = (fabsf(x) - 45.0f) / 45.0f;
            return 11.0f + 34.4f * (1.0f - d) * (1.0f - d);
        }
    };

    int cableNodes[2][181];
    for (int side = 0; side < 2; side++)
    {
        const float cy = (side == 0) ? -3.6f : +3.6f;
        int prevNode = -1;

        for (int i = 0; i <= 180; i++)
        {
            const float x = -90.0f + i * 1.0f;
            const float z = cableZ(x);

            // Compute tangent angle via finite differences and rotate block to follow curve
            const float zPrev = cableZ(x - 1.0f);
            const float zNext = cableZ(x + 1.0f);
            const float dz = (i == 0) ? (zNext - z) : (i == 180) ? (z - zPrev) : (zNext - zPrev) * 0.5f;
            const float dx = 1.0f;
            const float theta = atan2f(-dz, dx); // Invert dz to fix inverted block orientation
            const quat rot = {0.0f, sinf(theta * 0.5f), 0.0f, cosf(theta * 0.5f)};

            const float segLen = sqrtf(dx * dx + dz * dz);
            cableNodes[side][i] = s->addBody({segLen * 0.7f, 0.25f, 0.25f}, 0.6f, 0.5f, {x, cy, z},
                                             {0, 0, 0}, {0, 0, 0}, rot);

            if (prevNode >= 0)
            {
                s->ignoreCollision(prevNode, cableNodes[side][i]);
                vec3 mid = (s->bodyPos(prevNode) + s->bodyPos(cableNodes[side][i])) * 0.5f;
                s->addJointWorld(prevNode, cableNodes[side][i], mid,
                                 INFINITY, 0.0f, cableBreakForce);
            }

            // Connect cables to static shore cable anchorages
            if (i == 0)
            {
                int anc = (side == 0) ? leftAnchorL : leftAnchorR;
                s->addJointWorld(anc, cableNodes[side][0], {-90.0f, cy, 11.0f}, INFINITY, INFINITY, cableBreakForce * 2.0f);
            }
            if (i == 180)
            {
                int anc = (side == 0) ? rightAnchorL : rightAnchorR;
                s->addJointWorld(anc, cableNodes[side][180], {+90.0f, cy, 11.0f}, INFINITY, INFINITY, cableBreakForce * 2.0f);
            }

            // Connect cables to tower saddles
            if (fabsf(x - (-45.0f)) < 0.6f)
                s->addJointWorld(cableNodes[side][i], towerSaddles[0][side], {-45.0f, cy, 45.4f},
                                 INFINITY, INFINITY, cableBreakForce * 2.0f);
            if (fabsf(x - (+45.0f)) < 0.6f)
                s->addJointWorld(cableNodes[side][i], towerSaddles[1][side], {+45.0f, cy, 45.4f},
                                 INFINITY, INFINITY, cableBreakForce * 2.0f);

            prevNode = cableNodes[side][i];
        }
    }

    // Stiffening Deck Truss & Roadway across 144 bays
    const int numBays = 144;
    int deckFloorBeams[144];
    int prevBottomChordL = -1, prevBottomChordR = -1;
    int prevTopChordL = -1, prevTopChordR = -1;

    for (int b = 0; b < numBays; b++)
    {
        const float bx = -89.375f + b * 1.25f;

        // Bottom and top longitudinal chords (4 chords)
        int bChordL = s->addBody({1.25f, 0.35f, 0.35f}, 1.2f, 0.6f, {bx, -3.4f, 10.8f});
        int bChordR = s->addBody({1.25f, 0.35f, 0.35f}, 1.2f, 0.6f, {bx, +3.4f, 10.8f});
        int tChordL = s->addBody({1.25f, 0.35f, 0.35f}, 1.2f, 0.6f, {bx, -3.4f, 13.0f});
        int tChordR = s->addBody({1.25f, 0.35f, 0.35f}, 1.2f, 0.6f, {bx, +3.4f, 13.0f});

        if (prevBottomChordL >= 0)
        {
            s->addJointWorld(prevBottomChordL, bChordL, {bx - 0.625f, -3.4f, 10.8f}, INFINITY, INFINITY, deckBreakForce);
            s->addJointWorld(prevBottomChordR, bChordR, {bx - 0.625f, +3.4f, 10.8f}, INFINITY, INFINITY, deckBreakForce);
            s->addJointWorld(prevTopChordL, tChordL, {bx - 0.625f, -3.4f, 13.0f}, INFINITY, INFINITY, deckBreakForce);
            s->addJointWorld(prevTopChordR, tChordR, {bx - 0.625f, +3.4f, 13.0f}, INFINITY, INFINITY, deckBreakForce);
        }
        prevBottomChordL = bChordL;
        prevBottomChordR = bChordR;
        prevTopChordL = tChordL;
        prevTopChordR = tChordR;

        // Transverse floor beam
        deckFloorBeams[b] = s->addBody({0.35f, 7.6f, 0.35f}, 1.2f, 0.6f, {bx, 0.0f, 11.15f});
        s->addJointWorld(bChordL, deckFloorBeams[b], {bx, -3.4f, 10.975f}, INFINITY, INFINITY, deckBreakForce);
        s->addJointWorld(bChordR, deckFloorBeams[b], {bx, +3.4f, 10.975f}, INFINITY, INFINITY, deckBreakForce);

        // Connect deck ends to static road approach abutments
        if (b == 0)
        {
            s->addJointWorld(leftRoadAbutment, bChordL, {-90.0f, -3.4f, 10.8f}, INFINITY, INFINITY, deckBreakForce);
            s->addJointWorld(leftRoadAbutment, bChordR, {-90.0f, +3.4f, 10.8f}, INFINITY, INFINITY, deckBreakForce);
        }
        if (b == numBays - 1)
        {
            s->addJointWorld(rightRoadAbutment, bChordL, {+90.0f, -3.4f, 10.8f}, INFINITY, INFINITY, deckBreakForce);
            s->addJointWorld(rightRoadAbutment, bChordR, {+90.0f, +3.4f, 10.8f}, INFINITY, INFINITY, deckBreakForce);
        }

        // Vertical truss web posts (sitting flush on floor beam, under top chord)
        int postTL = s->addBody({0.2f, 0.2f, 1.50f}, 0.8f, 0.6f, {bx, -3.4f, 12.075f});
        int postTR = s->addBody({0.2f, 0.2f, 1.50f}, 0.8f, 0.6f, {bx, +3.4f, 12.075f});
        s->addJointWorld(deckFloorBeams[b], postTL, {bx, -3.4f, 11.325f}, INFINITY, INFINITY, deckBreakForce);
        s->addJointWorld(tChordL, postTL, {bx, -3.4f, 12.825f}, INFINITY, INFINITY, deckBreakForce);
        s->addJointWorld(deckFloorBeams[b], postTR, {bx, +3.4f, 11.325f}, INFINITY, INFINITY, deckBreakForce);
        s->addJointWorld(tChordR, postTR, {bx, +3.4f, 12.825f}, INFINITY, INFINITY, deckBreakForce);

        s->ignoreCollision(bChordL, postTL);
        s->ignoreCollision(bChordR, postTR);
        s->ignoreCollision(deckFloorBeams[b], postTL);
        s->ignoreCollision(deckFloorBeams[b], postTR);

        // Roadway deck surface planks (12 planks per bay)
        for (int p = 0; p < 12; p++)
        {
            const float py = -3.0f + p * 0.54f;
            int plank = s->addBody({1.22f, 0.52f, 0.12f}, 0.5f, 0.6f, {bx, py, 11.385f});
            s->addJointWorld(deckFloorBeams[b], plank, {bx, py, 11.325f}, INFINITY, INFINITY, minorBreakForce);
        }

        // Sidewalk curbs and railing barriers (4 blocks per bay)
        s->addBody({1.22f, 0.2f, 0.4f}, 0.4f, 0.6f, {bx, -3.3f, 11.645f});
        s->addBody({1.22f, 0.2f, 0.4f}, 0.4f, 0.6f, {bx, +3.3f, 11.645f});
        s->addBody({1.22f, 0.1f, 0.8f}, 0.3f, 0.5f, {bx, -3.5f, 13.4f});
        s->addBody({1.22f, 0.1f, 0.8f}, 0.3f, 0.5f, {bx, +3.5f, 13.4f});
    }

    // Vertical Hanger Suspenders made of chains of blocks
    for (int side = 0; side < 2; side++)
    {
        const float cy = (side == 0) ? -3.6f : +3.6f;

        for (int b = 2; b < numBays - 2; b++)
        {
            const float bx = -89.375f + b * 1.25f;
            if (fabsf(bx - (-45.0f)) < 1.6f || fabsf(bx - (+45.0f)) < 1.6f)
                continue; // Skip right at tower locations

            const int cableIdx = (int)fminf(fmaxf(roundf(bx + 90.0f), 0.0f), 180.0f);
            vec3 pCable = s->bodyPos(cableNodes[side][cableIdx]);
            vec3 pDeck = vec3{bx, cy, 11.325f};
            const float hTotal = pCable.z - pDeck.z;

            if (hTotal > 1.5f)
            {
                const int links = (int)fmaxf(1.0f, roundf(hTotal / 1.5f));
                int prevLink = cableNodes[side][cableIdx];

                for (int l = 1; l <= links; l++)
                {
                    const float t = (l - 0.5f) / (float)links;
                    vec3 linkPos = pCable * (1.0f - t) + pDeck * t;
                    const float hSeg = hTotal / (float)links;
                    int link = s->addBody({0.16f, 0.16f, hSeg * 0.7f}, 0.4f, 0.5f, linkPos);

                    vec3 midJoint = pCable * (1.0f - (l - 1.0f) / (float)links) + pDeck * ((l - 1.0f) / (float)links);
                    if (l == 1)
                        midJoint = pCable;

                    s->ignoreCollision(prevLink, link);
                    s->addJointWorld(prevLink, link, midJoint, INFINITY, 0.0f, hangerBreakForce);
                    prevLink = link;
                }
                s->addJointWorld(prevLink, deckFloorBeams[b], pDeck, INFINITY, 0.0f, hangerBreakForce);
            }
        }
    }

    // Traffic lane dividers and safety barrier blocks
    for (int b = 0; b < numBays; b++)
    {
        const float bx = -89.375f + b * 1.25f;
        for (int i = 0; i < 18; i++)
        {
            const float iy = -2.6f + i * 0.31f;
            s->addBody({0.25f, 0.15f, 0.2f}, 0.3f, 0.5f, {bx, iy, 11.4f});
        }
    }

    return {160.0f, 20.0f};
}
