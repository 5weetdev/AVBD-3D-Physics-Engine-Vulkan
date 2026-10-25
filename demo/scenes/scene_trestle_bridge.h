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

// Colossal 120m Railway Trestle (~10,000 blocks)
static SceneView sceneTrestleBridgeGpu(SceneBuilder *s)
{
    s->clear();

    const float breakForceMajor = 200000.0f;
    const float breakForceDeck = 100000.0f;
    const float breakForceMinor = 120000.0f;

    // Canyon floor & cliff abutments
    int ground = s->addBody({10000.0f, 10000.0f, 1.0f}, 0.0f, 0.5f, {0.0f, 0.0f, -0.5f});
    int leftCliff = s->addBody({24.0f, 16.0f, 24.0f}, 0.0f, 0.6f, {-72.0f, 0.0f, 12.0f}); // Left cliff top Z=24
    int rightCliff = s->addBody({24.0f, 16.0f, 24.0f}, 0.0f, 0.6f, {+72.0f, 0.0f, 12.0f}); // Right cliff top Z=24

    const int numBents = 25;
    const int numTiers = 6;
    const float bentSpacing = 4.8f;
    const float tierHeight = 4.0f;

    int bentTopCaps[25];

    // 1. Tower Bents (25 bents x 6 tiers x 10 beams = 1,500 frame beams)
    for (int b = 0; b < numBents; b++)
    {
        const float x = -57.6f + b * bentSpacing;
        int prevCap = -1;

        for (int tr = 0; tr < numTiers; tr++)
        {
            const float zBottom = tr * tierHeight;
            const float zCenter = zBottom + tierHeight * 0.5f;
            const float zTop = zBottom + tierHeight;

            // Battered posts along Y (spread wider at bottom, narrower at top)
            const float batter = 1.0f + (numTiers - 1 - tr) * 0.25f;
            const float yL = -2.0f * batter;
            const float yR = +2.0f * batter;

            int postL = s->addBody({0.4f, 0.4f, tierHeight * 0.95f}, 1.2f, 0.6f, {x, yL, zCenter});
            int postR = s->addBody({0.4f, 0.4f, tierHeight * 0.95f}, 1.2f, 0.6f, {x, yR, zCenter});

            // Anchor bottom tier posts to bedrock ground
            if (tr == 0)
            {
                s->addJointWorld(ground, postL, {x, yL, 0.10f}, INFINITY, INFINITY, breakForceMajor);
                s->addJointWorld(ground, postR, {x, yR, 0.10f}, INFINITY, INFINITY, breakForceMajor);
            }

            // Horizontal sill / cap beam
            int cap = s->addBody({0.45f, (yR - yL) + 1.2f, 0.35f}, 1.2f, 0.6f, {x, 0.0f, zTop});

            s->addJointWorld(postL, cap, {x, yL, zTop - 0.175f}, INFINITY, INFINITY, breakForceMajor);
            s->addJointWorld(postR, cap, {x, yR, zTop - 0.175f}, INFINITY, INFINITY, breakForceMajor);

            if (prevCap >= 0)
            {
                s->addJointWorld(prevCap, postL, {x, yL, zBottom + 0.175f}, INFINITY, INFINITY, breakForceMajor);
                s->addJointWorld(prevCap, postR, {x, yR, zBottom + 0.175f}, INFINITY, INFINITY, breakForceMajor);
            }

            // Transverse diagonal X-braces
            const float spanY = yR - yL;
            const float diagAngle = atan2f(tierHeight, spanY);
            const quat qBr1 = {sinf(diagAngle * 0.5f), 0.0f, 0.0f, cosf(diagAngle * 0.5f)};
            const quat qBr2 = {sinf(-diagAngle * 0.5f), 0.0f, 0.0f, cosf(-diagAngle * 0.5f)};
            const float braceLen = sqrtf(spanY * spanY + tierHeight * tierHeight);

            int br1 = s->addBody({0.2f, braceLen * 0.96f, 0.2f}, 0.8f, 0.6f, {x - 0.12f, 0.0f, zCenter}, {0, 0, 0}, {0, 0, 0}, qBr1);
            int br2 = s->addBody({0.2f, braceLen * 0.96f, 0.2f}, 0.8f, 0.6f, {x + 0.12f, 0.0f, zCenter}, {0, 0, 0}, {0, 0, 0}, qBr2);

            s->addJointWorld(postL, br1, {x - 0.12f, yL, zBottom + 0.10f}, INFINITY, 0.0f, breakForceMinor);
            s->addJointWorld(postR, br1, {x - 0.12f, yR, zTop - 0.10f}, INFINITY, 0.0f, breakForceMinor);
            s->addJointWorld(postL, br2, {x + 0.12f, yL, zTop - 0.10f}, INFINITY, 0.0f, breakForceMinor);
            s->addJointWorld(postR, br2, {x + 0.12f, yR, zBottom + 0.10f}, INFINITY, 0.0f, breakForceMinor);

            // Crossing braces must not collide with each other or cap beams
            s->ignoreCollision(br1, br2);
            s->ignoreCollision(cap, br1);
            s->ignoreCollision(cap, br2);
            if (prevCap >= 0)
            {
                s->ignoreCollision(prevCap, br1);
                s->ignoreCollision(prevCap, br2);
            }

            prevCap = cap;
            if (tr == numTiers - 1)
                bentTopCaps[b] = cap;
        }
    }

    // 2. Longitudinal girts and X-braces across 24 bays (24 x 6 x 8 = 1,152 beams)
    for (int b = 0; b < numBents - 1; b++)
    {
        const float x0 = -57.6f + b * bentSpacing;
        const float x1 = x0 + bentSpacing;
        const float cx = (x0 + x1) * 0.5f;

        for (int tr = 0; tr < numTiers; tr++)
        {
            const float zCenter = tr * tierHeight + tierHeight * 0.5f;
            const float zTop = (tr + 1) * tierHeight;

            // Longitudinal girt beams (left and right)
            int girtL = s->addBody({bentSpacing * 0.95f, 0.25f, 0.3f}, 1.0f, 0.6f, {cx, -2.0f, zTop});
            int girtR = s->addBody({bentSpacing * 0.95f, 0.25f, 0.3f}, 1.0f, 0.6f, {cx, +2.0f, zTop});

            // Longitudinal X-bracing (offset by 12cm in Y so diagonals do not collide)
            const float diagXAngle = atan2f(tierHeight, bentSpacing);
            const quat qX1 = {0.0f, sinf(diagXAngle * 0.5f), 0.0f, cosf(diagXAngle * 0.5f)};
            const quat qX2 = {0.0f, sinf(-diagXAngle * 0.5f), 0.0f, cosf(-diagXAngle * 0.5f)};
            const float xBraceLen = sqrtf(bentSpacing * bentSpacing + tierHeight * tierHeight) * 0.96f;

            int xbL1 = s->addBody({xBraceLen, 0.2f, 0.2f}, 0.8f, 0.6f, {cx, -2.12f, zCenter}, {0, 0, 0}, {0, 0, 0}, qX1);
            int xbL2 = s->addBody({xBraceLen, 0.2f, 0.2f}, 0.8f, 0.6f, {cx, -1.88f, zCenter}, {0, 0, 0}, {0, 0, 0}, qX2);
            int xbR1 = s->addBody({xBraceLen, 0.2f, 0.2f}, 0.8f, 0.6f, {cx, +1.88f, zCenter}, {0, 0, 0}, {0, 0, 0}, qX1);
            int xbR2 = s->addBody({xBraceLen, 0.2f, 0.2f}, 0.8f, 0.6f, {cx, +2.12f, zCenter}, {0, 0, 0}, {0, 0, 0}, qX2);

            // Crossing longitudinal braces must not collide with each other
            s->ignoreCollision(xbL1, xbL2);
            s->ignoreCollision(xbR1, xbR2);
        }
    }

    // 3. Sway tie ropes made of chains of blocks between adjacent bents (24 bays x 4 ropes x 8 blocks = 768 blocks)
    for (int b = 0; b < numBents - 1; b++)
    {
        const float x0 = -57.6f + b * bentSpacing;
        const float x1 = x0 + bentSpacing;

        for (int side = 0; side < 2; side++)
        {
            const float y = (side == 0) ? -2.4f : +2.4f;
            int prevNode = bentTopCaps[b];

            for (int r = 1; r <= 8; r++)
            {
                const float t = r / 9.0f;
                const float rx = x0 * (1.0f - t) + x1 * t;
                const float rz = 24.0f - 3.5f * sinf(t * 3.14159f);
                int rBlock = s->addBody({0.5f, 0.15f, 0.15f}, 0.5f, 0.5f, {rx, y, rz});
                vec3 mid = (s->bodyPos(prevNode) + s->bodyPos(rBlock)) * 0.5f;
                s->addJointWorld(prevNode, rBlock, mid, INFINITY, 0.0f, breakForceMinor);
                prevNode = rBlock;
            }
            vec3 midEnd = (s->bodyPos(prevNode) + s->bodyPos(bentTopCaps[b + 1])) * 0.5f;
            s->addJointWorld(prevNode, bentTopCaps[b + 1], midEnd, INFINITY, 0.0f, breakForceMinor);
        }
    }

    // 4. Longitudinal stringers (24 bays x 8 stringers = 192 long beams)
    int stringers[26][8];

    for (int b = 0; b < numBents - 1; b++)
    {
        const float x0 = -57.6f + b * bentSpacing;
        const float x1 = x0 + bentSpacing;
        const float cx = (x0 + x1) * 0.5f;
        for (int st = 0; st < 8; st++)
        {
            const float sy = -2.1f + st * 0.6f;
            int str = s->addBody({bentSpacing * 0.98f, 0.35f, 0.45f}, 1.2f, 0.6f, {cx, sy, 24.4f});
            stringers[b + 1][st] = str;
            s->addJointWorld(bentTopCaps[b], str, {x0, sy, 24.175f},
                             INFINITY, INFINITY, breakForceMajor);
            s->addJointWorld(bentTopCaps[b + 1], str, {x1, sy, 24.175f},
                             INFINITY, INFINITY, breakForceMajor);
        }
    }

    // Approach stringers connecting cliffs to end bents (16 long beams)
    for (int st = 0; st < 8; st++)
    {
        const float sy = -2.1f + st * 0.6f;
        int strL = s->addBody({2.35f, 0.35f, 0.45f}, 1.2f, 0.6f, {-58.8f, sy, 24.4f});
        stringers[0][st] = strL;
        s->addJointWorld(leftCliff, strL, {-60.0f, sy, 24.175f}, INFINITY, INFINITY, breakForceMajor);
        s->addJointWorld(bentTopCaps[0], strL, {-57.6f, sy, 24.175f}, INFINITY, INFINITY, breakForceMajor);

        int strR = s->addBody({2.35f, 0.35f, 0.45f}, 1.2f, 0.6f, {+58.8f, sy, 24.4f});
        stringers[25][st] = strR;
        s->addJointWorld(bentTopCaps[numBents - 1], strR, {+57.6f, sy, 24.175f}, INFINITY, INFINITY, breakForceMajor);
        s->addJointWorld(rightCliff, strR, {+60.0f, sy, 24.175f}, INFINITY, INFINITY, breakForceMajor);
    }

    // 5. Dense railroad cross-ties & deck structure (598 stations x 13 blocks = 7,774 blocks)
    const int numTies = 598;
    int prevTie = -1;

    for (int i = 0; i < numTies; i++)
    {
        const float px = -59.7f + i * 0.20f;

        // Railroad cross-tie beam
        int tie = s->addBody({0.18f, 4.4f, 0.22f}, 1.0f, 0.6f, {px, 0.0f, 24.75f});
        if (prevTie >= 0)
            s->addJointWorld(prevTie, tie, {px - 0.10f, 0.0f, 24.75f}, INFINITY, INFINITY, breakForceDeck);
        prevTie = tie;

        if (i == 0)
            s->addJointWorld(leftCliff, tie, {-59.79f, 0, 24.75f}, INFINITY, INFINITY, breakForceDeck);
        if (i == numTies - 1)
            s->addJointWorld(rightCliff, tie, {+59.79f, 0, 24.75f}, INFINITY, INFINITY, breakForceDeck);

        // Fasten tie to longitudinal stringers below it
        int spanIdx;
        if (px < -57.6f) spanIdx = 0;
        else if (px > +57.6f) spanIdx = 25;
        else spanIdx = 1 + (int)((px - (-57.6f)) / bentSpacing);
        
        // Attach to the outermost stringers (st=0 and st=7)
        s->addJointWorld(stringers[spanIdx][0], tie, {px, -2.1f, 24.6f}, INFINITY, INFINITY, breakForceDeck);
        s->addJointWorld(stringers[spanIdx][7], tie, {px, +2.1f, 24.6f}, INFINITY, INFINITY, breakForceDeck);

        // Walkway decking planks (3 left, 3 right = 6 planks)
        for (int p = 0; p < 3; p++)
        {
            const float pyL = -2.0f + p * 0.35f;
            const float pyR = +1.3f + p * 0.35f;
            int plankL = s->addBody({0.19f, 0.32f, 0.08f}, 0.4f, 0.6f, {px, pyL, 24.9f});
            int plankR = s->addBody({0.19f, 0.32f, 0.08f}, 0.4f, 0.6f, {px, pyR, 24.9f});
            s->addJointWorld(tie, plankL, {px, pyL, 24.86f}, INFINITY, INFINITY, breakForceMinor);
            s->addJointWorld(tie, plankR, {px, pyR, 24.86f}, INFINITY, INFINITY, breakForceMinor);
        }

        // Railing upright posts (2 posts)
        int postL = s->addBody({0.08f, 0.08f, 0.9f}, 0.3f, 0.5f, {px, -2.15f, 25.35f});
        int postR = s->addBody({0.08f, 0.08f, 0.9f}, 0.3f, 0.5f, {px, +2.15f, 25.35f});
        s->addJointWorld(tie, postL, {px, -2.15f, 24.86f}, INFINITY, INFINITY, breakForceMinor);
        s->addJointWorld(tie, postR, {px, +2.15f, 24.86f}, INFINITY, INFINITY, breakForceMinor);

        // Steel running rails & guard rails (4 rail blocks)
        s->addBody({0.19f, 0.12f, 0.14f}, 0.6f, 0.4f, {px, -0.75f, 24.93f});
        s->addBody({0.19f, 0.12f, 0.14f}, 0.6f, 0.4f, {px, +0.75f, 24.93f});
        s->addBody({0.19f, 0.10f, 0.12f}, 0.5f, 0.4f, {px, -0.45f, 24.92f});
        s->addBody({0.19f, 0.10f, 0.12f}, 0.5f, 0.4f, {px, +0.45f, 24.92f});
    }

    return {140.0f, 16.0f};
}
