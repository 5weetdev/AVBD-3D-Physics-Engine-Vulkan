#pragma once

#include "scene_common.h"

#include <vector>

struct TowerBlock
{
    int id;
    vec3 pos;
};

static void addCircularBrickTower(SceneBuilder *s, vec3 origin, float radiusInner, float radiusOuter, int blocksInner,
                                  int blocksOuter, int layers, float brickLength, float brickWidth, float brickHeight)
{
    const float twoPi = 6.283185307179586f;
    const float dThetaIn = twoPi / (float)blocksInner;
    const float dThetaOut = twoPi / (float)blocksOuter;

    for (int k = 0; k < layers; k++)
    {
        const float z = k * brickHeight + brickHeight * 0.5f;
        const float offsetIn = (k % 2) * (dThetaIn * 0.5f);
        const float offsetOut = (k % 2) * (dThetaOut * 0.5f);

        // Inner ring
        for (int i = 0; i < blocksInner; i++)
        {
            const float theta = i * dThetaIn + offsetIn;
            const float px = origin.x + radiusInner * cosf(theta);
            const float py = origin.y + radiusInner * sinf(theta);
            const float psi = theta + 1.5707963267948966f;
            const quat rot = {0.0f, 0.0f, sinf(psi * 0.5f), cosf(psi * 0.5f)};
            s->addBody({brickLength, brickWidth, brickHeight}, 1.0f, 0.5f, {px, py, z}, {0, 0, 0}, {0, 0, 0}, rot);
        }

        // Outer ring
        for (int i = 0; i < blocksOuter; i++)
        {
            const float theta = i * dThetaOut + offsetOut;
            const float px = origin.x + radiusOuter * cosf(theta);
            const float py = origin.y + radiusOuter * sinf(theta);
            const float psi = theta + 1.5707963267948966f;
            const quat rot = {0.0f, 0.0f, sinf(psi * 0.5f), cosf(psi * 0.5f)};
            s->addBody({brickLength, brickWidth, brickHeight}, 1.0f, 0.5f, {px, py, z}, {0, 0, 0}, {0, 0, 0}, rot);
        }
    }
}

// Two circular hollow brick towers (100 blocks/layer x 100 layers = 10,000 blocks each = 20,000 blocks total)
static SceneView sceneRoundTowers20kGpu(SceneBuilder *s)
{
    s->clear();

    const float groundSizeX = 140.0f;
    const float groundSizeY = 80.0f;
    s->addBody({10000.0f, 10000.0f, 1.0f}, 0.0f, 0.5f, {0.0f, 0.0f, -0.5f});

    const float radiusInner = 7.25f;
    const float radiusOuter = 7.77f;
    const int blocksInner = 48;
    const int blocksOuter = 52;
    const int layers = 100;
    const float brickLength = 0.90f;
    const float brickWidth = 0.50f;
    const float brickHeight = 0.50f;

    // Tower A on the left, Tower B on the right
    addCircularBrickTower(s, {-18.0f, 0.0f, 0.0f}, radiusInner, radiusOuter, blocksInner, blocksOuter, layers,
                          brickLength, brickWidth, brickHeight);
    addCircularBrickTower(s, {+18.0f, 0.0f, 0.0f}, radiusInner, radiusOuter, blocksInner, blocksOuter, layers,
                          brickLength, brickWidth, brickHeight);

    return {110.0f, 25.0f};
}

static void addSquareBrickTower(SceneBuilder *s, int ground, vec3 origin, int layers, float brickLength,
                                float brickWidth, float brickHeight)
{
    std::vector<TowerBlock> blocks;
    auto addBlock = [&](vec3 ext, vec3 pos)
    {
        int id = s->addBody(ext, 1.0f, 0.5f, pos);
        blocks.push_back({id, pos});
    };

    // A square double-brick hollow keep with authentic stretcher bond:
    // Outer footprint: 14.0m x 14.0m.
    // Inner hollow courtyard: 12.0m x 12.0m.
    // Wall thickness: 1.0m (outer 0.5m brick + inner 0.5m brick).
    // Exactly 104 bricks per layer (54 outer + 50 inner).
    // Every single brick rests on exactly 2 bricks below it (50% running bond).
    // Alternating corner overlap, zero in-plane overlap, zero gaps.

    const float halfWid = brickWidth * 0.5f;      // 0.25m
    const float outerWallOffset = 7.0f - halfWid; // 6.75m
    const float innerWallOffset = 6.5f - halfWid; // 6.25m

    const vec3 sizeX = {brickLength, brickWidth, brickHeight}; // {1.0, 0.5, 0.5}
    const vec3 sizeY = {brickWidth, brickLength, brickHeight}; // {0.5, 1.0, 0.5}

    for (int k = 0; k < layers; k++)
    {
        const float z = k * brickHeight + brickHeight * 0.5f;

        if (k % 2 == 0)
        {
            // Layer 0 (Even):
            // Outer North & South run full 14m along X (14 bricks each)
            for (int i = 0; i < 14; i++)
            {
                float x = -6.5f + i * 1.0f;
                addBlock(sizeX, {origin.x + x, origin.y + outerWallOffset, z});
                addBlock(sizeX, {origin.x + x, origin.y - outerWallOffset, z});
            }
            // Outer East & West fit between North/South along Y (13 bricks each)
            for (int i = 0; i < 13; i++)
            {
                float y = -6.0f + i * 1.0f;
                addBlock(sizeY, {origin.x + outerWallOffset, origin.y + y, z});
                addBlock(sizeY, {origin.x - outerWallOffset, origin.y + y, z});
            }

            // Inner North & South run full 13m along X (13 bricks each)
            for (int i = 0; i < 13; i++)
            {
                float x = -6.0f + i * 1.0f;
                addBlock(sizeX, {origin.x + x, origin.y + innerWallOffset, z});
                addBlock(sizeX, {origin.x + x, origin.y - innerWallOffset, z});
            }
            // Inner East & West fit between North/South along Y (12 bricks each)
            for (int i = 0; i < 12; i++)
            {
                float y = -5.5f + i * 1.0f;
                addBlock(sizeY, {origin.x + innerWallOffset, origin.y + y, z});
                addBlock(sizeY, {origin.x - innerWallOffset, origin.y + y, z});
            }
        }
        else
        {
            // Layer 1 (Odd):
            // Outer East & West run full 14m along Y (14 bricks each)
            for (int i = 0; i < 14; i++)
            {
                float y = -6.5f + i * 1.0f;
                addBlock(sizeY, {origin.x + outerWallOffset, origin.y + y, z});
                addBlock(sizeY, {origin.x - outerWallOffset, origin.y + y, z});
            }
            // Outer North & South fit between East/West along X (13 bricks each)
            for (int i = 0; i < 13; i++)
            {
                float x = -6.0f + i * 1.0f;
                addBlock(sizeX, {origin.x + x, origin.y + outerWallOffset, z});
                addBlock(sizeX, {origin.x + x, origin.y - outerWallOffset, z});
            }

            // Inner East & West run full 13m along Y (13 bricks each)
            for (int i = 0; i < 13; i++)
            {
                float y = -6.0f + i * 1.0f;
                addBlock(sizeY, {origin.x + innerWallOffset, origin.y + y, z});
                addBlock(sizeY, {origin.x - innerWallOffset, origin.y + y, z});
            }
            // Inner North & South fit between East/West along X (12 bricks each)
            for (int i = 0; i < 12; i++)
            {
                float x = -5.5f + i * 1.0f;
                addBlock(sizeX, {origin.x + x, origin.y + innerWallOffset, z});
                addBlock(sizeX, {origin.x + x, origin.y - innerWallOffset, z});
            }
        }
    }

    const float breakForce = 800.0f;
    const float panelAngStiffness = 1000000.0f;
    for (size_t i = 0; i < blocks.size(); i++)
    {
        if (blocks[i].pos.z < brickHeight)
            s->addJointWorld(ground, blocks[i].id, {blocks[i].pos.x, blocks[i].pos.y, 0.0f}, INFINITY, INFINITY,
                             breakForce);

        size_t max_j = i + 400;
        if (max_j > blocks.size())
            max_j = blocks.size();
        for (size_t j = i + 1; j < max_j; j++)
        {
            vec3 d = {blocks[i].pos.x - blocks[j].pos.x, blocks[i].pos.y - blocks[j].pos.y,
                      blocks[i].pos.z - blocks[j].pos.z};
            float dist = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
            if (dist <= 1.05f)
            {
                vec3 mid = {(blocks[i].pos.x + blocks[j].pos.x) * 0.5f, (blocks[i].pos.y + blocks[j].pos.y) * 0.5f,
                            (blocks[i].pos.z + blocks[j].pos.z) * 0.5f};
                s->addJointWorld(blocks[i].id, blocks[j].id, mid, INFINITY, panelAngStiffness, breakForce);
            }
        }
    }
}

// Two square hollow brick towers (100 blocks/layer x 100 layers = 10,000 blocks each = 20,000 blocks total)
static SceneView sceneSquareTowers20kGpu(SceneBuilder *s)
{
    s->clear();

    const float groundSizeX = 140.0f;
    const float groundSizeY = 80.0f;
    int ground = s->addBody({10000.0f, 10000.0f, 1.0f}, 0.0f, 0.5f, {0.0f, 0.0f, -0.5f});

    const int layers = 100;
    const float brickLength = 1.0f;
    const float brickWidth = 0.50f;
    const float brickHeight = 0.50f;

    // Tower A on the left, Tower B on the right
    addSquareBrickTower(s, ground, {-18.0f, 0.0f, 0.0f}, layers, brickLength, brickWidth, brickHeight);
    addSquareBrickTower(s, ground, {+18.0f, 0.0f, 0.0f}, layers, brickLength, brickWidth, brickHeight);

    return {110.0f, 25.0f};
}
