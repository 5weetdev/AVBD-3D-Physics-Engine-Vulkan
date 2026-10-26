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

// A curtain-wall skyscraper, centred on the origin.
//
// The tower is a real frame rather than a shell: square columns on a 4 x 3 bay grid
// carry cast floor slabs, and the facade hangs off those slabs as spandrel bands with
// deep vertical mullion fins between them. The fins run the full storey height and sit
// proud of the spandrels, so each face reads as the dense vertical striping of the
// reference tower, with a continuous ribbon window left open between each spandrel top
// and the slab above. The roof carries a parapet ring and a few mechanical plant boxes.
//
// Joints fracture at 1600 N, with the non-structural cladding at 800 N, so the facade
// sheds panels first and the frame follows once it is pushed hard enough.
static SceneView sceneSkyscraperGpu(SceneBuilder *s)
{
    s->clear();

    int ground = s->addBody({10000.0f, 10000.0f, 1.0f}, 0.0f, 0.5f, {0.0f, 0.0f, -0.5f});

    // Weaken joints by 0.75x across the board for more dramatic failure.
    const float breakForce = 2400.0f;
    const float breakForceMinor = 600.0f;
    const float panelAngStiffness = 1000000.0f;

    const int floors = 46;
    const float colHeight = 2.8f;
    const float slabThick = 0.2f;
    const float floorPitch = colHeight + slabThick; // 3.0m -> 138m to the parapet

    const int baysX = 4;
    const int baysY = 3;
    const float baySize = 3.0f;
    const float halfSpanX = baysX * 0.5f * baySize; // 6.0m
    const float halfSpanY = baysY * 0.5f * baySize; // 4.5m

    const float colSize = 0.35f;
    const float spandrelH = 1.05f; // band under each ribbon window
    const float spandrelT = 0.16f;
    const float finW = 0.14f; // fin thickness along the facade
    const float finD = 0.34f; // how far the fin stands proud of the glass line

    // The cladding is layered strictly outboard of the frame so nothing interpenetrates:
    // slab edge -> perimeter column -> spandrel band -> mullion fin, each layer resting
    // flush against the outer face of the one beneath it.
    const float faceX = halfSpanX + colSize * 0.5f; // outer face of the perimeter columns
    const float faceY = halfSpanY + colSize * 0.5f;
    const float spandrelCX = faceX + spandrelT * 0.5f; // spandrel centreline
    const float spandrelCY = faceY + spandrelT * 0.5f;
    const float finCX = faceX + spandrelT + finD * 0.5f; // fin centreline, proud of the spandrel
    const float finCY = faceY + spandrelT + finD * 0.5f;

    // Map a tower-local coordinate onto the bay index of the slab covering it.
    auto bayX = [&](float lx) {
        int i = (int)floorf(lx / baySize + baysX * 0.5f);
        return i < 0 ? 0 : (i > baysX - 1 ? baysX - 1 : i);
    };
    auto bayY = [&](float ly) {
        int j = (int)floorf(ly / baySize + baysY * 0.5f);
        return j < 0 ? 0 : (j > baysY - 1 ? baysY - 1 : j);
    };

    int prevSlabs[baysX][baysY];
    for (int bx = 0; bx < baysX; bx++)
        for (int by = 0; by < baysY; by++)
            prevSlabs[bx][by] = -1;

    for (int k = 0; k < floors; k++)
    {
        const float zBase = k * floorPitch; // top of the slab below
        const float zCol = zBase + colHeight * 0.5f;
        const float zSlab = zBase + colHeight + slabThick * 0.5f;
        const float zSlabTop = zBase + colHeight;

        // 1. Grade the structural strength by floor height.
        // Base floors are ~2.5x stronger to support the building, roof floors are weaker.
        const float heightFactor = 1.0f - (float)k / (float)floors;
        const float floorBreakForce = breakForce * (0.4f + 2.0f * heightFactor);

        // Slab-to-slab is weaker to allow concrete floors to fragment and shatter
        const float slabBreakForce = floorBreakForce * 0.5f;
        // Slab-to-column is weaker to allow floor pancaking (stripping floors from columns)
        const float slabToColBreakForce = floorBreakForce * 0.6f;
        // Column vertical angular stiffness allows swaying and bending before snapping
        const float colAngStiffness = 5000000.0f;

        int curSlabs[baysX][baysY];

        // 1. Floor slabs (4 x 3 = 12 per floor), tied edge to edge.
        for (int bx = 0; bx < baysX; bx++)
        {
            for (int by = 0; by < baysY; by++)
            {
                const float cx = (bx - (baysX - 1) * 0.5f) * baySize;
                const float cy = (by - (baysY - 1) * 0.5f) * baySize;
                // Floor slabs are thick concrete; density 1.2f adds weight for pancaking
                curSlabs[bx][by] = s->addBody({baySize, baySize, slabThick}, 1.2f, 0.6f, {cx, cy, zSlab});

                if (bx > 0)
                    s->addJointWorld(curSlabs[bx - 1][by], curSlabs[bx][by],
                                     {cx - baySize * 0.5f, cy, zSlab},
                                     INFINITY, panelAngStiffness, slabBreakForce);
                if (by > 0)
                    s->addJointWorld(curSlabs[bx][by - 1], curSlabs[bx][by],
                                     {cx, cy - baySize * 0.5f, zSlab},
                                     INFINITY, panelAngStiffness, slabBreakForce);
            }
        }

        // 2. Columns on every gridline (5 x 4 = 20 per floor).
        for (int i = 0; i <= baysX; i++)
        {
            const float cx = (i - baysX * 0.5f) * baySize;
            for (int j = 0; j <= baysY; j++)
            {
                const float ly = (j - baysY * 0.5f) * baySize;
                // Core structural columns; very dense reinforced concrete (1.5f)
                int col = s->addBody({colSize, colSize, colHeight}, 1.5f, 0.6f, {cx, ly, zCol});

                if (k == 0)
                    s->addJointWorld(ground, col, {cx, ly, 0.0f}, INFINITY, INFINITY, floorBreakForce);

                // Column head into every slab meeting this gridline, and foot into
                // the matching slabs of the floor below.
                for (int dx = -1; dx <= 0; dx++)
                {
                    int bx = i + dx;
                    if (bx < 0 || bx > baysX - 1)
                        continue;
                    for (int dy = -1; dy <= 0; dy++)
                    {
                        int by = j + dy;
                        if (by < 0 || by > baysY - 1)
                            continue;

                        // Head to current floor slab (pancaking joint)
                        s->addJointWorld(col, curSlabs[bx][by], {cx, ly, zSlabTop},
                                         INFINITY, panelAngStiffness, slabToColBreakForce);

                        if (prevSlabs[bx][by] >= 0)
                        {
                            // Foot to previous floor slab (vertical column stack continuity)
                            // We use colAngStiffness instead of INFINITY to allow bending
                            s->addJointWorld(prevSlabs[bx][by], col, {cx, ly, zBase},
                                             INFINITY, colAngStiffness, floorBreakForce);
                        }
                    }
                }
            }
        }

        // 3. Spandrel bands around the whole perimeter (14 per floor). Each sits on
        //    the slab below and stops short of the slab above, leaving a ribbon window.
        const float zSpandrel = zBase + spandrelH * 0.5f;

        // Faces normal to Y (the long 12m elevations).
        for (int bx = 0; bx < baysX; bx++)
        {
            const float cx = (bx - (baysX - 1) * 0.5f) * baySize;
            for (int side = 0; side < 2; side++)
            {
                const float sgn = (side == 0) ? -1.0f : +1.0f;
                const float wy = sgn * spandrelCY;
                const int by = (side == 0) ? 0 : baysY - 1;
                // Spandrel bands are lightweight curtain wall panels (0.4f)
                int band = s->addBody({baySize * 0.98f, spandrelT, spandrelH}, 0.4f, 0.6f, {cx, wy, zSpandrel});

                if (k == 0)
                    s->addJointWorld(ground, band, {cx, wy, 0.0f}, INFINITY, INFINITY, breakForceMinor);
                else if (prevSlabs[bx][by] >= 0)
                    s->addJointWorld(prevSlabs[bx][by], band, {cx, wy, zBase},
                                     INFINITY, panelAngStiffness, breakForceMinor);
            }
        }

        // Faces normal to X (the short 9m elevations).
        for (int by = 0; by < baysY; by++)
        {
            const float ly = (by - (baysY - 1) * 0.5f) * baySize;
            for (int side = 0; side < 2; side++)
            {
                const float sgn = (side == 0) ? -1.0f : +1.0f;
                const float wx = sgn * spandrelCX;
                const int bx = (side == 0) ? 0 : baysX - 1;
                // Spandrel bands are lightweight curtain wall panels (0.4f)
                int band = s->addBody({spandrelT, baySize * 0.98f, spandrelH}, 0.4f, 0.6f, {wx, ly, zSpandrel});

                if (k == 0)
                    s->addJointWorld(ground, band, {wx, ly, 0.0f}, INFINITY, INFINITY, breakForceMinor);
                else if (prevSlabs[bx][by] >= 0)
                    s->addJointWorld(prevSlabs[bx][by], band, {wx, ly, zBase},
                                     INFINITY, panelAngStiffness, breakForceMinor);
            }
        }

        // 4. Vertical mullion fins: one on every gridline plus one at each bay centre,
        //    full storey height and standing proud of the spandrel line (32 per floor).
        const float zFin = zBase + colHeight * 0.5f;

        for (int i = 0; i < baysX * 2 + 1; i++)
        {
            const float lx = (i * 0.5f - baysX * 0.5f) * baySize;
            const int bx = bayX(lx);
            for (int side = 0; side < 2; side++)
            {
                const float sgn = (side == 0) ? -1.0f : +1.0f;
                const float fy = sgn * finCY;
                const int by = (side == 0) ? 0 : baysY - 1;
                // Aluminum/glass mullion fins are very lightweight (0.2f)
                int fin = s->addBody({finW, finD, colHeight}, 0.2f, 0.6f, {lx, fy, zFin});

                s->addJointWorld(fin, curSlabs[bx][by], {lx, fy, zSlabTop},
                                 INFINITY, panelAngStiffness, breakForceMinor);
                if (k == 0)
                    s->addJointWorld(ground, fin, {lx, fy, 0.0f}, INFINITY, INFINITY, breakForceMinor);
                else if (prevSlabs[bx][by] >= 0)
                    s->addJointWorld(prevSlabs[bx][by], fin, {lx, fy, zBase},
                                     INFINITY, panelAngStiffness, breakForceMinor);
            }
        }

        for (int j = 0; j < baysY * 2 + 1; j++)
        {
            const float ly = (j * 0.5f - baysY * 0.5f) * baySize;
            const int by = bayY(ly);
            for (int side = 0; side < 2; side++)
            {
                const float sgn = (side == 0) ? -1.0f : +1.0f;
                const float fx = sgn * finCX;
                const int bx = (side == 0) ? 0 : baysX - 1;
                // Aluminum/glass mullion fins are very lightweight (0.2f)
                int fin = s->addBody({finD, finW, colHeight}, 0.2f, 0.6f, {fx, ly, zFin});

                s->addJointWorld(fin, curSlabs[bx][by], {fx, ly, zSlabTop},
                                 INFINITY, panelAngStiffness, breakForceMinor);
                if (k == 0)
                    s->addJointWorld(ground, fin, {fx, ly, 0.0f}, INFINITY, INFINITY, breakForceMinor);
                else if (prevSlabs[bx][by] >= 0)
                    s->addJointWorld(prevSlabs[bx][by], fin, {fx, ly, zBase},
                                     INFINITY, panelAngStiffness, breakForceMinor);
            }
        }

        for (int bx = 0; bx < baysX; bx++)
            for (int by = 0; by < baysY; by++)
                prevSlabs[bx][by] = curSlabs[bx][by];
    }

    // 5. Roof: a parapet ring standing on the top slab, then the mechanical plant.
    const float zRoof = floors * floorPitch; // top surface of the last slab
    const float parapetH = 1.1f;
    const float parapetT = 0.22f;
    const float zPar = zRoof + parapetH * 0.5f;

    // The ring is mitred rather than crossed: the long runs sit just inside the slab
    // edge, and the short runs are shortened to stop clear of them at the corners.
    const float parCY = halfSpanY - parapetT * 0.5f;
    const float parCX = halfSpanX - parapetT * 0.5f;
    const float parShortHalf = parCY - parapetT * 0.5f; // where the short runs must end
    const float parShortSeg = (parShortHalf * 2.0f) / (float)baysY;

    for (int side = 0; side < 2; side++)
    {
        const float sgn = (side == 0) ? -1.0f : +1.0f;
        const int edgeY = (side == 0) ? 0 : baysY - 1;
        const int edgeX = (side == 0) ? 0 : baysX - 1;

        // Long runs along X, one per bay.
        for (int b = 0; b < baysX; b++)
        {
            const float cx = (b - (baysX - 1) * 0.5f) * baySize;
            const float py = sgn * parCY;
            // Parapet cladding (0.4f)
            int par = s->addBody({baySize * 0.98f, parapetT, parapetH}, 0.4f, 0.6f, {cx, py, zPar});
            s->addJointWorld(prevSlabs[b][edgeY], par, {cx, py, zRoof},
                             INFINITY, panelAngStiffness, breakForceMinor);
        }

        // Short runs along Y, trimmed to fit between the long runs.
        for (int b = 0; b < baysY; b++)
        {
            const float cy = (b - (baysY - 1) * 0.5f) * parShortSeg;
            const float px = sgn * parCX;
            // Parapet cladding (0.4f)
            int par = s->addBody({parapetT, parShortSeg * 0.98f, parapetH}, 0.4f, 0.6f, {px, cy, zPar});
            s->addJointWorld(prevSlabs[edgeX][bayY(cy)], par, {px, cy, zRoof},
                             INFINITY, panelAngStiffness, breakForceMinor);
        }
    }

    // Rooftop plant boxes, as in the reference photo.
    const float mechH[3] = {1.6f, 1.2f, 1.2f};
    const float mechX[3] = {-3.0f, 0.6f, 3.6f};
    const float mechY[3] = {0.0f, 1.2f, -1.0f};
    const float mechSX[3] = {3.2f, 2.4f, 2.0f};
    const float mechSY[3] = {3.0f, 2.2f, 2.6f};
    for (int m = 0; m < 3; m++)
    {
        // Heavy mechanical equipment (2.0f)
        int box = s->addBody({mechSX[m], mechSY[m], mechH[m]}, 2.0f, 0.6f,
                             {mechX[m], mechY[m], zRoof + mechH[m] * 0.5f});
        s->addJointWorld(prevSlabs[bayX(mechX[m])][bayY(mechY[m])], box,
                         {mechX[m], mechY[m], zRoof}, INFINITY, panelAngStiffness, breakForce);
    }

    for (int i = 1; i < s->bodyCount(); i++)
    {
        s->setAwake(i, false);
    }

    return {230.0f, 62.0f};
}
