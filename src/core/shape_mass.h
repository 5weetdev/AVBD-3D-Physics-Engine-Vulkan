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

#include "collision.h"
#include "maths.h"

// ---------------------------------------------------------------------------
// Mass properties per shape.
//
// `half` is the shape's tight OBB in every case -- box (hx,hy,hz), sphere (r,r,r),
// capsule (r, r, r + halfSeg) about local +z -- which is the invariant the whole
// shape pipeline rests on (src/core/collision.h). The box branch is the arithmetic
// GpuSolver::addBody has always done, written out here unchanged so that adding shapes
// cannot move a box scene by a single ulp.
//
// A density of zero yields zero mass and zero moment, which is how a static body is
// spelled everywhere else.
// ---------------------------------------------------------------------------
inline void shapeMassProperties(int shape, const vec3 &half, float density,
                                float &massOut, vec3 &momentOut)
{
    const float kPi = 3.14159265358979323846f;

    if (shape == AVBD_SHAPE_SPHERE)
    {
        const float r = half.x;
        const float m = (4.0f / 3.0f) * kPi * r * r * r * density;
        const float i = 0.4f * m * r * r;
        massOut = m;
        momentOut = vec3{i, i, i};
        return;
    }

    if (shape == AVBD_SHAPE_CAPSULE)
    {
        const float r = half.x;
        const float hs = half.z - half.x > 0.0f ? half.z - half.x : 0.0f;
        const float L = 2.0f * hs;

        const float mCyl = kPi * r * r * L * density;
        const float mCap = (4.0f / 3.0f) * kPi * r * r * r * density; // both hemispheres

        // Along the axis the two hemispheres together are a whole sphere, so the axial
        // term is exact. Across it, the hemispheres are carried to the ends by the
        // parallel-axis theorem about their true centroid, 3r/8 out from the flat face
        // -- the 3/8*r*L cross term below is that offset, and dropping it (as treating
        // them as a point mass at L/2 would) understates a stubby capsule's inertia by
        // tens of percent.
        const float iAxial = 0.5f * mCyl * r * r + 0.4f * mCap * r * r;
        const float iLateral = mCyl * (L * L / 12.0f + r * r * 0.25f) +
                               mCap * (0.4f * r * r + 0.25f * L * L + 0.375f * r * L);

        massOut = mCyl + mCap;
        momentOut = vec3{iLateral, iLateral, iAxial};
        return;
    }

    const vec3 size = half * 2.0f;
    const float m = size.x * size.y * size.z * density;
    massOut = m;
    momentOut = vec3{
        (size.y * size.y + size.z * size.z) / 12.0f * m,
        (size.x * size.x + size.z * size.z) / 12.0f * m,
        (size.x * size.x + size.y * size.y) / 12.0f * m};
}
