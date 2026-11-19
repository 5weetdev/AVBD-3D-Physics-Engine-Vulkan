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

// Mass from shapes, host only. Each shape's mass data
// comes from Box3D's own b3ComputeHullMass / b3ComputeSphereMass / b3ComputeCapsuleMass; the
// shapes are summed about the combined centre with the parallel-axis theorem, and the 3x3
// inertia is diagonalised (Jacobi) into principal moments and a principal rotation, so the
// solver keeps its diag(moment) primal untouched.

#include "collision.h"
#include "maths.h"

struct b3HullData; // Box3D's hull blob; only pointers cross this header.
struct b3MeshData; // Box3D's cooked mesh blob.

struct ShapeMassInput
{
    int kind = AVBD_SHAPE_BOX; // AVBD_SHAPE_BOX / SPHERE / CAPSULE / HULL
    vec3 half{0, 0, 0};        // as SceneDesc::shapeHalf (box half extents, sphere r, capsule r,r,r+hs)
    vec3 localPos{0, 0, 0};    // shape frame in the body frame
    quat localRot{0, 0, 0, 1};
    const b3HullData *hull = nullptr; // AVBD_SHAPE_HULL only
    const b3MeshData *mesh = nullptr; // AVBD_SHAPE_MESH only: a closed mesh has volume mass, an open one none
    float density = -1.0f;            // < 0: the density passed to computeCompoundMass; 0: no mass
};

struct CompoundMass
{
    float mass = 0.0f;
    vec3 center{0, 0, 0};          // combined centre of mass, in the body frame
    vec3 moment{0, 0, 0};          // principal moments about the centre
    quat principalRot{0, 0, 0, 1}; // principal frame -> body frame
};

// True when the cooked mesh is a closed, consistently wound, non-degenerate surface with volume.
bool meshIsClosed(const b3MeshData *mesh);

// Sums `count` shapes, each at its own density or, if that is negative, at `density`. Returns
// false (and leaves `out` zero) when no shape has mass: no shapes, non-positive densities, or
// only kinds with no mass (meshes, a hull with no blob).
bool computeCompoundMass(const ShapeMassInput *shapes, int count, float density, CompoundMass &out);

// One shape's mass, centroid and inertia about that centroid, all in the body frame (row-major
// inertia). Returns false for a kind with no mass.
bool computeShapeMass(const ShapeMassInput &shape, float density, float &mass, vec3 &center, float inertia[3][3]);

// Principal moments and the principal rotation (principal frame -> input frame) of a symmetric
// row-major inertia tensor. A diagonal tensor gives exactly its diagonal and the identity.
void principalInertia(const float inertia[3][3], vec3 &moment, quat &rot);
