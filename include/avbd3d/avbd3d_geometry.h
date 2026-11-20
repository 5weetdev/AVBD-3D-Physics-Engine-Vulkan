/* SPDX-FileCopyrightText: 2026 avbd3d contributors
   SPDX-License-Identifier: MIT */

/* avbd3d geometry: convex hulls, triangle meshes, height fields and compounds, and the
   character mover helpers. Geometry is created by the caller, passed to the avbd3d_create_*_shape
   calls of avbd3d.h (which copy what they need), and destroyed by the caller. Destroying
   geometry that a shape was made from is allowed. */
#ifndef AVBD3D_GEOMETRY_H
#define AVBD3D_GEOMETRY_H

#include "avbd3d.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /* These are collision planes that can be fed to avbd3d_solve_planes. Normally
   this is assembled by the user from plane results in Avbd3dPlaneResult. */
    typedef struct Avbd3dCollisionPlane
    {
        /* The collision plane between the mover and some shape. */
        Avbd3dPlane plane;

        /* Setting this to FLT_MAX makes the plane as rigid as possible. Lower values can
	   make the plane collision soft. Usually in meters. */
        float pushLimit;

        /* The push on the mover determined by avbd3d_solve_planes. Usually in meters. */
        float push;

        /* Indicates if avbd3d_clip_vector should clip against this plane. Should be false for soft collision. */
        int32_t clipVelocity;
    } Avbd3dCollisionPlane;

    /* Result returned by avbd3d_solve_planes. */
    typedef struct Avbd3dPlaneSolverResult
    {
        /* The final relative translation. */
        Avbd3dVec3 delta;

        /* The number of iterations used by the plane solver. For diagnostics. */
        int32_t iterationCount;
    } Avbd3dPlaneSolverResult;

    /* This is used to create a re-usable collision mesh. No pointers
   are held to this data in Avbd3dMeshData. So all this data can be temporary. */
    typedef struct Avbd3dMeshDef
    {
        /* Triangle vertices. */
        Avbd3dVec3 *vertices;

        /* Stride between vertices. Use 0 for contiguous vertices. */
        size_t stride;

        /* Triangle vertex indices. 3 for each triangle. CCW winding unless CW is indicated below. */
        int32_t *indices;

        /* Triangle material index. 1 per triangle. Indexes into Avbd3dShapeDef::materials.
	   This allows different run-time material data to be associated with different
	   instances of this mesh. */
        uint8_t *materialIndices;

        /* Tolerance for vertex welding in length units. */
        float weldTolerance;

        /* The vertex count. Must be 3 or more. */
        int32_t vertexCount;

        /* The triangle count. Must be 1 or more. */
        int32_t triangleCount;

        /* Optionally weld nearby vertices. */
        int32_t weldVertices;

        /* Use the median split instead of SAH to speed up mesh creation. Good
	   for meshes that are structured like a grid. */
        int32_t useMedianSplit;

        /* Compute triangle adjacency information using shared edges */
        int32_t identifyEdges;

        /* Input indices have clockWise winding order. */
        int32_t clockWiseWinding;
    } Avbd3dMeshDef;

    /* Data used to create a height field. No pointers are held to this data. */
    typedef struct Avbd3dHeightFieldDef
    {
        /* Grid point heights
	   count = countX * countZ */
        float *heights;

        /* Grid cell material
	   A value of 0xFF is reserved for holes
	   count = (countX - 1) * (countZ - 1) */
        uint8_t *materialIndices;

        /* The height field scale. All components must be positive values. */
        Avbd3dVec3 scale;

        /* The number of grid lines along the x-axis. */
        int32_t countX;

        /* The number of grid lines along the z-axis. */
        int32_t countZ;

        /* Global minimum and maximum heights used for quantization. This is important
	   if you want height fields to be placed next to each other and line up exactly.
	   In that case, both height fields should use the same minimum and maximum heights.
	   All height values are clamped to this range.
	   These values are in unscaled space. */
        float globalMinimumHeight;

        /* The maximum. */
        float globalMaximumHeight;

        /* Use clock-wise winding. This effectively inverts the height-field along the y-axis. */
        int32_t clockwiseWinding;
    } Avbd3dHeightFieldDef;

    /* Definition for a capsule in a compound shape. */
    typedef struct Avbd3dCompoundCapsuleDef
    {
        /* Local capsule. */
        Avbd3dCapsule capsule;

        /* Material properties. */
        Avbd3dSurfaceMaterial material;
    } Avbd3dCompoundCapsuleDef;

    /* Definition for a convex hull in a compound shape. */
    typedef struct Avbd3dCompoundHullDef
    {
        /* Shared hull. */
        const Avbd3dHullData *hull;

        /* Transform of the shared hull into compound local space. */
        Avbd3dTransform transform;

        /* Material properties. */
        Avbd3dSurfaceMaterial material;
    } Avbd3dCompoundHullDef;

    /* Definition for a triangle mesh in a compound shape. */
    typedef struct Avbd3dCompoundMeshDef
    {
        /* Shared mesh. */
        const Avbd3dMeshData *meshData;

        /* Transform of the shared mesh into compound local space. */
        Avbd3dTransform transform;

        /* Local space non-uniform mesh scale. May have negative components. */
        Avbd3dVec3 scale;

        /* Material properties.
	   This array must line up with the material indices on the triangles. */
        const Avbd3dSurfaceMaterial *materials;

        /* Number of materials. */
        int32_t materialCount;
    } Avbd3dCompoundMeshDef;

    /* Definition for a sphere in a compound shape. */
    typedef struct Avbd3dCompoundSphereDef
    {
        /* Local sphere. */
        Avbd3dSphere sphere;

        /* Material properties. */
        Avbd3dSurfaceMaterial material;
    } Avbd3dCompoundSphereDef;

    /* Definition for creating a compound shape. All this data is fully cloned
   into the run-time compound shape. */
    typedef struct Avbd3dCompoundDef
    {
        /* Capsule instances. */
        Avbd3dCompoundCapsuleDef *capsules;

        /* Number of capsules. */
        int32_t capsuleCount;

        /* Hulls instances. */
        Avbd3dCompoundHullDef *hulls;

        /* Number of hull instances. */
        int32_t hullCount;

        /* Mesh instances. */
        Avbd3dCompoundMeshDef *meshes;

        /* Number of mesh instances. */
        int32_t meshCount;

        /* Sphere instances. */
        Avbd3dCompoundSphereDef *spheres;

        /* Number of spheres. */
        int32_t sphereCount;
    } Avbd3dCompoundDef;

    /* A box hull that lives in the caller's memory: no allocation, nothing to destroy. Opaque
   storage; read it as a hull with avbd3d_box_hull_data. It may be copied and moved. */
    typedef struct Avbd3dBoxHull
    {
        uint64_t storage[80];
    } Avbd3dBoxHull;

    /* ---------------------------------------------------------------------------
   Hulls
   --------------------------------------------------------------------------- */

    /* Builds the convex hull of a point cloud. max_vertex_count caps the hull's vertices (use 0 for
   no cap beyond the library limit). Fails with AVBD3D_ERR_INVALID_ARG if the points are
   degenerate. */
    AVBD3D_API Avbd3dResult avbd3d_create_hull(const Avbd3dVec3 *points, int32_t point_count, int32_t max_vertex_count,
                                               Avbd3dHullData **out_hull);

    AVBD3D_API Avbd3dResult avbd3d_destroy_hull(Avbd3dHullData *hull);

    /* A box hull with half extents hx, hy, hz. A pure helper: it allocates nothing. */
    AVBD3D_API Avbd3dBoxHull avbd3d_make_box_hull(float hx, float hy, float hz);

    /* The hull inside a box hull. Valid while the box hull is. Pass it to avbd3d_create_hull_shape;
   never to avbd3d_destroy_hull. */
    AVBD3D_API const Avbd3dHullData *avbd3d_box_hull_data(const Avbd3dBoxHull *box);

    /* ---------------------------------------------------------------------------
   Meshes
   --------------------------------------------------------------------------- */

    /* Builds a triangle mesh. Static bodies only. Fails with AVBD3D_ERR_INVALID_ARG if the
   definition is malformed or every triangle is degenerate. */
    AVBD3D_API Avbd3dResult avbd3d_create_mesh(const Avbd3dMeshDef *def, Avbd3dMeshData **out_mesh);

    AVBD3D_API Avbd3dResult avbd3d_destroy_mesh(Avbd3dMeshData *mesh);

    /* ---------------------------------------------------------------------------
   Height fields
   --------------------------------------------------------------------------- */

    AVBD3D_API Avbd3dResult avbd3d_create_height_field(const Avbd3dHeightFieldDef *def,
                                                       Avbd3dHeightFieldData **out_height_field);

    AVBD3D_API Avbd3dResult avbd3d_destroy_height_field(Avbd3dHeightFieldData *height_field);

    /* ---------------------------------------------------------------------------
   Compounds
   --------------------------------------------------------------------------- */

    /* Builds a compound of capsules, hulls, meshes and spheres. All input in the def is copied. */
    AVBD3D_API Avbd3dResult avbd3d_create_compound(const Avbd3dCompoundDef *def, Avbd3dCompoundData **out_compound);

    AVBD3D_API Avbd3dResult avbd3d_destroy_compound(Avbd3dCompoundData *compound);

    /* ---------------------------------------------------------------------------
   Character mover
   --------------------------------------------------------------------------- */

    /* Solves the position of a mover that satisfies the collision planes gathered by
   avbd3d_world_collide_mover. target_delta is the translation wanted from the position the planes
   were generated at. The planes are updated in place. A pure helper. */
    AVBD3D_API Avbd3dPlaneSolverResult avbd3d_solve_planes(Avbd3dVec3 target_delta, Avbd3dCollisionPlane *planes,
                                                           int32_t count);

    /* Clips a velocity against the collision planes. A pure helper. */
    AVBD3D_API Avbd3dVec3 avbd3d_clip_vector(Avbd3dVec3 vector, const Avbd3dCollisionPlane *planes, int32_t count);

#ifdef __cplusplus
}
#endif

#endif /* AVBD3D_GEOMETRY_H */
