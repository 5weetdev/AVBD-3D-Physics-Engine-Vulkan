// Creation and destruction of the collision builders the avbd3d ABI declares in avbd3d_collision.h:
// convex hulls, triangle meshes and height fields. Each one is Box3D's builder, compiled unmodified
// into avbd3d_box3d. The avbd3d plain-data types are Box3D's layout (static-asserted in
// avbd3d_api.cpp and the layout gate), so the casts below are the same bytes.
#include "avbd3d_internal.h"

#include "box3d/base.h"
#include "box3d/collision.h"
#include "box3d/math_functions.h"
#include "box3d/types.h"

#include <cstring>

static_assert(sizeof(avbd3d_Vec3) == sizeof(b3Vec3), "avbd3d_Vec3 layout");
static_assert(sizeof(avbd3d_MeshDef) == sizeof(b3MeshDef), "avbd3d_MeshDef layout");
static_assert(sizeof(avbd3d_HeightFieldDef) == sizeof(b3HeightFieldDef), "avbd3d_HeightFieldDef layout");

namespace
{

// Same bytes, two type names: avbd3d and Box3D structs are layout-identical (see the static_asserts above).
template <class To, class From> To avbd3d_bit(const From &from)
{
    static_assert(sizeof(To) == sizeof(From), "avbd3d and Box3D layouts differ");
    To to;
    std::memcpy(&to, &from, sizeof(To));
    return to;
}

} // namespace

avbd3d_HullData *avbd3d_CreateHull(const avbd3d_Vec3 *points, int pointCount, int maxVertexCount)
{
    return reinterpret_cast<avbd3d_HullData *>(
        b3CreateHull(reinterpret_cast<const b3Vec3 *>(points), pointCount, maxVertexCount));
}

void avbd3d_DestroyHull(avbd3d_HullData *hull)
{
    b3DestroyHull(reinterpret_cast<b3HullData *>(hull));
}

avbd3d_MeshData *avbd3d_CreateMesh(const avbd3d_MeshDef *def, int *degenerateTriangleIndices, int degenerateCapacity)
{
    return reinterpret_cast<avbd3d_MeshData *>(
        b3CreateMesh(reinterpret_cast<const b3MeshDef *>(def), degenerateTriangleIndices, degenerateCapacity));
}

void avbd3d_DestroyMesh(avbd3d_MeshData *mesh)
{
    b3DestroyMesh(reinterpret_cast<b3MeshData *>(mesh));
}

avbd3d_HeightFieldData *avbd3d_CreateHeightField(const avbd3d_HeightFieldDef *data)
{
    return reinterpret_cast<avbd3d_HeightFieldData *>(
        b3CreateHeightField(reinterpret_cast<const b3HeightFieldDef *>(data)));
}

void avbd3d_DestroyHeightField(avbd3d_HeightFieldData *heightField)
{
    b3DestroyHeightField(reinterpret_cast<b3HeightFieldData *>(heightField));
}

// Thin wrappers over Box3D's collision helpers (geometry, trees, shape builders, queries).

avbd3d_ChildShape avbd3d_GetCompoundChild(const avbd3d_CompoundData *compound, int childIndex)
{
    return avbd3d_bit<avbd3d_ChildShape>(
        b3GetCompoundChild(reinterpret_cast<const b3CompoundData *>(compound), childIndex));
}

const avbd3d_SurfaceMaterial *avbd3d_GetCompoundMaterials(const avbd3d_CompoundData *compound)
{
    return reinterpret_cast<const avbd3d_SurfaceMaterial *>(
        b3GetCompoundMaterials(reinterpret_cast<const b3CompoundData *>(compound)));
}

avbd3d_CompoundData *avbd3d_CreateCompound(const avbd3d_CompoundDef *def)
{
    return reinterpret_cast<avbd3d_CompoundData *>(b3CreateCompound(reinterpret_cast<const b3CompoundDef *>(def)));
}

void avbd3d_DestroyCompound(avbd3d_CompoundData *compound)
{
    b3DestroyCompound(reinterpret_cast<b3CompoundData *>(compound));
}

avbd3d_PlaneSolverResult avbd3d_SolvePlanes(avbd3d_Vec3 targetDelta, avbd3d_CollisionPlane *planes, int count)
{
    return avbd3d_bit<avbd3d_PlaneSolverResult>(
        b3SolvePlanes(avbd3d_bit<b3Vec3>(targetDelta), reinterpret_cast<b3CollisionPlane *>(planes), count));
}

avbd3d_Vec3 avbd3d_ClipVector(avbd3d_Vec3 vector, const avbd3d_CollisionPlane *planes, int count)
{
    return avbd3d_bit<avbd3d_Vec3>(
        b3ClipVector(avbd3d_bit<b3Vec3>(vector), reinterpret_cast<const b3CollisionPlane *>(planes), count));
}
