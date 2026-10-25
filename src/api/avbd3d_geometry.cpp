// The public geometry calls (avbd3d_geometry.h): hulls, meshes, height fields, compounds and the
// character mover helpers. Thin wrappers over the legacy collision layer.
#include "avbd3d_capi.h"

#include <vector>

static_assert(sizeof(Avbd3dBoxHull) == sizeof(avbd3d_BoxHull), "Avbd3dBoxHull storage");
static_assert(alignof(Avbd3dBoxHull) >= alignof(avbd3d_BoxHull), "Avbd3dBoxHull alignment");

namespace
{
template <class P, class L> Avbd3dResult made(L *p, P **out)
{
    if (!p)
        return AVBD3D_ERR_INVALID_ARG;
    *out = reinterpret_cast<P *>(p);
    return AVBD3D_OK;
}
} // namespace

Avbd3dResult avbd3d_create_hull(const Avbd3dVec3 *points, int32_t point_count, int32_t max_vertex_count,
                                Avbd3dHullData **out_hull)
{
    if (!out_hull)
        return AVBD3D_ERR_INVALID_ARG;
    *out_hull = nullptr;
    if (!points || point_count < 4 || max_vertex_count < 0)
        return AVBD3D_ERR_INVALID_ARG;
    static_assert(sizeof(Avbd3dVec3) == sizeof(avbd3d_Vec3), "vec3");
    return made(avbd3d_CreateHull(reinterpret_cast<const avbd3d_Vec3 *>(points), point_count,
                                          max_vertex_count == 0 ? AVBD3D_MAX_HULL_VERTICES : max_vertex_count),
                out_hull);
}

Avbd3dResult avbd3d_destroy_hull(Avbd3dHullData *hull)
{
    if (!hull)
        return AVBD3D_ERR_NULL_HANDLE;
    avbd3d_DestroyHull(reinterpret_cast<avbd3d_HullData *>(hull));
    return AVBD3D_OK;
}

Avbd3dBoxHull avbd3d_make_box_hull(float hx, float hy, float hz)
{
    const avbd3d_BoxHull b = avbd3d_MakeBoxHull(hx, hy, hz);
    Avbd3dBoxHull r;
    std::memcpy(&r, &b, sizeof r);
    return r;
}

const Avbd3dHullData *avbd3d_box_hull_data(const Avbd3dBoxHull *box)
{
    return box ? reinterpret_cast<const Avbd3dHullData *>(box) : nullptr;
}

Avbd3dResult avbd3d_create_mesh(const Avbd3dMeshDef *def, Avbd3dMeshData **out_mesh)
{
    if (!out_mesh)
        return AVBD3D_ERR_INVALID_ARG;
    *out_mesh = nullptr;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_MeshDef ld = lg(*def);
    return made(avbd3d_CreateMesh(&ld, nullptr, 0), out_mesh);
}

Avbd3dResult avbd3d_destroy_mesh(Avbd3dMeshData *mesh)
{
    if (!mesh)
        return AVBD3D_ERR_NULL_HANDLE;
    avbd3d_DestroyMesh(reinterpret_cast<avbd3d_MeshData *>(mesh));
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_create_height_field(const Avbd3dHeightFieldDef *def, Avbd3dHeightFieldData **out_height_field)
{
    if (!out_height_field)
        return AVBD3D_ERR_INVALID_ARG;
    *out_height_field = nullptr;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_HeightFieldDef ld = lg(*def);
    return made(avbd3d_CreateHeightField(&ld), out_height_field);
}

Avbd3dResult avbd3d_destroy_height_field(Avbd3dHeightFieldData *height_field)
{
    if (!height_field)
        return AVBD3D_ERR_NULL_HANDLE;
    avbd3d_DestroyHeightField(reinterpret_cast<avbd3d_HeightFieldData *>(height_field));
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_create_compound(const Avbd3dCompoundDef *def, Avbd3dCompoundData **out_compound)
{
    if (!out_compound)
        return AVBD3D_ERR_INVALID_ARG;
    *out_compound = nullptr;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    // The instance arrays hold public-layout records; convert each to the internal layout.
    if (def->capsuleCount < 0 || def->hullCount < 0 || def->meshCount < 0 || def->sphereCount < 0)
        return AVBD3D_ERR_INVALID_ARG;
    if ((def->capsuleCount && !def->capsules) || (def->hullCount && !def->hulls) ||
        (def->meshCount && !def->meshes) || (def->sphereCount && !def->spheres))
        return AVBD3D_ERR_INVALID_ARG;
    std::vector<avbd3d_CompoundCapsuleDef> capsules;
    std::vector<avbd3d_CompoundHullDef> hulls;
    std::vector<avbd3d_CompoundMeshDef> meshes;
    std::vector<avbd3d_CompoundSphereDef> spheres;
    std::vector<std::vector<avbd3d_SurfaceMaterial>> meshMaterials((size_t)def->meshCount);
    for (int32_t i = 0; i < def->capsuleCount; ++i)
        capsules.push_back(lg(def->capsules[i]));
    for (int32_t i = 0; i < def->hullCount; ++i)
        hulls.push_back(lg(def->hulls[i]));
    for (int32_t i = 0; i < def->sphereCount; ++i)
        spheres.push_back(lg(def->spheres[i]));
    for (int32_t i = 0; i < def->meshCount; ++i)
    {
        const Avbd3dCompoundMeshDef &m = def->meshes[i];
        if (m.materialCount < 0 || (m.materialCount && !m.materials))
            return AVBD3D_ERR_INVALID_ARG;
        for (int32_t k = 0; k < m.materialCount; ++k)
            meshMaterials[(size_t)i].push_back(lg(m.materials[k]));
        avbd3d_CompoundMeshDef lm = lg(m);
        lm.materials = meshMaterials[(size_t)i].empty() ? nullptr : meshMaterials[(size_t)i].data();
        meshes.push_back(lm);
    }
    avbd3d_CompoundDef ld;
    std::memset(&ld, 0, sizeof ld);
    ld.capsules = capsules.empty() ? nullptr : capsules.data();
    ld.capsuleCount = def->capsuleCount;
    ld.hulls = hulls.empty() ? nullptr : hulls.data();
    ld.hullCount = def->hullCount;
    ld.meshes = meshes.empty() ? nullptr : meshes.data();
    ld.meshCount = def->meshCount;
    ld.spheres = spheres.empty() ? nullptr : spheres.data();
    ld.sphereCount = def->sphereCount;
    return made(avbd3d_CreateCompound(&ld), out_compound);
}

Avbd3dResult avbd3d_destroy_compound(Avbd3dCompoundData *compound)
{
    if (!compound)
        return AVBD3D_ERR_NULL_HANDLE;
    avbd3d_DestroyCompound(reinterpret_cast<avbd3d_CompoundData *>(compound));
    return AVBD3D_OK;
}

Avbd3dPlaneSolverResult avbd3d_solve_planes(Avbd3dVec3 target_delta, Avbd3dCollisionPlane *planes, int32_t count)
{
    std::vector<avbd3d_CollisionPlane> tmp;
    if (planes && count > 0)
    {
        tmp.reserve((size_t)count);
        for (int32_t i = 0; i < count; ++i)
            tmp.push_back(lg(planes[i]));
    }
    const avbd3d_PlaneSolverResult r = avbd3d_SolvePlanes(lg(target_delta), tmp.data(), (int)tmp.size());
    for (size_t i = 0; i < tmp.size(); ++i)
        planes[i] = pb(tmp[i]);
    return pb(r);
}

Avbd3dVec3 avbd3d_clip_vector(Avbd3dVec3 vector, const Avbd3dCollisionPlane *planes, int32_t count)
{
    std::vector<avbd3d_CollisionPlane> tmp;
    if (planes && count > 0)
        for (int32_t i = 0; i < count; ++i)
            tmp.push_back(lg(planes[i]));
    return pb(avbd3d_ClipVector(lg(vector), tmp.data(), (int)tmp.size()));
}
