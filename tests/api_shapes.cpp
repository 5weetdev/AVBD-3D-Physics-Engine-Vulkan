// Gate for the avbd3d shape creation of general hulls, static meshes and height fields
// (src/api, Box3D-style C ABI). Y-up, the same convention as test_vk_api3d_sensors.cpp.
// Checks:
//   a) a non-box hull (a square pyramid) rests flat on a static ground box at the expected height
//      within 0.05 m, and a ray cast down onto it hits that shape.
//   b) a box rests on a static mesh (a 10x10 cell grid made with avbd3d_create_mesh) at the expected
//      height, a ray cast onto the mesh hits it, and CreateMeshShape on a dynamic body returns null.
//   c) a sphere rests on a static height field at the expected height, a ray cast onto the height
//      field hits it, and a small sphere dropped through a hole cell falls past the field.
//   d) a dynamic CLOSED mesh (cube, tetrahedron) is accepted with mass = volume * density and the
//      matching centre and inertia, lands on a static box and rests, and carries a box that rests
//      on top of it. An open mesh on a dynamic body is refused, and SetType refuses to make a body
//      that owns an open mesh dynamic.
// Mass from shapes is covered by test_vk_api3d_mass.cpp.



// (merged) // Gate for avbd3d_create_baked_compound_shape (src/api, Box3D-style C ABI). Y-up.
// // A static body carries a 3-child compound: three 1 m cubes at x = -2, 0, +2 whose tops are y = 0
// // (a gap of 1 m between neighbours). Checks:
// //   a) a small box dropped on the LAST child (x = +2) and one on the middle child rest on their
// //      tops, and one dropped in a gap (x = +1) falls through: the footprint is the three children.
// //   b) a ray down onto the last child hits the compound and returns the compound id; a ray down
// //      the gap misses; the body lists one shape.
// //   c) the compound AABB spans all children.
// //   d) a compound on a dynamic body is accepted (mass, centre and inertia in checkDynamicCompound).
// //   f) a dynamic 3-cube compound has the summed mass/centre/inertia and settles on ground.
// //   g) friction set on the compound applies to every child (slope, non-first child in contact).
// //   e) destroying the compound removes every child: its id goes invalid, a ray through each child
// //      misses, and a box over x = +2 now falls.
// 
// 
// 
#include "test_util.h"
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

namespace shapes
{
static int g_fail = 0;
#define CHECK(cond, ...)                                                                                              \
    do                                                                                                                \
    {                                                                                                                 \
        if (!(cond))                                                                                                  \
        {                                                                                                             \
            std::printf("FAIL %s:%d: %s  ", __FILE__, __LINE__, #cond);                                               \
            std::printf(__VA_ARGS__);                                                                                 \
            std::printf("\n");                                                                                        \
            g_fail++;                                                                                                 \
        }                                                                                                             \
    } while (0)

static bool sameShape(Avbd3dShapeId a, Avbd3dShapeId b) { return a.index1 == b.index1 && a.generation == b.generation; }

static Avbd3dBodyId makeBody(Avbd3dWorld *w, Avbd3dBodyType type, Avbd3dVec3 p)
{
    Avbd3dBodyDef bd = avbd3d_default_body_def();
    bd.type = type;
    bd.position = p;
    return t3d_create_body(w, &bd);
}

static Avbd3dShapeId addBox(Avbd3dBodyId b, float hx, float hy, float hz)
{
    Avbd3dShapeDef sd = avbd3d_default_shape_def();
    Avbd3dBoxHull hull = avbd3d_make_box_hull(hx, hy, hz);
    return t3d_create_hull_shape(b, &sd, avbd3d_box_hull_data(&hull));
}

static float bodyY(Avbd3dBodyId b) { return t3d_body_get_position(b).y; }

static void run(Avbd3dWorld *w, float seconds)
{
    const int steps = (int)std::lround(seconds * 60.0f);
    for (int i = 0; i < steps; i++)
        t3d_step(w, 1.0f / 60.0f);
}

// a) A square pyramid, base at local y = -0.5 and apex at +0.5, dropped onto a ground box whose top is y = 0.
static void checkHullRest()
{
    Avbd3dWorldDef wd = avbd3d_default_world_def();
    Avbd3dWorld *w = t3d_create_world(&wd);

    Avbd3dBodyId ground = makeBody(w, AVBD3D_STATIC_BODY, Avbd3dVec3{0, -0.5f, 0});
    Avbd3dShapeId groundShape = addBox(ground, 20.0f, 0.5f, 20.0f);
    CHECK(avbd3d_shape_is_valid(groundShape), "ground box");

    const Avbd3dVec3 pts[5] = {{-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, 0.5f},
                                {-0.5f, -0.5f, 0.5f},  {0.0f, 0.5f, 0.0f}};
    Avbd3dHullData *hull = t3d_create_hull(pts, 5, 128);
    CHECK(hull != nullptr, "pyramid hull");

    Avbd3dBodyId pyramid = makeBody(w, AVBD3D_DYNAMIC_BODY, Avbd3dVec3{0, 2.0f, 0});
    Avbd3dShapeDef sd = avbd3d_default_shape_def();
    Avbd3dShapeId pyramidShape = t3d_create_hull_shape(pyramid, &sd, hull);
    t3d_destroy_hull(hull);
    CHECK(avbd3d_shape_is_valid(pyramidShape), "general hull shape");
    CHECK(t3d_shape_get_type(pyramidShape) == AVBD3D_HULL_SHAPE, "general hull type");

    run(w, 3.0f);
    // The base sits on the ground top (y = 0), so the origin rests at y = 0.5.
    CHECK(std::fabs(bodyY(pyramid) - 0.5f) < 0.05f, "pyramid rest height %.4f", bodyY(pyramid));

    Avbd3dRayResult ray = t3d_world_cast_ray_closest(w, {0, 10, 0}, {0, -20, 0}, avbd3d_default_query_filter());
    CHECK(ray.hit && sameShape(ray.shapeId, pyramidShape), "ray hits the general hull");

    t3d_destroy_world(w);
}

// b) A box on a static mesh. The mesh is a 10x10 grid of 1 m cells centred on the origin, at y = 0.
static void checkMeshRest()
{
    Avbd3dWorldDef wd = avbd3d_default_world_def();
    Avbd3dWorld *w = t3d_create_world(&wd);

    const int n = 10;
    std::vector<Avbd3dVec3> verts;
    for (int j = 0; j <= n; j++)
        for (int i = 0; i <= n; i++)
            verts.push_back(Avbd3dVec3{(float)i - 5.0f, 0.0f, (float)j - 5.0f});
    std::vector<int32_t> idx;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
        {
            const int32_t a = j * (n + 1) + i, b = a + 1, c = a + (n + 1) + 1, d = a + (n + 1);
            // (c-a) x (b-a) points along +y, so the top faces point up.
            idx.insert(idx.end(), {a, c, b, a, d, c});
        }
    Avbd3dMeshDef md = {};
    md.vertices = verts.data();
    md.stride = 0;
    md.indices = idx.data();
    md.vertexCount = (int)verts.size();
    md.triangleCount = (int)idx.size() / 3;
    md.identifyEdges = true;
    md.weldVertices = false;
    Avbd3dMeshData *mesh = t3d_create_mesh(&md);
    CHECK(mesh != nullptr, "grid mesh");

    Avbd3dBodyId ground = makeBody(w, AVBD3D_STATIC_BODY, Avbd3dVec3{0, 0, 0});
    Avbd3dShapeDef sd = avbd3d_default_shape_def();
    Avbd3dShapeId meshShape = t3d_create_mesh_shape(ground, &sd, mesh, Avbd3dVec3{1, 1, 1});
    CHECK(avbd3d_shape_is_valid(meshShape), "static mesh shape");

    Avbd3dBodyId dyn = makeBody(w, AVBD3D_DYNAMIC_BODY, Avbd3dVec3{0, 2.0f, 0});
    Avbd3dShapeId refused;
    CHECK(avbd3d_create_mesh_shape(dyn, &sd, mesh, Avbd3dVec3{1, 1, 1}, &refused) == AVBD3D_ERR_INVALID_ARG,
          "mesh on a dynamic body is refused");

    Avbd3dShapeId boxShape = addBox(dyn, 0.5f, 0.5f, 0.5f);
    CHECK(avbd3d_shape_is_valid(boxShape), "box on the mesh body");

    run(w, 3.0f);
    CHECK(std::fabs(bodyY(dyn) - 0.5f) < 0.05f, "box rest height on mesh %.4f", bodyY(dyn));

    // Cast beside the resting box (x, z in [-0.5, 0.5]) so the closest hit is the mesh, not the box.
    Avbd3dRayResult ray = t3d_world_cast_ray_closest(w, {3.25f, 10, 3.25f}, {0, -20, 0}, avbd3d_default_query_filter());
    CHECK(ray.hit && sameShape(ray.shapeId, meshShape), "ray hits the mesh");

    t3d_destroy_mesh(mesh);
    t3d_destroy_world(w);
}

// c) A 10x10 height field of flat cells at y = 0, with one hole cell at column 4, row 4. The grid
// starts at the body origin, so the body sits at (-4.5, 0, -4.5) to centre the field, and the hole
// covers the world x and z in [-0.5, 0.5].
static void checkHeightField()
{
    Avbd3dWorldDef wd = avbd3d_default_world_def();
    Avbd3dWorld *w = t3d_create_world(&wd);

    const int rows = 10, cols = 10;
    std::vector<float> heights((size_t)rows * cols, 0.0f);
    std::vector<uint8_t> mats((size_t)(rows - 1) * (cols - 1), 0);
    mats[4 * (cols - 1) + 4] = ((uint8_t)0xFF);

    Avbd3dHeightFieldDef hd = {};
    hd.heights = heights.data();
    hd.materialIndices = mats.data();
    hd.scale = Avbd3dVec3{1, 1, 1};
    hd.countX = cols;
    hd.countZ = rows;
    hd.globalMinimumHeight = 0.0f;
    hd.globalMaximumHeight = 1.0f;
    Avbd3dHeightFieldData *hf = t3d_create_height_field(&hd);
    CHECK(hf != nullptr, "height field");

    Avbd3dBodyId ground = makeBody(w, AVBD3D_STATIC_BODY, Avbd3dVec3{-4.5f, 0, -4.5f});
    Avbd3dShapeDef sd = avbd3d_default_shape_def();
    Avbd3dShapeId hfShape = t3d_create_height_field_shape(ground, &sd, hf);
    CHECK(avbd3d_shape_is_valid(hfShape), "height-field shape");
    CHECK(t3d_shape_get_type(hfShape) == AVBD3D_HEIGHT_FIELD_SHAPE, "height-field type");

    Avbd3dBodyId ball = makeBody(w, AVBD3D_DYNAMIC_BODY, Avbd3dVec3{-2.0f, 2.0f, -2.0f});
    Avbd3dSphere sph{{0, 0, 0}, 0.25f};
    Avbd3dShapeId ballShape = t3d_create_sphere_shape(ball, &sd, &sph);
    CHECK(avbd3d_shape_is_valid(ballShape), "sphere on the field");

    Avbd3dBodyId faller = makeBody(w, AVBD3D_DYNAMIC_BODY, Avbd3dVec3{0.0f, 2.0f, 0.0f});
    t3d_create_sphere_shape(faller, &sd, &sph);

    run(w, 3.0f);
    CHECK(std::fabs(bodyY(ball) - 0.25f) < 0.05f, "sphere rest height on height field %.4f", bodyY(ball));
    CHECK(bodyY(faller) < -2.0f, "sphere falls through the hole %.4f", bodyY(faller));

    // Cast beside the resting sphere (at x = z = -2) so the closest hit is the field, not the sphere.
    Avbd3dRayResult ray = t3d_world_cast_ray_closest(w, {-3.0f, 10, -3.0f}, {0, -20, 0}, avbd3d_default_query_filter());
    CHECK(ray.hit && sameShape(ray.shapeId, hfShape), "ray hits the height field");

    t3d_destroy_height_field(hf);
    t3d_destroy_world(w);
}

// d) Dynamic closed meshes.
static Avbd3dMeshData *makeClosedMesh(const std::vector<Avbd3dVec3> &v, std::vector<int32_t> tris)
{
    // Orient every triangle outward from the vertex centroid (convex shapes only).
    Avbd3dVec3 c = {0, 0, 0};
    for (const auto &p : v)
    {
        c.x += p.x / (float)v.size();
        c.y += p.y / (float)v.size();
        c.z += p.z / (float)v.size();
    }
    for (size_t t = 0; t + 2 < tris.size(); t += 3)
    {
        const Avbd3dVec3 a = v[(size_t)tris[t]], b = v[(size_t)tris[t + 1]], d = v[(size_t)tris[t + 2]];
        const Avbd3dVec3 e1 = {b.x - a.x, b.y - a.y, b.z - a.z}, e2 = {d.x - a.x, d.y - a.y, d.z - a.z};
        const Avbd3dVec3 n = {e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z, e1.x * e2.y - e1.y * e2.x};
        if (n.x * (a.x - c.x) + n.y * (a.y - c.y) + n.z * (a.z - c.z) < 0.0f)
            std::swap(tris[t + 1], tris[t + 2]);
    }
    std::vector<Avbd3dVec3> verts = v;
    Avbd3dMeshDef md = {};
    md.vertices = verts.data();
    md.indices = tris.data();
    md.vertexCount = (int)verts.size();
    md.triangleCount = (int)tris.size() / 3;
    md.identifyEdges = true;
    md.weldVertices = false;
    return t3d_create_mesh(&md);
}

static Avbd3dMeshData *makeCubeMesh()
{
    std::vector<Avbd3dVec3> v;
    for (int i = 0; i < 8; i++)
        v.push_back(Avbd3dVec3{(i & 1) ? 0.5f : -0.5f, (i & 2) ? 0.5f : -0.5f, (i & 4) ? 0.5f : -0.5f});
    // two triangles per face; orientation is fixed up by makeClosedMesh
    const int32_t f[12][3] = {{0, 1, 3}, {0, 3, 2}, {4, 5, 7}, {4, 7, 6}, {0, 1, 5}, {0, 5, 4},
                              {2, 3, 7}, {2, 7, 6}, {0, 2, 6}, {0, 6, 4}, {1, 3, 7}, {1, 7, 5}};
    std::vector<int32_t> t;
    for (auto &q : f)
        t.insert(t.end(), {q[0], q[1], q[2]});
    return makeClosedMesh(v, t);
}

static void checkDynamicMesh()
{
    Avbd3dWorldDef wd = avbd3d_default_world_def();
    Avbd3dWorld *w = t3d_create_world(&wd);

    Avbd3dBodyId ground = makeBody(w, AVBD3D_STATIC_BODY, Avbd3dVec3{0, -0.5f, 0});
    addBox(ground, 20.0f, 0.5f, 20.0f);

    // Tetrahedron: volume 1/6, density 6 -> mass 1, centroid (0.25, 0.25, 0.25).
    {
        Avbd3dMeshData *tet = makeClosedMesh({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
                                              {0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3});
        CHECK(tet != nullptr, "tetra mesh");
        Avbd3dBodyId b = makeBody(w, AVBD3D_DYNAMIC_BODY, Avbd3dVec3{50, 5, 0});
        Avbd3dShapeDef sd = avbd3d_default_shape_def();
        sd.density = 6.0f;
        Avbd3dShapeId s = t3d_create_mesh_shape(b, &sd, tet, Avbd3dVec3{1, 1, 1});
        CHECK(avbd3d_shape_is_valid(s), "dynamic tetrahedron accepted");
        CHECK(std::fabs(t3d_body_get_mass(b) - 1.0f) < 1e-3f, "tetra mass %.4f", t3d_body_get_mass(b));
        const Avbd3dVec3 c = t3d_body_get_local_center(b);
        CHECK(std::fabs(c.x - 0.25f) < 1e-3f && std::fabs(c.y - 0.25f) < 1e-3f && std::fabs(c.z - 0.25f) < 1e-3f,
              "tetra centre %.4f %.4f %.4f", c.x, c.y, c.z);
        // Trace of the inertia about the centroid = 2 * tr(cov) * density, with
        // cov = V/20 * (sum p p^T + S S^T) - V c c^T, V = 1/6, sum |p|^2 = 3, |S|^2 = 3, |c|^2 = 3/16.
        const Avbd3dMatrix3 I = t3d_body_get_local_rotational_inertia(b);
        const double V = 1.0 / 6.0;
        const double trCov = V / 20.0 * (3.0 + 3.0) - V * (3.0 / 16.0);
        const double trWant = 2.0 * trCov * 6.0;
        const double trGot = I.cx.x + I.cy.y + I.cz.z;
        CHECK(std::fabs(trGot - trWant) < 2e-3, "tetra inertia trace %.5f want %.5f", trGot, trWant);
        t3d_destroy_body(b);
        t3d_destroy_mesh(tet);
    }

    Avbd3dMeshData *cube = makeCubeMesh();
    CHECK(cube != nullptr, "cube mesh");
    Avbd3dBodyId mb = makeBody(w, AVBD3D_DYNAMIC_BODY, Avbd3dVec3{0, 2.0f, 0});
    Avbd3dShapeDef sd = avbd3d_default_shape_def();
    sd.density = 2.0f;
    Avbd3dShapeId ms = t3d_create_mesh_shape(mb, &sd, cube, Avbd3dVec3{1, 1, 1});
    CHECK(avbd3d_shape_is_valid(ms), "dynamic cube mesh accepted");
    CHECK(std::fabs(t3d_body_get_mass(mb) - 2.0f) < 1e-3f, "cube mass %.4f", t3d_body_get_mass(mb));
    {
        const Avbd3dMatrix3 I = t3d_body_get_local_rotational_inertia(mb);
        CHECK(std::fabs(I.cx.x - 1.0f / 3.0f) < 2e-3f && std::fabs(I.cy.y - 1.0f / 3.0f) < 2e-3f &&
                  std::fabs(I.cz.z - 1.0f / 3.0f) < 2e-3f,
              "cube inertia %.4f %.4f %.4f", I.cx.x, I.cy.y, I.cz.z);
    }
    run(w, 3.0f);
    CHECK(std::fabs(bodyY(mb) - 0.5f) < 0.05f, "dynamic cube mesh rest height %.4f", bodyY(mb));
    {
        const Avbd3dVec3 v = t3d_body_get_linear_velocity(mb), a = t3d_body_get_angular_velocity(mb);
        CHECK(std::fabs(v.x) + std::fabs(v.y) + std::fabs(v.z) < 0.05f, "cube mesh at rest, v (%.3f %.3f %.3f)", v.x,
              v.y, v.z);
        CHECK(std::fabs(a.x) + std::fabs(a.y) + std::fabs(a.z) < 0.05f, "cube mesh not spinning");
    }

    // A box dropped on the resting dynamic mesh stays on it.
    Avbd3dBodyId top = makeBody(w, AVBD3D_DYNAMIC_BODY, Avbd3dVec3{0, 2.0f, 0});
    addBox(top, 0.3f, 0.3f, 0.3f);
    run(w, 3.0f);
    CHECK(std::fabs(bodyY(top) - 1.3f) < 0.06f, "box rests on the dynamic mesh, y %.4f", bodyY(top));
    CHECK(std::fabs(bodyY(mb) - 0.5f) < 0.06f, "mesh keeps its height under the box, y %.4f", bodyY(mb));

    // An open mesh is refused on a dynamic body; SetType refuses a switch for an open-mesh body.
    {
        std::vector<Avbd3dVec3> v = {{0, 0, 0}, {1, 0, 0}, {0, 0, 1}, {1, 0, 1}};
        std::vector<int32_t> t = {0, 2, 1, 1, 2, 3};
        Avbd3dMeshDef md = {};
        md.vertices = v.data();
        md.indices = t.data();
        md.vertexCount = 4;
        md.triangleCount = 2;
        md.identifyEdges = true;
        Avbd3dMeshData *open = t3d_create_mesh(&md);
        CHECK(open != nullptr, "open mesh");
        Avbd3dBodyId d = makeBody(w, AVBD3D_DYNAMIC_BODY, Avbd3dVec3{30, 5, 0});
        Avbd3dShapeId openRefused;
        CHECK(avbd3d_create_mesh_shape(d, &sd, open, Avbd3dVec3{1, 1, 1}, &openRefused) == AVBD3D_ERR_INVALID_ARG,
              "open mesh on a dynamic body refused");
        Avbd3dBodyId st = makeBody(w, AVBD3D_STATIC_BODY, Avbd3dVec3{40, 0, 0});
        CHECK(avbd3d_shape_is_valid(t3d_create_mesh_shape(st, &sd, open, Avbd3dVec3{1, 1, 1})), "open mesh on static");
        (void)avbd3d_body_set_type(st, AVBD3D_DYNAMIC_BODY); // refused: the body stays static
        CHECK(t3d_body_get_type(st) == AVBD3D_STATIC_BODY, "SetType refused for an open mesh body");
        t3d_destroy_mesh(open);
    }
    t3d_destroy_mesh(cube);
    t3d_destroy_world(w);
}

int run()
{
    checkHullRest();
    checkMeshRest();
    checkHeightField();
    checkDynamicMesh();
    if (g_fail)
    {
        std::printf("avbd_test_vk_api3d_shapes: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("avbd_test_vk_api3d_shapes: all checks passed\n");
    return 0;
}
#undef CHECK
} // namespace shapes

namespace compound
{
static int g_fail = 0;
#define CHECK(cond, ...)                                                                                              \
    do                                                                                                                \
    {                                                                                                                 \
        if (!(cond))                                                                                                  \
        {                                                                                                             \
            std::printf("FAIL %s:%d: %s  ", __FILE__, __LINE__, #cond);                                               \
            std::printf(__VA_ARGS__);                                                                                 \
            std::printf("\n");                                                                                        \
            g_fail++;                                                                                                 \
        }                                                                                                             \
    } while (0)

static bool sameShape(Avbd3dShapeId a, Avbd3dShapeId b) { return a.index1 == b.index1 && a.generation == b.generation; }

static Avbd3dBodyId makeBody(Avbd3dWorld *w, Avbd3dBodyType type, Avbd3dVec3 p)
{
    Avbd3dBodyDef bd = avbd3d_default_body_def();
    bd.type = type;
    bd.position = p;
    return t3d_create_body(w, &bd);
}

static Avbd3dBodyId dropBox(Avbd3dWorld *w, float x)
{
    Avbd3dBodyId b = makeBody(w, AVBD3D_DYNAMIC_BODY, Avbd3dVec3{x, 1.0f, 0});
    Avbd3dShapeDef sd = avbd3d_default_shape_def();
    Avbd3dBoxHull h = avbd3d_make_box_hull(0.15f, 0.15f, 0.15f);
    t3d_create_hull_shape(b, &sd, avbd3d_box_hull_data(&h));
    return b;
}

static void run(Avbd3dWorld *w, float seconds)
{
    const int steps = (int)std::lround(seconds * 60.0f);
    for (int i = 0; i < steps; i++)
        t3d_step(w, 1.0f / 60.0f);
}

static bool rayHits(Avbd3dWorld *w, float x, Avbd3dShapeId *shape)
{
    Avbd3dRayResult r = t3d_world_cast_ray_closest(w, Avbd3dVec3{x, 5, 0}, Avbd3dVec3{0, -10, 0},
                                                     avbd3d_default_query_filter());
    if (shape)
        *shape = r.shapeId;
    return r.hit;
}

// f) A dynamic compound: three 1 m cubes (density 1) at x = 0, 2, 4. Mass = 3, centre (2, 0, 0),
// principal inertia diag(0.5, 8.5, 8.5) about the centre (3 * 1/6 about x; plus 2 * 4 about y and z).
// It falls on a static ground and settles without sinking or spinning.
static void checkDynamicCompound()
{
    Avbd3dWorldDef wd = avbd3d_default_world_def();
    Avbd3dWorld *w = t3d_create_world(&wd);
    Avbd3dBodyId ground = makeBody(w, AVBD3D_STATIC_BODY, Avbd3dVec3{0, -0.5f, 0});
    {
        Avbd3dShapeDef sd = avbd3d_default_shape_def();
        Avbd3dBoxHull g = avbd3d_make_box_hull(30.0f, 0.5f, 30.0f);
        t3d_create_hull_shape(ground, &sd, avbd3d_box_hull_data(&g));
    }

    Avbd3dBoxHull cube = avbd3d_make_box_hull(0.5f, 0.5f, 0.5f);
    Avbd3dCompoundHullDef hulls[3];
    for (int i = 0; i < 3; i++)
    {
        hulls[i].hull = avbd3d_box_hull_data(&cube);
        hulls[i].transform = t3d_transform_identity;
        hulls[i].transform.p = Avbd3dVec3{2.0f * (float)i, 0.0f, 0};
        hulls[i].material = avbd3d_default_surface_material();
    }
    Avbd3dCompoundDef cd = {};
    cd.hulls = hulls;
    cd.hullCount = 3;
    Avbd3dCompoundData *compound = t3d_create_compound(&cd);

    Avbd3dBodyId b = makeBody(w, AVBD3D_DYNAMIC_BODY, Avbd3dVec3{0, 3.0f, 0});
    Avbd3dShapeDef sd = avbd3d_default_shape_def();
    Avbd3dShapeId cid = t3d_create_baked_compound_shape(b, &sd, compound);
    CHECK(avbd3d_shape_is_valid(cid), "dynamic compound accepted");
    CHECK(std::fabs(t3d_body_get_mass(b) - 3.0f) < 1e-3f, "compound mass %.4f", t3d_body_get_mass(b));
    const Avbd3dVec3 c = t3d_body_get_local_center(b);
    CHECK(std::fabs(c.x - 2.0f) < 1e-3f && std::fabs(c.y) < 1e-3f && std::fabs(c.z) < 1e-3f,
          "compound centre %.4f %.4f %.4f", c.x, c.y, c.z);
    const Avbd3dMatrix3 I = t3d_body_get_local_rotational_inertia(b);
    CHECK(std::fabs(I.cx.x - 0.5f) < 5e-3f && std::fabs(I.cy.y - 8.5f) < 5e-3f && std::fabs(I.cz.z - 8.5f) < 5e-3f,
          "compound inertia %.4f %.4f %.4f", I.cx.x, I.cy.y, I.cz.z);

    // Sphere children have no mass until the compound recomputes it once after the loop.
    {
        Avbd3dCompoundSphereDef sph[2];
        for (int i = 0; i < 2; i++)
        {
            sph[i].sphere = Avbd3dSphere{Avbd3dVec3{i ? 1.0f : -1.0f, 0, 0}, 0.5f};
            sph[i].material = avbd3d_default_surface_material();
        }
        Avbd3dCompoundDef sd2 = {};
        sd2.spheres = sph;
        sd2.sphereCount = 2;
        Avbd3dCompoundData *sc = t3d_create_compound(&sd2);
        Avbd3dBodyId sb = makeBody(w, AVBD3D_DYNAMIC_BODY, Avbd3dVec3{50, 5, 0});
        CHECK(avbd3d_shape_is_valid(t3d_create_baked_compound_shape(sb, &sd, sc)), "sphere compound accepted");
        const float want = 2.0f * (4.0f / 3.0f) * 3.14159265f * 0.125f;
        CHECK(std::fabs(t3d_body_get_mass(sb) - want) < 2e-3f, "sphere compound mass %.4f want %.4f",
              t3d_body_get_mass(sb), want);
        t3d_destroy_body(sb);
        t3d_destroy_compound(sc);
    }

    run(w, 4.0f);
    const Avbd3dVec3 p = t3d_body_get_position(b);
    CHECK(std::fabs(p.y - 0.5f) < 0.05f, "compound rest height %.4f", p.y);
    const Avbd3dQuat q = t3d_body_get_rotation(b);
    CHECK(std::fabs(q.v.x) + std::fabs(q.v.y) + std::fabs(q.v.z) < 0.03f, "compound not tilted/spun (%.3f %.3f %.3f)",
          q.v.x, q.v.y, q.v.z);
    const Avbd3dVec3 a = t3d_body_get_angular_velocity(b);
    CHECK(std::fabs(a.x) + std::fabs(a.y) + std::fabs(a.z) < 0.05f, "compound angular velocity %.3f %.3f %.3f", a.x, a.y,
          a.z);
    t3d_destroy_compound(compound);
    t3d_destroy_world(w);
}

// g) Friction set on a compound reaches every child. The compound is a 1 m cube (child 1) plus a
// tiny box high above it (child 0, the id's first child), on a 30 degree slope. Friction 0 on the
// compound slides (the contact is child 1, not the first), friction 1 sticks.
static float slopeDrift(float friction)
{
    Avbd3dWorldDef wd = avbd3d_default_world_def();
    Avbd3dWorld *w = t3d_create_world(&wd);
    const float th = 30.0f * 3.14159265f / 180.0f;
    const Avbd3dQuat qs = {{0, 0, std::sin(0.5f * th)}, std::cos(0.5f * th)};
    const Avbd3dVec3 n = {-std::sin(th), std::cos(th), 0};  // slope normal
    const Avbd3dVec3 down = {-std::cos(th), -std::sin(th), 0}; // downhill (the slope rises toward +x)

    Avbd3dBodyDef gd = avbd3d_default_body_def();
    gd.type = AVBD3D_STATIC_BODY;
    gd.rotation = qs;
    Avbd3dBodyId slope = t3d_create_body(w, &gd);
    {
        Avbd3dShapeDef sd = avbd3d_default_shape_def();
        Avbd3dBoxHull g = avbd3d_make_box_hull(30.0f, 0.5f, 30.0f);
        t3d_create_hull_shape(slope, &sd, avbd3d_box_hull_data(&g));
    }
    // The surface is at +0.5 along the normal; the cube centre sits 0.5 above it.
    Avbd3dBodyDef bd = avbd3d_default_body_def();
    bd.type = AVBD3D_DYNAMIC_BODY;
    bd.rotation = qs;
    bd.position = Avbd3dVec3{n.x * 1.01f, n.y * 1.01f, 0};
    Avbd3dBodyId b = t3d_create_body(w, &bd);

    Avbd3dBoxHull big = avbd3d_make_box_hull(0.5f, 0.5f, 0.5f), tiny = avbd3d_make_box_hull(0.05f, 0.05f, 0.05f);
    Avbd3dCompoundHullDef hulls[2];
    hulls[0].hull = avbd3d_box_hull_data(&tiny); // child 0: tiny, high above the slope
    hulls[0].transform = t3d_transform_identity;
    hulls[0].transform.p = Avbd3dVec3{0, 1.5f, 0};
    hulls[1].hull = avbd3d_box_hull_data(&big); // child 1: the one that touches
    hulls[1].transform = t3d_transform_identity;
    for (auto &h : hulls)
        h.material = avbd3d_default_surface_material();
    Avbd3dCompoundDef cd = {};
    cd.hulls = hulls;
    cd.hullCount = 2;
    Avbd3dCompoundData *compound = t3d_create_compound(&cd);
    Avbd3dShapeDef sd = avbd3d_default_shape_def();
    Avbd3dShapeId cid = t3d_create_baked_compound_shape(b, &sd, compound);
    CHECK(avbd3d_shape_is_valid(cid), "slope compound");
    t3d_shape_set_friction(cid, friction);

    const Avbd3dVec3 start = t3d_body_get_position(b);
    run(w, 1.0f);
    const Avbd3dVec3 end = t3d_body_get_position(b);
    const float d = (end.x - start.x) * down.x + (end.y - start.y) * down.y;
    t3d_destroy_compound(compound);
    t3d_destroy_world(w);
    return d;
}

static void checkCompoundFriction()
{
    // The ground's own friction is 0.6; the mix is the geometric mean, so friction 1 gives
    // sqrt(0.6) = 0.77 > tan 30 = 0.58 (sticks) and friction 0 gives 0 (slides ~2.5 m in 1 s).
    const float slide = slopeDrift(0.0f), stick = slopeDrift(1.0f);
    CHECK(slide > 1.0f, "friction 0 on the compound slides, drift %.3f", slide);
    CHECK(stick < 0.15f, "friction 1 on the compound sticks, drift %.3f", stick);
}

int run()
{
    Avbd3dWorldDef wd = avbd3d_default_world_def();
    Avbd3dWorld *w = t3d_create_world(&wd);

    Avbd3dBoxHull cube = avbd3d_make_box_hull(0.5f, 0.5f, 0.5f);
    Avbd3dCompoundHullDef hulls[3];
    const float xs[3] = {-2.0f, 0.0f, 2.0f};
    for (int i = 0; i < 3; i++)
    {
        hulls[i].hull = avbd3d_box_hull_data(&cube);
        hulls[i].transform = t3d_transform_identity;
        hulls[i].transform.p = Avbd3dVec3{xs[i], -0.5f, 0};
        hulls[i].material = avbd3d_default_surface_material();
    }
    Avbd3dCompoundDef cd = {};
    cd.hulls = hulls;
    cd.hullCount = 3;
    Avbd3dCompoundData *compound = t3d_create_compound(&cd);
    CHECK(compound != nullptr, "compound baked");

    checkDynamicCompound();
    checkCompoundFriction();

    Avbd3dBodyId ground = makeBody(w, AVBD3D_STATIC_BODY, Avbd3dVec3{0, 0, 0});
    Avbd3dShapeDef sd = avbd3d_default_shape_def();
    Avbd3dShapeId cid = t3d_create_baked_compound_shape(ground, &sd, compound);
    CHECK(avbd3d_shape_is_valid(cid), "compound shape created");
    CHECK(t3d_body_get_shape_count(ground) == 1, "one shape listed, got %d", t3d_body_get_shape_count(ground));

    // d) a dynamic body takes a compound too (checked in detail by checkDynamicCompound).
    Avbd3dBodyId dyn = makeBody(w, AVBD3D_DYNAMIC_BODY, Avbd3dVec3{0, 20, 0});
    CHECK(avbd3d_shape_is_valid(t3d_create_baked_compound_shape(dyn, &sd, compound)), "dynamic body accepted");
    t3d_destroy_body(dyn);

    Avbd3dBodyId last = dropBox(w, 2.0f), mid = dropBox(w, 0.0f), gap = dropBox(w, 1.0f);
    run(w, 3.0f);
    // a) the box half-extent is 0.15 and the tops are y = 0.
    CHECK(std::fabs(t3d_body_get_position(last).y - 0.15f) < 0.03f, "last child rest %.4f",
          t3d_body_get_position(last).y);
    CHECK(std::fabs(t3d_body_get_position(mid).y - 0.15f) < 0.03f, "mid child rest %.4f",
          t3d_body_get_position(mid).y);
    CHECK(t3d_body_get_position(gap).y < -1.0f, "gap box falls through, y %.4f", t3d_body_get_position(gap).y);

    // b) rays return the compound id for any child.
    Avbd3dShapeId hit{};
    CHECK(rayHits(w, 2.3f, &hit) && sameShape(hit, cid), "ray on the last child returns the compound id");
    CHECK(rayHits(w, -2.0f, &hit) && sameShape(hit, cid), "ray on the first child returns the compound id");
    {
        Avbd3dRayResult r = t3d_world_cast_ray_closest(w, Avbd3dVec3{1.0f, 0.5f, 0}, Avbd3dVec3{0, 0.5f, 0},
                                                         avbd3d_default_query_filter());
        CHECK(!r.hit, "no hit in the gap above the compound");
    }

    // c) the AABB spans x from -2.5 to 2.5.
    Avbd3dAABB bb = t3d_shape_get_aabb(cid);
    CHECK(bb.lowerBound.x < -2.4f && bb.upperBound.x > 2.4f, "compound aabb %.3f..%.3f", bb.lowerBound.x, bb.upperBound.x);

    // e) destroy removes every child.
    t3d_destroy_shape(cid, true);
    CHECK(!avbd3d_shape_is_valid(cid), "destroyed id invalid");
    CHECK(t3d_body_get_shape_count(ground) == 0, "no shapes left");
    CHECK(!rayHits(w, 2.3f, nullptr), "last child gone");
    {
        Avbd3dRayResult r = t3d_world_cast_ray_closest(w, Avbd3dVec3{-2, 5, 0}, Avbd3dVec3{0, -10, 0},
                                                         avbd3d_default_query_filter());
        CHECK(!r.hit, "first child gone");
    }
    Avbd3dBodyId after = dropBox(w, 2.0f);
    run(w, 1.5f);
    CHECK(t3d_body_get_position(after).y < -1.0f, "box falls once the compound is destroyed, y %.4f",
          t3d_body_get_position(after).y);

    t3d_destroy_compound(compound);
    t3d_destroy_world(w);
    if (g_fail == 0)
        std::printf("avbd_test_vk_api3d_compound: PASS\n");
    return g_fail == 0 ? 0 : 1;
}
#undef CHECK
} // namespace compound

int main()
{
    int failed = 0;
    failed |= shapes::run() != 0;
    failed |= compound::run() != 0;
    return failed ? 1 : 0;
}
