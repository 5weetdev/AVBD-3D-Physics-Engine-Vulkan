// Gate for the avbd3d world queries (src/api/avbd3d_query.cpp): ray and closest-ray casts,
// filter exclusion, AABB and shape overlaps, a shape cast, a dynamic body's proxy after motion,
// a mover cast into a wall, mover collision with the ground, debug draw counts, and the shape
// ray cast. Y-up, the same convention as test_vk_api3d.cpp.

#include "test_util.h"

#include <cmath>
#include <cstdint>
#include <cstdio>

static int g_fail = 0;
#define CHECK(cond, ...)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            std::printf("FAIL %s:%d: %s  ", __FILE__, __LINE__, #cond);                                                \
            std::printf(__VA_ARGS__);                                                                                  \
            std::printf("\n");                                                                                         \
            g_fail++;                                                                                                  \
        }                                                                                                              \
    } while (0)

static Avbd3dBodyDef bodyDef(Avbd3dBodyType type, Avbd3dVec3 p)
{
    Avbd3dBodyDef bd = avbd3d_default_body_def();
    bd.type = type;
    bd.position = p;
    return bd;
}

static Avbd3dBodyId makeBody(Avbd3dWorld *w, Avbd3dBodyType type, Avbd3dVec3 p)
{
    Avbd3dBodyDef bd = bodyDef(type, p);
    return t3d_create_body(w, &bd);
}

static Avbd3dBodyId makeBodyBox(Avbd3dWorld *w, const Avbd3dBodyDef &bd, float hx, float hy, float hz)
{
    Avbd3dBodyId b = t3d_create_body(w, &bd);
    Avbd3dShapeDef sd = avbd3d_default_shape_def();
    Avbd3dBoxHull hull = avbd3d_make_box_hull(hx, hy, hz);
    t3d_create_hull_shape(b, &sd, avbd3d_box_hull_data(&hull));
    return b;
}

static void step(Avbd3dWorld *w, int n = 1)
{
    for (int i = 0; i < n; i++)
        t3d_step(w, 1.0f / 60.0f);
}

static bool near(float a, float b, float tol)
{
    return std::fabs(a - b) <= tol;
}

// Angle (rad) of the rotation taking b to a, from the relative quaternion (well conditioned near 0, unlike acos(dot)).
static double angleBetween(Avbd3dQuat a, Avbd3dQuat b)
{
    // a * conj(b)
    const double x = (double)a.s * -b.v.x + (double)a.v.x * b.s + (double)a.v.y * -b.v.z - (double)a.v.z * -b.v.y;
    const double y = (double)a.s * -b.v.y + (double)a.v.y * b.s + (double)a.v.z * -b.v.x - (double)a.v.x * -b.v.z;
    const double z = (double)a.s * -b.v.z + (double)a.v.z * b.s + (double)a.v.x * -b.v.y - (double)a.v.y * -b.v.x;
    const double s = (double)a.s * b.s + (double)a.v.x * b.v.x + (double)a.v.y * b.v.y + (double)a.v.z * b.v.z;
    return 2.0 * std::atan2(std::sqrt(x * x + y * y + z * z), std::fabs(s));
}

// ---- callbacks -----------------------------------------------------------------------------

static int g_overlapCount = 0;
static Avbd3dShapeId g_overlapLast{};
static int32_t overlapFcn(Avbd3dShapeId s, void *)
{
    g_overlapCount++;
    g_overlapLast = s;
    return true;
}

static Avbd3dShapeId g_castShape{};
static float g_castFraction = -1.0f;
static float castFcn(Avbd3dShapeId s, Avbd3dVec3, Avbd3dVec3, float fraction, uint64_t, int, int, void *)
{
    g_castShape = s;
    g_castFraction = fraction;
    return fraction; // clip the search to this hit
}

static int g_planeCalls = 0;
static float g_planeNormalY = -2.0f;
static float g_planeOffset = 0.0f; // offset of the plane with the largest normal y
static int32_t planeFcn(Avbd3dShapeId, const Avbd3dPlaneResult *planes, int count, void *)
{
    g_planeCalls++;
    for (int i = 0; i < count; i++)
        if (planes[i].plane.normal.y > g_planeNormalY)
        {
            g_planeNormalY = planes[i].plane.normal.y;
            g_planeOffset = planes[i].plane.offset;
        }
    return true;
}

static float g_moverFraction = -1.0f;

static int g_drawShapes = 0;
static void drawShape(void *, Avbd3dTransform, Avbd3dHexColor, void *)
{
    g_drawShapes++;
}

int main()
{
    Avbd3dWorldDef wd = avbd3d_default_world_def();
    Avbd3dWorld *w = t3d_create_world(&wd);
    if (!(w != NULL))
    {
        std::printf("FAIL: world create\n");
        return 1;
    }

    // Feature status: every query feature this gate covers is supported.

    // ---- static scene -----------------------------------------------------------------------
    // Ground: top face at y = 0, spanning x, z in [-10, 10].
    makeBodyBox(w, bodyDef(AVBD3D_STATIC_BODY, {0, -0.5f, 0}), 10, 0.5f, 10);

    // Sphere at (5, 1, 0), radius 0.5, top at y = 1.5. Category 2 so a filter can exclude it.
    Avbd3dBodyId sphereBody = makeBody(w, AVBD3D_STATIC_BODY, {5, 1, 0});
    Avbd3dShapeDef sphereDef = avbd3d_default_shape_def();
    sphereDef.filter.categoryBits = 2;
    Avbd3dSphere sph{{0, 0, 0}, 0.5f};
    Avbd3dShapeId sphereShape = t3d_create_sphere_shape(sphereBody, &sphereDef, &sph);
    CHECK(sphereShape.index1 != 0, "sphere shape created");

    // Capsule along y at x = -5: centres y 0.5..1.5, radius 0.25, top at y = 1.75.
    Avbd3dBodyId capBody = makeBody(w, AVBD3D_STATIC_BODY, {-5, 0, 0});
    Avbd3dShapeDef capDef = avbd3d_default_shape_def();
    Avbd3dCapsule cap{{0, 0.5f, 0}, {0, 1.5f, 0}, 0.25f}; // body-local, body at x = -5
    Avbd3dShapeId capShape = t3d_create_capsule_shape(capBody, &capDef, &cap);
    CHECK(capShape.index1 != 0, "capsule shape created");

    // Wall: x in [2, 3], y in [0, 4], z in [-2, 2]. Its face at x = 2.
    makeBodyBox(w, bodyDef(AVBD3D_STATIC_BODY, {2.5f, 2, 0}), 0.5f, 2, 2);

    // ---- closest ray: ground ----------------------------------------------------------------
    Avbd3dQueryFilter all = avbd3d_default_query_filter();
    Avbd3dRayResult r = t3d_world_cast_ray_closest(w, {0, 10, 0}, {0, -20, 0}, all);
    CHECK(r.hit, "ray down hits the ground");
    CHECK(near(r.point.y, 0.0f, 1e-3f), "ground hit y=%g (want 0)", r.point.y);
    CHECK(near(r.normal.y, 1.0f, 1e-3f), "ground normal y=%g (want 1)", r.normal.y);
    CHECK(near(r.fraction, 0.5f, 1e-3f), "ground fraction %g (want 0.5)", r.fraction);

    // ---- closest ray: sphere in front of the ground -----------------------------------------
    r = t3d_world_cast_ray_closest(w, {5, 10, 0}, {0, -20, 0}, all);
    CHECK(r.hit && r.shapeId.index1 == sphereShape.index1, "ray hits the sphere shape");
    CHECK(near(r.point.y, 1.5f, 1e-3f), "sphere top y=%g (want 1.5)", r.point.y);
    CHECK(near(r.fraction, (10.0f - 1.5f) / 20.0f, 1e-3f), "sphere fraction %g", r.fraction);

    // ---- closest ray: capsule top -----------------------------------------------------------
    r = t3d_world_cast_ray_closest(w, {-5, 10, 0}, {0, -20, 0}, all);
    CHECK(r.hit && r.shapeId.index1 == capShape.index1, "ray hits the capsule shape");
    CHECK(near(r.point.y, 1.75f, 1e-3f), "capsule top y=%g (want 1.75)", r.point.y);

    // ---- filter exclusion: maskBits 1 rejects the category-2 sphere, so the ground is hit ---
    Avbd3dQueryFilter onlyOne = all;
    onlyOne.maskBits = 1;
    r = t3d_world_cast_ray_closest(w, {5, 10, 0}, {0, -20, 0}, onlyOne);
    CHECK(r.hit && near(r.point.y, 0.0f, 1e-3f), "filter skips the sphere, ground hit y=%g", r.point.y);
    Avbd3dQueryFilter nothing = all;
    nothing.maskBits = 0;
    r = t3d_world_cast_ray_closest(w, {0, 10, 0}, {0, -20, 0}, nothing);
    CHECK(!r.hit, "maskBits 0 hits nothing");

    // Shape ray cast of the sphere: origin above it, same hit as the world query.
    Avbd3dCastOutput so = t3d_shape_ray_cast(sphereShape, {5, 10, 0}, {0, -20, 0});
    CHECK(so.hit && near(so.point.y, 1.5f, 1e-3f), "Shape_RayCast sphere hit y=%g", so.point.y);
    Avbd3dCastOutput miss = t3d_shape_ray_cast(sphereShape, {0, 10, 0}, {0, -20, 0});
    CHECK(!miss.hit, "Shape_RayCast misses the sphere off to the side");

    // ---- overlaps ---------------------------------------------------------------------------
    g_overlapCount = 0;
    Avbd3dAABB box{{4.2f, 0.4f, -0.6f}, {5.8f, 1.6f, 0.6f}};
    t3d_world_overlap_aabb(w, box, all, overlapFcn, nullptr);
    CHECK(g_overlapCount == 1 && g_overlapLast.index1 == sphereShape.index1, "AABB overlap finds only the sphere (%d)",
          g_overlapCount);

    Avbd3dVec3 pt{5, 1, 0};
    Avbd3dShapeProxy probe{&pt, 1, 0.1f};
    g_overlapCount = 0;
    t3d_world_overlap_shape(w, {0, 0, 0}, &probe, all, overlapFcn, nullptr);
    CHECK(g_overlapCount == 1 && g_overlapLast.index1 == sphereShape.index1, "shape overlap finds the sphere (%d)",
          g_overlapCount);

    // ---- shape cast: a small sphere proxy straight down onto the ground ---------------------
    Avbd3dVec3 start{0, 5, 0};
    Avbd3dShapeProxy drop{&start, 1, 0.1f};
    g_castShape = t3d_null_shape;
    g_castFraction = -1.0f;
    t3d_world_cast_shape(w, {0, 0, 0}, &drop, {0, -10, 0}, all, castFcn, nullptr);
    CHECK(g_castShape.index1 != 0, "shape cast hits something");
    CHECK(near(g_castFraction, 0.49f, 0.01f), "shape cast fraction %g (want 0.49)", g_castFraction);

    // ---- dynamic body: a falling sphere, so the tree must refit it -------------------------
    Avbd3dBodyId fall = makeBody(w, AVBD3D_DYNAMIC_BODY, {-2, 10, 0});
    Avbd3dShapeDef fallDef = avbd3d_default_shape_def();
    Avbd3dSphere fs{{0, 0, 0}, 0.5f};
    Avbd3dShapeId fallShape = t3d_create_sphere_shape(fall, &fallDef, &fs);
    step(w, 60);
    Avbd3dVec3 fp = t3d_body_get_position(fall);
    CHECK(fp.y < 9.5f && fp.y > 3.0f, "falling sphere y=%g after 60 steps", fp.y);
    r = t3d_world_cast_ray_closest(w, {-2, 20, 0}, {0, -30, 0}, all);
    CHECK(r.hit && r.shapeId.index1 == fallShape.index1, "ray finds the moved sphere (hit=%d shape=%d)", r.hit,
          r.shapeId.index1);
    CHECK(near(r.point.y, fp.y + 0.5f, 0.2f), "moved sphere top y=%g vs body y=%g", r.point.y, fp.y);

    // ---- mover cast: a capsule slides toward the wall at x = 2 ------------------------------
    Avbd3dCapsule mover{{0, 0.5f, 0}, {0, 1.5f, 0}, 0.25f};
    g_moverFraction = t3d_world_cast_mover(w, {0, 0, 0}, &mover, {6, 0, 0}, all, nullptr, nullptr);
    CHECK(g_moverFraction < 1.0f, "mover cast stops before the wall (fraction %g)", g_moverFraction);
    CHECK(near(g_moverFraction, 1.75f / 6.0f, 0.02f), "mover fraction %g (want %g)", g_moverFraction, 1.75f / 6.0f);
    float free = t3d_world_cast_mover(w, {0, 0, 0}, &mover, {0, 0, 1}, all, nullptr, nullptr);
    CHECK(near(free, 1.0f, 1e-4f), "mover sideways misses the wall (fraction %g)", free);

    // ---- mover collision: a capsule whose bottom is in the ground ---------------------------
    Avbd3dCapsule sink{{0, 0.1f, 0}, {0, 1.1f, 0}, 0.25f};
    g_planeCalls = 0;
    g_planeNormalY = -2.0f;
    t3d_world_collide_mover(w, {0, 0, 0}, &sink, all, planeFcn, nullptr);
    CHECK(g_planeCalls >= 1, "collide mover reports the ground");
    CHECK(g_planeNormalY > 0.9f, "ground plane normal y=%g (want ~1)", g_planeNormalY);
    // Box3D's plane offset is the penetration depth (b3SolvePlanes reads dot(n, delta) - offset
    // with delta from the mover's current place): bottom at 0.1 - 0.25 is 0.15 into the ground.
    CHECK(near(g_planeOffset, 0.15f, 2e-3f), "ground plane offset %g (want penetration 0.15)", g_planeOffset);

    // ---- debug draw: one DrawShapeFcn call per shape ----------------------------------------
    Avbd3dDebugDraw draw = avbd3d_default_debug_draw();
    draw.drawShapes = true;
    draw.drawBounds = false;
    draw.DrawShapeFcn = drawShape;
    draw.drawingBounds = Avbd3dAABB{{-100, -100, -100}, {100, 100, 100}};
    g_drawShapes = 0;
    t3d_world_draw(w, &draw, UINT64_MAX);
    // Ground, sphere, capsule, wall and the falling sphere.
    CHECK(g_drawShapes == 5, "debug draw called %d times (want 5)", g_drawShapes);

    // ---- per-body queries (Box3D b3Body_*) --------------------------------------------------
    {
        Avbd3dWorld *w2 = t3d_create_world(&wd);
        const Avbd3dQueryFilter qf = avbd3d_default_query_filter();
        Avbd3dBodyId A = makeBodyBox(w2, bodyDef(AVBD3D_STATIC_BODY, {0, 1, 0}), 1, 1, 1);
        Avbd3dBodyId B = makeBodyBox(w2, bodyDef(AVBD3D_STATIC_BODY, {0, 1, -5}), 1, 1, 1);
        const Avbd3dTransform idA{{0, 1, 0}, {{0, 0, 0}, 1}};
        const Avbd3dTransform idB{{0, 1, -5}, {{0, 0, 0}, 1}};

        // The world ray stops at B (z = -6); the body ray at A ignores B and hits A (z = -1).
        Avbd3dRayResult wr = t3d_world_cast_ray_closest(w2, {0, 1, -10}, {0, 0, 20}, qf);
        CHECK(wr.hit && near(wr.fraction, 4.0f / 20.0f, 1e-4f), "world ray hits B first (%g)", wr.fraction);
        Avbd3dBodyCastResult ra = t3d_body_cast_ray(A, {0, 1, -10}, {0, 0, 20}, qf, 1.0f, idA);
        CHECK(ra.hit && near(ra.fraction, 9.0f / 20.0f, 1e-4f) && near(ra.point.z, -1.0f, 1e-3f) &&
                  near(ra.normal.z, -1.0f, 1e-4f),
              "body ray hits A only: hit %d f %g z %g", (int)ra.hit, ra.fraction, ra.point.z);
        Avbd3dShapeId shapesA[2];
        CHECK(t3d_body_get_shapes(A, shapesA, 2) == 1 && ra.shapeId.index1 == shapesA[0].index1,
              "body ray shape id is A's");
        Avbd3dBodyCastResult rb = t3d_body_cast_ray(B, {0, 1, -10}, {0, 0, 20}, qf, 1.0f, idB);
        CHECK(rb.hit && near(rb.fraction, 4.0f / 20.0f, 1e-4f), "body ray hits B (%g)", rb.fraction);
        // A ray that misses A, and a maxFraction that stops short of it.
        CHECK(!t3d_body_cast_ray(A, {5, 1, -10}, {0, 0, 20}, qf, 1.0f, idA).hit, "body ray miss");
        CHECK(!t3d_body_cast_ray(A, {0, 1, -10}, {0, 0, 20}, qf, 0.3f, idA).hit, "body ray maxFraction");
        // The supplied body transform replaces the body's own: A moved to z = 10 is hit at z = 9.
        Avbd3dBodyCastResult rm =
            t3d_body_cast_ray(A, {0, 1, -10}, {0, 0, 30}, qf, 1.0f, Avbd3dTransform{{0, 1, 10}, {{0, 0, 0}, 1}});
        CHECK(rm.hit && near(rm.point.z, 9.0f, 1e-3f), "body ray at a given transform (z %g)", rm.point.z);

        // Shape cast: a sphere (r 0.5) along +z from z = -10 touches A's face at z = -1 with its centre at -1.5.
        Avbd3dVec3 sc[1] = {{0, 0, 0}};
        Avbd3dShapeProxy sp{sc, 1, 0.5f};
        Avbd3dBodyCastResult cs = t3d_body_cast_shape(A, {0, 1, -10}, &sp, {0, 0, 20}, qf, 1.0f, false, idA);
        CHECK(cs.hit && near(cs.fraction, 8.5f / 20.0f, 2e-3f), "body shape cast fraction %g (want 0.425)",
              cs.fraction);

        // Overlap: a sphere at A's centre overlaps A and not B; at x = 3 it misses, unless A is moved there.
        Avbd3dVec3 pc[1] = {{0, 0, 0}};
        Avbd3dShapeProxy pr{pc, 1, 0.5f};
        CHECK(t3d_body_overlap_shape(A, {0, 1, 0}, &pr, qf, idA), "body overlap A");
        CHECK(!t3d_body_overlap_shape(B, {0, 1, 0}, &pr, qf, idB), "body overlap B (no)");
        CHECK(!t3d_body_overlap_shape(A, {3, 1, 0}, &pr, qf, idA), "body overlap A far (no)");
        CHECK(t3d_body_overlap_shape(A, {3, 1, 0}, &pr, qf, Avbd3dTransform{{3, 1, 0}, {{0, 0, 0}, 1}}),
              "body overlap at a given transform");
        CHECK(t3d_body_overlap_shape(A, {1.4f, 1, 0}, &pr, qf, idA), "body overlap by radius");
        CHECK(!t3d_body_overlap_shape(A, {1.6f, 1, 0}, &pr, qf, idA), "body overlap past radius (no)");

        // Closest point on the unit box (half 1 at (0,1,0)).
        Avbd3dVec3 cp{};
        float dist = t3d_body_get_closest_point(A, &cp, {4, 1, 0});
        CHECK(near(dist, 3.0f, 1e-4f) && near(cp.x, 1.0f, 1e-4f) && near(cp.y, 1.0f, 1e-4f) && near(cp.z, 0.0f, 1e-4f),
              "closest point side: d %g (%g %g %g)", dist, cp.x, cp.y, cp.z);
        dist = t3d_body_get_closest_point(A, &cp, {3, 5, 0});
        CHECK(near(dist, std::sqrt(4.0f + 9.0f), 1e-4f) && near(cp.x, 1.0f, 1e-4f) && near(cp.y, 2.0f, 1e-4f),
              "closest point edge: d %g (%g %g %g)", dist, cp.x, cp.y, cp.z);
        dist = t3d_body_get_closest_point(A, &cp, {0.2f, 1, 0});
        CHECK(dist == 0.0f, "closest point inside d %g", dist);

        // Mover colliding with A's +x face: capsule axis along y at x = 1.2, radius 0.25 penetrates 0.05.
        Avbd3dCapsule mv{{1.2f, 0.5f, 0}, {1.2f, 1.5f, 0}, 0.25f};
        Avbd3dBodyPlaneResult planes[4];
        int np = t3d_body_collide_mover(A, planes, 4, {0, 0, 0}, &mv, qf, idA);
        CHECK(np == 1 && planes[0].result.plane.normal.x > 0.99f && near(planes[0].result.plane.offset, 0.05f, 2e-3f),
              "body mover: n %d nx %g off %g", np, planes[0].result.plane.normal.x, planes[0].result.plane.offset);
        CHECK(t3d_body_collide_mover(B, planes, 4, {0, 0, 0}, &mv, qf, idB) == 0, "body mover B (no)");

        // ---- SetTargetTransform: a kinematic platform driven along a circle, yawing ---------------
        const float dt = 1.0f / 60.0f;
        const float R = 2.0f, om = 1.0f; // 2 m circle at 1 rad/s: 2 m/s
        Avbd3dBodyId plat = makeBodyBox(w2, bodyDef(AVBD3D_KINEMATIC_BODY, {R, 10, 0}), 2.0f, 0.1f, 2.0f);
        Avbd3dVec3 axis{0.3f, 1.0f, 0.2f};
        const float al = std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
        axis = {axis.x / al, axis.y / al, axis.z / al};
        // The rider sits on the platform top (y = 10.1).
        Avbd3dBodyId rider = makeBodyBox(w2, bodyDef(AVBD3D_DYNAMIC_BODY, {R, 10.1f + 0.25f, 0}), 0.25f, 0.25f, 0.25f);
        double worstPos = 0.0, worstRot = 0.0;
        for (int i = 1; i <= 240; i++)
        {
            const float t = i * dt;
            Avbd3dTransform target{{R * std::cos(om * t), 10, R * std::sin(om * t)},
                                   t3d_quat_axis_angle({0, 1, 0}, 0.5f * om * t)};
            t3d_body_set_target_transform(plat, target, dt, true);
            step(w2);
            const Avbd3dTransform got = t3d_body_get_transform(plat);
            const double dx = got.p.x - target.p.x, dy = got.p.y - target.p.y, dz = got.p.z - target.p.z;
            worstPos = std::fmax(worstPos, std::sqrt(dx * dx + dy * dy + dz * dz));
            worstRot = std::fmax(worstRot, angleBetween(got.q, target.q));
        }
        CHECK(worstPos < 1e-3, "platform position error %g (want < 1e-3)", worstPos);
        CHECK(worstRot < 1e-3, "platform rotation error %g rad (want < 1e-3)", worstRot);
        const Avbd3dVec3 pp = t3d_body_get_position(plat);
        const Avbd3dVec3 rp = t3d_body_get_position(rider);
        const double ddx = rp.x - pp.x, ddz = rp.z - pp.z;
        CHECK(std::sqrt(ddx * ddx + ddz * ddz) < 1.5 && near(rp.y, 10.35f, 0.1f), "rider carried: offset (%g %g) y %g",
              ddx, ddz, rp.y);

        // A kinematic body lands on a target that rotates about a general axis.
        Avbd3dBodyId spin = makeBodyBox(w2, bodyDef(AVBD3D_KINEMATIC_BODY, {-20, 30, 0}), 1, 1, 1);
        double worstSpin = 0.0;
        for (int i = 1; i <= 120; i++)
        {
            const float t = i * dt;
            Avbd3dTransform target{{-20.0f + 0.5f * t, 30, 0}, t3d_quat_axis_angle(axis, 1.5f * t)};
            t3d_body_set_target_transform(spin, target, dt, true);
            step(w2);
            const Avbd3dTransform got = t3d_body_get_transform(spin);
            worstSpin = std::fmax(worstSpin, angleBetween(got.q, target.q));
            CHECK(near(got.p.x, target.p.x, 1e-3f), "spin body x %g vs %g", got.p.x, target.p.x);
        }
        CHECK(worstSpin < 1e-3, "tilted rotation error %g rad", worstSpin);

        // A static body and timeStep <= 0 are ignored.
        t3d_body_set_target_transform(A, Avbd3dTransform{{9, 9, 9}, {{0, 0, 0}, 1}}, dt, true);
        step(w2);
        CHECK(near(t3d_body_get_position(A).x, 0.0f, 1e-6f), "static body ignores a target");
        t3d_body_set_target_transform(spin, Avbd3dTransform{{0, 0, 0}, {{0, 0, 0}, 1}}, 0.0f, true);
        step(w2);
        CHECK(t3d_body_get_position(spin).y > 29.9f, "timeStep 0 ignored");
        t3d_destroy_world(w2);
    }

    t3d_destroy_world(w);
    if (g_fail == 0)
        std::printf("avbd_test_vk_api3d_query: all checks passed\n");
    else
        std::printf("avbd_test_vk_api3d_query: %d failure(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
