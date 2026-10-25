// avbd3d world queries: ray, shape and mover casts, overlaps, mover collision, debug draw.
//
// The shape-level math is Box3D's (per-type ray, shape cast, overlap and mover functions from
// box3d/collision.h and the mover functions in external/box3d/src/shape.h), compiled unmodified
// into avbd3d_box3d. The traversal and the per-shape dispatch follow Box3D's physics_world.c and
// shape.c: the tree is traversed in world float space, and every per-shape test is done in a
// frame that is relative to the query origin, so results stay precise far from the world origin.
#include "avbd3d_world.h"

#include "box3d/collision.h"
#include "box3d/math_functions.h"
#include "box3d/types.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstdint>
#include <iterator>
#include <vector>

// Non-public Box3D mover functions (external/box3d/src/shape.h). They take a shape in its own
// local frame and a mover in the same frame, and return the plane count.
extern "C" int b3CollideMoverAndSphere(b3PlaneResult *result, const b3Sphere *shape, const b3Capsule *mover);
extern "C" int b3CollideMoverAndCapsule(b3PlaneResult *result, const b3Capsule *shape, const b3Capsule *mover);
extern "C" int b3CollideMoverAndHull(b3PlaneResult *result, const b3HullData *shape, const b3Capsule *mover);
extern "C" int b3CollideMoverAndMesh(b3PlaneResult *planes, int capacity, const b3Mesh *shape, const b3Capsule *mover);
extern "C" int b3CollideMoverAndHeightField(b3PlaneResult *results, int capacity, const b3HeightFieldData *shape,
                                            const b3Capsule *mover);
// Non-public Box3D projected area of a hull (external/box3d/src/hull.c), used by explosions.
extern "C" float b3ComputeHullProjectedArea(const b3HullData *hull, b3Vec3 direction);

using namespace avbd3d_impl;

namespace
{
constexpr int kMaxPlanesPerShape = 64;

b3Vec3 toB(avbd3d_Vec3 v) { return b3Vec3{v.x, v.y, v.z}; }
avbd3d_Vec3 fromB(b3Vec3 v) { return avbd3d_Vec3{v.x, v.y, v.z}; }

// Box3D's shape filter rule: the shape and the query must each accept the other.
// Sensors are invisible to queries, as in Box3D.
bool shouldQuery(const ShapeRec &r, avbd3d_QueryFilter f)
{
    if (r.isSensor)
        return false;
    return (r.filter.categoryBits & f.maskBits) != 0 && (r.filter.maskBits & f.categoryBits) != 0;
}

// Shape transform relative to the query origin.
b3Transform relTransform(const World &w, int s, avbd3d_Pos origin)
{
    b3Transform xf = w.shapeWorldTransform(s);
    xf.p = b3Sub(xf.p, toB(origin));
    return xf;
}

b3CastOutput rayCastLocal(const ShapeRec &r, const b3RayCastInput &in)
{
    const b3Mesh meshView = r.geom.mesh ? r.geom.meshView() : b3Mesh{nullptr, b3Vec3{1.0f, 1.0f, 1.0f}};
    switch (r.type)
    {
        case avbd3d_sphereShape:
            return b3RayCastSphere(&r.geom.sphere, &in);
        case avbd3d_capsuleShape:
            return b3RayCastCapsule(&r.geom.capsule, &in);
        case avbd3d_meshShape:
            return b3RayCastMesh(&meshView, &in);
        case avbd3d_heightShape:
            return b3RayCastHeightField(r.geom.heightField.get(), &in);
        default:
            return b3RayCastHull(r.geom.hullData(), &in);
    }
}

b3CastOutput shapeCastLocal(const ShapeRec &r, const b3ShapeCastInput &in)
{
    const b3Mesh meshView = r.geom.mesh ? r.geom.meshView() : b3Mesh{nullptr, b3Vec3{1.0f, 1.0f, 1.0f}};
    switch (r.type)
    {
        case avbd3d_sphereShape:
            return b3ShapeCastSphere(&r.geom.sphere, &in);
        case avbd3d_capsuleShape:
            return b3ShapeCastCapsule(&r.geom.capsule, &in);
        case avbd3d_meshShape:
            return b3ShapeCastMesh(&meshView, &in);
        case avbd3d_heightShape:
            return b3ShapeCastHeightField(r.geom.heightField.get(), &in);
        default:
            return b3ShapeCastHull(r.geom.hullData(), &in);
    }
}

bool overlapLocal(const ShapeRec &r, b3Transform xf, const b3ShapeProxy *proxy)
{
    const b3Mesh meshView = r.geom.mesh ? r.geom.meshView() : b3Mesh{nullptr, b3Vec3{1.0f, 1.0f, 1.0f}};
    switch (r.type)
    {
        case avbd3d_sphereShape:
            return b3OverlapSphere(&r.geom.sphere, xf, proxy);
        case avbd3d_capsuleShape:
            return b3OverlapCapsule(&r.geom.capsule, xf, proxy);
        case avbd3d_meshShape:
            return b3OverlapMesh(&meshView, xf, proxy);
        case avbd3d_heightShape:
            return b3OverlapHeightField(r.geom.heightField.get(), xf, proxy);
        default:
            return b3OverlapHull(r.geom.hullData(), xf, proxy);
    }
}

int collideMoverLocal(const ShapeRec &r, b3PlaneResult *out, const b3Capsule *mover)
{
    switch (r.type)
    {
        case avbd3d_sphereShape:
            return b3CollideMoverAndSphere(out, &r.geom.sphere, mover);
        case avbd3d_capsuleShape:
            return b3CollideMoverAndCapsule(out, &r.geom.capsule, mover);
        case avbd3d_meshShape:
        {
            const b3Mesh meshView = r.geom.meshView();
            return b3CollideMoverAndMesh(out, kMaxPlanesPerShape, &meshView, mover);
        }
        case avbd3d_heightShape:
            return b3CollideMoverAndHeightField(out, kMaxPlanesPerShape, r.geom.heightField.get(), mover);
        default:
            return b3CollideMoverAndHull(out, r.geom.hullData(), mover);
    }
}

b3AABB shapeAABB(const ShapeRec &r, b3Transform xf)
{
    switch (r.type)
    {
        case avbd3d_sphereShape:
            return b3ComputeSphereAABB(&r.geom.sphere, xf);
        case avbd3d_capsuleShape:
            return b3ComputeCapsuleAABB(&r.geom.capsule, xf);
        case avbd3d_meshShape:
            return b3ComputeMeshAABB(r.geom.mesh.get(), xf, b3Vec3{1.0f, 1.0f, 1.0f});
        case avbd3d_heightShape:
            return b3ComputeHeightFieldAABB(r.geom.heightField.get(), xf);
        default:
            return b3ComputeHullAABB(r.geom.hullData(), xf);
    }
}

// Box of a shape proxy whose points are relative to the origin, lifted to world float space.
b3AABB proxyBox(const avbd3d_ShapeProxy *proxy, avbd3d_Pos origin)
{
    b3Vec3 lo{0, 0, 0}, hi{0, 0, 0};
    const b3Vec3 o = toB(origin);
    for (int i = 0; i < proxy->count; i++)
    {
        const b3Vec3 p = b3Add(toB(proxy->points[i]), o);
        if (i == 0)
            lo = hi = p;
        else
        {
            lo = b3Min(lo, p);
            hi = b3Max(hi, p);
        }
    }
    const b3Vec3 r{proxy->radius, proxy->radius, proxy->radius};
    return b3AABB{b3Sub(lo, r), b3Add(hi, r)};
}

avbd3d_TreeStats toStats(b3TreeStats s)
{
    return avbd3d_TreeStats{s.nodeVisits, s.leafVisits};
}

avbd3d_WorldCastOutput toWorldOutput(const b3CastOutput &o, avbd3d_Pos origin, b3Transform xf)
{
    avbd3d_WorldCastOutput out{};
    out.normal = fromB(b3RotateVector(xf.q, o.normal));
    out.point = fromB(b3Add(b3TransformPoint(xf, o.point), toB(origin)));
    out.fraction = o.fraction;
    out.iterations = o.iterations;
    out.triangleIndex = o.triangleIndex;
    out.childIndex = o.childIndex;
    out.materialIndex = o.materialIndex;
    out.hit = o.hit;
    return out;
}

// ---- ray cast

struct RayCtx
{
    World *w;
    avbd3d_Pos origin;
    avbd3d_QueryFilter filter;
    avbd3d_CastResultFcn *fcn;
    void *user;
};

float rayCastCallback(const b3RayCastInput *input, int, uint64_t userData, void *ctx)
{
    RayCtx &c = *static_cast<RayCtx *>(ctx);
    const int s = (int)userData;
    const ShapeRec &r = c.w->shapes[s];
    if (!r.alive || !shouldQuery(r, c.filter))
        return input->maxFraction;

    const b3Transform xf = relTransform(*c.w, s, c.origin);
    b3RayCastInput local;
    local.origin = b3InvTransformPoint(xf, b3Vec3_zero);
    local.translation = b3InvRotateVector(xf.q, input->translation);
    local.maxFraction = input->maxFraction;
    const b3CastOutput out = rayCastLocal(r, local);
    if (!out.hit)
        return input->maxFraction;

    const avbd3d_WorldCastOutput w = toWorldOutput(out, c.origin, xf);
    return c.fcn(c.w->shapeId(s), w.point, w.normal, w.fraction, r.material.userMaterialId, w.triangleIndex, w.childIndex,
                 c.user);
}

struct ClosestCtx
{
    World *w;
    avbd3d_Pos origin;
    avbd3d_QueryFilter filter;
    avbd3d_RayResult result;
};

float rayClosestCallback(const b3RayCastInput *input, int, uint64_t userData, void *ctx)
{
    ClosestCtx &c = *static_cast<ClosestCtx *>(ctx);
    const int s = (int)userData;
    const ShapeRec &r = c.w->shapes[s];
    if (!r.alive || !shouldQuery(r, c.filter))
        return input->maxFraction;

    const b3Transform xf = relTransform(*c.w, s, c.origin);
    b3RayCastInput local;
    local.origin = b3InvTransformPoint(xf, b3Vec3_zero);
    local.translation = b3InvRotateVector(xf.q, input->translation);
    local.maxFraction = input->maxFraction;
    const b3CastOutput out = rayCastLocal(r, local);
    if (!out.hit)
        return input->maxFraction;

    const avbd3d_WorldCastOutput w = toWorldOutput(out, c.origin, xf);
    c.result.shapeId = c.w->shapeId(s);
    c.result.point = w.point;
    c.result.normal = w.normal;
    c.result.userMaterialId = r.material.userMaterialId;
    c.result.fraction = w.fraction;
    c.result.triangleIndex = w.triangleIndex;
    c.result.childIndex = w.childIndex;
    c.result.hit = true;
    return w.fraction; // clip the search to the closest hit so far
}

// ---- shape cast (proxy swept through the tree)

struct ShapeCastCtx
{
    World *w;
    avbd3d_Pos origin;
    avbd3d_QueryFilter filter;
    const avbd3d_ShapeProxy *proxy; // origin relative
    avbd3d_Vec3 translation;
    avbd3d_CastResultFcn *fcn;
    void *user;
};

float shapeCastCallback(const b3BoxCastInput *input, int, uint64_t userData, void *ctx)
{
    ShapeCastCtx &c = *static_cast<ShapeCastCtx *>(ctx);
    const int s = (int)userData;
    const ShapeRec &r = c.w->shapes[s];
    if (!r.alive || !shouldQuery(r, c.filter))
        return input->maxFraction;

    const b3Transform xf = relTransform(*c.w, s, c.origin);
    std::vector<b3Vec3> pts(c.proxy->count);
    for (int i = 0; i < c.proxy->count; i++)
        pts[i] = b3InvTransformPoint(xf, toB(c.proxy->points[i]));
    b3ShapeCastInput local;
    local.proxy.points = pts.data();
    local.proxy.count = c.proxy->count;
    local.proxy.radius = c.proxy->radius;
    local.translation = b3InvRotateVector(xf.q, toB(c.translation));
    local.maxFraction = input->maxFraction;
    local.canEncroach = false;
    const b3CastOutput out = shapeCastLocal(r, local);
    if (!out.hit)
        return input->maxFraction;

    const avbd3d_WorldCastOutput w = toWorldOutput(out, c.origin, xf);
    return c.fcn(c.w->shapeId(s), w.point, w.normal, w.fraction, r.material.userMaterialId, w.triangleIndex, w.childIndex,
                 c.user);
}

// ---- mover cast (capsule swept through the tree; Box3D MoverCastCallback)

struct MoverCastCtx
{
    World *w;
    avbd3d_Pos origin;
    avbd3d_QueryFilter filter;
    const avbd3d_Capsule *mover; // origin relative
    avbd3d_Vec3 translation;
    avbd3d_MoverFilterFcn *fcn;
    void *user;
    float fraction;
};

float moverCastCallback(const b3BoxCastInput *input, int, uint64_t userData, void *ctx)
{
    MoverCastCtx &c = *static_cast<MoverCastCtx *>(ctx);
    const int s = (int)userData;
    const ShapeRec &r = c.w->shapes[s];
    if (!r.alive || !shouldQuery(r, c.filter))
        return c.fraction;

    if (c.fcn && !c.fcn(c.w->shapeId(s), c.user))
        return c.fraction;

    const b3Transform xf = relTransform(*c.w, s, c.origin);
    const b3Vec3 pts[2] = {b3InvTransformPoint(xf, toB(c.mover->center1)), b3InvTransformPoint(xf, toB(c.mover->center2))};
    b3ShapeCastInput local;
    local.proxy.points = pts;
    local.proxy.count = 2;
    local.proxy.radius = c.mover->radius;
    local.translation = b3InvRotateVector(xf.q, toB(c.translation));
    local.maxFraction = input->maxFraction;
    local.canEncroach = c.mover->radius > 0.0f;
    const b3CastOutput out = shapeCastLocal(r, local);
    if (out.fraction == 0.0f)
        return c.fraction; // overlapping shapes are ignored

    c.fraction = out.fraction;
    return out.fraction;
}

// ---- overlap (origin-relative proxy against the tree)

struct OverlapShapeCtx
{
    World *w;
    avbd3d_Pos origin;
    avbd3d_QueryFilter filter;
    const avbd3d_ShapeProxy *proxy;
    avbd3d_OverlapResultFcn *fcn;
    void *user;
};

bool overlapShapeCallback(int, uint64_t userData, void *ctx)
{
    OverlapShapeCtx &c = *static_cast<OverlapShapeCtx *>(ctx);
    const int s = (int)userData;
    const ShapeRec &r = c.w->shapes[s];
    if (!r.alive || !shouldQuery(r, c.filter))
        return true;
    const b3ShapeProxy proxy{(const b3Vec3 *)c.proxy->points, c.proxy->count, c.proxy->radius};
    if (!overlapLocal(r, relTransform(*c.w, s, c.origin), &proxy))
        return true;
    return c.fcn(c.w->shapeId(s), c.user);
}

struct OverlapAABBCtx
{
    World *w;
    avbd3d_QueryFilter filter;
    avbd3d_OverlapResultFcn *fcn;
    void *user;
};

bool overlapAABBCallback(int, uint64_t userData, void *ctx)
{
    OverlapAABBCtx &c = *static_cast<OverlapAABBCtx *>(ctx);
    const int s = (int)userData;
    const ShapeRec &r = c.w->shapes[s];
    if (!r.alive || !shouldQuery(r, c.filter))
        return true;
    return c.fcn(c.w->shapeId(s), c.user);
}

// ---- mover collision (origin-relative planes)

struct CollideMoverCtx
{
    World *w;
    avbd3d_Pos origin;
    avbd3d_QueryFilter filter;
    avbd3d_Capsule mover; // origin relative
    avbd3d_PlaneResultFcn *fcn;
    void *user;
};

bool collideMoverCallback(int, uint64_t userData, void *ctx)
{
    CollideMoverCtx &c = *static_cast<CollideMoverCtx *>(ctx);
    const int s = (int)userData;
    const ShapeRec &r = c.w->shapes[s];
    if (!r.alive || !shouldQuery(r, c.filter))
        return true;

    const b3Transform xf = relTransform(*c.w, s, c.origin);
    b3Capsule local;
    local.center1 = b3InvTransformPoint(xf, toB(c.mover.center1));
    local.center2 = b3InvTransformPoint(xf, toB(c.mover.center2));
    local.radius = c.mover.radius;

    b3PlaneResult buffer[kMaxPlanesPerShape];
    const int count = collideMoverLocal(r, buffer, &local);
    if (count <= 0)
        return true;

    std::array<avbd3d_PlaneResult, kMaxPlanesPerShape> planes;
    for (int i = 0; i < count; i++)
    {
        const b3PlaneResult &src = buffer[i];
        avbd3d_PlaneResult &dst = planes[i];
        const b3Vec3 n = b3RotateVector(xf.q, src.plane.normal);
        const b3Vec3 p = b3TransformPoint(xf, src.point);
        dst.plane.normal = fromB(n);
        // The offset is the penetration depth (totalRadius - distance), which a rigid transform
        // does not change; b3SolvePlanes reads it against the mover's displacement. Kept as is.
        dst.plane.offset = src.plane.offset;
        dst.point = fromB(p);
        dst.triangleIndex = src.triangleIndex;
        dst.childIndex = src.childIndex;
        dst.materialIndex = src.materialIndex;
    }
    return c.fcn(c.w->shapeId(s), planes.data(), count, c.user);
}

// ---- debug draw

struct DrawCtx
{
    World *w;
    avbd3d_DebugDraw *draw;
    int *shapeCount;
};

bool drawCallback(int, uint64_t userData, void *ctx)
{
    DrawCtx &c = *static_cast<DrawCtx *>(ctx);
    const int s = (int)userData;
    const ShapeRec &r = c.w->shapes[s];
    if (!r.alive)
        return true;

    const b3Transform xf = c.w->shapeWorldTransform(s);
    const avbd3d_HexColor color = c.w->bodies[r.body].type == avbd3d_staticBody ? avbd3d_colorDimGray : avbd3d_colorWhite;
    if (c.draw->drawShapes && c.draw->DrawShapeFcn)
    {
        avbd3d_WorldTransform t{fromB(xf.p), avbd3d_Quat{fromB(xf.q.v), xf.q.s}};
        c.draw->DrawShapeFcn(r.user, t, color, c.draw->context);
        (*c.shapeCount)++;
    }
    if (c.draw->drawBounds && c.draw->DrawBoundsFcn)
    {
        const b3AABB box = shapeAABB(r, xf);
        c.draw->DrawBoundsFcn(avbd3d_AABB{fromB(box.lowerBound), fromB(box.upperBound)}, avbd3d_colorYellow, c.draw->context);
    }
    return true;
}
} // namespace

avbd3d_QueryFilter avbd3d_DefaultQueryFilter(void)
{
    return avbd3d_QueryFilter{B3_DEFAULT_CATEGORY_BITS, B3_DEFAULT_MASK_BITS, 0, "default"};
}

// Zero-initialised like b3DefaultDebugDraw: every callback stays null (World_Draw skips a null
// callback), the flags are off, and the bounds and scales match Box3D's defaults.
avbd3d_DebugDraw avbd3d_DefaultDebugDraw(void)
{
    avbd3d_DebugDraw draw{};
    const float h = 100.0f;
    draw.drawingBounds = avbd3d_AABB{{-h, -h, -h}, {h, h, h}};
    draw.jointScale = 1.0f;
    draw.forceScale = 1.0f;
    return draw;
}

avbd3d_TreeStats avbd3d_World_CastRay(avbd3d_WorldId worldId, avbd3d_Pos origin, avbd3d_Vec3 translation,
                                      avbd3d_QueryFilter filter, avbd3d_CastResultFcn *fcn, void *context)
{
    World *w = settled(worldId);
    if (!w)
        return avbd3d_TreeStats{};
    RayCtx ctx{w, origin, filter, fcn, context};
    b3RayCastInput input{toB(origin), toB(translation), 1.0f};
    return toStats(b3DynamicTree_RayCast(&w->queryTree(), &input, filter.maskBits, false, rayCastCallback, &ctx));
}

avbd3d_RayResult avbd3d_World_CastRayClosest(avbd3d_WorldId worldId, avbd3d_Pos origin, avbd3d_Vec3 translation,
                                             avbd3d_QueryFilter filter)
{
    avbd3d_RayResult none{};
    World *w = settled(worldId);
    if (!w)
        return none;
    ClosestCtx ctx{w, origin, filter, none};
    b3RayCastInput input{toB(origin), toB(translation), 1.0f};
    const b3TreeStats st =
        b3DynamicTree_RayCast(&w->queryTree(), &input, filter.maskBits, false, rayClosestCallback, &ctx);
    ctx.result.nodeVisits = st.nodeVisits;
    ctx.result.leafVisits = st.leafVisits;
    return ctx.result;
}

avbd3d_TreeStats avbd3d_World_CastShape(avbd3d_WorldId worldId, avbd3d_Pos origin, const avbd3d_ShapeProxy *proxy,
                                        avbd3d_Vec3 translation, avbd3d_QueryFilter filter, avbd3d_CastResultFcn *fcn,
                                        void *context)
{
    World *w = settled(worldId);
    if (!w || !proxy || proxy->count <= 0)
        return avbd3d_TreeStats{};
    ShapeCastCtx ctx{w, origin, filter, proxy, translation, fcn, context};
    // Sweep the proxy's world box along the translation through the tree.
    const b3AABB box = proxyBox(proxy, origin);
    b3BoxCastInput input{box, toB(translation), 1.0f};
    return toStats(b3DynamicTree_BoxCast(&w->queryTree(), &input, filter.maskBits, false, shapeCastCallback, &ctx));
}

avbd3d_TreeStats avbd3d_World_OverlapAABB(avbd3d_WorldId worldId, avbd3d_AABB aabb, avbd3d_QueryFilter filter,
                                          avbd3d_OverlapResultFcn *fcn, void *context)
{
    World *w = settled(worldId);
    if (!w)
        return avbd3d_TreeStats{};
    OverlapAABBCtx ctx{w, filter, fcn, context};
    const b3AABB box{toB(aabb.lowerBound), toB(aabb.upperBound)};
    return toStats(b3DynamicTree_Query(&w->queryTree(), box, filter.maskBits, false, overlapAABBCallback, &ctx));
}

avbd3d_TreeStats avbd3d_World_OverlapShape(avbd3d_WorldId worldId, avbd3d_Pos origin, const avbd3d_ShapeProxy *proxy,
                                           avbd3d_QueryFilter filter, avbd3d_OverlapResultFcn *fcn, void *context)
{
    World *w = settled(worldId);
    if (!w || !proxy || proxy->count <= 0)
        return avbd3d_TreeStats{};
    OverlapShapeCtx ctx{w, origin, filter, proxy, fcn, context};
    const b3AABB box = proxyBox(proxy, origin);
    return toStats(b3DynamicTree_Query(&w->queryTree(), box, filter.maskBits, false, overlapShapeCallback, &ctx));
}

float avbd3d_World_CastMover(avbd3d_WorldId worldId, avbd3d_Pos origin, const avbd3d_Capsule *mover, avbd3d_Vec3 translation,
                             avbd3d_QueryFilter filter, avbd3d_MoverFilterFcn *fcn, void *context)
{
    World *w = settled(worldId);
    if (!w || !mover)
        return 1.0f;
    MoverCastCtx ctx{w, origin, filter, mover, translation, fcn, context, 1.0f};

    // Box of the capsule in origin relative space, lifted to world float space.
    const avbd3d_Vec3 c[2] = {mover->center1, mover->center2};
    avbd3d_Vec3 pts[2] = {c[0], c[1]};
    avbd3d_ShapeProxy proxy{pts, 2, mover->radius};
    const b3AABB box = proxyBox(&proxy, origin);
    b3BoxCastInput input{box, toB(translation), 1.0f};
    b3DynamicTree_BoxCast(&w->queryTree(), &input, filter.maskBits, false, moverCastCallback, &ctx);
    return ctx.fraction;
}

void avbd3d_World_CollideMover(avbd3d_WorldId worldId, avbd3d_Pos origin, const avbd3d_Capsule *mover,
                               avbd3d_QueryFilter filter, avbd3d_PlaneResultFcn *fcn, void *context)
{
    World *w = settled(worldId);
    if (!w || !mover)
        return;
    CollideMoverCtx ctx{w, origin, filter, *mover, fcn, context};
    const avbd3d_Vec3 lo = {std::min(mover->center1.x, mover->center2.x) - mover->radius,
                            std::min(mover->center1.y, mover->center2.y) - mover->radius,
                            std::min(mover->center1.z, mover->center2.z) - mover->radius};
    const avbd3d_Vec3 hi = {std::max(mover->center1.x, mover->center2.x) + mover->radius,
                            std::max(mover->center1.y, mover->center2.y) + mover->radius,
                            std::max(mover->center1.z, mover->center2.z) + mover->radius};
    const b3Vec3 o = toB(origin);
    const b3AABB box{b3Add(toB(lo), o), b3Add(toB(hi), o)};
    b3DynamicTree_Query(&w->queryTree(), box, filter.maskBits, false, collideMoverCallback, &ctx);
}

void avbd3d_World_Draw(avbd3d_WorldId worldId, avbd3d_DebugDraw *draw, uint64_t maskBits)
{
    World *w = settled(worldId);
    if (!w || !draw)
        return;
    int shapeCount = 0;
    DrawCtx ctx{w, draw, &shapeCount};
    const avbd3d_AABB &db = draw->drawingBounds;
    const b3AABB box{toB(db.lowerBound), toB(db.upperBound)};
    b3DynamicTree_Query(&w->queryTree(), box, maskBits, false, drawCallback, &ctx);
}

avbd3d_WorldCastOutput avbd3d_Shape_RayCast(avbd3d_ShapeId shapeId, avbd3d_Pos origin, avbd3d_Vec3 translation)
{
    avbd3d_WorldCastOutput none{};
    int slot = -1;
    World *w = worldOfShape(shapeId, slot); // settles an in-flight step
    if (!w)
        return none;
    const ShapeRec &r = w->shapes[slot];
    const b3Transform xf = relTransform(*w, slot, origin);
    b3RayCastInput local;
    local.origin = b3InvTransformPoint(xf, b3Vec3_zero);
    local.translation = b3InvRotateVector(xf.q, toB(translation));
    local.maxFraction = 1.0f;
    return toWorldOutput(rayCastLocal(r, local), origin, xf);
}

// ---- sensors and explosions (docs/SOLVER_3D_ROADMAP.md Phase 4b-2) -------------------------

namespace avbd3d_impl
{
namespace
{
constexpr float kPi = 3.14159265358979f;

// Box3D's filter rule (b3ShouldShapesCollide, external/box3d/src/shape.h): a nonzero equal group
// decides alone, otherwise each shape's mask must accept the other's category.
bool filtersCollide(const avbd3d_Filter &a, const avbd3d_Filter &b)
{
    if (a.groupIndex == b.groupIndex && a.groupIndex != 0)
        return a.groupIndex > 0;
    return (a.maskBits & b.categoryBits) != 0 && (a.categoryBits & b.maskBits) != 0;
}

// The convex proxy points of a shape in the frame xf maps it to. A sphere is one point, a capsule
// two, a box its eight hull vertices.
std::vector<b3Vec3> proxyPoints(const ShapeRec &r, b3Transform xf)
{
    std::vector<b3Vec3> out;
    if (r.type == avbd3d_sphereShape)
    {
        out.push_back(b3TransformPoint(xf, r.geom.sphere.center));
    }
    else if (r.type == avbd3d_capsuleShape)
    {
        out.push_back(b3TransformPoint(xf, r.geom.capsule.center1));
        out.push_back(b3TransformPoint(xf, r.geom.capsule.center2));
    }
    else if (r.type == avbd3d_meshShape || r.type == avbd3d_heightShape)
    {
        // Meshes and height fields have no convex proxy points (gap: no debug proxy drawn).
    }
    else
    {
        const b3HullData *h = r.geom.hullData();
        const b3Vec3 *p = b3GetHullPoints(h);
        for (int i = 0; i < h->vertexCount; i++)
            out.push_back(b3TransformPoint(xf, p[i]));
    }
    return out;
}

float proxyRadius(const ShapeRec &r)
{
    if (r.type == avbd3d_sphereShape)
        return r.geom.sphere.radius;
    if (r.type == avbd3d_capsuleShape)
        return r.geom.capsule.radius;
    return 0.0f;
}

vec3 toVv(b3Vec3 v) { return vec3{v.x, v.y, v.z}; }

// Box3D's sensor visitor rules (b3SensorQueryCallback in external/box3d/src/sensor.c): the
// visitor is not the sensor, is not on the sensor's body, has enableSensorEvents, passes the
// filter, and overlaps the sensor's shape exactly. Sensors are not visitors here, following the
// Phase 4b-2 spec; Box3D's docs say sensors do detect sensors.
struct SensorCtx
{
    World *w;
    int sensor;
    std::vector<b3Vec3> points; // sensor proxy, world space
    float radius;
    std::vector<uint64_t> *out;
};

bool sensorVisitorCallback(int, uint64_t userData, void *ctx)
{
    SensorCtx &c = *static_cast<SensorCtx *>(ctx);
    const int v = (int)userData;
    if (v == c.sensor)
        return true;
    const ShapeRec &sr = c.w->shapes[c.sensor];
    const ShapeRec &vr = c.w->shapes[v];
    if (!vr.alive || vr.isSensor || !vr.sensorEvents || vr.body == sr.body)
        return true;
    if (!c.w->bodies[vr.body].alive || !c.w->bodies[vr.body].enabled)
        return true;
    if (!filtersCollide(sr.filter, vr.filter))
        return true;
    const b3ShapeProxy proxy{c.points.data(), (int)c.points.size(), c.radius};
    const avbd3d_Pos zero{};
    if (!overlapLocal(vr, relTransform(*c.w, v, zero), &proxy))
        return true;
    c.out->push_back(((uint64_t)v << 16) | vr.gen);
    return true;
}

struct ExplodeCtx
{
    World *w;
    b3Vec3 position;
    float radius, falloff, impulsePerArea;
};

// One shape in an explosion: Box3D's ExplosionCallback (external/box3d/src/physics_world.c).
bool explodeCallback(int, uint64_t userData, void *ctx)
{
    ExplodeCtx &c = *static_cast<ExplodeCtx *>(ctx);
    const int s = (int)userData;
    const ShapeRec &r = c.w->shapes[s];
    if (!r.alive || r.explosionScale == 0.0f)
        return true;
    const BodyRec &br = c.w->bodies[r.body];
    if (!br.alive || !br.enabled || br.type != avbd3d_dynamicBody)
        return true;

    // Everything below is relative to the explosion point, so distances stay precise far from
    // the world origin.
    const b3Transform xf = c.w->shapeWorldTransform(s);
    std::vector<b3Vec3> pts = proxyPoints(r, xf);
    for (b3Vec3 &p : pts)
        p = b3Sub(p, c.position);
    const b3ShapeProxy proxyA{pts.data(), (int)pts.size(), proxyRadius(r)};
    const b3Vec3 origin{0.0f, 0.0f, 0.0f};
    const b3ShapeProxy proxyB{&origin, 1, 0.0f};
    b3DistanceInput input;
    input.proxyA = proxyA;
    input.proxyB = proxyB;
    input.transform = b3Transform_identity;
    input.useRadii = true;
    b3SimplexCache cache = {0};
    const b3DistanceOutput out = b3ShapeDistance(&input, &cache, NULL, 0);

    const float radius = c.radius, falloff = c.falloff;
    if (out.distance > radius + falloff)
        return true;

    // Witness point on the shape, relative to the explosion. A point inside the shape falls back
    // to the shape's centre.
    b3Vec3 closest = out.pointA;
    if (out.distance == 0.0f)
        closest = b3Sub(xf.p, c.position);

    // Outward direction from the explosion to the witness point.
    b3Vec3 dir = closest;
    if (b3LengthSquared(dir) > 100.0f * FLT_EPSILON * FLT_EPSILON)
        dir = b3Normalize(dir);
    else
        dir = b3Vec3{1.0f, 0.0f, 0.0f};

    // Projected area of the shape facing the explosion, computed in the shape frame.
    const b3Vec3 dirLocal = b3InvRotateVector(xf.q, dir);
    float area = 0.0f;
    if (r.type == avbd3d_sphereShape)
    {
        area = kPi * r.geom.sphere.radius * r.geom.sphere.radius;
    }
    else if (r.type == avbd3d_capsuleShape)
    {
        const float rad = r.geom.capsule.radius;
        const b3Vec3 axis = b3Sub(r.geom.capsule.center2, r.geom.capsule.center1);
        const float projected = b3Length(b3Cross(axis, dirLocal));
        area = kPi * rad * rad + 2.0f * rad * projected;
    }
    else
    {
        // Meshes and height fields have no projected area here (gap: they receive no blast impulse).
        area = (r.type == avbd3d_meshShape || r.type == avbd3d_heightShape)
                   ? 0.0f
                   : b3ComputeHullProjectedArea(r.geom.hullData(), dirLocal);
    }

    float scale = 1.0f;
    if (out.distance > radius && falloff > 0.0f)
        scale = std::clamp((radius + falloff - out.distance) / falloff, 0.0f, 1.0f);

    const float magnitude = c.impulsePerArea * area * scale * r.explosionScale;
    const b3Vec3 impulse = b3MulSV(magnitude, dir);
    // Through the queued impulse path: it wakes the body (Box3D wakes every body in range, asleep or
    // not) and applies the impulse at the witness point, so the spin comes from the lever arm.
    const b3Vec3 point = b3Add(c.position, closest);
    c.w->applyImpulse(r.body, toVv(impulse), toVv(point), true);
    return true;
}
} // namespace

void World::collectSensors()
{
    sensorBegin.clear();
    sensorEnd.assign(m_pendingSensorEnd.begin(), m_pendingSensorEnd.end());
    m_pendingSensorEnd.clear();

    std::vector<uint64_t> now, begun, ended;
    for (int s : m_sensors)
    {
        ShapeRec &sr = shapes[s];
        // A sensor reports nothing while it or its body is off, so every overlap ends.
        now.clear();
        const bool active = sr.sensorEvents && sr.proxy >= 0 && bodies[sr.body].alive && bodies[sr.body].enabled;
        if (active)
        {
            SensorCtx ctx{this, s, proxyPoints(sr, shapeWorldTransform(s)), proxyRadius(sr), &now};
            const b3AABB box = b3DynamicTree_GetAABB(&queryTree(), sr.proxy);
            b3DynamicTree_Query(&queryTree(), box, sr.filter.maskBits, false, sensorVisitorCallback, &ctx);
            std::sort(now.begin(), now.end());
        }
        // Box3D's sorted diff by (id, generation): the sets are sorted by key, so a visitor that
        // left and came back again (a new generation) is a new overlap.
        begun.clear();
        ended.clear();
        std::set_difference(now.begin(), now.end(), sr.overlaps.begin(), sr.overlaps.end(), std::back_inserter(begun));
        std::set_difference(sr.overlaps.begin(), sr.overlaps.end(), now.begin(), now.end(), std::back_inserter(ended));
        for (uint64_t k : begun)
            sensorBegin.push_back(avbd3d_SensorBeginTouchEvent{shapeIdAt(s, sr.gen), shapeIdAt((int)(k >> 16), (uint16_t)(k & 0xFFFFu))});
        for (uint64_t k : ended)
            sensorEnd.push_back(avbd3d_SensorEndTouchEvent{shapeIdAt(s, sr.gen), shapeIdAt((int)(k >> 16), (uint16_t)(k & 0xFFFFu))});
        sr.overlaps.swap(now);
    }
}

void World::explode(const avbd3d_ExplosionDef &def)
{
    const float extent = def.radius + def.falloff;
    const b3Vec3 p = toB(def.position);
    const b3AABB box{b3Vec3{p.x - extent, p.y - extent, p.z - extent}, b3Vec3{p.x + extent, p.y + extent, p.z + extent}};
    ExplodeCtx ctx{this, p, def.radius, def.falloff, def.impulsePerArea};
    b3DynamicTree_Query(&queryTree(), box, def.maskBits, false, explodeCallback, &ctx);
}
} // namespace avbd3d_impl

// ---- per-body queries (Box3D b3Body_CastRay / CastShape / OverlapShape / CollideMover / GetClosestPoint)
//
// Same per-shape math as the world queries, run over one body's shapes at a caller supplied body
// transform (the body origin pose), so a query can ask about where a body will be.

namespace
{
// A shape's transform for a given body transform, relative to the query origin.
b3Transform bodyShapeXf(const ShapeRec &r, avbd3d_WorldTransform bt, avbd3d_Pos origin)
{
    b3Transform body;
    body.p = b3Sub(toB(bt.p), toB(origin));
    body.q = b3Quat{{bt.q.v.x, bt.q.v.y, bt.q.v.z}, bt.q.s};
    b3Transform local;
    local.p = b3Vec3{r.localPos.x, r.localPos.y, r.localPos.z};
    local.q = b3Quat{{r.localRot.x, r.localRot.y, r.localRot.z}, r.localRot.w};
    return b3MulTransforms(body, local);
}

avbd3d_BodyCastResult toBodyResult(const World &w, int s, const b3CastOutput &out, avbd3d_Pos origin, b3Transform xf)
{
    const avbd3d_WorldCastOutput o = toWorldOutput(out, origin, xf);
    avbd3d_BodyCastResult res{};
    res.shapeId = w.shapeId(s);
    res.point = o.point;
    res.normal = o.normal;
    res.fraction = o.fraction;
    res.triangleIndex = o.triangleIndex;
    res.userMaterialId = w.shapes[s].material.userMaterialId;
    res.iterations = o.iterations;
    res.hit = true;
    return res;
}

bool convexShape(const ShapeRec &r)
{
    return r.type != avbd3d_meshShape && r.type != avbd3d_heightShape;
}
} // namespace

avbd3d_BodyCastResult avbd3d_Body_CastRay(avbd3d_BodyId bodyId, avbd3d_Pos origin, avbd3d_Vec3 translation,
                                          avbd3d_QueryFilter filter, float maxFraction, avbd3d_WorldTransform bodyTransform)
{
    avbd3d_BodyCastResult result{};
    int b = -1;
    World *w = worldOfBody(bodyId, b);
    if (!w)
        return result;
    for (int s : w->bodies[b].shapes)
    {
        const ShapeRec &r = w->shapes[s];
        if (!r.alive || !shouldQuery(r, filter))
            continue;
        const b3Transform xf = bodyShapeXf(r, bodyTransform, origin);
        b3RayCastInput local;
        local.origin = b3InvTransformPoint(xf, b3Vec3_zero);
        local.translation = b3InvRotateVector(xf.q, toB(translation));
        local.maxFraction = maxFraction;
        const b3CastOutput out = rayCastLocal(r, local);
        if (!out.hit || out.fraction > maxFraction)
            continue;
        result = toBodyResult(*w, s, out, origin, xf);
        maxFraction = out.fraction; // keep the closest
    }
    return result;
}

avbd3d_BodyCastResult avbd3d_Body_CastShape(avbd3d_BodyId bodyId, avbd3d_Pos origin, const avbd3d_ShapeProxy *proxy,
                                            avbd3d_Vec3 translation, avbd3d_QueryFilter filter, float maxFraction,
                                            bool canEncroach, avbd3d_WorldTransform bodyTransform)
{
    avbd3d_BodyCastResult result{};
    int b = -1;
    World *w = worldOfBody(bodyId, b);
    if (!w || !proxy || proxy->count <= 0)
        return result;
    std::vector<b3Vec3> pts(proxy->count);
    for (int s : w->bodies[b].shapes)
    {
        const ShapeRec &r = w->shapes[s];
        if (!r.alive || !shouldQuery(r, filter))
            continue;
        const b3Transform xf = bodyShapeXf(r, bodyTransform, origin);
        for (int i = 0; i < proxy->count; i++)
            pts[i] = b3InvTransformPoint(xf, toB(proxy->points[i]));
        b3ShapeCastInput local;
        local.proxy.points = pts.data();
        local.proxy.count = proxy->count;
        local.proxy.radius = proxy->radius;
        local.translation = b3InvRotateVector(xf.q, toB(translation));
        local.maxFraction = maxFraction;
        local.canEncroach = canEncroach;
        const b3CastOutput out = shapeCastLocal(r, local);
        if (!out.hit || out.fraction > maxFraction)
            continue;
        result = toBodyResult(*w, s, out, origin, xf);
        maxFraction = out.fraction;
    }
    return result;
}

bool avbd3d_Body_OverlapShape(avbd3d_BodyId bodyId, avbd3d_Pos origin, const avbd3d_ShapeProxy *proxy,
                              avbd3d_QueryFilter filter, avbd3d_WorldTransform bodyTransform)
{
    int b = -1;
    World *w = worldOfBody(bodyId, b);
    if (!w || !proxy || proxy->count <= 0)
        return false;
    const b3ShapeProxy bp{(const b3Vec3 *)proxy->points, proxy->count, proxy->radius};
    for (int s : w->bodies[b].shapes)
    {
        const ShapeRec &r = w->shapes[s];
        if (!r.alive || !shouldQuery(r, filter))
            continue;
        if (overlapLocal(r, bodyShapeXf(r, bodyTransform, origin), &bp))
            return true;
    }
    return false;
}

int avbd3d_Body_CollideMover(avbd3d_BodyId bodyId, avbd3d_BodyPlaneResult *bodyPlanes, int planeCapacity, avbd3d_Pos origin,
                             const avbd3d_Capsule *mover, avbd3d_QueryFilter filter, avbd3d_WorldTransform bodyTransform)
{
    int b = -1;
    World *w = worldOfBody(bodyId, b);
    if (!w || !mover || !bodyPlanes || planeCapacity <= 0)
        return 0;
    int count = 0;
    for (int s : w->bodies[b].shapes)
    {
        const ShapeRec &r = w->shapes[s];
        if (!r.alive || !shouldQuery(r, filter) || !convexShape(r))
            continue;
        const b3Transform xf = bodyShapeXf(r, bodyTransform, origin);
        b3Capsule local;
        local.center1 = b3InvTransformPoint(xf, toB(mover->center1));
        local.center2 = b3InvTransformPoint(xf, toB(mover->center2));
        local.radius = mover->radius;
        b3PlaneResult buffer[kMaxPlanesPerShape];
        if (collideMoverLocal(r, buffer, &local) <= 0)
            continue;
        const b3PlaneResult &src = buffer[0];
        avbd3d_PlaneResult &dst = bodyPlanes[count].result;
        bodyPlanes[count].shapeId = w->shapeId(s);
        dst.plane.normal = fromB(b3RotateVector(xf.q, src.plane.normal));
        dst.plane.offset = src.plane.offset;
        dst.point = fromB(b3TransformPoint(xf, src.point));
        dst.triangleIndex = src.triangleIndex;
        dst.childIndex = src.childIndex;
        dst.materialIndex = src.materialIndex;
        if (++count == planeCapacity)
            break;
    }
    return count;
}

// Unlike Box3D (whose useRadii = false measures to a sphere's or capsule's core), the distance and
// the point are to the surface of rounded shapes.
float avbd3d_Body_GetClosestPoint(avbd3d_BodyId bodyId, avbd3d_Vec3 *result, avbd3d_Vec3 target)
{
    int b = -1;
    World *w = worldOfBody(bodyId, b);
    if (!w)
    {
        if (result)
            *result = avbd3d_Vec3{0, 0, 0};
        return 0.0f;
    }
    float closest = FLT_MAX;
    const vec3 o = w->originPos(b);
    b3Vec3 closestPoint = b3Vec3{o.x, o.y, o.z};
    const b3Vec3 tp = toB(target);

    b3DistanceInput input{};
    input.proxyA = b3ShapeProxy{&tp, 1, 0.0f};
    input.transform = b3Transform_identity;
    input.useRadii = true;
    for (int s : w->bodies[b].shapes)
    {
        const ShapeRec &r = w->shapes[s];
        if (!r.alive || !convexShape(r))
            continue;
        const std::vector<b3Vec3> pts = proxyPoints(r, w->shapeWorldTransform(s));
        if (pts.empty())
            continue;
        input.proxyB = b3ShapeProxy{pts.data(), (int)pts.size(), proxyRadius(r)};
        b3SimplexCache cache{};
        const b3DistanceOutput out = b3ShapeDistance(&input, &cache, nullptr, 0);
        if (out.distance < closest)
        {
            closest = out.distance;
            closestPoint = out.pointB;
        }
    }
    if (result)
        *result = fromB(closestPoint);
    return closest;
}
