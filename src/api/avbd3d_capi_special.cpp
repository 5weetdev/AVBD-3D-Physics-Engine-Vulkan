// The public C API calls that are more than a one-line wrapper: world lifetime, stepping, stats,
// capabilities, the callback queries, debug draw, validity checks and the Vulkan GPU calls. The
// mechanical wrappers are in avbd3d_capi_gen.cpp.
#include "avbd3d_capi.h"

#include "vk_constraints.h"

#include <cmath>

// ---- version and result text --------------------------------------------------------------

uint32_t avbd3d_abi_version(void)
{
    return ((uint32_t)AVBD3D_ABI_VERSION_MAJOR << 16) | (uint32_t)AVBD3D_ABI_VERSION_MINOR;
}

const char *avbd3d_result_string(Avbd3dResult result)
{
    switch (result)
    {
    case AVBD3D_OK:
        return "AVBD3D_OK";
    case AVBD3D_ERR_NULL_HANDLE:
        return "AVBD3D_ERR_NULL_HANDLE";
    case AVBD3D_ERR_INVALID_ARG:
        return "AVBD3D_ERR_INVALID_ARG";
    case AVBD3D_ERR_UNSUPPORTED:
        return "AVBD3D_ERR_UNSUPPORTED";
    case AVBD3D_ERR_DEVICE:
        return "AVBD3D_ERR_DEVICE";
    case AVBD3D_ERR_CAPACITY:
        return "AVBD3D_ERR_CAPACITY";
    case AVBD3D_ERR_DEVICE_LOST:
        return "AVBD3D_ERR_DEVICE_LOST";
    case AVBD3D_ERR_OUT_OF_MEMORY:
        return "AVBD3D_ERR_OUT_OF_MEMORY";
    }
    return "AVBD3D_ERR_UNKNOWN";
}

// ---- defaults -----------------------------------------------------------------------------

Avbd3dDebugDraw avbd3d_default_debug_draw(void)
{
    const avbd3d_DebugDraw d = avbd3d_DefaultDebugDraw();
    Avbd3dDebugDraw r;
    std::memset(&r, 0, sizeof r);
    r.drawingBounds = pb(d.drawingBounds);
    return r;
}

// ---- world lifetime and stepping ----------------------------------------------------------

static Avbd3dResult createdWorld(avbd3d_WorldId id, avbd3d_Error err, Avbd3dWorld **out_world)
{
    if (id.index1 == 0)
        return err != avbd3d_errorNone ? mapError(err) : AVBD3D_ERR_CAPACITY;
    *out_world = worldHandle(id);
    return *out_world ? AVBD3D_OK : AVBD3D_ERR_DEVICE;
}

Avbd3dResult avbd3d_create_world(const Avbd3dWorldDef *def, Avbd3dWorld **out_world)
{
    if (!out_world)
        return AVBD3D_ERR_INVALID_ARG;
    *out_world = nullptr;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_WorldDef ld = lg(*def);
    avbd3d_Error err = avbd3d_errorNone;
    const avbd3d_WorldId id = avbd3d_CreateWorld(&ld, &err);
    return createdWorld(id, err, out_world);
}

Avbd3dResult avbd3d_destroy_world(Avbd3dWorld *world)
{
    if (!world || !avbd3d_impl::validWorld(world))
        return AVBD3D_ERR_NULL_HANDLE;
    avbd3d_DestroyWorld(avbd3d_impl::worldIdOf(reinterpret_cast<World *>(world)));
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_step(Avbd3dWorld *world, float dt)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!(dt >= 0.0f) || !std::isfinite(dt))
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_World_Step(lw, dt);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    return w->stepResult().ok ? AVBD3D_OK : AVBD3D_ERR_CAPACITY;
}

Avbd3dResult avbd3d_step_begin(Avbd3dWorld *world, float dt)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!(dt >= 0.0f) || !std::isfinite(dt))
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_World_StepBegin(lw, dt);
    return leave(w);
}

Avbd3dResult avbd3d_step_end(Avbd3dWorld *world)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    const bool wasInFlight = w->inFlight();
    w->stepEnd();
    if (Avbd3dResult r_ = leave(w))
        return r_;
    return wasInFlight && !w->stepResult().ok ? AVBD3D_ERR_CAPACITY : AVBD3D_OK;
}

// ---- stats and capabilities ---------------------------------------------------------------

Avbd3dResult avbd3d_get_stats(Avbd3dWorld *world, Avbd3dStats *out)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    w->settle();
    if (Avbd3dResult r_ = leave(w))
        return r_;
    const avbdvk::VkWorld &v = w->vk();
    Avbd3dStats s;
    std::memset(&s, 0, sizeof s);
    s.bodies = v.bodyCount();
    s.awake_bodies = v.activeBodyCount();
    s.joint_rows = v.jointCount();
    s.springs = v.springCount();
    s.pairs = v.pairCount();
    s.manifolds = v.manifoldSlots();
    s.contacts = v.contactTotal();
    s.colors = v.colorCount();
    s.coloring_rounds = v.coloringRounds();
    s.pair_capacity = v.pairCapacity();
    s.contact_capacity = v.contactCapacity();
    s.manifold_capacity = v.manifoldCapacity();
    s.cell_size = w->desc.cellSize;
    const avbdvk::VkWorld::StepResult &sr = w->stepResult();
    s.ok = sr.ok ? 1 : 0;
    s.truncated = sr.truncated ? 1 : 0;
    s.submits = sr.submits;
    s.capacity_reruns = sr.capacityReruns;
    s.flood_reruns = sr.floodReruns;
    s.island_reruns = sr.islandReruns;
    s.jp_reruns = sr.jpReruns;
    s.color_reruns = sr.colorReruns;
    s.contact_events_truncated = w->contactTruncated ? 1 : 0;
    s.device_bytes = v.deviceBytesAllocated();
    const avbdvk::VkWorld::StepTiming &t = v.lastTiming();
    s.step_ms = t.stepMs;
    s.gpu_ms = t.gpuMs;
    s.front_ms = t.frontMs;
    s.broadphase_ms = t.broadMs;
    s.narrowphase_ms = t.narrowMs;
    s.build_ms = t.buildMs;
    s.gpu_build_ms = t.gpuBuildMs;
    s.solve_ms = t.solveMs;
    s.iterations_ms = t.iterMs;
    s.primal_ms = t.primalMs;
    s.dual_ms = t.dualMs;
    s.tail_ms = t.tailMs;
    s.host_pre_ms = t.hostPreMs;
    s.host_record_ms = t.hostRecordMs;
    s.host_submit_ms = t.hostSubmitMs;
    s.host_wait_ms = t.hostWaitMs;
    *out = s;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_get_capabilities(Avbd3dWorld *world, Avbd3dCapabilities *out)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!out || out->struct_size != sizeof(Avbd3dCapabilities))
        return AVBD3D_ERR_INVALID_ARG;
    const avbdvk::Device &d = w->vk().device();
    Avbd3dCapabilities c;
    std::memset(&c, 0, sizeof c);
    c.struct_size = sizeof c;
    std::strncpy(c.device_name, d.deviceName(), sizeof c.device_name - 1);
    c.vendor_id = d.vendorId();
    c.subgroup_size = d.subgroupSize();
    c.device_memory_bytes = d.deviceMemoryBytes();
    c.max_hull_vertices = AVBD3D_MAX_HULL_VERTICES;
    c.max_hull_faces = AVBD3D_MAX_HULL_FACES;
    c.max_hull_edges = AVBD3D_MAX_HULL_EDGES;
    c.max_manifold_points = AVBD3D_MAX_MANIFOLD_POINTS;
    c.max_child_shapes = AVBD3D_MAX_CHILD_SHAPES;
    c.max_colors = avbdvk::AVBD_MAX_COLORS_VK;
    *out = c;
    return AVBD3D_OK;
}

// ---- debug draw ---------------------------------------------------------------------------

namespace
{
struct DrawCtx
{
    const Avbd3dDebugDraw *draw;
};

void drawShapeTramp(void *userShape, avbd3d_WorldTransform xf, avbd3d_HexColor color, void *context)
{
    const Avbd3dDebugDraw *d = static_cast<DrawCtx *>(context)->draw;
    d->DrawShapeFcn(userShape, pb(xf), (Avbd3dHexColor)color, d->context);
}

void drawBoundsTramp(avbd3d_AABB aabb, avbd3d_HexColor color, void *context)
{
    const Avbd3dDebugDraw *d = static_cast<DrawCtx *>(context)->draw;
    d->DrawBoundsFcn(pb(aabb), (Avbd3dHexColor)color, d->context);
}
} // namespace

Avbd3dResult avbd3d_world_draw(Avbd3dWorld *world, const Avbd3dDebugDraw *draw, uint64_t maskBits)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!draw)
        return AVBD3D_ERR_INVALID_ARG;
    DrawCtx ctx{draw};
    avbd3d_DebugDraw ld = avbd3d_DefaultDebugDraw();
    ld.DrawShapeFcn = draw->DrawShapeFcn ? drawShapeTramp : nullptr;
    ld.DrawBoundsFcn = draw->DrawBoundsFcn ? drawBoundsTramp : nullptr;
    ld.drawingBounds = lg(draw->drawingBounds);
    ld.drawShapes = draw->drawShapes != 0;
    ld.drawBounds = draw->drawBounds != 0;
    ld.context = &ctx;
    avbd3d_World_Draw(lw, &ld, maskBits);
    return leave(w);
}

// ---- queries with callbacks ---------------------------------------------------------------

namespace
{
struct OverlapCtx
{
    Avbd3dOverlapResultFcn *fcn;
    void *context;
};
bool overlapTramp(avbd3d_ShapeId id, void *c)
{
    OverlapCtx *x = static_cast<OverlapCtx *>(c);
    return x->fcn(pb(id), x->context) != 0;
}

struct CastCtx
{
    Avbd3dCastResultFcn *fcn;
    void *context;
};
float castTramp(avbd3d_ShapeId id, avbd3d_Pos point, avbd3d_Vec3 normal, float fraction, uint64_t userMaterialId,
                int triangleIndex, int childIndex, void *c)
{
    CastCtx *x = static_cast<CastCtx *>(c);
    return x->fcn(pb(id), pb(point), pb(normal), fraction, userMaterialId, triangleIndex, childIndex, x->context);
}

struct MoverCtx
{
    Avbd3dMoverFilterFcn *fcn;
    void *context;
};
bool moverTramp(avbd3d_ShapeId id, void *c)
{
    MoverCtx *x = static_cast<MoverCtx *>(c);
    return x->fcn(pb(id), x->context) != 0;
}

struct PlaneCtx
{
    Avbd3dPlaneResultFcn *fcn;
    void *context;
};
bool planeTramp(avbd3d_ShapeId id, const avbd3d_PlaneResult *plane, int count, void *c)
{
    PlaneCtx *x = static_cast<PlaneCtx *>(c);
    static_assert(sizeof(Avbd3dPlaneResult) == sizeof(avbd3d_PlaneResult), "plane result layout");
    return x->fcn(pb(id), reinterpret_cast<const Avbd3dPlaneResult *>(plane), count, x->context) != 0;
}

bool proxyOk(const Avbd3dShapeProxy *p)
{
    return p && p->points && p->count >= 1 && p->count <= AVBD3D_MAX_SHAPE_CAST_POINTS;
}
} // namespace

Avbd3dResult avbd3d_world_overlap_aabb(Avbd3dWorld *world, Avbd3dAABB aabb, Avbd3dQueryFilter filter,
                                       Avbd3dOverlapResultFcn *fcn, void *context)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!fcn)
        return AVBD3D_ERR_INVALID_ARG;
    OverlapCtx ctx{fcn, context};
    avbd3d_World_OverlapAABB(lw, lg(aabb), lg(filter), overlapTramp, &ctx);
    return leave(w);
}

Avbd3dResult avbd3d_world_overlap_shape(Avbd3dWorld *world, Avbd3dVec3 origin, const Avbd3dShapeProxy *proxy,
                                        Avbd3dQueryFilter filter, Avbd3dOverlapResultFcn *fcn, void *context)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!fcn || !proxyOk(proxy))
        return AVBD3D_ERR_INVALID_ARG;
    OverlapCtx ctx{fcn, context};
    const avbd3d_ShapeProxy lp = lg(*proxy);
    avbd3d_World_OverlapShape(lw, lg(origin), &lp, lg(filter), overlapTramp, &ctx);
    return leave(w);
}

Avbd3dResult avbd3d_world_cast_ray(Avbd3dWorld *world, Avbd3dVec3 origin, Avbd3dVec3 translation,
                                   Avbd3dQueryFilter filter, Avbd3dCastResultFcn *fcn, void *context)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!fcn)
        return AVBD3D_ERR_INVALID_ARG;
    CastCtx ctx{fcn, context};
    avbd3d_World_CastRay(lw, lg(origin), lg(translation), lg(filter), castTramp, &ctx);
    return leave(w);
}

Avbd3dResult avbd3d_world_cast_shape(Avbd3dWorld *world, Avbd3dVec3 origin, const Avbd3dShapeProxy *proxy,
                                     Avbd3dVec3 translation, Avbd3dQueryFilter filter, Avbd3dCastResultFcn *fcn,
                                     void *context)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!fcn || !proxyOk(proxy))
        return AVBD3D_ERR_INVALID_ARG;
    CastCtx ctx{fcn, context};
    const avbd3d_ShapeProxy lp = lg(*proxy);
    avbd3d_World_CastShape(lw, lg(origin), &lp, lg(translation), lg(filter), castTramp, &ctx);
    return leave(w);
}

Avbd3dResult avbd3d_world_cast_mover(Avbd3dWorld *world, Avbd3dVec3 origin, const Avbd3dCapsule *mover,
                                     Avbd3dVec3 translation, Avbd3dQueryFilter filter, Avbd3dMoverFilterFcn *fcn,
                                     void *context, float *out_fraction)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!mover || !out_fraction)
        return AVBD3D_ERR_INVALID_ARG;
    MoverCtx ctx{fcn, context};
    const avbd3d_Capsule lc = lg(*mover);
    const float f =
        avbd3d_World_CastMover(lw, lg(origin), &lc, lg(translation), lg(filter), fcn ? moverTramp : nullptr, &ctx);
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out_fraction = f;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_world_collide_mover(Avbd3dWorld *world, Avbd3dVec3 origin, const Avbd3dCapsule *mover,
                                        Avbd3dQueryFilter filter, Avbd3dPlaneResultFcn *fcn, void *context)
{
    World *w;
    avbd3d_WorldId lw;
    if (Avbd3dResult r_ = enterWorld(world, w, lw))
        return r_;
    if (!mover || !fcn)
        return AVBD3D_ERR_INVALID_ARG;
    PlaneCtx ctx{fcn, context};
    const avbd3d_Capsule lc = lg(*mover);
    avbd3d_World_CollideMover(lw, lg(origin), &lc, lg(filter), planeTramp, &ctx);
    return leave(w);
}

// ---- validity, closest point, joint destroy -----------------------------------------------

int32_t avbd3d_body_is_valid(Avbd3dBodyId id)
{
    World *w;
    avbd3d_BodyId lid;
    return enterBody(id, w, lid) == AVBD3D_OK ? 1 : 0;
}

int32_t avbd3d_shape_is_valid(Avbd3dShapeId id)
{
    World *w;
    avbd3d_ShapeId lid;
    return enterShape(id, w, lid) == AVBD3D_OK ? 1 : 0;
}

int32_t avbd3d_joint_is_valid(Avbd3dJointId id)
{
    World *w;
    avbd3d_JointId lid;
    return enterJoint(id, w, lid) == AVBD3D_OK ? 1 : 0;
}

int32_t avbd3d_contact_is_valid(Avbd3dContactId id)
{
    World *w;
    avbd3d_ContactId lid;
    return enterContact(id, w, lid) == AVBD3D_OK ? 1 : 0;
}

Avbd3dResult avbd3d_body_get_closest_point(Avbd3dBodyId id, Avbd3dVec3 target, Avbd3dVec3 *out_point,
                                           float *out_distance)
{
    World *w;
    avbd3d_BodyId lid;
    if (Avbd3dResult r_ = enterBody(id, w, lid))
        return r_;
    if (!out_point || !out_distance)
        return AVBD3D_ERR_INVALID_ARG;
    avbd3d_Vec3 p{};
    const float d = avbd3d_Body_GetClosestPoint(lid, &p, lg(target));
    if (Avbd3dResult r_ = leave(w))
        return r_;
    *out_point = pb(p);
    *out_distance = d;
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_destroy_joint(Avbd3dJointId id, int32_t wake_bodies)
{
    World *w;
    avbd3d_JointId lid;
    if (Avbd3dResult r_ = enterJoint(id, w, lid))
        return r_;
    avbd3d_DestroyJoint(lid, wake_bodies != 0);
    return leave(w);
}

// ---- Vulkan GPU ---------------------------------------------------------------------------

Avbd3dGpuDef avbd3d_default_gpu_def(void)
{
    Avbd3dGpuDef d = pb(avbd3d_DefaultGpuDef());
    d.struct_size = sizeof(Avbd3dGpuDef);
    return d;
}

Avbd3dResult avbd3d_create_gpu(const Avbd3dGpuDef *def, Avbd3dGpu **out_gpu)
{
    if (!out_gpu)
        return AVBD3D_ERR_INVALID_ARG;
    *out_gpu = nullptr;
    if (!def || def->struct_size != sizeof(Avbd3dGpuDef))
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_GpuDef ld = lg(*def);
    avbd3d_Error err = avbd3d_errorNone;
    avbd3d_Gpu *gpu = avbd3d_CreateGpu(&ld, &err);
    if (!gpu)
        return err != avbd3d_errorNone ? mapError(err) : AVBD3D_ERR_DEVICE;
    *out_gpu = reinterpret_cast<Avbd3dGpu *>(gpu);
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_destroy_gpu(Avbd3dGpu *gpu)
{
    if (!gpu)
        return AVBD3D_ERR_NULL_HANDLE;
    avbd3d_DestroyGpu(reinterpret_cast<avbd3d_Gpu *>(gpu));
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_create_world_on_gpu(Avbd3dGpu *gpu, const Avbd3dWorldDef *def, Avbd3dWorld **out_world)
{
    if (!out_world)
        return AVBD3D_ERR_INVALID_ARG;
    *out_world = nullptr;
    if (!gpu)
        return AVBD3D_ERR_NULL_HANDLE;
    if (!def)
        return AVBD3D_ERR_INVALID_ARG;
    const avbd3d_WorldDef ld = lg(*def);
    avbd3d_Error err = avbd3d_errorNone;
    const avbd3d_WorldId id = avbd3d_CreateWorldOnGpu(&ld, reinterpret_cast<avbd3d_Gpu *>(gpu), &err);
    return createdWorld(id, err, out_world);
}

Avbd3dResult avbd3d_gpu_get_vulkan_context(const Avbd3dGpu *gpu, Avbd3dVulkanContext *out)
{
    if (!gpu)
        return AVBD3D_ERR_NULL_HANDLE;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    *out = pb(avbd3d_Gpu_GetVulkanContext(reinterpret_cast<const avbd3d_Gpu *>(gpu)));
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_gpu_get_allocated_bytes(const Avbd3dGpu *gpu, uint64_t *out)
{
    if (!gpu)
        return AVBD3D_ERR_NULL_HANDLE;
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    *out = avbd3d_Gpu_GetAllocatedBytes(reinterpret_cast<const avbd3d_Gpu *>(gpu));
    return AVBD3D_OK;
}

Avbd3dResult avbd3d_get_validation_message_count(int32_t *out)
{
    if (!out)
        return AVBD3D_ERR_INVALID_ARG;
    *out = avbd3d_GetValidationMessageCount();
    return AVBD3D_OK;
}
