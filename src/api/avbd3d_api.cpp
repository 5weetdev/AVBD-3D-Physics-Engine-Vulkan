// The avbd3d C API on the world layer in avbd3d_world.{h,cpp}. Only the supported subset is
// implemented (see avbd3d_ext.h capabilities); everything else in the headers is declared and
// not defined.

#include "avbd3d_world.h"

#include "shape_mass.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>

using namespace avbd3d_impl;

// Mirror of Box3D's internal b3Triangle (external/box3d/src/math_internal.h). Box3D's
// b3GetHeightFieldTriangle (external/box3d/src/height_field.c, src/shape.h) is compiled into
// avbd3d_box3d and is not in the public headers.
struct b3Triangle
{
    b3Vec3 vertices[3];
    int i1, i2, i3;
    int flags;
};
extern "C" b3Triangle b3GetHeightFieldTriangle(const b3HeightFieldData *heightField, int triangleIndex);

namespace avbd3d_impl
{
namespace
{
constexpr int kInternalWorld = 1152023; // Box3D's B3_SECRET_COOKIE

struct Slot
{
    std::unique_ptr<World> world;
    uint16_t gen = 1;
};
std::vector<Slot> &table()
{
    static std::vector<Slot> t(AVBD3D_MAX_WORLDS);
    return t;
}
} // namespace

World *lookupWorld(avbd3d_WorldId id)
{
    if (id.index1 < 1 || id.index1 > (int)table().size())
        return nullptr;
    Slot &s = table()[id.index1 - 1];
    return s.world && s.gen == id.generation ? s.world.get() : nullptr;
}

World *byIndex(int w0)
{
    if (w0 < 0 || w0 >= (int)table().size())
        return nullptr;
    return table()[w0].world.get();
}

World *worldOfBody(avbd3d_BodyId id, int &slot)
{
    World *w = byIndex(id.world0);
    if (!w || !w->bodyValid(id))
        return nullptr;
    // No call waits for a step in flight: the body reads the newest finished step.
    slot = id.index1;
    w->syncBody(slot);
    return w;
}

World *worldOfShape(avbd3d_ShapeId id, int &slot)
{
    World *w = byIndex(id.world0);
    if (!w || !w->shapeValid(id))
        return nullptr;
    slot = id.index1 - 1;
    w->syncBody(w->shapes[slot].body);
    return w;
}

World *worldOfJoint(avbd3d_JointId id, int &rec)
{
    World *w = byIndex(id.world0);
    if (!w || !w->jointValid(id))
        return nullptr;
    rec = id.index1 - 1;
    w->syncBody(w->joints[rec].a); // a < 0 (the world) is skipped
    w->syncBody(w->joints[rec].b);
    return w;
}

// The public id of a live world (its table slot and current generation).
avbd3d_WorldId worldIdOf(const World *w)
{
    return avbd3d_WorldId{(uint16_t)(w->worldIndex + 1), table()[w->worldIndex].gen};
}
} // namespace avbd3d_impl

namespace
{
quat arcFromZ(vec3 d)
{
    // Shortest rotation taking +z to the unit vector d.
    const float c = d.z;
    if (c > 0.999999f)
        return quat{0, 0, 0, 1};
    if (c < -0.999999f)
        return quat{1, 0, 0, 0};
    const vec3 a{-d.y, d.x, 0.0f}; // z x d
    quat q{a.x, a.y, a.z, 1.0f + c};
    return normalize(q);
}

avbd3d_Quat identityQ() { return avbd3d_Quat{avbd3d_Vec3{0, 0, 0}, 1.0f}; }
} // namespace

extern "C"
{

// ---- defaults -----------------------------------------------------------------------------

avbd3d_WorldDef avbd3d_DefaultWorldDef(void)
{
    avbd3d_WorldDef d;
    std::memset(&d, 0, sizeof d);
    d.gravity = avbd3d_Vec3{0.0f, -10.0f, 0.0f};
    d.restitutionThreshold = 1.0f;
    d.hitEventThreshold = 1.0f;
    d.enableSleep = true;
    d.enableContinuous = true;
    d.internalValue = kInternalWorld;
    return d;
}

avbd3d_BodyDef avbd3d_DefaultBodyDef(void)
{
    avbd3d_BodyDef d;
    std::memset(&d, 0, sizeof d);
    d.type = avbd3d_staticBody;
    d.rotation = identityQ();
    d.gravityScale = 1.0f;
    d.sleepThreshold = 0.05f;
    d.enableSleep = true;
    d.isAwake = true;
    d.isEnabled = true;
    d.internalValue = kInternalWorld;
    return d;
}

avbd3d_Filter avbd3d_DefaultFilter(void)
{
    avbd3d_Filter f;
    f.categoryBits = 1u;
    f.maskBits = 0xFFFFFFFFFFFFFFFFull;
    f.groupIndex = 0;
    return f;
}

avbd3d_SurfaceMaterial avbd3d_DefaultSurfaceMaterial(void)
{
    avbd3d_SurfaceMaterial m;
    std::memset(&m, 0, sizeof m);
    m.friction = 0.6f;
    return m;
}

avbd3d_ShapeDef avbd3d_DefaultShapeDef(void)
{
    avbd3d_ShapeDef d;
    std::memset(&d, 0, sizeof d);
    d.baseMaterial = avbd3d_DefaultSurfaceMaterial();
    d.density = 1.0f;
    d.explosionScale = 1.0f;
    d.filter = avbd3d_DefaultFilter();
    d.isSensor = false;
    d.enableSensorEvents = false; // opt-in, as in Box3D: a visitor or sensor must ask for it
    d.enableContactEvents = false; // opt-in: a flagged shape makes the step read back its manifolds
    d.updateBodyMass = true;
    d.invokeContactCreation = true;
    d.enableSpeculativeContact = true;
    d.internalValue = kInternalWorld;
    return d;
}

static avbd3d_JointDef defaultJointBase()
{
    avbd3d_JointDef b;
    std::memset(&b, 0, sizeof b);
    b.localFrameA.q = identityQ();
    b.localFrameB.q = identityQ();
    b.forceThreshold = 3.4028235e38f; // Box3D: FLT_MAX is off, 0 reports every awake joint
    b.torqueThreshold = 3.4028235e38f;
    b.drawScale = 1.0f;
    b.internalValue = kInternalWorld;
    return b;
}

avbd3d_DistanceJointDef avbd3d_DefaultDistanceJointDef(void)
{
    avbd3d_DistanceJointDef d;
    std::memset(&d, 0, sizeof d);
    d.base = defaultJointBase();
    d.length = 1.0f;
    d.lowerSpringForce = -3.4028235e38f;
    d.upperSpringForce = 3.4028235e38f;
    d.minLength = 0.0f;
    d.maxLength = 100000.0f;
    return d;
}

avbd3d_MotorJointDef avbd3d_DefaultMotorJointDef(void)
{
    avbd3d_MotorJointDef d;
    std::memset(&d, 0, sizeof d);
    d.base = defaultJointBase();
    return d;
}

avbd3d_FilterJointDef avbd3d_DefaultFilterJointDef(void)
{
    avbd3d_FilterJointDef d;
    std::memset(&d, 0, sizeof d);
    d.base = defaultJointBase();
    return d;
}

avbd3d_ParallelJointDef avbd3d_DefaultParallelJointDef(void)
{
    avbd3d_ParallelJointDef d;
    std::memset(&d, 0, sizeof d);
    d.base = defaultJointBase();
    d.hertz = 1.0f;
    d.dampingRatio = 1.0f;
    d.maxTorque = 1000.0f;
    return d;
}

avbd3d_PrismaticJointDef avbd3d_DefaultPrismaticJointDef(void)
{
    avbd3d_PrismaticJointDef d;
    std::memset(&d, 0, sizeof d);
    d.base = defaultJointBase();
    return d;
}

avbd3d_RevoluteJointDef avbd3d_DefaultRevoluteJointDef(void)
{
    avbd3d_RevoluteJointDef d;
    std::memset(&d, 0, sizeof d);
    d.base = defaultJointBase();
    return d;
}

avbd3d_SphericalJointDef avbd3d_DefaultSphericalJointDef(void)
{
    avbd3d_SphericalJointDef d;
    std::memset(&d, 0, sizeof d);
    d.base = defaultJointBase();
    d.targetRotation = identityQ();
    return d;
}

avbd3d_WeldJointDef avbd3d_DefaultWeldJointDef(void)
{
    avbd3d_WeldJointDef d;
    std::memset(&d, 0, sizeof d);
    d.base = defaultJointBase();
    return d;
}

avbd3d_WheelJointDef avbd3d_DefaultWheelJointDef(void)
{
    avbd3d_WheelJointDef d;
    std::memset(&d, 0, sizeof d);
    d.base = defaultJointBase();
    d.enableSuspensionSpring = true;
    d.suspensionHertz = 1.0f;
    d.suspensionDampingRatio = 0.7f;
    return d;
}

// ---- hulls --------------------------------------------------------------------------------

avbd3d_BoxHull avbd3d_MakeBoxHull(float hx, float hy, float hz)
{
    // A descriptor only: enough for avbd3d_CreateHullShape to recognise a box. The vertex, edge
    // and face arrays of the full Box3D hull are not built.
    avbd3d_BoxHull h;
    std::memset(&h, 0, sizeof h);
    avbd3d_HullData &d = h.base;
    d.version = AVBD3D_HULL_VERSION;
    d.aabb.lowerBound = avbd3d_Vec3{-hx, -hy, -hz};
    d.aabb.upperBound = avbd3d_Vec3{hx, hy, hz};
    d.volume = 8.0f * hx * hy * hz;
    d.surfaceArea = 8.0f * (hx * hy + hy * hz + hx * hz);
    d.innerRadius = std::min(hx, std::min(hy, hz));
    d.vertexCount = 8;
    d.edgeCount = 24;
    d.faceCount = 6;
    d.byteCount = (int32_t)sizeof(avbd3d_HullData);
    const float m = d.volume;
    d.centralInertia.cx = avbd3d_Vec3{m * (hy * hy + hz * hz) / 3.0f, 0, 0};
    d.centralInertia.cy = avbd3d_Vec3{0, m * (hx * hx + hz * hz) / 3.0f, 0};
    d.centralInertia.cz = avbd3d_Vec3{0, 0, m * (hx * hx + hy * hy) / 3.0f};
    uint64_t hsh = 1469598103934665603ull;
    const float vals[3] = {hx, hy, hz};
    for (float v : vals)
    {
        uint32_t u;
        std::memcpy(&u, &v, 4);
        hsh = (hsh ^ u) * 1099511628211ull;
    }
    d.hash = hsh;
    return h;
}

// ---- worlds -------------------------------------------------------------------------------

avbd3d_WorldId avbd3d_CreateWorld(const avbd3d_WorldDef *def, avbd3d_Error *outErr) { return createWorldOn(def, nullptr, nullptr, *outErr); }

} // extern "C"

void avbd3d_impl::destroyWorldsOn(void *gpu)
{
    for (Slot &s : table())
        if (s.world && s.world->gpu == gpu)
        {
            s.world.reset();
            s.gen = (uint16_t)(s.gen + 1 ? s.gen + 1 : 1);
        }
}

bool avbd3d_impl::validWorld(const void *p)
{
    for (Slot &s : table())
        if (s.world && s.world.get() == p)
            return true;
    return false;
}

avbd3d_WorldId avbd3d_impl::createWorldOn(const avbd3d_WorldDef *def, Device *device, void *gpu, avbd3d_Error &err)
{
    avbd3d_WorldId null{0, 0};
    err = avbd3d_errorNone;
    if (!def || def->internalValue != kInternalWorld)
        return null;
    auto &t = table();
    for (size_t i = 0; i < t.size(); i++)
    {
        if (t[i].world)
            continue;
        try
        {
            t[i].world.reset(new World(*def, device));
        }
        catch (...)
        {
            std::string text;
            err = errorOfCurrentException(text);
            fprintf(stderr, "[avbd3d] avbd3d_CreateWorld failed: %s\n", text.c_str());
            return null;
        }
        t[i].world->worldIndex = (int)i;
        t[i].world->gpu = gpu;
        avbd3d_WorldId id;
        id.index1 = (uint16_t)(i + 1);
        id.generation = t[i].gen;
        return id;
    }
    return null;
}

extern "C"
{

void avbd3d_DestroyWorld(avbd3d_WorldId id)
{
    if (!lookupWorld(id))
        return;
    Slot &s = table()[id.index1 - 1];
    s.world.reset();
    s.gen = (uint16_t)(s.gen + 1 ? s.gen + 1 : 1);
}




void avbd3d_World_Step(avbd3d_WorldId id, float dt)
{
    if (World *w = lookupWorld(id))
    {
        w->stepBegin(dt);
        w->stepEnd();
    }
}

void avbd3d_World_StepBegin(avbd3d_WorldId id, float dt)
{
    if (World *w = lookupWorld(id))
        w->stepBegin(dt);
}


bool avbd3d_World_IsStepInFlight(avbd3d_WorldId id)
{
    World *w = lookupWorld(id);
    return w && w->inFlight();
}




// ---- contact events -----------------------------------------------------------------------

avbd3d_ContactEvents avbd3d_World_GetContactEvents(avbd3d_WorldId id)
{
    avbd3d_ContactEvents e;
    std::memset(&e, 0, sizeof e);
    if (World *w = settled(id))
    {
        e.beginEvents = w->contactBegin.data();
        e.beginCount = (int)w->contactBegin.size();
        e.endEvents = w->contactEnd.data();
        e.endCount = (int)w->contactEnd.size();
        e.hitEvents = w->contactHits.data();
        e.hitCount = (int)w->contactHits.size();
    }
    return e;
}


// ---- sensor events and explosions (Phase 4b-2) ----------------------------------------------

avbd3d_SensorEvents avbd3d_World_GetSensorEvents(avbd3d_WorldId id)
{
    avbd3d_SensorEvents e;
    std::memset(&e, 0, sizeof e);
    if (World *w = settled(id))
    {
        e.beginEvents = w->sensorBegin.data();
        e.beginCount = (int)w->sensorBegin.size();
        e.endEvents = w->sensorEnd.data();
        e.endCount = (int)w->sensorEnd.size();
    }
    return e;
}

avbd3d_ExplosionDef avbd3d_DefaultExplosionDef(void)
{
    avbd3d_ExplosionDef d;
    std::memset(&d, 0, sizeof d);
    d.maskBits = 0xFFFFFFFFFFFFFFFFull;
    return d;
}

void avbd3d_World_Explode(avbd3d_WorldId id, const avbd3d_ExplosionDef *explosionDef)
{
    if (!explosionDef)
        return;
    if (World *w = settled(id))
        w->explode(*explosionDef);
}

bool avbd3d_Shape_IsSensor(avbd3d_ShapeId id)
{
    int s;
    World *w = worldOfShape(id, s);
    return w && w->shapes[s].isSensor;
}

void avbd3d_Shape_EnableSensorEvents(avbd3d_ShapeId id, bool flag)
{
    int s;
    if (World *w = worldOfShape(id, s))
        w->forShapeGroup(s, [&](int k) { w->shapes[k].sensorEvents = flag; });
}


int avbd3d_Shape_GetSensorCapacity(avbd3d_ShapeId id)
{
    int s;
    World *w = worldOfShape(id, s);
    return w && w->shapes[s].isSensor ? (int)w->shapes[s].overlaps.size() : 0;
}

int avbd3d_Shape_GetSensorData(avbd3d_ShapeId id, avbd3d_ShapeId *visitorIds, int capacity)
{
    int s;
    World *w = worldOfShape(id, s);
    if (!w || !w->shapes[s].isSensor || !visitorIds || capacity <= 0)
        return 0;
    const std::vector<uint64_t> &ov = w->shapes[s].overlaps;
    const int n = std::min(capacity, (int)ov.size());
    for (int i = 0; i < n; i++)
        visitorIds[i] = w->shapeIdAt((int)(ov[i] >> 16), (uint16_t)(ov[i] & 0xFFFFu));
    return n;
}

void avbd3d_World_SetHitEventThreshold(avbd3d_WorldId id, float value)
{
    if (World *w = settled(id))
        w->hitThreshold = value;
}


void avbd3d_Body_EnableHitEvents(avbd3d_BodyId id, bool flag)
{
    int b;
    if (World *w = worldOfBody(id, b))
        for (int s : w->bodies[b].shapes)
        {
            w->shapes[s].hitEvents = flag;
            w->setShapeReport(s);
        }
}

int avbd3d_Body_GetContactCapacity(avbd3d_BodyId id)
{
    int b;
    World *w = worldOfBody(id, b);
    return w ? (int)w->contactsOf(b, true).size() : 0;
}

int avbd3d_Body_GetContactData(avbd3d_BodyId id, avbd3d_ContactData *data, int capacity)
{
    int b;
    World *w = worldOfBody(id, b);
    if (!w || !data || capacity <= 0)
        return 0;
    const std::vector<int> list = w->contactsOf(b, true);
    const int n = std::min(capacity, (int)list.size());
    for (int i = 0; i < n; i++)
        data[i] = w->contactData(list[i]);
    return n;
}

void avbd3d_Shape_EnableContactEvents(avbd3d_ShapeId id, bool flag)
{
    int s;
    if (World *w = worldOfShape(id, s))
        w->forShapeGroup(s, [&](int k) {
            w->shapes[k].contactEvents = flag;
            w->setShapeReport(k);
        });
}


void avbd3d_Shape_EnableHitEvents(avbd3d_ShapeId id, bool flag)
{
    int s;
    if (World *w = worldOfShape(id, s))
        w->forShapeGroup(s, [&](int k) {
            w->shapes[k].hitEvents = flag;
            w->setShapeReport(k);
        });
}


int avbd3d_Shape_GetContactCapacity(avbd3d_ShapeId id)
{
    int s;
    World *w = worldOfShape(id, s);
    return w ? (int)w->contactsOf(s, false).size() : 0;
}

int avbd3d_Shape_GetContactData(avbd3d_ShapeId id, avbd3d_ContactData *data, int capacity)
{
    int s;
    World *w = worldOfShape(id, s);
    if (!w || !data || capacity <= 0)
        return 0;
    const std::vector<int> list = w->contactsOf(s, false);
    const int n = std::min(capacity, (int)list.size());
    for (int i = 0; i < n; i++)
        data[i] = w->contactData(list[i]);
    return n;
}


avbd3d_ContactData avbd3d_Contact_GetData(avbd3d_ContactId id)
{
    avbd3d_ContactData d;
    std::memset(&d, 0, sizeof d);
    World *w = byIndex(id.world0);
    if (w && w->contactValid(id))
        d = w->contactData(id.index1 - 1);
    return d;
}

avbd3d_BodyEvents avbd3d_World_GetBodyEvents(avbd3d_WorldId id)
{
    avbd3d_BodyEvents e{nullptr, 0};
    if (World *w = settled(id))
    {
        std::vector<avbd3d_BodyMoveEvent> &ev = w->moveEvents();
        e.moveEvents = ev.data();
        e.moveCount = (int)ev.size();
    }
    return e;
}

void avbd3d_World_SetGravity(avbd3d_WorldId id, avbd3d_Vec3 g)
{
    if (World *w = settled(id))
        w->gravity = g;
}

avbd3d_Vec3 avbd3d_World_GetGravity(avbd3d_WorldId id)
{
    World *w = settled(id);
    return w ? w->gravity : avbd3d_Vec3{0, 0, 0};
}

void avbd3d_World_EnableSleeping(avbd3d_WorldId id, bool flag)
{
    World *w = settled(id);
    if (!w)
        return;
    // As Box3D: turning sleep off wakes every sleeping body, so nothing stays parked.
    if (!flag && w->sleepEnabled)
        for (int b = 1; b < (int)w->bodies.size(); b++)
            if (w->bodies[b].alive && w->bodies[b].enabled && !w->isAwake(b))
                w->setAwake(b, true);
    w->sleepEnabled = flag;
}

// ---- solver def ---------------------------------------------------------------------------

avbd3d_SolverDef avbd3d_DefaultSolverDef(void)
{
    const SolverParams p = defaultSolverParams();
    avbd3d_SolverDef d;
    std::memset(&d, 0, sizeof d);
    d.iterations = p.iterations;
    d.primalLanes = p.primalLanes;
    d.alpha = p.alpha;
    d.betaLin = p.betaLin;
    d.betaAng = p.betaAng;
    d.gamma = p.gamma;
    d.autoKStart = p.autoKStart;
    d.dualDamping = p.dualDamping;
    d.dualDampVel = p.dualDampVel;
    d.sleepFrames = p.sleepFrames;
    d.sleepLinearVelocity = p.sleepLinVel;
    d.sleepAngularVelocity = p.sleepAngVel;
    d.sleepDisplacement = p.sleepDisp;
    d.sleepFracture = p.sleepFracture;
    d.sleepLeak = p.sleepLeak;
    d.enableKillFallen = p.killFallenEnabled;
    d.killHeight = p.killHeight;
    d.grabStrength = p.grabStrength;
    d.contactCap = p.contactCap;
    d.coarseIterations = p.coarseIterations;
    d.coarseCell = p.coarseCell;
    d.coarseDamping = p.coarseDamping;
    d.worldFrameInertia = p.worldFrameInertia;
    d.internalValue = kInternalWorld;
    return d;
}

void avbd3d_World_SetSolverDef(avbd3d_WorldId id, const avbd3d_SolverDef *d)
{
    World *w = settled(id);
    if (!w || !d || d->internalValue != kInternalWorld)
        return;
    SolverParams &p = w->params;
    p.iterations = d->iterations;
    p.primalLanes = d->primalLanes;
    p.alpha = d->alpha;
    p.betaLin = d->betaLin;
    p.betaAng = d->betaAng;
    p.gamma = d->gamma;
    p.autoKStart = d->autoKStart;
    p.dualDamping = d->dualDamping;
    p.dualDampVel = d->dualDampVel;
    p.sleepFrames = d->sleepFrames;
    p.sleepLinVel = d->sleepLinearVelocity;
    p.sleepAngVel = d->sleepAngularVelocity;
    p.sleepDisp = d->sleepDisplacement;
    p.sleepFracture = d->sleepFracture;
    p.sleepLeak = d->sleepLeak;
    p.killFallenEnabled = d->enableKillFallen;
    p.killHeight = d->killHeight;
    p.grabStrength = d->grabStrength;
    p.contactCap = d->contactCap;
    p.coarseIterations = d->coarseIterations;
    p.coarseCell = d->coarseCell;
    p.coarseDamping = d->coarseDamping;
    p.worldFrameInertia = d->worldFrameInertia;
}


// ---- batched body access ------------------------------------------------------------------

extern "C++" {
// One body read through the mirror (syncBody pulls it from the snapshot when stale).
static void readBodyMirror(World *w, int b, avbd3d_BodyDataType type, unsigned char *out, size_t sz)
{
    w->syncBody(b);
    switch (type)
    {
    case avbd3d_bodyDataPosition: { avbd3d_Vec3 v = fromV(w->originPos(b)); std::memcpy(out, &v, sz); break; }
    case avbd3d_bodyDataRotation: { avbd3d_Quat q = fromQ(w->originRot(b)); std::memcpy(out, &q, sz); break; }
    case avbd3d_bodyDataTransform:
    {
        avbd3d_Transform t{fromV(w->originPos(b)), fromQ(w->originRot(b))};
        std::memcpy(out, &t, sz);
        break;
    }
    case avbd3d_bodyDataLinearVelocity: { avbd3d_Vec3 v = fromV(w->desc.velLin[b]); std::memcpy(out, &v, sz); break; }
    case avbd3d_bodyDataAngularVelocity: { avbd3d_Vec3 v = fromV(w->desc.velAng[b]); std::memcpy(out, &v, sz); break; }
    case avbd3d_bodyDataAwake: *out = w->isAwake(b) ? 1 : 0; break;
    default: break;
    }
}

// The batch loop with the type fixed at compile time. A movable body the mirror is behind on
// (and no host write holds) is read straight from the finished snapshot: no mirror or proxy
// bookkeeping, same values syncBody would have copied. Anything else goes through the mirror.
template <avbd3d_BodyDataType T, size_t SZ>
static int readBodies(World *w, const avbd3d_BodyId *ids, int count, unsigned char *out)
{
    int valid = 0;
    for (int i = 0; i < count; i++, out += SZ)
    {
        bool plain = false;
        const VkWorld::SnapshotView *sv = w->snapshotFor(ids[i], &plain);
        const int b = ids[i].index1;
        if (!sv)
        {
            std::memset(out, 0, SZ);
            if (!w->bodyValid(ids[i]))
                continue;
            valid++;
            readBodyMirror(w, b, T, out, SZ);
            continue;
        }
        valid++;
        if (T == avbd3d_bodyDataLinearVelocity) { avbd3d_Vec3 v = fromV(sv->velLin[b]); std::memcpy(out, &v, SZ); }
        else if (T == avbd3d_bodyDataAngularVelocity) { avbd3d_Vec3 v = fromV(sv->velAng[b]); std::memcpy(out, &v, SZ); }
        else if (T == avbd3d_bodyDataAwake) *out = sv->awake[b] ? 1 : 0;
        else
        {
            // plain: the origin is the centre of mass and the principal frame is the body frame
            const quat oq = plain ? sv->rot[b] : sv->rot[b] * conjugate(w->desc.bodyPrincipalRot[b]);
            if (T == avbd3d_bodyDataRotation) { avbd3d_Quat q = fromQ(oq); std::memcpy(out, &q, SZ); }
            else
            {
                const avbd3d_Vec3 p = plain ? fromV(sv->pos[b]) : fromV(sv->pos[b] - rotate(oq, w->desc.bodyLocalCenter[b]));
                if (T == avbd3d_bodyDataPosition) std::memcpy(out, &p, SZ);
                else { avbd3d_Transform t{p, fromQ(oq)}; std::memcpy(out, &t, SZ); }
            }
        }
    }
    return valid;
}

} // extern "C++"

int avbd3d_World_GetBodyData(avbd3d_WorldId id, const avbd3d_BodyId *ids, int count, avbd3d_BodyDataType type, void *data)
{
    World *w = settled(id);
    if (!w || type < 0 || type >= avbd3d_bodyDataTypeCount)
        return 0;
    unsigned char *out = (unsigned char *)data;
    switch (type)
    {
    case avbd3d_bodyDataPosition: return readBodies<avbd3d_bodyDataPosition, sizeof(avbd3d_Vec3)>(w, ids, count, out);
    case avbd3d_bodyDataRotation: return readBodies<avbd3d_bodyDataRotation, sizeof(avbd3d_Quat)>(w, ids, count, out);
    case avbd3d_bodyDataTransform: return readBodies<avbd3d_bodyDataTransform, sizeof(avbd3d_Transform)>(w, ids, count, out);
    case avbd3d_bodyDataLinearVelocity: return readBodies<avbd3d_bodyDataLinearVelocity, sizeof(avbd3d_Vec3)>(w, ids, count, out);
    case avbd3d_bodyDataAngularVelocity: return readBodies<avbd3d_bodyDataAngularVelocity, sizeof(avbd3d_Vec3)>(w, ids, count, out);
    case avbd3d_bodyDataAwake: return readBodies<avbd3d_bodyDataAwake, 1>(w, ids, count, out);
    default: return 0;
    }
}

int avbd3d_World_SetBodyData(avbd3d_WorldId id, const avbd3d_BodyId *ids, int count, avbd3d_BodyDataType type, const void *data)
{
    World *w = settled(id);
    if (!w || type < 0 || type >= avbd3d_bodyDataAwake)
        return 0;
    static const size_t sizes[] = {sizeof(avbd3d_Vec3), sizeof(avbd3d_Quat), sizeof(avbd3d_Transform),
                                   sizeof(avbd3d_Vec3), sizeof(avbd3d_Vec3)};
    const size_t sz = sizes[type];
    const unsigned char *in = (const unsigned char *)data;
    int valid = 0;
    for (int i = 0; i < count; i++, in += sz)
    {
        if (!w->bodyValid(ids[i]))
            continue;
        const int b = ids[i].index1;
        valid++;
        switch (type)
        {
        case avbd3d_bodyDataPosition:
        {
            avbd3d_Vec3 v;
            std::memcpy(&v, in, sz);
            w->setTransform(b, toV(v), w->originRot(b));
            break;
        }
        case avbd3d_bodyDataRotation:
        {
            avbd3d_Quat q;
            std::memcpy(&q, in, sz);
            w->setTransform(b, w->originPos(b), toQ(q));
            break;
        }
        case avbd3d_bodyDataTransform:
        {
            avbd3d_Transform t;
            std::memcpy(&t, in, sz);
            w->setTransform(b, toV(t.p), toQ(t.q));
            break;
        }
        case avbd3d_bodyDataLinearVelocity:
        {
            avbd3d_Vec3 v;
            std::memcpy(&v, in, sz);
            w->setLinearVelocity(b, toV(v));
            break;
        }
        case avbd3d_bodyDataAngularVelocity:
        {
            avbd3d_Vec3 v;
            std::memcpy(&v, in, sz);
            w->setAngularVelocity(b, toV(v));
            break;
        }
        default: break;
        }
    }
    return valid;
}

// ---- capabilities -------------------------------------------------------------------------

static avbd3d_FeatureStatus featureStatus(avbd3d_Feature f)
{
    switch (f)
    {
    case avbd3d_featureSphereShape:
    case avbd3d_featureCapsuleShape:
    case avbd3d_featureHullShape:
    case avbd3d_featureMeshShape:        // static bodies only
    case avbd3d_featureHeightFieldShape: // static bodies only, cooked as a mesh
    case avbd3d_featureWeldJoint:
    case avbd3d_featureRevoluteJoint:
    case avbd3d_featurePrismaticJoint:
    case avbd3d_featureWheelJoint:
    case avbd3d_featureSphericalJoint:
    case avbd3d_featureDistanceJoint:
    case avbd3d_featureMotorJoint:
    case avbd3d_featureParallelJoint:
    case avbd3d_featureFilterJoint:
    case avbd3d_featureBodyEvents:
    case avbd3d_featureCompoundShape:
    case avbd3d_featureJointEvents:
    case avbd3d_featureContactEvents:
    case avbd3d_featureHitEvents:
    case avbd3d_featureContactData:
    case avbd3d_featureSensors:
    case avbd3d_featureSensorEvents:
    case avbd3d_featureExplosions:
    case avbd3d_featureAsyncStep:
    case avbd3d_featureSolverDef:
    case avbd3d_featureBatchedBodyAccess:
    case avbd3d_featureRayCast:
    case avbd3d_featureShapeCast:
    case avbd3d_featureOverlap:
    case avbd3d_featureCharacterMover:
    case avbd3d_featureDebugDraw:
    case avbd3d_featureBodyDamping:   // BodyProps, applied before the inertial target
    case avbd3d_featureGravityScale:  // BodyProps
    case avbd3d_featureKinematicBody: // BODY_KINEMATIC, advanced by csIntegrate
    case avbd3d_featureRestitution:   // velocity pass after the solve (restitution.slang)
    case avbd3d_featureGrab:          // the drag joint pool (VkWorld::dragBodies)
    case avbd3d_featureAvbdJoint:     // raw joint and spring rows (avbd3d_CreateAvbdJoint)
    case avbd3d_featureMotionLocks:        // BODY_LOCK_* props flags
    case avbd3d_featureRollingResistance:  // primal.slang rolling row, BodyProps.rollingResistance
    case avbd3d_featureTangentVelocity:    // narrowphase belt velocity (body-frame vec3)
    case avbd3d_featureFrictionMixing:     // avbd3d_World_SetFrictionMixing / SetRestitutionMixing
    case avbd3d_featureJointFracture: // avbd3d_AvbdJointDef fractureForce / fractureTorque
    case avbd3d_featureSpeculativeContacts: // the 1 cm narrowphase margin (docs/SOLVER_3D_ROADMAP.md Phase 5)
    case avbd3d_featureContinuous:    // post-solve time-of-impact clamp (ccd.slang)
    case avbd3d_featureBullets:       // BODY_BULLET: also swept against non-bullet dynamics
    case avbd3d_featureSpringDamping: // joint spring dampingRatio: implicit damper (forces.slang)
    case avbd3d_featureWorldFrameInertia: // SolverParams.worldFrameInertia, default on (primal.slang)
        return avbd3d_featureSupported;
    default:
        return avbd3d_featureUnsupported;
    }
}





// ---- bodies -------------------------------------------------------------------------------

avbd3d_BodyId avbd3d_CreateBody(avbd3d_WorldId id, const avbd3d_BodyDef *def)
{
    World *w = settled(id);
    if (!w || !def)
        return avbd3d_nullBodyId;
    return w->createBody(*def);
}

void avbd3d_DestroyBody(avbd3d_BodyId id)
{
    int b;
    if (World *w = worldOfBody(id, b))
        w->destroyBody(b);
}


avbd3d_BodyType avbd3d_Body_GetType(avbd3d_BodyId id)
{
    int b;
    World *w = worldOfBody(id, b);
    return w ? (avbd3d_BodyType)w->bodies[b].type : avbd3d_staticBody;
}

void avbd3d_Body_SetType(avbd3d_BodyId id, avbd3d_BodyType type)
{
    int b;
    if (World *w = worldOfBody(id, b))
        w->setType(b, type);
}

void avbd3d_Body_SetUserData(avbd3d_BodyId id, void *u)
{
    int b;
    if (World *w = worldOfBody(id, b))
        w->setBodyUser(b, u);
}

void *avbd3d_Body_GetUserData(avbd3d_BodyId id)
{
    int b;
    World *w = worldOfBody(id, b);
    return w ? w->bodies[b].user : nullptr;
}

avbd3d_Pos avbd3d_Body_GetPosition(avbd3d_BodyId id)
{
    int b;
    World *w = worldOfBody(id, b);
    return w ? fromV(w->originPos(b)) : avbd3d_Vec3{0, 0, 0};
}

avbd3d_Quat avbd3d_Body_GetRotation(avbd3d_BodyId id)
{
    int b;
    World *w = worldOfBody(id, b);
    return w ? fromQ(w->originRot(b)) : identityQ();
}

avbd3d_WorldTransform avbd3d_Body_GetTransform(avbd3d_BodyId id)
{
    return avbd3d_WorldTransform{avbd3d_Body_GetPosition(id), avbd3d_Body_GetRotation(id)};
}

void avbd3d_Body_SetTransform(avbd3d_BodyId id, avbd3d_Pos p, avbd3d_Quat q)
{
    int b;
    if (World *w = worldOfBody(id, b))
        w->setTransform(b, toV(p), toQ(q));
}

avbd3d_Vec3 avbd3d_Body_GetLocalPoint(avbd3d_BodyId id, avbd3d_Pos p)
{
    int b;
    World *w = worldOfBody(id, b);
    if (!w)
        return p;
    return fromV(rotate(inverse(w->originRot(b)), toV(p) - w->originPos(b)));
}

avbd3d_Pos avbd3d_Body_GetWorldPoint(avbd3d_BodyId id, avbd3d_Vec3 p)
{
    int b;
    World *w = worldOfBody(id, b);
    if (!w)
        return p;
    return fromV(w->originPos(b) + rotate(w->originRot(b), toV(p)));
}

avbd3d_Vec3 avbd3d_Body_GetLocalVector(avbd3d_BodyId id, avbd3d_Vec3 v)
{
    int b;
    World *w = worldOfBody(id, b);
    return w ? fromV(rotate(inverse(w->originRot(b)), toV(v))) : v;
}

avbd3d_Vec3 avbd3d_Body_GetWorldVector(avbd3d_BodyId id, avbd3d_Vec3 v)
{
    int b;
    World *w = worldOfBody(id, b);
    return w ? fromV(rotate(w->originRot(b), toV(v))) : v;
}

avbd3d_Vec3 avbd3d_Body_GetLinearVelocity(avbd3d_BodyId id)
{
    int b;
    World *w = worldOfBody(id, b);
    return w ? fromV(w->desc.velLin[b]) : avbd3d_Vec3{0, 0, 0};
}

avbd3d_Vec3 avbd3d_Body_GetAngularVelocity(avbd3d_BodyId id)
{
    int b;
    World *w = worldOfBody(id, b);
    return w ? fromV(w->desc.velAng[b]) : avbd3d_Vec3{0, 0, 0};
}

void avbd3d_Body_SetLinearVelocity(avbd3d_BodyId id, avbd3d_Vec3 v)
{
    int b;
    if (World *w = worldOfBody(id, b))
        w->setLinearVelocity(b, toV(v));
}

void avbd3d_Body_SetAngularVelocity(avbd3d_BodyId id, avbd3d_Vec3 v)
{
    int b;
    if (World *w = worldOfBody(id, b))
        w->setAngularVelocity(b, toV(v));
}

// Box3D's rule (b3Body_SetTargetTransform): the velocity that carries the body's centre of mass and
// orientation to the target in timeStep. The angular velocity is the exact inverse of the solver's
// orientation update q' = normalize(q + 0.5 dt (w,0) q) (integrate.slang), so a kinematic body lands on
// the target rather than close to it: the shortest arc target is rescaled to q2.q1 = 1, which makes the
// scalar part of (q2 - q1) q1^-1 vanish and the vector part exactly the needed 2 w dt.
void avbd3d_Body_SetTargetTransform(avbd3d_BodyId id, avbd3d_WorldTransform target, float timeStep, bool wake)
{
    int b;
    World *w = worldOfBody(id, b);
    if (!w)
        return;
    BodyRec &r = w->bodies[b];
    if (!r.enabled || r.type == avbd3d_staticBody || !(timeStep > 0.0f))
        return;
    const bool awakeNow = w->isAwake(b);
    if (!awakeNow && !wake)
        return;

    const quat q1 = w->originRot(b);
    quat q2 = normalize(toQ(target.q));
    const vec3 lc = w->desc.bodyLocalCenter[b];
    const vec3 center1 = w->originPos(b) + rotate(q1, lc);
    const vec3 center2 = toV(target.p) + rotate(q2, lc);
    const float invDt = 1.0f / timeStep;
    const vec3 linear = (center2 - center1) * invDt;

    float d = q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w;
    if (d < 0.0f)
    {
        q2 = quat{-q2.x, -q2.y, -q2.z, -q2.w};
        d = -d;
    }
    // Rescale to q2'.q1 = 1 (needs d > 0; a half-turn step has no exact inverse, keep Box3D's form).
    const float k = d > 0.1f ? 1.0f / d : 1.0f;
    const quat dq{(q2.x * k) - q1.x, (q2.y * k) - q1.y, (q2.z * k) - q1.z, (q2.w * k) - q1.w};
    const quat om = dq * conjugate(q1);
    const vec3 angular = vec3{om.x, om.y, om.z} * (2.0f * invDt);

    if (!awakeNow)
    {
        // Box3D: a sleeping body only wakes if the movement is above its sleep threshold.
        const avbd3d_AABB box = avbd3d_Body_ComputeAABB(id);
        const vec3 ext{0.5f * (box.upperBound.x - box.lowerBound.x), 0.5f * (box.upperBound.y - box.lowerBound.y),
                       0.5f * (box.upperBound.z - box.lowerBound.z)};
        const float maxVelocity = length(linear) + length(angular) * length(ext);
        if (maxVelocity < r.sleepThreshold)
            return;
    }
    w->setLinearVelocity(b, linear);
    w->setAngularVelocity(b, angular);
}

avbd3d_Vec3 avbd3d_Body_GetWorldPointVelocity(avbd3d_BodyId id, avbd3d_Pos p)
{
    int b;
    World *w = worldOfBody(id, b);
    if (!w)
        return avbd3d_Vec3{0, 0, 0};
    const vec3 r = toV(p) - w->desc.pos[b];
    return fromV(w->desc.velLin[b] + cross(w->desc.velAng[b], r));
}

avbd3d_Vec3 avbd3d_Body_GetLocalPointVelocity(avbd3d_BodyId id, avbd3d_Vec3 p)
{
    return avbd3d_Body_GetWorldPointVelocity(id, avbd3d_Body_GetWorldPoint(id, p));
}

void avbd3d_Body_ApplyForce(avbd3d_BodyId id, avbd3d_Vec3 f, avbd3d_Pos p, bool wake)
{
    int b;
    if (World *w = worldOfBody(id, b))
        w->applyForce(b, toV(f), toV(p), wake);
}

void avbd3d_Body_ApplyForceToCenter(avbd3d_BodyId id, avbd3d_Vec3 f, bool wake)
{
    int b;
    if (World *w = worldOfBody(id, b))
        w->applyForce(b, toV(f), w->desc.pos[b], wake);
}

void avbd3d_Body_ApplyTorque(avbd3d_BodyId id, avbd3d_Vec3 t, bool wake)
{
    int b;
    if (World *w = worldOfBody(id, b))
        w->applyTorque(b, toV(t), wake);
}

void avbd3d_Body_ApplyLinearImpulse(avbd3d_BodyId id, avbd3d_Vec3 j, avbd3d_Pos p, bool wake)
{
    int b;
    if (World *w = worldOfBody(id, b))
        w->applyImpulse(b, toV(j), toV(p), wake);
}

void avbd3d_Body_ApplyLinearImpulseToCenter(avbd3d_BodyId id, avbd3d_Vec3 j, bool wake)
{
    int b;
    if (World *w = worldOfBody(id, b))
        w->applyImpulse(b, toV(j), w->desc.pos[b], wake);
}

void avbd3d_Body_ApplyAngularImpulse(avbd3d_BodyId id, avbd3d_Vec3 j, bool wake)
{
    int b;
    if (World *w = worldOfBody(id, b))
        w->applyAngularImpulse(b, toV(j), wake);
}

float avbd3d_Body_GetMass(avbd3d_BodyId id)
{
    int b;
    World *w = worldOfBody(id, b);
    return w ? w->desc.mass[b] : 0.0f;
}

float avbd3d_Body_GetInverseMass(avbd3d_BodyId id)
{
    const float m = avbd3d_Body_GetMass(id);
    return m > 0 ? 1.0f / m : 0.0f;
}

avbd3d_Matrix3 avbd3d_Body_GetLocalRotationalInertia(avbd3d_BodyId id)
{
    int b;
    World *w = worldOfBody(id, b);
    avbd3d_Matrix3 m;
    std::memset(&m, 0, sizeof m);
    if (w)
        m = principalToMatrix(w->desc.moment[b], w->desc.bodyPrincipalRot[b]);
    return m;
}

avbd3d_Vec3 avbd3d_Body_GetLocalCenter(avbd3d_BodyId id)
{
    int b;
    World *w = worldOfBody(id, b);
    return w ? fromV(w->desc.bodyLocalCenter[b]) : avbd3d_Vec3{0, 0, 0};
}

avbd3d_Pos avbd3d_Body_GetWorldCenter(avbd3d_BodyId id)
{
    int b;
    World *w = worldOfBody(id, b);
    return w ? fromV(w->desc.pos[b]) : avbd3d_Vec3{0, 0, 0};
}

void avbd3d_Body_SetLinearDamping(avbd3d_BodyId id, float v)
{
    int b;
    if (World *w = worldOfBody(id, b))
    {
        w->bodies[b].linearDamping = v;
        w->syncProps(b);
    }
}


void avbd3d_Body_SetAngularDamping(avbd3d_BodyId id, float v)
{
    int b;
    if (World *w = worldOfBody(id, b))
    {
        w->bodies[b].angularDamping = v;
        w->syncProps(b);
    }
}


void avbd3d_Body_SetGravityScale(avbd3d_BodyId id, float v)
{
    int b;
    if (World *w = worldOfBody(id, b))
    {
        w->bodies[b].gravityScale = v;
        w->syncProps(b);
    }
}


void avbd3d_Body_SetMotionLocks(avbd3d_BodyId id, avbd3d_MotionLocks locks)
{
    int b;
    if (World *w = worldOfBody(id, b))
    {
        w->bodies[b].locks = locks;
        w->syncProps(b);
    }
}


bool avbd3d_Body_IsAwake(avbd3d_BodyId id)
{
    int b;
    World *w = worldOfBody(id, b);
    return w && w->bodies[b].enabled && w->isAwake(b);
}

void avbd3d_Body_SetAwake(avbd3d_BodyId id, bool awake)
{
    int b;
    if (World *w = worldOfBody(id, b))
        w->setAwake(b, awake);
}





bool avbd3d_Body_IsEnabled(avbd3d_BodyId id)
{
    int b;
    World *w = worldOfBody(id, b);
    return w && w->bodies[b].enabled;
}

void avbd3d_Body_Disable(avbd3d_BodyId id)
{
    int b;
    if (World *w = worldOfBody(id, b))
        w->setEnabled(b, false);
}

void avbd3d_Body_Enable(avbd3d_BodyId id)
{
    int b;
    if (World *w = worldOfBody(id, b))
        w->setEnabled(b, true);
}

avbd3d_WorldId avbd3d_Body_GetWorld(avbd3d_BodyId id)
{
    World *w = byIndex(id.world0);
    avbd3d_WorldId r{0, 0};
    if (w)
    {
        r.index1 = (uint16_t)(id.world0 + 1);
        r.generation = table()[id.world0].gen;
    }
    return r;
}

int avbd3d_Body_GetShapeCount(avbd3d_BodyId id)
{
    int b;
    World *w = worldOfBody(id, b);
    if (!w)
        return 0;
    int n = 0;
    for (int s : w->bodies[b].shapes)
        n += w->shapes[s].parent < 0;
    return n;
}

int avbd3d_Body_GetShapes(avbd3d_BodyId id, avbd3d_ShapeId *out, int capacity)
{
    int b;
    World *w = worldOfBody(id, b);
    if (!w)
        return 0;
    int n = 0;
    for (int s : w->bodies[b].shapes)
        if (w->shapes[s].parent < 0 && n < capacity)
            out[n++] = w->shapeId(s);
    return n;
}

// ---- shapes -------------------------------------------------------------------------------

avbd3d_ShapeId avbd3d_CreateSphereShape(avbd3d_BodyId id, const avbd3d_ShapeDef *def, const avbd3d_Sphere *s)
{
    int b;
    World *w = worldOfBody(id, b);
    if (!w || !def || !s || !(s->radius > 0.0f))
        return avbd3d_nullShapeId;
    return w->createShape(b, *def, AVBD_SHAPE_SPHERE, vec3{s->radius, s->radius, s->radius}, toV(s->center),
                          quat{0, 0, 0, 1});
}

avbd3d_ShapeId avbd3d_CreateCapsuleShape(avbd3d_BodyId id, const avbd3d_ShapeDef *def, const avbd3d_Capsule *c)
{
    int b;
    World *w = worldOfBody(id, b);
    if (!w || !def || !c || !(c->radius > 0.0f))
        return avbd3d_nullShapeId;
    const vec3 p1 = toV(c->center1), p2 = toV(c->center2);
    const vec3 axis = p2 - p1;
    const float len = length(axis);
    const quat q = len > 1e-6f ? arcFromZ(axis * (1.0f / len)) : quat{0, 0, 0, 1};
    return w->createShape(b, *def, AVBD_SHAPE_CAPSULE, vec3{c->radius, c->radius, c->radius + 0.5f * len},
                          (p1 + p2) * 0.5f, q);
}

// Box3D's b3MakeBoxHull-shaped hull test: exactly the box fast path the solver already had.
static bool isBoxHull(const avbd3d_HullData *h)
{
    const avbd3d_Vec3 lo = h->aabb.lowerBound, hi = h->aabb.upperBound;
    const float ex = hi.x - lo.x, ey = hi.y - lo.y, ez = hi.z - lo.z;
    const float vol = ex * ey * ez;
    return h->vertexCount == 8 && h->faceCount == 6 && vol > 0.0f && std::fabs(h->volume - vol) <= 1e-3f * vol;
}

static avbd3d_ShapeId createBoxHull(avbd3d_BodyId id, const avbd3d_ShapeDef *def, const avbd3d_HullData *h,
                                    const avbd3d_Transform *t, avbd3d_Vec3 scale)
{
    int b;
    World *w = worldOfBody(id, b);
    if (!w || !def || !h)
        return avbd3d_nullShapeId;
    const avbd3d_Vec3 lo = h->aabb.lowerBound, hi = h->aabb.upperBound;
    const float ex = hi.x - lo.x, ey = hi.y - lo.y, ez = hi.z - lo.z;
    const vec3 centre{0.5f * (lo.x + hi.x) * scale.x, 0.5f * (lo.y + hi.y) * scale.y, 0.5f * (lo.z + hi.z) * scale.z};
    const vec3 half{0.5f * ex * std::fabs(scale.x), 0.5f * ey * std::fabs(scale.y), 0.5f * ez * std::fabs(scale.z)};
    const quat q = t ? normalize(toQ(t->q)) : quat{0, 0, 0, 1};
    const vec3 p = (t ? toV(t->p) : vec3{0, 0, 0}) + rotate(q, centre);
    return w->createShape(b, *def, AVBD_SHAPE_BOX, half, p, q);
}

// Any other convex hull: a Box3D clone with the scale baked in, stored as an AVBD_SHAPE_HULL blob.
// The hull lives in the shape frame, so the transform is the shape's local pose.
static avbd3d_ShapeId createGeneralHull(avbd3d_BodyId id, const avbd3d_ShapeDef *def, const avbd3d_HullData *h,
                                        const avbd3d_Transform *t, avbd3d_Vec3 scale)
{
    int b;
    World *w = worldOfBody(id, b);
    if (!w || !def || !h || h->vertexCount < 1 || h->vertexCount > AVBD3D_MAX_HULL_VERTICES)
        return avbd3d_nullShapeId;
    const b3HullData *src = reinterpret_cast<const b3HullData *>(h);
    if (src->version != B3_HULL_VERSION)
        return avbd3d_nullShapeId;
    if (!(std::fabs(scale.x) > 1e-6f && std::fabs(scale.y) > 1e-6f && std::fabs(scale.z) > 1e-6f))
        return avbd3d_nullShapeId;
    const b3Transform identity{b3Vec3{0, 0, 0}, b3Quat{b3Vec3{0, 0, 0}, 1.0f}};
    b3HullData *clone = b3CloneAndTransformHull(src, identity, b3Vec3{scale.x, scale.y, scale.z});
    if (clone == nullptr)
        return avbd3d_nullShapeId;
    if (clone->vertexCount > AVBD3D_MAX_HULL_VERTICES)
    {
        b3DestroyHull(clone); // refused, never truncated
        return avbd3d_nullShapeId;
    }
    const int blob = w->internGeometryBlob(reinterpret_cast<const unsigned char *>(clone), (size_t)clone->byteCount,
                                           clone->hash);
    const auto amax = [](float a, float b2) { return std::max(std::fabs(a), std::fabs(b2)); };
    const vec3 half{amax(clone->aabb.lowerBound.x, clone->aabb.upperBound.x),
                    amax(clone->aabb.lowerBound.y, clone->aabb.upperBound.y),
                    amax(clone->aabb.lowerBound.z, clone->aabb.upperBound.z)};
    const quat q = t ? normalize(toQ(t->q)) : quat{0, 0, 0, 1};
    const vec3 p = t ? toV(t->p) : vec3{0, 0, 0};
    const avbd3d_ShapeId sid = w->createShape(b, *def, AVBD_SHAPE_HULL, half, p, q);
    int s;
    if (World *ws = worldOfShape(sid, s))
    {
        ShapeGeom g;
        g.hull.reset(clone);
        ws->attachGeometry(s, avbd3d_hullShape, blob, std::move(g));
    }
    else
    {
        b3DestroyHull(clone);
    }
    return sid;
}

static avbd3d_ShapeId createHullAny(avbd3d_BodyId id, const avbd3d_ShapeDef *def, const avbd3d_HullData *h,
                                    const avbd3d_Transform *t, avbd3d_Vec3 scale)
{
    if (h && isBoxHull(h))
        return createBoxHull(id, def, h, t, scale);
    return createGeneralHull(id, def, h, t, scale);
}

avbd3d_ShapeId avbd3d_CreateHullShape(avbd3d_BodyId id, const avbd3d_ShapeDef *def, const avbd3d_HullData *hull)
{
    return createHullAny(id, def, hull, nullptr, avbd3d_Vec3{1, 1, 1});
}

avbd3d_ShapeId avbd3d_CreateTransformedHullShape(avbd3d_BodyId id, const avbd3d_ShapeDef *def,
                                                 const avbd3d_HullData *hull, avbd3d_Transform t, avbd3d_Vec3 scale)
{
    return createHullAny(id, def, hull, &t, scale);
}

// Height-field shapes are static only. A triangle mesh may sit on a kinematic body, or on a
// dynamic one when it is closed (it needs a volume for mass); mesh-vs-mesh pairs are skipped.
static bool isStaticBody(const World *w, int b)
{
    return w->bodies[b].type == avbd3d_staticBody;
}

// Cooks a triangle mesh (already in body/shape coordinates) with Box3D, as SceneBuilder does.
static b3MeshData *cookTriangles(const std::vector<b3Vec3> &verts, const std::vector<int32_t> &tris)
{
    b3MeshDef def = {};
    def.vertices = const_cast<b3Vec3 *>(verts.data());
    def.vertexCount = (int)verts.size();
    def.indices = const_cast<int32_t *>(tris.data());
    def.triangleCount = (int)(tris.size() / 3);
    def.identifyEdges = true; // per-triangle edge flags, as SceneBuilder::cookMesh
    def.weldVertices = false; // keep the caller's indices
    b3MeshData *mesh = b3CreateMesh(&def, nullptr, 0);
    if (mesh != nullptr && (mesh->version != B3_MESH_VERSION || mesh->byteCount <= (int)sizeof(b3MeshData)))
    {
        b3DestroyMesh(mesh);
        return nullptr;
    }
    return mesh;
}

static vec3 meshHalf(const b3MeshData *m)
{
    const auto amax = [](float a, float b2) { return std::max(std::fabs(a), std::fabs(b2)); };
    return vec3{amax(m->bounds.lowerBound.x, m->bounds.upperBound.x),
                amax(m->bounds.lowerBound.y, m->bounds.upperBound.y),
                amax(m->bounds.lowerBound.z, m->bounds.upperBound.z)};
}

static avbd3d_ShapeId createMeshImpl(avbd3d_BodyId id, const avbd3d_ShapeDef *def, const avbd3d_MeshData *mesh,
                                     avbd3d_Vec3 scale, const avbd3d_Transform *xf)
{
    int b;
    World *w = worldOfBody(id, b);
    // Mesh scale is baked: the vertices are scaled and re-cooked. A mirroring scale flips winding.
    if (!w || !def || !mesh)
        return avbd3d_nullShapeId;
    if (!(std::fabs(scale.x) > 1e-6f && std::fabs(scale.y) > 1e-6f && std::fabs(scale.z) > 1e-6f))
        return avbd3d_nullShapeId;
    const int vc = mesh->vertexCount, tc = mesh->triangleCount;
    if (vc < 3 || tc < 1 || tc >= (1 << 21) - 1)
        return avbd3d_nullShapeId;
    const avbd3d_Vec3 *src = avbd3d_GetMeshVertices(mesh);
    const avbd3d_MeshTriangle *st = avbd3d_GetMeshTriangles(mesh);
    std::vector<b3Vec3> verts((size_t)vc);
    for (int i = 0; i < vc; i++)
    {
        vec3 v{src[i].x * scale.x, src[i].y * scale.y, src[i].z * scale.z};
        if (xf) // a compound child's pose, baked into the vertices
            v = toV(xf->p) + rotate(normalize(toQ(xf->q)), v);
        verts[(size_t)i] = b3Vec3{v.x, v.y, v.z};
    }
    const bool mirrored = ((scale.x < 0.0f) != (scale.y < 0.0f)) != (scale.z < 0.0f);
    std::vector<int32_t> tris((size_t)tc * 3);
    for (int t = 0; t < tc; t++)
    {
        const int32_t a = st[t].index1, bi = st[t].index2, c = st[t].index3;
        if (a < 0 || a >= vc || bi < 0 || bi >= vc || c < 0 || c >= vc)
            return avbd3d_nullShapeId;
        tris[(size_t)t * 3 + 0] = a;
        tris[(size_t)t * 3 + 1] = mirrored ? c : bi;
        tris[(size_t)t * 3 + 2] = mirrored ? bi : c;
    }
    b3MeshData *cooked = cookTriangles(verts, tris);
    if (cooked == nullptr)
        return avbd3d_nullShapeId;
    if (w->bodies[b].type == avbd3d_dynamicBody && !meshIsClosed(cooked))
    {
#ifndef NDEBUG
        fprintf(stderr, "[avbd3d] a dynamic body needs a closed mesh (every edge shared by two triangles, volume > 0)\n");
#endif
        b3DestroyMesh(cooked);
        return avbd3d_nullShapeId;
    }
    const int blob = w->internGeometryBlob(reinterpret_cast<const unsigned char *>(cooked), (size_t)cooked->byteCount,
                                           cooked->hash);
    const avbd3d_ShapeId sid = w->createShape(b, *def, AVBD_SHAPE_MESH, meshHalf(cooked), vec3{0, 0, 0},
                                              quat{0, 0, 0, 1});
    int s;
    if (World *ws = worldOfShape(sid, s))
    {
        ShapeGeom g;
        g.mesh.reset(cooked);
        ws->attachGeometry(s, avbd3d_meshShape, blob, std::move(g));
    }
    else
    {
        b3DestroyMesh(cooked);
    }
    return sid;
}

avbd3d_ShapeId avbd3d_CreateMeshShape(avbd3d_BodyId id, const avbd3d_ShapeDef *def, const avbd3d_MeshData *mesh,
                                      avbd3d_Vec3 scale)
{
    return createMeshImpl(id, def, mesh, scale, nullptr);
}

// A baked compound: each child becomes a shape of its own on the body, with the child pose and
// material; the first child's id is the compound id and every child maps to it
// (ShapeRec::parent). A static, kinematic or dynamic body may own one; a dynamic body needs
// closed meshes. A failed child undoes the ones already made. The per-shape setters
// (density, friction, restitution, material, filter, events, user data) apply to every child.
// Mass is recomputed once, after the loop, when def->updateBodyMass.
avbd3d_ShapeId avbd3d_CreateBakedCompoundShape(avbd3d_BodyId id, avbd3d_ShapeDef *def,
                                               const avbd3d_CompoundData *compound)
{
    int b;
    World *w = worldOfBody(id, b);
    if (!w || !def || !compound || compound->version != AVBD3D_COMPOUND_VERSION)
        return avbd3d_nullShapeId;
    const int count = compound->capsuleCount + compound->hullCount + compound->meshCount + compound->sphereCount;
    if (count < 1)
        return avbd3d_nullShapeId;
    const avbd3d_SurfaceMaterial *mats = avbd3d_GetCompoundMaterials(compound);
    std::vector<avbd3d_ShapeId> made;
    for (int i = 0; i < count; i++)
    {
        const avbd3d_ChildShape c = avbd3d_GetCompoundChild(compound, i);
        avbd3d_ShapeDef d = *def;
        d.updateBodyMass = false;
        if (mats && compound->materialCount > 0)
            d.baseMaterial = mats[std::min(std::max(c.materialIndices[0], 0), compound->materialCount - 1)];
        const quat q = normalize(toQ(c.transform.q));
        const vec3 p = toV(c.transform.p);
        avbd3d_ShapeId sid = avbd3d_nullShapeId;
        switch (c.type)
        {
        case avbd3d_sphereShape:
        {
            avbd3d_Sphere sp = c.sphere;
            const vec3 cc = p + rotate(q, toV(sp.center));
            sp.center = avbd3d_Vec3{cc.x, cc.y, cc.z};
            sid = avbd3d_CreateSphereShape(id, &d, &sp);
            break;
        }
        case avbd3d_capsuleShape:
        {
            avbd3d_Capsule cp = c.capsule;
            const vec3 a = p + rotate(q, toV(cp.center1)), e = p + rotate(q, toV(cp.center2));
            cp.center1 = avbd3d_Vec3{a.x, a.y, a.z};
            cp.center2 = avbd3d_Vec3{e.x, e.y, e.z};
            sid = avbd3d_CreateCapsuleShape(id, &d, &cp);
            break;
        }
        case avbd3d_hullShape:
            sid = createHullAny(id, &d, c.hull, &c.transform, avbd3d_Vec3{1, 1, 1});
            break;
        case avbd3d_meshShape:
            sid = createMeshImpl(id, &d, c.mesh.data, c.mesh.scale, &c.transform);
            break;
        default:
            break;
        }
        if (sid.index1 == 0)
        {
            for (size_t k = made.size(); k-- > 0;)
            {
                int s;
                if (World *ws = worldOfShape(made[k], s))
                    ws->destroyShape(s, false);
            }
            return avbd3d_nullShapeId;
        }
        made.push_back(sid);
    }
    const int first = made[0].index1 - 1;
    for (size_t k = 1; k < made.size(); k++)
    {
        const int s = made[k].index1 - 1;
        w->shapes[s].parent = first;
        w->shapes[first].members.push_back(s);
    }
    if (def->updateBodyMass)
        w->updateMass(b);
    return w->shapeId(first);
}

// The triangles of a height field, as a Box3D mesh. Holes are skipped. Per-triangle materials are not
// carried (gap: the cooked mesh has no material indices).
avbd3d_ShapeId avbd3d_CreateHeightFieldShape(avbd3d_BodyId id, const avbd3d_ShapeDef *def,
                                             const avbd3d_HeightFieldData *hf)
{
    int b;
    World *w = worldOfBody(id, b);
    if (!w || !def || !hf || !isStaticBody(w, b))
        return avbd3d_nullShapeId;
    const b3HeightFieldData *h = reinterpret_cast<const b3HeightFieldData *>(hf);
    if (h->version != B3_HEIGHT_FIELD_VERSION || h->columnCount < 2 || h->rowCount < 2 || h->byteCount <= 0)
        return avbd3d_nullShapeId;
    const int cols = h->columnCount, rows = h->rowCount;
    const int cellCount = (cols - 1) * (rows - 1);
    if (2 * cellCount >= (1 << 21) - 1)
        return avbd3d_nullShapeId;
    const uint8_t *mats = b3GetHeightFieldMaterialIndices(h);
    std::vector<b3Vec3> verts((size_t)rows * cols, b3Vec3{0, 0, 0});
    std::vector<int32_t> tris;
    tris.reserve((size_t)2 * cellCount * 3);
    for (int t = 0; t < 2 * cellCount; t++)
    {
        if (mats[t >> 1] == B3_HEIGHT_FIELD_HOLE)
            continue;
        const b3Triangle tri = b3GetHeightFieldTriangle(h, t);
        verts[(size_t)tri.i1] = tri.vertices[0];
        verts[(size_t)tri.i2] = tri.vertices[1];
        verts[(size_t)tri.i3] = tri.vertices[2];
        tris.push_back(tri.i1);
        tris.push_back(tri.i2);
        tris.push_back(tri.i3);
    }
    if (tris.empty())
        return avbd3d_nullShapeId;
    b3MeshData *cooked = cookTriangles(verts, tris);
    if (cooked == nullptr)
        return avbd3d_nullShapeId;
    // The original height field is kept for the queries, as a byte copy the shape owns.
    void *copy = std::malloc((size_t)h->byteCount);
    if (copy == nullptr)
    {
        b3DestroyMesh(cooked);
        return avbd3d_nullShapeId;
    }
    std::memcpy(copy, h, (size_t)h->byteCount);
    const int blob = w->internGeometryBlob(reinterpret_cast<const unsigned char *>(cooked), (size_t)cooked->byteCount,
                                           cooked->hash);
    const avbd3d_ShapeId sid = w->createShape(b, *def, AVBD_SHAPE_MESH, meshHalf(cooked), vec3{0, 0, 0},
                                              quat{0, 0, 0, 1});
    int s;
    if (World *ws = worldOfShape(sid, s))
    {
        ShapeGeom g;
        g.mesh.reset(cooked);
        g.heightField.reset(static_cast<b3HeightFieldData *>(copy));
        ws->attachGeometry(s, avbd3d_heightShape, blob, std::move(g));
    }
    else
    {
        b3DestroyMesh(cooked);
        std::free(copy);
    }
    return sid;
}

void avbd3d_DestroyShape(avbd3d_ShapeId id, bool updateBodyMass)
{
    int s;
    if (World *w = worldOfShape(id, s))
        w->destroyShape(s, updateBodyMass);
}


avbd3d_ShapeType avbd3d_Shape_GetType(avbd3d_ShapeId id)
{
    int s;
    World *w = worldOfShape(id, s);
    return w ? (avbd3d_ShapeType)w->shapes[s].type : avbd3d_hullShape;
}

avbd3d_BodyId avbd3d_Shape_GetBody(avbd3d_ShapeId id)
{
    int s;
    World *w = worldOfShape(id, s);
    return w ? w->bodyId(w->shapes[s].body) : avbd3d_nullBodyId;
}

avbd3d_WorldId avbd3d_Shape_GetWorld(avbd3d_ShapeId id)
{
    avbd3d_BodyId b{1, id.world0, 0};
    return avbd3d_Body_GetWorld(b);
}

void avbd3d_Shape_SetUserData(avbd3d_ShapeId id, void *u)
{
    int s;
    if (World *w = worldOfShape(id, s))
        w->forShapeGroup(s, [&](int k) { w->shapes[k].user = u; });
}

void *avbd3d_Shape_GetUserData(avbd3d_ShapeId id)
{
    int s;
    World *w = worldOfShape(id, s);
    return w ? w->shapes[s].user : nullptr;
}

void avbd3d_Shape_SetFriction(avbd3d_ShapeId id, float f)
{
    int s;
    if (World *w = worldOfShape(id, s))
        w->forShapeGroup(s, [&](int k) {
            w->shapes[k].material.friction = f;
            w->setShapeFriction(k, f);
        });
}

void avbd3d_Shape_SetFilter(avbd3d_ShapeId id, avbd3d_Filter filter)
{
    int s;
    if (World *w = worldOfShape(id, s))
        w->forShapeGroup(s, [&](int k) { w->setShapeFilter(k, filter); });
}

void avbd3d_Shape_SetRestitution(avbd3d_ShapeId id, float restitution)
{
    int s;
    if (World *w = worldOfShape(id, s))
        w->forShapeGroup(s, [&](int k) {
            w->shapes[k].material.restitution = restitution;
            w->syncShapeMaterial(k);
        });
}


// Friction goes through setShapeFriction; restitution and rolling resistance go to the shape's
// device arrays (syncShapeMaterial, mixed per contact), and the tangent velocity reaches the
// body's BodyProps when this is the primary shape (rotated from the shape frame to the body frame
// in syncProps).
void avbd3d_Shape_SetSurfaceMaterial(avbd3d_ShapeId id, avbd3d_SurfaceMaterial m)
{
    int s;
    if (World *w = worldOfShape(id, s))
    {
        w->forShapeGroup(s, [&](int k) {
            w->shapes[k].material = m;
            w->setShapeFriction(k, m.friction);
            w->syncShapeMaterial(k);
            w->syncProps(w->shapes[k].body); // the tangent velocity follows the primary shape
        });
    }
}



// ---- joints -------------------------------------------------------------------------------

avbd3d_AvbdJointDef avbd3d_DefaultAvbdJointDef(void)
{
    avbd3d_AvbdJointDef d;
    std::memset(&d, 0, sizeof d);
    d.base = defaultJointBase();
    d.kind = avbd3d_avbdJointPoint;
    d.linearStiffness = INFINITY;
    d.angularStiffness = 0.0f;
    d.fractureForce = INFINITY;
    d.fractureTorque = -1.0f;
    return d;
}

avbd3d_JointId avbd3d_CreateAvbdJoint(avbd3d_WorldId wid, const avbd3d_AvbdJointDef *d)
{
    World *w = settled(wid);
    if (!w || !d || d->kind < avbd3d_avbdJointPoint || d->kind > avbd3d_avbdJointSwingTwist)
        return avbd3d_nullJointId;
    return w->createJoint(avbd3d_avbdJoint, d, sizeof *d);
}

avbd3d_AvbdSpringDef avbd3d_DefaultAvbdSpringDef(void)
{
    avbd3d_AvbdSpringDef d;
    std::memset(&d, 0, sizeof d);
    d.base = defaultJointBase();
    d.stiffness = 100.0f;
    d.restLength = -1.0f;
    return d;
}

avbd3d_JointId avbd3d_CreateAvbdSpring(avbd3d_WorldId wid, const avbd3d_AvbdSpringDef *d)
{
    World *w = settled(wid);
    if (!w || !d || !(d->stiffness >= 0.0f))
        return avbd3d_nullJointId;
    return w->createJoint(avbd3d_avbdSpring, d, sizeof *d);
}

avbd3d_JointId avbd3d_CreateWeldJoint(avbd3d_WorldId wid, const avbd3d_WeldJointDef *d)
{
    World *w = settled(wid);
    if (!w || !d)
        return avbd3d_nullJointId;
    return w->createJoint(avbd3d_weldJoint, d, sizeof *d);
}

avbd3d_JointId avbd3d_CreateRevoluteJoint(avbd3d_WorldId wid, const avbd3d_RevoluteJointDef *d)
{
    World *w = settled(wid);
    if (!w || !d)
        return avbd3d_nullJointId;
    return w->createJoint(avbd3d_revoluteJoint, d, sizeof *d);
}

avbd3d_JointId avbd3d_CreatePrismaticJoint(avbd3d_WorldId wid, const avbd3d_PrismaticJointDef *d)
{
    World *w = settled(wid);
    if (!w || !d)
        return avbd3d_nullJointId;
    return w->createJoint(avbd3d_prismaticJoint, d, sizeof *d);
}

avbd3d_JointId avbd3d_CreateWheelJoint(avbd3d_WorldId wid, const avbd3d_WheelJointDef *d)
{
    World *w = settled(wid);
    if (!w || !d)
        return avbd3d_nullJointId;
    return w->createJoint(avbd3d_wheelJoint, d, sizeof *d);
}

avbd3d_JointId avbd3d_CreateSphericalJoint(avbd3d_WorldId wid, const avbd3d_SphericalJointDef *d)
{
    World *w = settled(wid);
    if (!w || !d)
        return avbd3d_nullJointId;
    return w->createJoint(avbd3d_sphericalJoint, d, sizeof *d);
}

avbd3d_JointId avbd3d_CreateDistanceJoint(avbd3d_WorldId wid, const avbd3d_DistanceJointDef *d)
{
    World *w = settled(wid);
    if (!w || !d)
        return avbd3d_nullJointId;
    return w->createJoint(avbd3d_distanceJoint, d, sizeof *d);
}

avbd3d_JointId avbd3d_CreateMotorJoint(avbd3d_WorldId wid, const avbd3d_MotorJointDef *d)
{
    World *w = settled(wid);
    if (!w || !d)
        return avbd3d_nullJointId;
    return w->createJoint(avbd3d_motorJoint, d, sizeof *d);
}

avbd3d_JointId avbd3d_CreateParallelJoint(avbd3d_WorldId wid, const avbd3d_ParallelJointDef *d)
{
    World *w = settled(wid);
    if (!w || !d)
        return avbd3d_nullJointId;
    return w->createJoint(avbd3d_parallelJoint, d, sizeof *d);
}

avbd3d_JointId avbd3d_CreateFilterJoint(avbd3d_WorldId wid, const avbd3d_FilterJointDef *d)
{
    World *w = settled(wid);
    if (!w || !d)
        return avbd3d_nullJointId;
    return w->createJoint(avbd3d_filterJoint, d, sizeof *d);
}

void avbd3d_DestroyJoint(avbd3d_JointId id, bool wakeAttached)
{
    int r;
    if (World *w = worldOfJoint(id, r))
    {
        const int a = w->joints[r].a, b = w->joints[r].b;
        w->destroyJoint(r);
        if (wakeAttached)
        {
            if (a >= 0)
                w->setAwake(a, true);
            w->setAwake(b, true);
        }
    }
}


} // extern "C"
