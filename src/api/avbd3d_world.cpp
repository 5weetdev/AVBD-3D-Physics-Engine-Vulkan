#include "avbd3d_world.h"
#include "vk_guard.h"

#include "collision.h"
#include "box3d/constants.h"
#include "shape_mass.h"
#include "shape_mass_b3.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace avbd3d_impl
{

namespace
{
size_t strideOf(BodyWrite t)
{
    switch (t)
    {
    case BodyWrite::Transform:
        return sizeof(BodyTransform);
    case BodyWrite::Impulse:
    case BodyWrite::Force:
        return sizeof(BodyPointVector);
    case BodyWrite::Awake:
        return 1;
    default:
        return sizeof(vec3);
    }
}

bool isIdentity(vec3 p, quat q)
{
    return p.x == 0 && p.y == 0 && p.z == 0 && q.x == 0 && q.y == 0 && q.z == 0 && q.w == 1;
}
} // namespace

World::World(const avbd3d_WorldDef &def, Device *shared) : m_vk(shared)
{
    params = defaultSolverParams();
    gravity = def.gravity;
    params.gravity = toV(def.gravity);
    sleepEnabled = def.enableSleep;
    params.sleepEnabled = def.enableSleep;
    user = def.userData;
    hitThreshold = def.hitEventThreshold;
    // Host-only numeric properties (read back by avbd3d_props.cpp).
    props["restitutionThreshold"] = def.restitutionThreshold;
    props["continuous"] = def.enableContinuous ? 1.0 : 0.0;

    // Slot 0 is the graveyard: a static body that is in neither body list. A destroyed shape is
    // re-pointed at it, so the shape lists (built from the body lists) never see it again.
    SceneDesc &d = desc;
    d.params = params;
    bodies.emplace_back();
    bodies[0].type = avbd3d_staticBody;
    bodies[0].enabled = false;
    d.count = 1;
    d.pos.push_back(vec3{0, 0, 0});
    d.velLin.push_back(vec3{0, 0, 0});
    d.velAng.push_back(vec3{0, 0, 0});
    d.halfExtents.push_back(vec3{0.5f, 0.5f, 0.5f});
    d.moment.push_back(vec3{0, 0, 0});
    d.rot.push_back(quat{0, 0, 0, 1});
    d.mass.push_back(0.0f);
    d.friction.push_back(0.5f);
    d.shapeType.push_back(AVBD_SHAPE_BOX);
    d.renderGroup.push_back(-1);
    d.tier.push_back((unsigned char)BodyTier::Interactive);
    d.asleep.push_back(1);
    d.bodyPrimaryShape.push_back(-1);
    d.bodyLocalCenter.push_back(vec3{0, 0, 0});
    d.bodyPrincipalRot.push_back(quat{0, 0, 0, 1});
    d.bodyProps.assign(1, BodyProps{});
    d.cellSize = 1.0f;
    awake.push_back(0);
    m_bodySerial.push_back(0);
    m_bodyHold.push_back(0);

    // Test hook: the Nth step submission of this world reports VK_ERROR_DEVICE_LOST.
    const char *f = shared ? nullptr : std::getenv("AVBD_DEBUG_FAIL_SUBMIT");
    if (f)
        m_vk.device().setDebugFailSubmit((uint64_t)std::strtoull(f, nullptr, 10));
    m_vk.init();
    m_vk.build(d, params);
    m_vk.enableSnapshot(true);
    m_vk.setRestitutionThreshold(def.restitutionThreshold);
    m_vkBodies = 1;
    m_tree = b3DynamicTree_Create(64);
}

World::~World()
{
    if (m_inFlight)
        m_vk.stepEnd();
    b3DynamicTree_Destroy(&m_tree);
    m_vk.destroy();
}

// ---- ids ----------------------------------------------------------------------------------

avbd3d_BodyId World::bodyId(int slot) const
{
    avbd3d_BodyId id;
    id.index1 = slot;
    id.world0 = (uint16_t)worldIndex;
    id.generation = bodies[slot].gen;
    return id;
}

avbd3d_ShapeId World::shapeId(int slot) const
{
    if (shapes[slot].parent >= 0) // a compound child is named by its compound
        slot = shapes[slot].parent;
    avbd3d_ShapeId id;
    id.index1 = slot + 1;
    id.world0 = (uint16_t)worldIndex;
    id.generation = shapes[slot].gen;
    return id;
}

avbd3d_ShapeId World::shapeIdAt(int slot, uint16_t gen) const
{
    if (slot >= 0 && slot < (int)shapes.size() && shapes[slot].parent >= 0)
    {
        slot = shapes[slot].parent;
        gen = shapes[slot].gen;
    }
    avbd3d_ShapeId id;
    id.index1 = slot + 1;
    id.world0 = (uint16_t)worldIndex;
    id.generation = gen;
    return id;
}

bool World::bodyValid(avbd3d_BodyId id) const
{
    return id.index1 >= 1 && id.index1 < (int)bodies.size() && id.world0 == worldIndex && bodies[id.index1].alive &&
           bodies[id.index1].gen == id.generation;
}

bool World::shapeValid(avbd3d_ShapeId id) const
{
    return id.index1 >= 1 && id.index1 <= (int)shapes.size() && id.world0 == worldIndex &&
           shapes[id.index1 - 1].alive && shapes[id.index1 - 1].gen == id.generation;
}

bool World::jointValid(avbd3d_JointId id) const
{
    return id.index1 >= 1 && id.index1 <= (int)joints.size() && id.world0 == worldIndex &&
           joints[id.index1 - 1].alive && joints[id.index1 - 1].gen == id.generation;
}

// ---- slot allocation ----------------------------------------------------------------------

void World::markBody(int slot)
{
    holdBody(slot);
    BodyRec &r = bodies[slot];
    if (slot < m_vkBodies && !r.refresh)
    {
        r.refresh = true;
        m_refreshBodies.push_back(slot);
    }
}

void World::markShape(int slot)
{
    ShapeRec &r = shapes[slot];
    if (slot < m_vkShapes && !r.refresh)
    {
        r.refresh = true;
        m_refreshShapes.push_back(slot);
    }
}

int World::allocBody()
{
    SceneDesc &d = desc;
    if (!m_freeBodies.empty())
    {
        int s = m_freeBodies.back();
        m_freeBodies.pop_back();
        holdBody(s);
        return s;
    }
    const int s = d.count++;
    bodies.emplace_back();
    d.pos.push_back(vec3{0, 0, 0});
    d.velLin.push_back(vec3{0, 0, 0});
    d.velAng.push_back(vec3{0, 0, 0});
    d.halfExtents.push_back(vec3{0.5f, 0.5f, 0.5f});
    d.moment.push_back(vec3{1, 1, 1});
    d.rot.push_back(quat{0, 0, 0, 1});
    d.mass.push_back(0.0f);
    d.friction.push_back(0.5f);
    d.shapeType.push_back(AVBD_SHAPE_BOX);
    d.renderGroup.push_back(-1);
    d.tier.push_back((unsigned char)BodyTier::Interactive);
    d.asleep.push_back(0);
    d.bodyPrimaryShape.push_back(-1);
    d.bodyLocalCenter.push_back(vec3{0, 0, 0});
    d.bodyPrincipalRot.push_back(quat{0, 0, 0, 1});
    d.bodyProps.resize(d.count, BodyProps{});
    awake.push_back(1);
    m_bodySerial.push_back(0);
    m_bodyHold.push_back(m_vk.stepSerial() + 1);
    return s;
}

int World::allocShape()
{
    SceneDesc &d = desc;
    if (!m_freeShapes.empty())
    {
        int s = m_freeShapes.back();
        m_freeShapes.pop_back();
        return s;
    }
    const int s = d.shapeCount++;
    shapes.emplace_back();
    d.shapeBody.push_back(0);
    d.shapeLocalPos.push_back(vec3{0, 0, 0});
    d.shapeLocalRot.push_back(quat{0, 0, 0, 1});
    d.shapeKind.push_back(AVBD_SHAPE_BOX);
    d.shapeHalf.push_back(vec3{0.5f, 0.5f, 0.5f});
    d.shapeFriction.push_back(0.5f);
    d.shapeRestitution.push_back(0.0f);
    d.shapeRolling.push_back(0.0f);
    d.shapeCategory.push_back(1u);
    d.shapeMask.push_back(~0ull);
    d.shapeGroup.push_back(0);
    d.shapeFlags.push_back(AVBD_SHAPE_FLAG_IDENTITY);
    d.shapeGeom.push_back(-1);
    return s;
}

// ---- bodies -------------------------------------------------------------------------------

avbd3d_BodyId World::createBody(const avbd3d_BodyDef &def)
{
    const int s = allocBody();
    SceneDesc &d = desc;
    BodyRec &r = bodies[s];
    const uint16_t gen = r.gen;
    r = BodyRec();
    r.gen = gen;
    r.alive = true;
    r.type = def.type;
    r.enabled = def.isEnabled;
    r.user = def.userData;
    r.linearDamping = def.linearDamping;
    r.angularDamping = def.angularDamping;
    r.gravityScale = def.gravityScale;
    r.sleepThreshold = def.sleepThreshold;
    r.locks = def.motionLocks;
    r.bullet = def.isBullet;

    d.pos[s] = toV(def.position);
    d.rot[s] = normalize(toQ(def.rotation));
    d.velLin[s] = def.type == avbd3d_staticBody ? vec3{0, 0, 0} : toV(def.linearVelocity);
    d.velAng[s] = def.type == avbd3d_staticBody ? vec3{0, 0, 0} : toV(def.angularVelocity);
    d.halfExtents[s] = vec3{0.5f, 0.5f, 0.5f};
    d.moment[s] = vec3{1, 1, 1};
    d.mass[s] = 0.0f;
    d.friction[s] = 0.5f;
    d.shapeType[s] = AVBD_SHAPE_BOX;
    d.asleep[s] = def.isAwake ? 0 : 1;
    d.bodyPrimaryShape[s] = -1;
    d.bodyLocalCenter[s] = vec3{0, 0, 0};
    d.bodyPrincipalRot[s] = quat{0, 0, 0, 1};
    awake[s] = def.isAwake && def.type != avbd3d_staticBody;
    d.bodyProps[s] = BodyProps{};
    syncProps(s);
    markBody(s);
    m_listsDirty = true;
    return bodyId(s);
}

void World::applyMass(int slot)
{
    SceneDesc &d = desc;
    const BodyRec &r = bodies[slot];
    if (r.primary < 0)
    {
        d.mass[slot] = 0.0f;
        d.moment[slot] = vec3{1, 1, 1};
    }
    else if (r.type == avbd3d_dynamicBody)
    {
        d.mass[slot] = r.massDyn;
        d.moment[slot] = r.momentDyn;
    }
    else
    {
        d.mass[slot] = 0.0f;
        d.moment[slot] = vec3{0, 0, 0};
    }
}

void World::setPrimary(int bodySlot, int shapeSlot)
{
    SceneDesc &d = desc;
    BodyRec &r = bodies[bodySlot];
    r.primary = shapeSlot;
    d.bodyPrimaryShape[bodySlot] = shapeSlot;
    if (shapeSlot >= 0)
    {
        r.density = shapes[shapeSlot].density;
        d.halfExtents[bodySlot] = d.shapeHalf[shapeSlot];
        d.shapeType[bodySlot] = (unsigned char)d.shapeKind[shapeSlot];
        d.friction[bodySlot] = d.shapeFriction[shapeSlot];
    }
    updateMass(bodySlot);
    m_listsDirty = true;
}

void World::updateMass(int b)
{
    SceneDesc &d = desc;
    const BodyRec &r = bodies[b];
    float mass = 0.0f;
    vec3 moment{0, 0, 0}, center{0, 0, 0};
    quat rp{0, 0, 0, 1};
    if (r.primary >= 0)
    {
        // One centred primitive keeps the closed form the SceneBuilder path uses.
        const int p = r.primary;
        const ShapeRec &ps = shapes[p];
        const int kind = d.shapeKind[p];
        const bool primitive = kind == AVBD_SHAPE_BOX || kind == AVBD_SHAPE_SPHERE || kind == AVBD_SHAPE_CAPSULE;
        bool done = false;
        if (r.shapes.size() == 1 && primitive && isIdentity(ps.localPos, ps.localRot))
        {
            shapeMassProperties(kind, d.shapeHalf[p], ps.density, mass, moment);
            done = true;
        }
        else
        {
            m_massScratch.clear();
            for (int s : r.shapes)
            {
                const ShapeRec &sr = shapes[s];
                ShapeMassInput in;
                in.kind = d.shapeKind[s];
                in.half = d.shapeHalf[s];
                in.localPos = sr.localPos;
                in.localRot = sr.localRot;
                in.density = sr.density;
                if (in.kind == AVBD_SHAPE_HULL)
                    in.hull = (sr.geom.hull || sr.geom.box) ? sr.geom.hullData() : nullptr;
                if (in.kind == AVBD_SHAPE_MESH)
                    in.mesh = sr.geom.mesh.get();
                m_massScratch.push_back(in);
            }
            CompoundMass cm;
            if (computeCompoundMass(m_massScratch.data(), (int)m_massScratch.size(), 1.0f, cm))
            {
                mass = cm.mass;
                moment = cm.moment;
                // Only a dynamic body moves its centre: a static or kinematic body keeps its
                // origin as its pose, as Box3D does.
                if (r.type == avbd3d_dynamicBody)
                {
                    center = cm.center;
                    rp = cm.principalRot;
                }
                done = true;
            }
        }
        // No shape has mass yet (a hull before its geometry, a mesh): the primary's closed form.
        if (!done)
            shapeMassProperties(kind, d.shapeHalf[p], ps.density, mass, moment);
    }
    applyMassFrame(b, mass, center, moment, rp);
}

void World::applyMassFrame(int b, float mass, vec3 center, vec3 moment, quat principalRot)
{
    SceneDesc &d = desc;
    BodyRec &r = bodies[b];
    const vec3 c0 = d.bodyLocalCenter[b];
    const quat q0 = d.bodyPrincipalRot[b];
    const bool moved = c0.x != center.x || c0.y != center.y || c0.z != center.z || q0.x != principalRot.x ||
                       q0.y != principalRot.y || q0.z != principalRot.z || q0.w != principalRot.w;
    if (moved)
    {
        const vec3 o = originPos(b);
        const quat oq = originRot(b);
        const vec3 oldPos = d.pos[b];
        d.bodyLocalCenter[b] = center;
        d.bodyPrincipalRot[b] = principalRot;
        d.rot[b] = normalize(oq * principalRot);
        d.pos[b] = o + rotate(oq, center);
        // The origin keeps its velocity; the centre of mass moves with the body's spin.
        d.velLin[b] = d.velLin[b] + cross(d.velAng[b], d.pos[b] - oldPos);
        for (int s : r.shapes)
            syncShapePose(s);
    }
    r.massDyn = mass;
    r.momentDyn = moment;
    applyMass(b);
    syncProps(b);
    markBody(b);
}

void World::syncShapePose(int s)
{
    SceneDesc &d = desc;
    const ShapeRec &r = shapes[s];
    const vec3 c = d.bodyLocalCenter[r.body];
    const quat rp = d.bodyPrincipalRot[r.body];
    vec3 p = r.localPos;
    quat q = r.localRot;
    if (!isIdentity(c, rp))
    {
        const quat ip = conjugate(rp);
        p = rotate(ip, r.localPos - c);
        q = normalize(ip * r.localRot);
    }
    d.shapeLocalPos[s] = p;
    d.shapeLocalRot[s] = q;
    unsigned f = d.shapeFlags[s] & ~AVBD_SHAPE_FLAG_IDENTITY;
    if (isIdentity(p, q))
        f |= AVBD_SHAPE_FLAG_IDENTITY;
    d.shapeFlags[s] = f;
    markShape(s);
}

avbd3d_ShapeId World::createShape(int bodySlot, const avbd3d_ShapeDef &def, int kind, vec3 half, vec3 localPos,
                                  quat localRot)
{
    SceneDesc &d = desc;
    const int s = allocShape();
    ShapeRec &r = shapes[s];
    const uint16_t gen = r.gen;
    r = ShapeRec();
    r.gen = gen;
    r.alive = true;
    r.body = bodySlot;
    r.type = kind == AVBD_SHAPE_SPHERE    ? avbd3d_sphereShape
             : kind == AVBD_SHAPE_CAPSULE ? avbd3d_capsuleShape
                                          : avbd3d_hullShape;
    r.density = def.density > 0.0f ? def.density : 1.0f;
    r.user = def.userData;
    r.filter = def.filter;
    r.material = def.baseMaterial;

    d.shapeBody[s] = bodySlot;
    d.shapeKind[s] = (unsigned char)kind;
    d.shapeHalf[s] = half;
    d.shapeFriction[s] = def.baseMaterial.friction;
    syncShapeMaterial(s);
    // A sensor takes part in no device contact: category 0 and mask 0 reject every pair in the
    // narrowphase filter, and group 0 keeps the equal-group override from applying. Its host
    // filter (r.filter) is kept for the overlap query.
    r.isSensor = def.isSensor;
    if (r.isSensor)
        m_sensors.insert(std::lower_bound(m_sensors.begin(), m_sensors.end(), s), s);
    r.sensorEvents = def.enableSensorEvents;
    r.explosionScale = def.explosionScale;
    d.shapeCategory[s] = def.isSensor ? 0u : def.filter.categoryBits;
    d.shapeMask[s] = def.isSensor ? 0u : def.filter.maskBits;
    d.shapeGroup[s] = def.isSensor ? 0 : def.filter.groupIndex;
    d.shapeFlags[s] = 0u;
    d.shapeGeom[s] = -1;
    r.localPos = localPos;
    r.localRot = localRot;
    syncShapePose(s); // device pose relative to the body's current centre of mass
    r.contactEvents = def.enableContactEvents;
    r.hitEvents = def.enableHitEvents;
    setShapeReport(s); // ORs AVBD_SHAPE_FLAG_REPORT into the flags word and marks the shape
    markShape(s);

    // Host geometry for the query layer, in the shape frame (the device encoding of `half`).
    if (kind == AVBD_SHAPE_SPHERE)
    {
        r.geom.sphere.center = b3Vec3{0, 0, 0};
        r.geom.sphere.radius = half.x;
    }
    else if (kind == AVBD_SHAPE_CAPSULE)
    {
        const float seg = half.z - half.x; // half segment length
        r.geom.capsule.center1 = b3Vec3{0, 0, -seg};
        r.geom.capsule.center2 = b3Vec3{0, 0, seg};
        r.geom.capsule.radius = half.x;
    }
    else if (kind == AVBD_SHAPE_BOX)
    {
        r.geom.box.reset(new b3BoxHull(b3MakeBoxHull(half.x, half.y, half.z)));
    }
    // A hull, mesh or height field has no geometry yet: attachGeometry gives it one and the proxy.
    if (kind != AVBD_SHAPE_HULL && kind != AVBD_SHAPE_MESH)
        refreshShapeProxy(s);

    bodies[bodySlot].shapes.push_back(s);
    if (bodies[bodySlot].primary < 0)
        setPrimary(bodySlot, s);
    else
    {
        if (def.updateBodyMass)
            updateMass(bodySlot);
        m_listsDirty = true;
    }
    return shapeId(s);
}

int World::internGeometryBlob(const unsigned char *bytes, size_t n, uint64_t hash)
{
    const uint64_t h = hash;
    SceneDesc &d = desc;
    for (size_t i = 0; i < d.geomBlob.size(); i++)
        if (d.geomHash[i] == h && d.geomBlob[i].size() == n && std::memcmp(d.geomBlob[i].data(), bytes, n) == 0)
            return (int)i;
    d.geomBlob.emplace_back(bytes, bytes + n);
    d.geomHash.push_back(h);
    return (int)d.geomBlob.size() - 1;
}

void World::attachGeometry(int s, int type, int blob, ShapeGeom &&geom)
{
    ShapeRec &r = shapes[s];
    r.type = type;
    r.geom = std::move(geom);
    desc.shapeGeom[s] = blob;
    refreshShapeProxy(s);
    // A hull or a (closed) mesh has no mass until its geometry is here.
    if (type == avbd3d_hullShape || (type == avbd3d_meshShape && bodies[r.body].type == avbd3d_dynamicBody))
        updateMass(r.body);
}

void World::refreshShapeProxy(int s)
{
    ShapeRec &r = shapes[s];
    if (!r.alive)
        return;
    const b3Transform xf = shapeWorldTransform(s);
    b3AABB aabb;
    if (r.type == avbd3d_sphereShape)
        aabb = b3ComputeSphereAABB(&r.geom.sphere, xf);
    else if (r.type == avbd3d_capsuleShape)
        aabb = b3ComputeCapsuleAABB(&r.geom.capsule, xf);
    else if (r.type == avbd3d_meshShape)
        aabb = b3ComputeMeshAABB(r.geom.mesh.get(), xf, b3Vec3{1.0f, 1.0f, 1.0f});
    else if (r.type == avbd3d_heightShape)
        aabb = b3ComputeHeightFieldAABB(r.geom.heightField.get(), xf);
    else
        aabb = b3ComputeHullAABB(r.geom.hullData(), xf);
    r.aabb = aabb;

    if (r.proxy >= 0 && b3AABB_Contains(b3DynamicTree_GetAABB(&m_tree, r.proxy), aabb))
        return;
    // Box3D's margin: a fraction of the extent, capped at 5 cm.
    const b3Vec3 ext = b3Sub(aabb.upperBound, aabb.lowerBound);
    const float margin = std::min(B3_MAX_AABB_MARGIN, B3_AABB_MARGIN_FRACTION * 0.5f * b3Length(ext));
    b3AABB fat = aabb;
    fat.lowerBound = b3Sub(fat.lowerBound, b3Vec3{margin, margin, margin});
    fat.upperBound = b3Add(fat.upperBound, b3Vec3{margin, margin, margin});
    if (r.proxy < 0)
        r.proxy = b3DynamicTree_CreateProxy(&m_tree, fat, r.filter.categoryBits, (uint64_t)s);
    else
        b3DynamicTree_MoveProxy(&m_tree, r.proxy, fat);
}

void World::ensureProxies()
{
    syncBodies();
    for (int b : m_proxyDirty)
    {
        m_proxyFlag[b] = 0;
        if (bodies[b].alive)
            for (int s : bodies[b].shapes)
                refreshShapeProxy(s);
    }
    m_proxyDirty.clear();
}

void World::syncRenderGroups()
{
    // A group set on a body the device does not have yet is applied by the append that adds it.
    if (!renderGroupsDirty || desc.shapeCount > m_vkShapes)
        return;
    settle();
    m_vk.computeRenderGroups(desc);
    renderGroupsDirty = false;
}

std::vector<avbd3d_BodyMoveEvent> &World::moveEvents()
{
    // The bodies awake before or after the newest finished step: the snapshot carries both
    // flags, so a body asleep through the step is skipped without touching its pose.
    const uint64_t latest = m_vk.snapshotSerial();
    if (m_eventsSerial != latest)
    {
        events.clear();
        events.reserve(m_movable.size());
        const VkWorld::SnapshotView &v = view();
        for (int b : m_movable)
        {
            if (b >= v.count || (!v.awake[b] && !v.prevAwake[b]))
                continue;
            syncBody(b);
            const SceneDesc &d = desc;
            const quat q = d.rot[b] * conjugate(d.bodyPrincipalRot[b]); // originRot, already synced
            avbd3d_BodyMoveEvent e;
            e.userData = m_hot[b].user;
            e.transform.p = fromV(d.pos[b] - rotate(q, d.bodyLocalCenter[b]));
            e.transform.q = fromQ(q);
            e.bodyId.index1 = b;
            e.bodyId.world0 = (uint16_t)worldIndex;
            e.bodyId.generation = m_hot[b].gen;
            e.fellAsleep = !v.awake[b];
            events.push_back(e);
        }
        m_eventsSerial = latest;
    }
    return events;
}

b3Transform World::shapeWorldTransform(int s) const
{
    const ShapeRec &r = shapes[s];
    const vec3 o = originPos(r.body);
    const quat oq = originRot(r.body);
    const quat q = oq * r.localRot;
    const vec3 p = o + rotate(oq, r.localPos);
    b3Transform xf;
    xf.p = b3Vec3{p.x, p.y, p.z};
    xf.q = b3Quat{{q.x, q.y, q.z}, q.w};
    return xf;
}

void World::destroyShape(int s, bool updateBodyMass)
{
    // A compound goes with all its children.
    if (!shapes[s].members.empty())
    {
        const std::vector<int> kids = shapes[s].members;
        shapes[s].members.clear();
        for (int k : kids)
        {
            shapes[k].parent = -1;
            destroyShape(k, false);
        }
    }
    SceneDesc &d = desc;
    ShapeRec &r = shapes[s];
    const int b = r.body;
    if (r.proxy >= 0)
        b3DynamicTree_DestroyProxy(&m_tree, r.proxy);
    r.proxy = -1;
    r.geom.reset();
    // Contacts on this shape are not reported after this, so the next stepEnd ends them.
    r.contactEvents = false;
    r.hitEvents = false;
    setShapeReport(s);
    // A destroyed sensor ends every overlap it had. Its id is named with the generation it had,
    // since the slot's generation is bumped below. (A destroyed visitor is ended by the next
    // StepEnd, which finds it gone from the sensor's query.)
    if (r.isSensor)
    {
        m_sensors.erase(std::lower_bound(m_sensors.begin(), m_sensors.end(), s));
        for (uint64_t k : r.overlaps)
            m_pendingSensorEnd.push_back(
                avbd3d_SensorEndTouchEvent{shapeIdAt(s, r.gen), shapeIdAt((int)(k >> 16), (uint16_t)(k & 0xFFFFu))});
        r.overlaps.clear();
    }
    r.alive = false;
    r.gen = (uint16_t)(r.gen + 1 ? r.gen + 1 : 1);
    d.shapeBody[s] = 0;
    d.shapeHalf[s] = vec3{0.01f, 0.01f, 0.01f};
    markShape(s);
    m_pendingShapes.push_back(s);
    BodyRec &br = bodies[b];
    br.shapes.erase(std::remove(br.shapes.begin(), br.shapes.end(), s), br.shapes.end());
    if (br.primary == s)
        setPrimary(b, br.shapes.empty() ? -1 : br.shapes.front());
    else
    {
        if (updateBodyMass)
            updateMass(b);
        m_listsDirty = true;
    }
}

void World::destroyBody(int b)
{
    SceneDesc &d = desc;
    BodyRec &r = bodies[b];
    while (!r.shapes.empty())
        destroyShape(r.shapes.back(), false);
    while (!r.joints.empty())
        destroyJoint(r.joints.back());
    r.alive = false;
    r.gen = (uint16_t)(r.gen + 1 ? r.gen + 1 : 1);
    r.enabled = false;
    r.primary = -1;
    d.bodyPrimaryShape[b] = -1;
    d.mass[b] = 0.0f;
    d.velLin[b] = vec3{0, 0, 0};
    d.velAng[b] = vec3{0, 0, 0};
    awake[b] = 0;
    markBody(b);
    m_pendingBodies.push_back(b);
    m_listsDirty = true;
}

// ---- state --------------------------------------------------------------------------------

// The mirror is a cache of the device state: reading it may pull the body (syncBody).
quat World::originRot(int b) const
{
    const_cast<World *>(this)->syncBody(b);
    return desc.rot[b] * conjugate(desc.bodyPrincipalRot[b]);
}

vec3 World::originPos(int b) const
{
    return desc.pos[b] - rotate(originRot(b), desc.bodyLocalCenter[b]);
}

bool World::isAwake(int b) const
{
    const_cast<World *>(this)->syncBody(b);
    return awake[b] != 0;
}

void World::queue(int body, BodyWrite type, const void *data, size_t bytes)
{
    holdBody(body);
    Op op;
    op.body = body;
    op.type = type;
    std::memcpy(op.data, data, bytes);
    m_ops.push_back(op);
}

void World::setTransform(int b, vec3 origin, quat rot)
{
    SceneDesc &d = desc;
    const quat q = normalize(rot);
    d.rot[b] = q * d.bodyPrincipalRot[b];
    d.pos[b] = origin + rotate(q, d.bodyLocalCenter[b]);
    awake[b] = bodies[b].type != avbd3d_staticBody;
    BodyTransform t{d.pos[b], d.rot[b]};
    queue(b, BodyWrite::Transform, &t, sizeof t);
    for (int s : bodies[b].shapes)
        refreshShapeProxy(s);
}

void World::setLinearVelocity(int b, vec3 v)
{
    if (bodies[b].type == avbd3d_staticBody)
        return;
    desc.velLin[b] = v;
    awake[b] = 1;
    queue(b, BodyWrite::LinearVelocity, &v, sizeof v);
}

void World::setAngularVelocity(int b, vec3 w)
{
    if (bodies[b].type == avbd3d_staticBody)
        return;
    desc.velAng[b] = w;
    awake[b] = 1;
    queue(b, BodyWrite::AngularVelocity, &w, sizeof w);
}

// World-frame inverse inertia times a world vector, from the diagonal principal-frame moment.
static vec3 invInertiaMul(const SceneDesc &d, int b, vec3 w)
{
    const vec3 m = d.moment[b];
    if (m.x <= 0 || m.y <= 0 || m.z <= 0)
        return vec3{0, 0, 0};
    const quat q = d.rot[b];
    const vec3 l = rotate(inverse(q), w);
    return rotate(q, vec3{l.x / m.x, l.y / m.y, l.z / m.z});
}

void World::applyImpulse(int b, vec3 j, vec3 point, bool wake)
{
    if (bodies[b].type != avbd3d_dynamicBody || (!wake && !awake[b]))
        return;
    SceneDesc &d = desc;
    const float m = d.mass[b];
    if (m > 0)
    {
        d.velLin[b] = d.velLin[b] + j * (1.0f / m);
        d.velAng[b] = d.velAng[b] + invInertiaMul(d, b, cross(point - d.pos[b], j));
    }
    awake[b] = 1;
    BodyPointVector p{j, point};
    queue(b, BodyWrite::Impulse, &p, sizeof p);
}

void World::applyForce(int b, vec3 f, vec3 point, bool wake)
{
    if (bodies[b].type != avbd3d_dynamicBody || (!wake && !awake[b]))
        return;
    awake[b] = 1;
    BodyPointVector p{f, point};
    queue(b, BodyWrite::Force, &p, sizeof p);
}

void World::applyTorque(int b, vec3 t, bool wake)
{
    if (bodies[b].type != avbd3d_dynamicBody || (!wake && !awake[b]))
        return;
    awake[b] = 1;
    queue(b, BodyWrite::Torque, &t, sizeof t);
}

void World::applyAngularImpulse(int b, vec3 j, bool wake)
{
    if (bodies[b].type != avbd3d_dynamicBody || (!wake && !awake[b]))
        return;
    setAngularVelocity(b, desc.velAng[b] + invInertiaMul(desc, b, j));
}

void World::setAwake(int b, bool a)
{
    if (bodies[b].type == avbd3d_staticBody)
        return;
    awake[b] = a;
    if (!a)
    {
        desc.velLin[b] = vec3{0, 0, 0};
        desc.velAng[b] = vec3{0, 0, 0};
    }
    unsigned char v = a ? 1 : 0;
    queue(b, BodyWrite::Awake, &v, 1);
}

void World::setEnabled(int b, bool e)
{
    BodyRec &r = bodies[b];
    if (r.enabled == e)
        return;
    holdBody(b);
    r.enabled = e;
    if (!e)
        awake[b] = 0;
    else if (r.type != avbd3d_staticBody)
        setAwake(b, true);
    m_listsDirty = true;
}

void World::setType(int b, int type)
{
    BodyRec &r = bodies[b];
    if (r.type == type)
        return;
    if (type != avbd3d_staticBody)
    {
        // A height field cannot move, and a mesh needs a closed volume to have mass: refuse.
        for (int s : r.shapes)
        {
            const ShapeRec &sr = shapes[s];
            if (desc.shapeKind[s] != AVBD_SHAPE_MESH)
                continue;
            if (sr.geom.heightField || (sr.geom.mesh && !meshIsClosed(sr.geom.mesh.get())))
            {
#ifndef NDEBUG
                fprintf(stderr, "[avbd3d] Body_SetType refused: the body owns a height field or an open mesh\n");
#endif
                return;
            }
        }
    }
    r.type = type;
    updateMass(b); // a static or kinematic body keeps its origin as its centre
    if (type == avbd3d_staticBody)
    {
        desc.velLin[b] = vec3{0, 0, 0};
        desc.velAng[b] = vec3{0, 0, 0};
        awake[b] = 0;
    }
    else if (type == avbd3d_kinematicBody)
    {
        // Kinematic keeps the velocity it was given and moves with it, unsolved by gravity.
        setAwake(b, true);
    }
    else
        awake[b] = 1;
    syncProps(b);
    // A kinematic bit flip moves the body's shapes between the dynamic and static shape lists.
    for (int s : r.shapes)
        markShape(s);
    markBody(b);
    m_listsDirty = true;
}

// avbd3d_MixingRule (geometric, average, min, max, multiply) -> FrictionMix, and the restitution
// code (FrictionMix + 1, 0 = the default larger-of-two).
unsigned World::mixCode() const
{
    static const unsigned kFriction[5] = {FRICTION_MIX_GEOMETRIC, FRICTION_MIX_AVERAGE, FRICTION_MIX_MIN,
                                          FRICTION_MIX_MAX, FRICTION_MIX_MULTIPLY};
    const int f = std::min(std::max(frictionMixing, 0), 4);
    const int r = std::min(std::max(restitutionMixing, 0), 4);
    const unsigned rest = r == 3 ? 0u : kFriction[r] + 1u;
    return kFriction[f] | (rest << 8);
}

void World::setMixing(int friction, int restitution)
{
    if (friction == frictionMixing && restitution == restitutionMixing)
        return;
    frictionMixing = friction;
    restitutionMixing = restitution;
    for (int b = 1; b < (int)bodies.size(); b++)
        if (bodies[b].alive)
            syncProps(b);
}

void World::syncProps(int b)
{
    SceneDesc &d = desc;
    if ((int)d.bodyProps.size() < d.count)
        d.bodyProps.resize(d.count, BodyProps{});
    const BodyRec &r = bodies[b];
    BodyProps &p = d.bodyProps[b];
    p.linearDamping = r.linearDamping;
    p.angularDamping = r.angularDamping;
    p.gravityScale = r.gravityScale;

    unsigned flags = p.flags & ~(unsigned)(BODY_KINEMATIC | BODY_LOCK_ANY);
    if (r.type == avbd3d_kinematicBody)
        flags |= BODY_KINEMATIC;
    if (r.locks.linearX)
        flags |= BODY_LOCK_LIN_X;
    if (r.locks.linearY)
        flags |= BODY_LOCK_LIN_Y;
    if (r.locks.linearZ)
        flags |= BODY_LOCK_LIN_Z;
    if (r.locks.angularX)
        flags |= BODY_LOCK_ANG_X;
    if (r.locks.angularY)
        flags |= BODY_LOCK_ANG_Y;
    if (r.locks.angularZ)
        flags |= BODY_LOCK_ANG_Z;
    if (r.bullet)
        flags |= BODY_BULLET; // swept against non-bullet dynamics too (ccd.slang)
    p.flags = flags;

    // Restitution and rolling resistance are per shape (syncShapeMaterial), mixed per contact on
    // the device from the manifold's two shapes; the body-level fields stay 0.
    const ShapeRec *prim = r.primary >= 0 ? &shapes[r.primary] : nullptr;
    p.restitution = 0.0f;
    p.rollingResistance = 0.0f;
    // Surface conveyor velocity: the primary shape's tangentVelocity is in the shape frame, the
    // device wants it in the body frame.
    {
        vec3 t = prim ? toV(prim->material.tangentVelocity) : vec3{0, 0, 0};
        if (prim)
            t = rotate(prim->localRot, t);
        p.tangentSpeed = t.x;
        p.tangentY = t.y;
        p.tangentZ = t.z;
    }
    p.frictionMix = mixCode();

    markBody(b);
}

// The shape's restitution and rolling resistance into the device mirror. Rolling resistance acts
// on spheres and capsules only (primal.slang), so any other shape carries 0.
void World::syncShapeMaterial(int s)
{
    const ShapeRec &r = shapes[s];
    const int kind = desc.shapeKind[s];
    desc.shapeRestitution[s] = r.material.restitution;
    desc.shapeRolling[s] =
        (kind == AVBD_SHAPE_SPHERE || kind == AVBD_SHAPE_CAPSULE) ? r.material.rollingResistance : 0.0f;
    markShape(s);
}

void World::setShapeFriction(int s, float f)
{
    desc.shapeFriction[s] = f;
    const int b = shapes[s].body;
    if (bodies[b].primary == s)
        desc.friction[b] = f;
    markShape(s);
    markBody(b);
}

// A sensor stays at category/mask/group 0 on the device (it takes part in no device contact);
// its authored filter is only kept on the host for the sensor and query layers. A contact the new
// filter rejects is simply not reported next step, which ends it (and fires the end event when
// the shape has contact events). The body is woken so a resting pair is re-evaluated.
void World::setShapeFilter(int s, const avbd3d_Filter &filter)
{
    ShapeRec &r = shapes[s];
    r.filter = filter;
    if (r.proxy >= 0)
        b3DynamicTree_SetCategoryBits(&m_tree, r.proxy, filter.categoryBits);
    if (!r.isSensor)
    {
        desc.shapeCategory[s] = filter.categoryBits;
        desc.shapeMask[s] = filter.maskBits;
        desc.shapeGroup[s] = filter.groupIndex;
        markShape(s);
    }
    setAwake(r.body, true);
}

// ---- joints -------------------------------------------------------------------------------

void World::appendRows(const SceneDesc &sd, int a, int b, const std::vector<unsigned long long> &)
{
    SceneDesc &d = desc;
    auto remap = [&](int i) { return i < 0 ? -1 : (i == 0 && a >= 0 ? a : b); };
    for (int i = 0; i < sd.jointCount; i++)
    {
        d.jointA.push_back(remap(sd.jointA[i]));
        d.jointB.push_back(remap(sd.jointB[i]));
        d.jointRA.push_back(sd.jointRA[i]);
        d.jointRB.push_back(sd.jointRB[i]);
        d.jointKLin.push_back(sd.jointKLin[i]);
        d.jointKAng.push_back(sd.jointKAng[i]);
        d.jointFrac.push_back(sd.jointFrac[i]);
        d.jointFracAng.push_back(sd.jointFracAng[i]);
        d.jointArm.push_back(sd.jointArm[i]);
        d.jointRestAng.push_back(sd.jointRestAng[i]);
        d.jointFrameA.push_back(sd.jointFrameA[i]);
        d.jointFrameB.push_back(sd.jointFrameB[i]);
        d.jointGeneric.push_back(sd.jointGeneric[i]);
        for (int k = 0; k < 6; k++)
        {
            d.jointAxisMode.push_back(sd.jointAxisMode[6 * i + k]);
            d.jointAxisLo.push_back(sd.jointAxisLo[6 * i + k]);
            d.jointAxisHi.push_back(sd.jointAxisHi[6 * i + k]);
        }
    }
    d.jointCount += sd.jointCount;
    for (int i = 0; i < sd.springCount; i++)
    {
        d.springA.push_back(remap(sd.springA[i]));
        d.springB.push_back(remap(sd.springB[i]));
        d.springRA.push_back(sd.springRA[i]);
        d.springRB.push_back(sd.springRB[i]);
        d.springRest.push_back(sd.springRest[i]);
        d.springK.push_back(sd.springK[i]);
    }
    d.springCount += sd.springCount;
}

namespace
{

// Adds the preset rows for one joint type from its stored typed def. The field copies are the
// typed API's field names, one case per def type.
int addTypedJoint(SceneBuilder &sb, const jointpresets::JointBase &base, int type, const unsigned char *bytes)
{
    switch (type)
    {
    case avbd3d_weldJoint:
    {
        avbd3d_WeldJointDef d;
        std::memcpy(&d, bytes, sizeof d);
        jointpresets::WeldDef j;
        j.base = base;
        j.linearHertz = d.linearHertz;
        j.angularHertz = d.angularHertz;
        j.linearDampingRatio = d.linearDampingRatio;
        j.angularDampingRatio = d.angularDampingRatio;
        return jointpresets::addWeld(sb, j);
    }
    case avbd3d_revoluteJoint:
    {
        avbd3d_RevoluteJointDef d;
        std::memcpy(&d, bytes, sizeof d);
        jointpresets::RevoluteDef j;
        j.base = base;
        j.targetAngle = d.targetAngle;
        j.enableSpring = d.enableSpring;
        j.hertz = d.hertz;
        j.dampingRatio = d.dampingRatio;
        j.enableLimit = d.enableLimit;
        j.lowerAngle = d.lowerAngle;
        j.upperAngle = d.upperAngle;
        j.enableMotor = d.enableMotor;
        j.maxMotorTorque = d.maxMotorTorque;
        j.motorSpeed = d.motorSpeed;
        return jointpresets::addRevolute(sb, j);
    }
    case avbd3d_prismaticJoint:
    {
        avbd3d_PrismaticJointDef d;
        std::memcpy(&d, bytes, sizeof d);
        jointpresets::PrismaticDef j;
        j.base = base;
        j.enableSpring = d.enableSpring;
        j.hertz = d.hertz;
        j.dampingRatio = d.dampingRatio;
        j.targetTranslation = d.targetTranslation;
        j.enableLimit = d.enableLimit;
        j.lowerTranslation = d.lowerTranslation;
        j.upperTranslation = d.upperTranslation;
        j.enableMotor = d.enableMotor;
        j.maxMotorForce = d.maxMotorForce;
        j.motorSpeed = d.motorSpeed;
        return jointpresets::addPrismatic(sb, j);
    }
    case avbd3d_wheelJoint:
    {
        avbd3d_WheelJointDef d;
        std::memcpy(&d, bytes, sizeof d);
        jointpresets::WheelDef j;
        j.base = base;
        j.enableSuspensionSpring = d.enableSuspensionSpring;
        j.suspensionHertz = d.suspensionHertz;
        j.suspensionDampingRatio = d.suspensionDampingRatio;
        j.enableSuspensionLimit = d.enableSuspensionLimit;
        j.lowerSuspensionLimit = d.lowerSuspensionLimit;
        j.upperSuspensionLimit = d.upperSuspensionLimit;
        j.enableSpinMotor = d.enableSpinMotor;
        j.maxSpinTorque = d.maxSpinTorque;
        j.spinSpeed = d.spinSpeed;
        j.enableSteering = d.enableSteering;
        j.steeringHertz = d.steeringHertz;
        j.steeringDampingRatio = d.steeringDampingRatio;
        j.targetSteeringAngle = d.targetSteeringAngle;
        j.maxSteeringTorque = d.maxSteeringTorque;
        j.enableSteeringLimit = d.enableSteeringLimit;
        j.lowerSteeringLimit = d.lowerSteeringLimit;
        j.upperSteeringLimit = d.upperSteeringLimit;
        return jointpresets::addWheel(sb, j);
    }
    case avbd3d_sphericalJoint:
    {
        avbd3d_SphericalJointDef d;
        std::memcpy(&d, bytes, sizeof d);
        jointpresets::SphericalDef j;
        j.base = base;
        j.enableSpring = d.enableSpring;
        j.hertz = d.hertz;
        j.dampingRatio = d.dampingRatio;
        j.targetRotation = toQ(d.targetRotation);
        j.enableConeLimit = d.enableConeLimit;
        j.coneAngle = d.coneAngle;
        j.enableTwistLimit = d.enableTwistLimit;
        j.lowerTwistAngle = d.lowerTwistAngle;
        j.upperTwistAngle = d.upperTwistAngle;
        j.enableMotor = d.enableMotor;
        j.maxMotorTorque = d.maxMotorTorque;
        j.motorVelocity = toV(d.motorVelocity);
        return jointpresets::addSpherical(sb, j);
    }
    case avbd3d_distanceJoint:
    {
        avbd3d_DistanceJointDef d;
        std::memcpy(&d, bytes, sizeof d);
        jointpresets::DistanceDef j;
        j.base = base;
        j.length = d.length;
        j.enableSpring = d.enableSpring;
        j.lowerSpringForce = d.lowerSpringForce;
        j.upperSpringForce = d.upperSpringForce;
        j.hertz = d.hertz;
        j.dampingRatio = d.dampingRatio;
        j.enableLimit = d.enableLimit;
        j.minLength = d.minLength;
        j.maxLength = d.maxLength;
        j.enableMotor = d.enableMotor;
        j.maxMotorForce = d.maxMotorForce;
        j.motorSpeed = d.motorSpeed;
        return jointpresets::addDistance(sb, j);
    }
    case avbd3d_motorJoint:
    {
        avbd3d_MotorJointDef d;
        std::memcpy(&d, bytes, sizeof d);
        jointpresets::MotorDef j;
        j.base = base;
        j.linearVelocity = toV(d.linearVelocity);
        j.maxVelocityForce = d.maxVelocityForce;
        j.angularVelocity = toV(d.angularVelocity);
        j.maxVelocityTorque = d.maxVelocityTorque;
        j.linearHertz = d.linearHertz;
        j.linearDampingRatio = d.linearDampingRatio;
        j.maxSpringForce = d.maxSpringForce;
        j.angularHertz = d.angularHertz;
        j.angularDampingRatio = d.angularDampingRatio;
        j.maxSpringTorque = d.maxSpringTorque;
        return jointpresets::addMotor(sb, j);
    }
    case avbd3d_parallelJoint:
    {
        avbd3d_ParallelJointDef d;
        std::memcpy(&d, bytes, sizeof d);
        jointpresets::ParallelDef j;
        j.base = base;
        j.hertz = d.hertz;
        j.dampingRatio = d.dampingRatio;
        j.maxTorque = d.maxTorque;
        return jointpresets::addParallel(sb, j);
    }
    case avbd3d_filterJoint:
    {
        jointpresets::FilterDef j;
        j.base = base;
        jointpresets::addFilter(sb, j);
        return 0; // no rows, only the ignore pair
    }
    case avbd3d_avbdJoint:
    {
        avbd3d_AvbdJointDef d;
        std::memcpy(&d, bytes, sizeof d);
        int j;
        if (d.kind == avbd3d_avbdJointPoint)
            j = sb.addJoint(base.bodyA, base.bodyB, base.localFrameA.p, base.localFrameB.p, d.linearStiffness,
                            d.angularStiffness, d.fractureForce);
        else
        {
            JointDesc jd;
            jd.rA = base.localFrameA.p;
            jd.rB = base.localFrameB.p;
            jd.frameA = base.localFrameA.q;
            jd.frameB = base.localFrameB.q;
            for (int k = 0; k < 6; k++)
                jd.axis[k] = JointAxis{(int)d.axes[k].mode, d.axes[k].lo, d.axes[k].hi};
            jd.fracture = d.fractureForce;
            j = sb.addJoint6(base.bodyA, base.bodyB, jd);
            sb.setJointGeneric(j, (int)d.kind);
            sb.setJointForceBounds(j, d.linearStiffness, d.angularStiffness);
        }
        if (d.fractureTorque >= 0.0f)
            sb.setJointFractureAng(j, d.fractureTorque);
        if (base.collideConnected)
            sb.allowCollision(base.bodyA, base.bodyB);
        return j;
    }
    case avbd3d_avbdSpring:
    {
        avbd3d_AvbdSpringDef d;
        std::memcpy(&d, bytes, sizeof d);
        const int j =
            sb.addSpring(base.bodyA, base.bodyB, base.localFrameA.p, base.localFrameB.p, d.stiffness, d.restLength);
        if (base.collideConnected)
            sb.allowCollision(base.bodyA, base.bodyB);
        return j;
    }
    default:
        return -1;
    }
}

} // namespace

std::vector<unsigned long long> World::ignoreOf(const SceneDesc &sd, int a, int b) const
{
    std::vector<unsigned long long> out;
    for (unsigned long long k : sd.ignorePairs)
    {
        const int lo = (int)(k >> 32), hi = (int)(k & 0xFFFFFFFFu);
        const int x = lo == 0 && a >= 0 ? a : b, y = hi == 0 && a >= 0 ? a : b;
        if (x != y)
        {
            const int mn = std::min(x, y), mx = std::max(x, y);
            out.push_back(((unsigned long long)(unsigned)mn << 32) | (unsigned)mx);
        }
    }
    return out;
}

SceneDesc World::buildJointRows(const JointRec &rec)
{
    avbd3d_JointDef base;
    std::memcpy(&base, rec.def.data(), sizeof base);
    avbd3d_Transform fa = base.localFrameA, fb = base.localFrameB;
    if (rec.swapped)
        std::swap(fa, fb);

    SceneBuilder sb;
    // A body at its creation pose: the rows derive their rest orientation and arm from it.
    auto copyBody = [&](int slot, int which)
    {
        const BodyRec &r = bodies[slot];
        const SceneDesc &d = desc;
        const float dens = r.type == avbd3d_dynamicBody ? (r.density > 0 ? r.density : 1.0f) : 0.0f;
        const vec3 h = d.halfExtents[slot];
        const int kind = r.primary >= 0 ? d.shapeType[slot] : AVBD_SHAPE_BOX;
        const vec3 pos = rec.snapPos[which];
        const quat rot = rec.snapRot[which];
        if (kind == AVBD_SHAPE_SPHERE)
            return sb.addSphere(h.x, dens, d.friction[slot], pos, vec3{0, 0, 0}, vec3{0, 0, 0}, rot);
        if (kind == AVBD_SHAPE_CAPSULE)
            return sb.addCapsule(h.x, h.z - h.x, dens, d.friction[slot], pos, vec3{0, 0, 0}, vec3{0, 0, 0}, rot);
        return sb.addBody(h * 2.0f, dens, d.friction[slot], pos, vec3{0, 0, 0}, vec3{0, 0, 0}, rot);
    };
    jointpresets::JointBase jb;
    if (rec.a >= 0)
        jb.bodyA = copyBody(rec.a, 0);
    jb.bodyB = copyBody(rec.b, 1);
    jb.localFrameA.p = toV(fa.p);
    jb.localFrameA.q = toQ(fa.q);
    jb.localFrameB.p = toV(fb.p);
    jb.localFrameB.q = toQ(fb.q);
    jb.forceThreshold = base.forceThreshold;
    jb.torqueThreshold = base.torqueThreshold;
    jb.collideConnected = base.collideConnected;
    if (addTypedJoint(sb, jb, rec.type, rec.def.data()) < 0)
        return SceneDesc{};
    return sb.finalize();
}

avbd3d_JointId World::createJoint(int type, const void *def, size_t bytes)
{
    syncBodies(); // the rows are built from both bodies' poses
    avbd3d_JointDef base;
    if (!def || bytes < sizeof base)
        return avbd3d_nullJointId;
    std::memcpy(&base, def, sizeof base);

    avbd3d_BodyId ia = base.bodyIdA, ib = base.bodyIdB;
    const bool swapped = ib.index1 == 0;
    if (swapped)
        std::swap(ia, ib);
    if (!bodyValid(ib) || (ia.index1 != 0 && !bodyValid(ia)) || ib.index1 == ia.index1)
        return avbd3d_nullJointId;
    const int b = ib.index1;
    const int a = ia.index1 != 0 ? ia.index1 : -1;

    JointRec rec;
    rec.alive = true;
    rec.a = a;
    rec.b = b;
    rec.swapped = swapped;
    rec.type = type;
    rec.def.assign((const unsigned char *)def, (const unsigned char *)def + bytes);
    if (a >= 0)
    {
        rec.snapPos[0] = desc.pos[a];
        rec.snapRot[0] = desc.rot[a];
    }
    rec.snapPos[1] = desc.pos[b];
    rec.snapRot[1] = desc.rot[b];

    const SceneDesc sd = buildJointRows(rec);
    if (sd.jointCount == 0 && sd.springCount == 0 && type != avbd3d_filterJoint)
        return avbd3d_nullJointId;

    rec.firstRow = desc.jointCount;
    rec.rowCount = sd.jointCount;
    rec.firstSpring = desc.springCount;
    rec.springCount = sd.springCount;
    rec.ignore = ignoreOf(sd, a, b);
    appendRows(sd, a, b, rec.ignore);
    joints.push_back(rec);
    const int idx = (int)joints.size() - 1;
    if (a >= 0)
        bodies[a].joints.push_back(idx);
    bodies[b].joints.push_back(idx);
    m_newJoints.push_back(idx);
    m_listsDirty = true;
    avbd3d_JointId id;
    id.index1 = idx + 1;
    id.world0 = (uint16_t)worldIndex;
    id.generation = joints[idx].gen;
    return id;
}

void World::markJoint(int rec)
{
    JointRec &j = joints[rec];
    if (!j.alive || j.dirty)
        return;
    j.dirty = true;
    m_dirtyJoints.push_back(rec);
}

void World::touchJointThresholds(int rec)
{
    if (joints[rec].alive)
        m_newJoints.push_back(rec);
}

void World::rebuildJoint(int idx)
{
    JointRec &rec = joints[idx];
    const SceneDesc sd = buildJointRows(rec);
    if (sd.jointCount == 0 && rec.type != avbd3d_filterJoint && rec.type != avbd3d_avbdSpring)
        return; // the def no longer builds: the rows stay as they were

    if (sd.jointCount == rec.rowCount)
    {
        // Same rows: copy the authored fields of each row over. The dual state is kept, so a
        // changed motor speed does not restart the row's warm start. Thresholds are not touched
        // here: they only change through touchJointThresholds.
        std::vector<JointWriteRecGpu> recs;
        recs.reserve(sd.jointCount);
        for (int i = 0; i < sd.jointCount; i++)
        {
            const int row = rec.firstRow + i;
            desc.jointRA[row] = sd.jointRA[i];
            desc.jointRB[row] = sd.jointRB[i];
            desc.jointKLin[row] = sd.jointKLin[i];
            desc.jointKAng[row] = sd.jointKAng[i];
            desc.jointFrac[row] = sd.jointFrac[i];
            desc.jointFracAng[row] = sd.jointFracAng[i];
            desc.jointArm[row] = sd.jointArm[i];
            desc.jointRestAng[row] = sd.jointRestAng[i];
            desc.jointFrameA[row] = sd.jointFrameA[i];
            desc.jointFrameB[row] = sd.jointFrameB[i];
            desc.jointGeneric[row] = sd.jointGeneric[i];
            for (int k = 0; k < 6; k++)
            {
                desc.jointAxisMode[6 * row + k] = sd.jointAxisMode[6 * i + k];
                desc.jointAxisLo[6 * row + k] = sd.jointAxisLo[6 * i + k];
                desc.jointAxisHi[6 * row + k] = sd.jointAxisHi[6 * i + k];
            }
            if (row >= m_vkJoints)
                continue; // not on the device yet: this flush's append uploads the new values
            JointWriteRecGpu g{};
            g.row = (uint32_t)row;
            g.rA[0] = sd.jointRA[i].x;
            g.rA[1] = sd.jointRA[i].y;
            g.rA[2] = sd.jointRA[i].z;
            g.rB[0] = sd.jointRB[i].x;
            g.rB[1] = sd.jointRB[i].y;
            g.rB[2] = sd.jointRB[i].z;
            const quat ra = sd.jointRestAng[i], fa = sd.jointFrameA[i], fb = sd.jointFrameB[i];
            g.restAng[0] = ra.x;
            g.restAng[1] = ra.y;
            g.restAng[2] = ra.z;
            g.restAng[3] = ra.w;
            g.frameA[0] = fa.x;
            g.frameA[1] = fa.y;
            g.frameA[2] = fa.z;
            g.frameA[3] = fa.w;
            g.frameB[0] = fb.x;
            g.frameB[1] = fb.y;
            g.frameB[2] = fb.z;
            g.frameB[3] = fb.w;
            g.kLin = sd.jointKLin[i];
            g.kAng = sd.jointKAng[i];
            g.frac = sd.jointFrac[i];
            g.fracAng = sd.jointFracAng[i];
            g.arm = sd.jointArm[i];
            g.generic = sd.jointGeneric[i];
            for (int k = 0; k < 6; k++)
            {
                g.axisMode[k] = sd.jointAxisMode[6 * i + k];
                g.axisLo[k] = sd.jointAxisLo[6 * i + k];
                g.axisHi[k] = sd.jointAxisHi[6 * i + k];
            }
            recs.push_back(g);
        }
        if (!recs.empty())
            m_vk.setJointRows(recs.data(), (int)recs.size());
    }
    else
    {
        // Different row count: the old rows go inert and fresh rows are appended. The flush's
        // append uploads them, and the broken flags are set after it.
        for (int r = 0; r < rec.rowCount; r++)
            m_breakRows.push_back(rec.firstRow + r);
        rec.firstRow = desc.jointCount;
        rec.rowCount = sd.jointCount;
        rec.ignore = ignoreOf(sd, rec.a, rec.b);
        appendRows(sd, rec.a, rec.b, rec.ignore);
        m_newJoints.push_back(idx); // thresholds for the new first row
    }
    rec.ignore = ignoreOf(sd, rec.a, rec.b);
    m_listsDirty = true;
}

void World::destroyJoint(int rec)
{
    JointRec &j = joints[rec];
    j.alive = false;
    j.gen = (uint16_t)(j.gen + 1 ? j.gen + 1 : 1);
    for (int r = 0; r < j.rowCount; r++)
        m_breakRows.push_back(j.firstRow + r);
    for (int r = 0; r < j.springCount; r++)
        desc.springK[j.firstSpring + r] = 0.0f; // springs have no broken flag: an inert spring pulls with 0
    if (j.springCount > 0)
        m_springsDirty = true;
    if (j.a >= 0)
    {
        auto &v = bodies[j.a].joints;
        v.erase(std::remove(v.begin(), v.end(), rec), v.end());
    }
    auto &v = bodies[j.b].joints;
    v.erase(std::remove(v.begin(), v.end(), rec), v.end());
    j.ignore.clear();
    m_listsDirty = true;
}

// ---- flush and step -----------------------------------------------------------------------

void World::rebuildLists()
{
    SceneDesc &d = desc;
    d.dynamicIdx.clear();
    d.staticIdx.clear();
    m_movable.clear();
    m_hot.assign(d.count, BodyHot{});
    float maxHalf = 0.0f;
    for (int b = 1; b < d.count; b++)
    {
        const BodyRec &r = bodies[b];
        m_hot[b].user = r.user;
        m_hot[b].gen = r.gen;
        if (!r.alive || !r.enabled || r.primary < 0)
            continue;
        if (r.type != avbd3d_staticBody)
        {
            m_movable.push_back(b);
            m_hot[b].movable = 1;
        }
        if (r.type == avbd3d_dynamicBody && d.mass[b] > 0.0f)
        {
            d.dynamicIdx.push_back(b);
            for (int s : r.shapes)
                maxHalf = std::max(maxHalf, length(d.shapeHalf[s]));
        }
        else
            d.staticIdx.push_back(b);
    }
    d.cellSize = std::max(2.0f * maxHalf + 2.0f * COLLISION_MARGIN, 1.0f);

    d.ignorePairs.clear();
    for (const JointRec &j : joints)
        if (j.alive)
            d.ignorePairs.insert(d.ignorePairs.end(), j.ignore.begin(), j.ignore.end());
    std::sort(d.ignorePairs.begin(), d.ignorePairs.end());
    d.ignorePairs.erase(std::unique(d.ignorePairs.begin(), d.ignorePairs.end()), d.ignorePairs.end());
}

void World::flush()
{
    // Joints whose def changed: rebuild their rows first, so the append below uploads them.
    for (int idx : m_dirtyJoints)
    {
        joints[idx].dirty = false;
        if (joints[idx].alive)
            rebuildJoint(idx);
    }
    m_dirtyJoints.clear();

    // A slot freed since the last step is held back one step: the device purges nothing, but the
    // narrowphase only keeps the previous step's manifolds, so by the time the slot is reused
    // none of them names its old occupant.
    m_quarBodies.insert(m_quarBodies.end(), m_pendingBodies.begin(), m_pendingBodies.end());
    m_quarShapes.insert(m_quarShapes.end(), m_pendingShapes.begin(), m_pendingShapes.end());
    m_pendingBodies.clear();
    m_pendingShapes.clear();

    const bool lists = m_listsDirty;
    if (lists)
        rebuildLists();
    m_listsDirty = false;

    // Slots that already exist on the device and changed.
    m_scratchBodies.clear();
    m_scratchShapes.clear();
    for (int s : m_refreshBodies)
    {
        bodies[s].refresh = false;
        if (s < m_vkBodies)
            m_scratchBodies.push_back(s);
    }
    for (int s : m_refreshShapes)
    {
        shapes[s].refresh = false;
        if (s < m_vkShapes)
            m_scratchShapes.push_back(s);
    }
    m_refreshBodies.clear();
    m_refreshShapes.clear();
    if (!m_scratchBodies.empty() || !m_scratchShapes.empty())
        m_vk.refreshSlots(desc, m_scratchBodies, m_scratchShapes);

    if (desc.count > m_vkBodies || desc.shapeCount > m_vkShapes || desc.jointCount > m_vkJoints ||
        desc.springCount > m_vkSprings || m_springsDirty)
    {
        m_vk.append(desc);
        m_vkBodies = desc.count;
        m_vkShapes = desc.shapeCount;
        m_vkJoints = desc.jointCount;
        m_vkSprings = desc.springCount;
        m_springsDirty = false;
    }
    if (lists)
        m_vk.refreshLists(desc);

    for (int idx : m_newJoints)
    {
        if (!joints[idx].alive || joints[idx].rowCount == 0)
            continue;
        avbd3d_JointDef base;
        std::memcpy(&base, joints[idx].def.data(), sizeof base);
        m_vk.setJointThresholds(joints[idx].firstRow, base.forceThreshold, base.torqueThreshold);
    }
    m_newJoints.clear();
    syncRenderGroups();
    for (int row : m_breakRows)
        m_vk.setJointBroken(row, true);
    m_breakRows.clear();

    // Replay the body writes in order, one setBodyData per run of the same kind.
    size_t i = 0;
    while (i < m_ops.size())
    {
        const BodyWrite t = m_ops[i].type;
        const size_t stride = strideOf(t);
        m_opIdx.clear();
        m_opData.clear();
        while (i < m_ops.size() && m_ops[i].type == t)
        {
            const Op &op = m_ops[i++];
            const BodyRec &r = bodies[op.body];
            if (!r.alive || !r.enabled || op.body >= m_vkBodies)
                continue;
            m_opIdx.push_back(op.body);
            m_opData.insert(m_opData.end(), op.data, op.data + stride);
        }
        if (!m_opIdx.empty())
            m_vk.setBodyData(m_opIdx.data(), m_opData.data(), t, (int)m_opIdx.size());
    }
    m_ops.clear();
}

void World::stepBegin(float dt)
{
    if (m_inFlight)
        stepEnd();
    // A dead world makes no Vulkan call: the queued writes stay queued and are never applied.
    if (m_error != avbd3d_errorNone)
        return;
    try
    {
        flush();
        params.dt = dt;
        params.gravity = toV(gravity);
        params.sleepEnabled = sleepEnabled;
        {
            const auto ci = props.find("continuous");
            params.enableContinuous = ci == props.end() || ci->second != 0.0;
        }
        m_vk.enableContactReport(m_reportShapes > 0);
        m_vk.stepBegin(params); // records a Vulkan failure in the world instead of throwing
        m_inFlight = true;
        m_view = VkWorld::SnapshotView{}; // the begin may have regrown the host copy
        m_jview = VkWorld::JointSnapshotView{};
    }
    catch (...)
    {
        noteException();
    }
}

void World::syncDeviceError()
{
    const VkResult r = m_vk.deviceError();
    if (r == VK_SUCCESS || m_error != avbd3d_errorNone)
        return;
    m_error = r == VK_ERROR_DEVICE_LOST                                                  ? avbd3d_errorDeviceLost
              : (r == VK_ERROR_OUT_OF_DEVICE_MEMORY || r == VK_ERROR_OUT_OF_HOST_MEMORY) ? avbd3d_errorOutOfMemory
                                                                                         : avbd3d_errorDevice;
    m_errorText = m_vk.deviceErrorWhat();
}

void World::noteException()
{
    std::string text;
    const avbd3d_Error e = errorOfCurrentException(text);
    if (m_error != avbd3d_errorNone)
        return;
    m_error = e;
    m_errorText = text;
    fprintf(stderr, "[avbd3d] world is dead: %s\n", text.c_str());
}

void World::stepEnd()
{
    if (!m_inFlight)
        return;
    m_result = m_vk.stepEnd();
    m_inFlight = false;
    syncDeviceError();
    // The failed step published nothing: the mirror keeps the last good state.
    if (m_error != avbd3d_errorNone)
        return;

    m_freeBodies.insert(m_freeBodies.end(), m_quarBodies.begin(), m_quarBodies.end());
    m_freeShapes.insert(m_freeShapes.end(), m_quarShapes.begin(), m_quarShapes.end());
    m_quarBodies.clear();
    m_quarShapes.clear();

    collectContacts();
    // No body is mirrored here: the read model pulls them on first use (syncBody).
    // The sensor queries go through queryTree(), which syncs and refits the moved bodies first.
    if (!m_sensors.empty() || !m_pendingSensorEnd.empty() || !sensorBegin.empty() || !sensorEnd.empty())
        collectSensors();
    // A game that read the mirror after the last step will read it after this one: start the
    // host copy now, on the copy queue, so the first read finds it landed. One that never reads
    // never pays for it.
    if (m_wantSnapshot)
    {
        m_vk.requestSnapshotCopy();
        m_wantSnapshot = false;
    }
}

const VkWorld::SnapshotView &World::view()
{
    if (m_view.serial != m_vk.snapshotSerial())
        m_view = m_vk.snapshotView(); // waits for the DMA only, never for a step in flight
    m_wantSnapshot = true;
    return m_view;
}

bool World::jointForce(int rec, vec3 &force, vec3 &torque)
{
    force = torque = vec3{0, 0, 0};
    if (rec < 0 || rec >= (int)joints.size() || !joints[rec].alive || joints[rec].rowCount == 0)
        return false;
    if (m_jview.serial != m_vk.snapshotSerial())
        m_jview = m_vk.jointSnapshotView();
    m_wantSnapshot = true;
    const JointRec &j = joints[rec];
    const int row = j.firstRow;
    if (m_jview.serial == 0 || row >= m_jview.count)
        return false;
    syncBody(j.b);
    if (j.a >= 0)
        syncBody(j.a);
    const SceneDesc &d = desc;
    const JointForce f =
        VkWorld::jointForce(d.jointGeneric[row], m_jview.lambdaLin[row], m_jview.lambdaAng[row], d.jointArm[row],
                            d.jointFrameA[row], d.jointFrameB[row], d.jointRA[row], d.jointRB[row], j.a >= 0,
                            j.a >= 0 ? d.pos[j.a] : vec3{0, 0, 0}, j.a >= 0 ? d.rot[j.a] : quat{0, 0, 0, 1}, d.pos[j.b],
                            d.rot[j.b], d.jointAxisMode[row * 6 + 3] == JointAxisCone);
    force = f.force;
    torque = f.torque;
    return true;
}

bool World::jointMotorLambda(int rec, int axis, float &lambda, float &arm)
{
    lambda = 0.0f;
    arm = 1.0f;
    if (rec < 0 || rec >= (int)joints.size() || !joints[rec].alive || joints[rec].rowCount == 0)
        return false;
    if (m_jview.serial != m_vk.snapshotSerial())
        m_jview = m_vk.jointSnapshotView();
    m_wantSnapshot = true;
    const JointRec &j = joints[rec];
    for (int i = 0; i < j.rowCount; i++)
    {
        const int row = j.firstRow + i;
        if ((desc.jointAxisMode[row * 6 + axis] & 7) != JointAxisVelocityMotor)
            continue;
        if (m_jview.serial == 0 || row >= m_jview.count)
            return false;
        lambda = axis < 3 ? (&m_jview.lambdaLin[row].x)[axis] : (&m_jview.lambdaAng[row].x)[axis - 3];
        arm = desc.jointArm[row];
        return true;
    }
    return false;
}

std::vector<avbd3d_JointEvent> &World::jointEvents()
{
    m_jevents.clear();
    for (int i = 0; i < (int)joints.size(); i++)
    {
        const JointRec &j = joints[i];
        if (!j.alive || j.rowCount == 0)
            continue;
        avbd3d_JointDef base;
        std::memcpy(&base, j.def.data(), sizeof base);
        if (!(base.forceThreshold < FLT_MAX || base.torqueThreshold < FLT_MAX))
            continue;
        if (!((j.a >= 0 && isAwake(j.a)) || isAwake(j.b)))
            continue;
        vec3 f, t;
        if (!jointForce(i, f, t))
            continue;
        if (length(f) >= base.forceThreshold || length(t) >= base.torqueThreshold)
        {
            avbd3d_JointEvent e;
            e.jointId.index1 = i + 1;
            e.jointId.world0 = (uint16_t)worldIndex;
            e.jointId.generation = j.gen;
            e.userData = base.userData;
            m_jevents.push_back(e);
        }
    }
    return m_jevents;
}

void World::pullBody(int b)
{
    const VkWorld::SnapshotView &v = view();
    // Held: the host wrote this body after the snapshot's step began.
    if (v.serial == 0 || v.serial < m_bodyHold[b] || b >= v.count)
        return;
    // Kinematic bodies are advanced by the device too (by their own velocity); static and
    // disabled ones never move there and keep the host's values.
    if (b < (int)m_hot.size() && m_hot[b].movable)
    {
        // A body asleep before and after a step the mirror already followed did not move.
        const bool moved = v.awake[b] || v.prevAwake[b] || m_bodySerial[b] + 1 != v.serial;
        if (moved)
        {
            SceneDesc &d = desc;
            d.pos[b] = v.pos[b];
            d.rot[b] = v.rot[b];
            d.velLin[b] = v.velLin[b];
            d.velAng[b] = v.velAng[b];
            // Query proxies are refit lazily (ensureProxies), only when a query needs them.
            if ((int)m_proxyFlag.size() <= b)
                m_proxyFlag.resize(desc.count, 0);
            if (!m_proxyFlag[b])
            {
                m_proxyFlag[b] = 1;
                m_proxyDirty.push_back(b);
            }
        }
        awake[b] = v.awake[b];
    }
    m_bodySerial[b] = v.serial;
}

void World::syncBodies()
{
    const uint64_t latest = m_vk.snapshotSerial();
    if (m_allSerial == latest)
        return;
    m_allSerial = latest;
    for (int b : m_movable)
        syncBody(b);
}

// ---- contact events ----------------------------------------------------------------------

void World::setShapeReport(int s)
{
    ShapeRec &r = shapes[s];
    const bool on = r.alive && (r.contactEvents || r.hitEvents);
    if (on != r.reporting)
    {
        r.reporting = on;
        m_reportShapes += on ? 1 : -1;
    }
    unsigned f = desc.shapeFlags[s] & ~AVBD_SHAPE_FLAG_REPORT;
    if (on)
        f |= AVBD_SHAPE_FLAG_REPORT;
    if (f != desc.shapeFlags[s])
    {
        desc.shapeFlags[s] = f;
        markShape(s);
    }
}

avbd3d_ContactId World::contactId(int c) const
{
    avbd3d_ContactId id;
    id.index1 = c + 1;
    id.world0 = (uint16_t)worldIndex;
    id.padding = 0;
    id.generation = m_contacts[c].gen;
    return id;
}

bool World::contactValid(avbd3d_ContactId id) const
{
    return id.index1 >= 1 && id.index1 <= (int)m_contacts.size() && id.world0 == worldIndex &&
           m_contacts[id.index1 - 1].alive && m_contacts[id.index1 - 1].gen == id.generation;
}

std::vector<int> World::contactsOf(int slot, bool isBody) const
{
    std::vector<int> out;
    for (int c = 0; c < (int)m_contacts.size(); c++)
    {
        const ContactRec &k = m_contacts[c];
        if (!k.alive)
            continue;
        const bool hit = isBody ? (k.bodyA == slot || k.bodyB == slot) : (k.shapeA == slot || k.shapeB == slot);
        if (hit)
            out.push_back(c);
    }
    return out;
}

avbd3d_ContactData World::contactData(int c) const
{
    const ContactRec &k = m_contacts[c];
    avbd3d_ContactData d;
    std::memset(&d, 0, sizeof d);
    d.contactId = contactId(c);
    d.shapeIdA = shapeId(k.shapeA);
    d.shapeIdB = shapeId(k.shapeB);
    d.manifolds = &k.manifold;
    d.manifoldCount = 1;
    return d;
}

int World::allocContact()
{
    if (!m_freeContacts.empty())
    {
        const int c = m_freeContacts.back();
        m_freeContacts.pop_back();
        return c;
    }
    m_contacts.emplace_back();
    return (int)m_contacts.size() - 1;
}

void World::releaseContact(int c)
{
    ContactRec &k = m_contacts[c];
    m_pairs.erase(k.key);
    k.alive = false;
    k.gen = k.gen + 1 ? k.gen + 1 : 1;
    m_freeContacts.push_back(c);
}

static uint64_t pairKey(int a, int b)
{
    if (a > b)
        std::swap(a, b);
    return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
}

// One device manifold as an avbd3d_Manifold. The record carries the sums and the first four
// points; per-point fields the record does not hold are filled from the record level.
static void fillManifold(const ContactReportRecordGpu &g, avbd3d_Manifold &m)
{
    std::memset(&m, 0, sizeof m);
    m.normal = avbd3d_Vec3{g.nx, g.ny, g.nz};
    m.pointCount = (int)std::min<uint32_t>(g.pointCount, AVBD3D_MAX_MANIFOLD_POINTS);
    for (int i = 0; i < m.pointCount; i++)
    {
        const ContactReportPointGpu &p = g.points[i];
        avbd3d_ManifoldPoint &mp = m.points[i];
        mp.anchorA = avbd3d_Vec3{p.ax, p.ay, p.az};
        mp.anchorB = avbd3d_Vec3{p.bx, p.by, p.bz};
        mp.separation = p.separation;
        mp.baseSeparation = p.separation;
        mp.normalImpulse = p.normalImpulse;
        mp.totalNormalImpulse = p.normalImpulse;
        mp.normalVelocity = -g.approachSpeed; // negative while the pair approaches
        mp.triangleIndex = -1;
    }
}

void World::collectContacts()
{
    m_step++;
    contactBegin.clear();
    contactEnd.clear();
    contactHits.clear();

    const VkWorld::ContactReportView v = m_vk.contactReportView();
    contactTruncated = v.truncated;
    const int nShapes = (int)shapes.size();
    const int nBodies = (int)bodies.size();
    for (int i = 0; i < v.count; i++)
    {
        const ContactReportRecordGpu &g = v.records[i];
        const int sa = g.shapeA, sb = g.shapeB;
        if (sa < 0 || sb < 0 || sa >= nShapes || sb >= nShapes || sa == sb)
            continue;
        const ShapeRec &ra = shapes[sa], &rb = shapes[sb];
        // A destroyed shape can still appear in the previous step's manifold; its contact ends below.
        if (!ra.alive || !rb.alive || g.bodyA < 0 || g.bodyB < 0 || g.bodyA >= nBodies || g.bodyB >= nBodies)
            continue;
        const bool evContact = ra.contactEvents || rb.contactEvents;
        const bool evHit = ra.hitEvents || rb.hitEvents;
        if (!evContact && !evHit)
            continue;

        const uint64_t key = pairKey(sa, sb);
        int c;
        auto it = m_pairs.find(key);
        if (it == m_pairs.end())
        {
            c = allocContact();
            ContactRec &nk = m_contacts[c];
            nk.alive = true;
            nk.key = key;
            nk.genA = shapes[sa].gen;
            nk.genB = shapes[sb].gen;
            m_pairs.emplace(key, c);
            if (evContact)
                contactBegin.push_back(avbd3d_ContactBeginTouchEvent{shapeId(sa), shapeId(sb), contactId(c)});
        }
        else
        {
            c = it->second;
        }
        ContactRec &k = m_contacts[c];
        k.stamp = m_step;
        k.events = evContact;
        k.shapeA = sa;
        k.shapeB = sb;
        k.bodyA = g.bodyA;
        k.bodyB = g.bodyB;
        fillManifold(g, k.manifold);

        // Hit: closing speed at or above the threshold with a compressive impulse this step. The
        // point is the midpoint of the first contact, from the body positions at step start (desc
        // still holds them: the move readout is copied in below).
        if (evHit && g.approachSpeed >= hitThreshold && g.normalImpulse > 0.0f)
        {
            const vec3 pa = desc.pos[g.bodyA], pb = desc.pos[g.bodyB];
            const ContactReportPointGpu &p0 = g.points[0];
            const vec3 wa = vec3{pa.x + p0.ax, pa.y + p0.ay, pa.z + p0.az};
            const vec3 wb = vec3{pb.x + p0.bx, pb.y + p0.by, pb.z + p0.bz};
            avbd3d_ContactHitEvent h;
            std::memset(&h, 0, sizeof h);
            h.shapeIdA = shapeId(sa);
            h.shapeIdB = shapeId(sb);
            h.contactId = contactId(c);
            h.point = fromV(vec3{0.5f * (wa.x + wb.x), 0.5f * (wa.y + wb.y), 0.5f * (wa.z + wb.z)});
            h.normal = avbd3d_Vec3{g.nx, g.ny, g.nz};
            h.approachSpeed = g.approachSpeed;
            h.userMaterialIdA = ra.material.userMaterialId;
            h.userMaterialIdB = rb.material.userMaterialId;
            contactHits.push_back(h);
        }
    }

    // A touching pair that was not reported this step has ended. Skipped when the buffer
    // overflowed (missing records are not ends). Kept while both bodies sleep, since a sleeping
    // pair is not solved and so not reported.
    if (contactTruncated)
        return;
    for (int c = 0; c < (int)m_contacts.size(); c++)
    {
        const ContactRec &k = m_contacts[c];
        if (!k.alive || k.stamp == m_step)
            continue;
        syncBody(k.bodyA);
        syncBody(k.bodyB);
        const bool bothAsleep =
            shapes[k.shapeA].alive && shapes[k.shapeB].alive && awake[k.bodyA] == 0 && awake[k.bodyB] == 0;
        if (bothAsleep)
        {
            m_contacts[c].stamp = m_step;
            continue;
        }
        if (k.events)
        {
            // Name the shapes as the begin event did: a destroyed shape has a newer generation now.
            avbd3d_ShapeId ia{}, ib{};
            ia.index1 = k.shapeA + 1;
            ia.world0 = (uint16_t)worldIndex;
            ia.generation = k.genA;
            ib.index1 = k.shapeB + 1;
            ib.world0 = (uint16_t)worldIndex;
            ib.generation = k.genB;
            contactEnd.push_back(avbd3d_ContactEndTouchEvent{ia, ib, contactId(c)});
        }
        releaseContact(c);
    }
}

avbd3d_Error errorOfCurrentException(std::string &text)
{
    const avbdvk::Classified c = avbdvk::classifyCurrentException();
    text = c.what;
    switch (c.kind)
    {
    case avbdvk::Failure::HostOutOfMemory:
    case avbdvk::Failure::DeviceOutOfMemory:
        return avbd3d_errorOutOfMemory;
    case avbdvk::Failure::DeviceLost:
        return avbd3d_errorDeviceLost;
    default:
        return avbd3d_errorDevice;
    }
}

World *settled(avbd3d_WorldId id)
{
    return lookupWorld(id);
}

World *settledDevice(avbd3d_WorldId id)
{
    World *w = lookupWorld(id);
    if (w)
        w->settle();
    return w;
}

} // namespace avbd3d_impl
