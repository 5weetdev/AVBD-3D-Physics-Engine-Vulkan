// avbd3d world layer: one VkWorld per avbd3d world, a host SceneDesc mirror, generational ids,
// and a batch of structural changes and body writes applied between steps.
//
// Everything a caller does between steps (create/destroy bodies, shapes and joints, setters)
// edits the mirror at once and is queued for the device. The queue is applied at the top of the
// next StepBegin, before the step is recorded. After StepEnd the poses, velocities and sleep
// flags come back through the move readout the step copies inside its own submission, so the
// getters read the mirror and a step is still one queue submission.
#pragma once

#include "avbd3d_internal.h"

#include "box3d/collision.h"
#include "box3d/types.h"

#include "joint_presets.h"
#include "scene_desc.h"
#include "shape_mass_b3.h"
#include "vk_world.h"

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace avbd3d_impl
{
using avbdvk::ContactReportPointGpu;
using avbdvk::Device;
using avbdvk::ContactReportRecordGpu;
using avbdvk::JointWriteRecGpu;
using avbdvk::VkWorld;
}
namespace avbd3d_impl
{

struct BodyRec
{
    bool alive = false;
    uint16_t gen = 1;
    int type = avbd3d_dynamicBody;
    bool enabled = true;
    void *user = nullptr;
    int primary = -1;     // shape slot that fixes size and mass, -1 = none
    float density = 1.0f; // of the primary shape
    float massDyn = 0.0f; // mass as a dynamic body
    vec3 momentDyn{0, 0, 0};
    float linearDamping = 0, angularDamping = 0, gravityScale = 1, sleepThreshold = 0.05f;
    avbd3d_MotionLocks locks{};
    std::vector<int> shapes;
    std::vector<int> joints; // joint record indices
    bool refresh = false;    // slot < vkBodies and changed: re-upload at the next flush
    bool bullet = false;
};

// Host geometry for the query layer. The device keeps its own copy; this one is only read by
// the casts, overlaps, mover and draw in avbd3d_query.cpp.
// Owned host geometry. A general hull is a clone (scale baked in), a mesh is a cooked b3MeshData
// (scale baked in by re-cooking), a height field is a byte copy of the caller's height field.
struct HullFree
{
    void operator()(b3HullData *h) const { b3DestroyHull(h); }
};
struct MeshFree
{
    void operator()(b3MeshData *m) const { b3DestroyMesh(m); }
};
struct HeightFieldFree
{
    void operator()(b3HeightFieldData *h) const { std::free(h); } // malloc'd byte copy
};

struct ShapeGeom
{
    b3Sphere sphere{};
    b3Capsule capsule{};
    std::unique_ptr<b3BoxHull> box; // heap-allocated: b3MakeBoxHull returns by value
    std::unique_ptr<b3HullData, HullFree> hull;            // general convex hull (not a box)
    std::unique_ptr<b3MeshData, MeshFree> mesh;            // mesh shape, or height-field triangles
    std::unique_ptr<b3HeightFieldData, HeightFieldFree> heightField; // original height field

    // The hull the query layer reads: the general hull, or the box hull.
    const b3HullData *hullData() const { return hull ? hull.get() : &box->base; }
    b3Mesh meshView() const { return b3Mesh{mesh.get(), b3Vec3{1.0f, 1.0f, 1.0f}}; }
    void reset()
    {
        box.reset();
        hull.reset();
        mesh.reset();
        heightField.reset();
    }
};

struct ShapeRec
{
    bool alive = false;
    uint16_t gen = 1;
    int body = 0;
    int type = avbd3d_hullShape;
    float density = 1.0f;
    void *user = nullptr;
    avbd3d_Filter filter{};
    avbd3d_SurfaceMaterial material{};
    bool refresh = false;
    vec3 localPos{0, 0, 0}; // shape frame relative to the body origin frame
    quat localRot{0, 0, 0, 1};
    ShapeGeom geom;
    int proxy = -1;      // b3DynamicTree proxy id, -1 = none
    // The tight world AABB. The tree holds it fattened by a margin (Box3D's fatAABB), so a shape
    // that stays inside its fat box does not move its proxy.
    b3AABB aabb{};
    bool contactEvents = false; // enableContactEvents
    bool hitEvents = false;     // enableHitEvents
    bool reporting = false;     // carries AVBD_SHAPE_FLAG_REPORT on the device
    // Sensors (docs/SOLVER_3D_ROADMAP.md Phase 4b-2). A sensor is uploaded with category 0, mask 0
    // and group 0, so the device never makes a contact for it; its overlaps are found on the host.
    bool isSensor = false;
    bool sensorEvents = false;  // enableSensorEvents: gates the sensor and its visitor role alike
    float explosionScale = 1.0f;
    // Sensor only: the visitors inside it at the last step, sorted, one key per visitor:
    // (shape slot << 16) | generation.
    std::vector<uint64_t> overlaps;
    // Baked compound (avbd3d_CreateBakedCompoundShape): every child is a shape of its own on the
    // device. The first is the primary and its id is the compound id; the others have `parent` set
    // to it and are never named by an id (shapeId() maps them to the primary). `members` is on the
    // primary only.
    int parent = -1;
    std::vector<int> members;
};

struct JointRec
{
    bool alive = false;
    uint16_t gen = 1;
    int a = -1, b = 0;     // body slots, a < 0: the world
    bool swapped = false;  // the def named the world-side body as A: its frames are swapped on build
    int firstRow = 0, rowCount = 0;
    int firstSpring = 0, springCount = 0; // avbd3d_avbdSpring: its rows in desc's spring arrays
    int type = 0;          // avbd3d_JointType
    std::vector<unsigned long long> ignore;
    // The typed def as the caller passed it (avbd3d_<Type>JointDef, base first). The typed setters
    // write fields here; the rows are rebuilt from these bytes at the next flush.
    std::vector<unsigned char> def;
    bool dirty = false;
    // Body poses at creation, [0] = body A, [1] = body B. A row's rest orientation and torque arm
    // come from the poses the bodies had when the joint was built, so a rebuild uses these.
    vec3 snapPos[2]{};
    quat snapRot[2]{};
};

struct Op
{
    int body;
    BodyWrite type;
    unsigned char data[28];
};

class World
{
public:
    // Restitution threshold (m/s): stored as a world prop and pushed to the solver.
    void setRestitutionThreshold(float v)
    {
        props["restitutionThreshold"] = v;
        settle();
        m_vk.setRestitutionThreshold(v);
    }
    // shared != null: the world runs on that (initialised) device and leaves it alive.
    explicit World(const avbd3d_WorldDef &def, Device *shared = nullptr);
    ~World();

    // ---- bodies / shapes / joints
    avbd3d_BodyId createBody(const avbd3d_BodyDef &def);
    void destroyBody(int slot);
    avbd3d_ShapeId createShape(int bodySlot, const avbd3d_ShapeDef &def, int kind, vec3 half, vec3 localPos, quat localRot);
    // Attaches cooked or cloned geometry to a shape made by createShape: the host copy for the
    // queries, the device blob (desc.geomBlob index, -1 = none), and the refreshed proxy.
    void attachGeometry(int shapeSlot, int type, int blob, ShapeGeom &&geom);
    // Appends a blob to the scene geometry unless an identical one is already there.
    int internGeometryBlob(const unsigned char *bytes, size_t n, uint64_t hash);
    // updateBodyMass false leaves the body's mass alone (destroyBody, or the caller's choice).
    void destroyShape(int shapeSlot, bool updateBodyMass = true);
    // `def` is the typed def (avbd3d_<Type>JointDef) for `type`, `bytes` its size.
    avbd3d_JointId createJoint(int type, const void *def, size_t bytes);
    void destroyJoint(int rec);
    // Marks a joint whose def changed: its rows are rebuilt at the next flush.
    void markJoint(int rec);
    // Queues a threshold write for a joint at the next flush (no row rebuild).
    void touchJointThresholds(int rec);
    // The read model for joints: the constraint force and torque on body B (world frame) from
    // the newest finished step's multipliers (VkWorld::jointSnapshotView, a small DMA that
    // never waits for a step in flight). False when the joint has no rows or the snapshot does
    // not have it yet (created since); the outputs are zero then.
    bool jointForce(int rec, vec3 &force, vec3 &torque);
    // The multiplier of the joint's velocity-motor row on `axis` (0-2 linear, 3-5 angular, in
    // joint frame A) from the same snapshot, with the row's torque arm. False when no row of the
    // joint carries a motor on that axis or the snapshot has no data for it; lambda is 0 then.
    bool jointMotorLambda(int rec, int axis, float &lambda, float &arm);
    // Joints that are awake and whose force >= forceThreshold or torque >= torqueThreshold in
    // the newest finished step (Box3D: a threshold of 0 reports every awake joint, FLT_MAX is
    // off). Computed when asked.
    std::vector<avbd3d_JointEvent> &jointEvents();

    bool bodyValid(avbd3d_BodyId id) const;
    bool shapeValid(avbd3d_ShapeId id) const;
    bool jointValid(avbd3d_JointId id) const;
    avbd3d_BodyId bodyId(int slot) const;
    avbd3d_ShapeId shapeId(int slot) const;
    // A shape id for a slot at a given generation, which may be stale (end events name shapes
    // that were destroyed after the overlap began).
    avbd3d_ShapeId shapeIdAt(int slot, uint16_t gen) const;

    // ---- body state (origin frame)
    void setTransform(int b, vec3 origin, quat rot);
    void setLinearVelocity(int b, vec3 v);
    void setAngularVelocity(int b, vec3 w);
    void applyForce(int b, vec3 f, vec3 worldPoint, bool wake);
    void applyImpulse(int b, vec3 j, vec3 worldPoint, bool wake);
    void applyTorque(int b, vec3 t, bool wake);
    void applyAngularImpulse(int b, vec3 j, bool wake);
    void setAwake(int b, bool awake);
    void setEnabled(int b, bool enabled);
    void setType(int b, int type);
    void setShapeFriction(int s, float f);
    // Calls f(slot) for shape `s` and, if it names a baked compound, each of its children: a
    // per-shape setter acts on the whole compound.
    template <class F> void forShapeGroup(int s, F f)
    {
        f(s);
        for (size_t i = 0; i < shapes[s].members.size(); i++)
            f(shapes[s].members[i]);
    }
    void syncShapeMaterial(int s); // restitution + rolling resistance into the device shape arrays
    void setShapeFilter(int s, const avbd3d_Filter &filter);
    // Query layer: rebuilds a shape's proxy AABB from its body pose (on create, on setTransform,
    // and lazily for the bodies a step moved).
    void refreshShapeProxy(int shapeSlot);
    b3Transform shapeWorldTransform(int shapeSlot) const;
    // Refits the proxies of the bodies the steps since the last call moved. StepEnd only marks
    // them: refitting 100k proxies every step costs ~90 ms on the host, and a game that does not
    // query between steps never pays it.
    void ensureProxies();
    // The query tree, refit. Every query reaches the tree through here.
    b3DynamicTree &queryTree()
    {
        ensureProxies();
        return m_tree;
    }
    // Recomputes the Phase 2 BodyProps of a body from its BodyRec and its primary shape, then
    // marks it for re-upload. Only the fields the avbd3d API maps are written; the rest keep
    // their value (sleep threshold, tangent speed, friction mix, user material id).
    void syncProps(int b);
    // Recomputes the body's mass, centre of mass and principal inertia from all its shapes (each
    // with its own density) and applies them with applyMassFrame.
    void updateMass(int b);
    // Sets the mass, the centre of mass `center` (body-origin frame) and the principal moments
    // and axes. The body origin stays where it is: the device pose (centre of mass, principal
    // axes) and every shape's device-frame pose are rewritten around it.
    void applyMassFrame(int b, float mass, vec3 center, vec3 moment, quat principalRot);
    // Writes a shape's device-frame pose (relative to the body's centre of mass and principal
    // axes) from its body-origin pose, and its identity flag.
    void syncShapePose(int s);
    vec3 originPos(int b) const;
    quat originRot(int b) const;
    bool isAwake(int b) const;

    // ---- stepping
    void stepBegin(float dt);
    void stepEnd();
    bool inFlight() const { return m_inFlight; }
    // Ends a step in flight. Only the calls that use the device directly need it (render
    // writes, drag, stats); everything else reads the mirror below.
    void settle()
    {
        if (m_inFlight)
            stepEnd();
    }

    // ---- the read model: a lazy mirror of the newest finished step
    // The device publishes every step's poses as a VRAM snapshot (VkWorld::enableSnapshot);
    // the host pulls one only when something reads it, by a copy-queue DMA that never waits
    // for a step in flight. A body's desc pose, velocity and awake flag are refreshed from it
    // on first use after each step (syncBody), so reading 10 bodies of 100k costs 10 copies.
    // A body the host wrote since (a setter, a create, a property change) holds its written
    // values until the step that applies them has finished (m_bodyHold).
    void syncBody(int b)
    {
        if (b > 0 && b < (int)m_bodySerial.size() && m_bodySerial[b] < m_vk.snapshotSerial())
            pullBody(b);
    }
    // Read fast path: the finished snapshot when body b is movable, not held by a host write
    // and its mirror is behind the snapshot (syncBody would copy exactly these values). Writes
    // no mirror or proxy bookkeeping. Returns null otherwise (use the mirror).
    // Also checks the id from the packed hot records (no BodyRec touched); a destroyed body
    // clears its movable flag, a reused slot has a new generation.
    const VkWorld::SnapshotView *snapshotFor(avbd3d_BodyId id, bool *plain = nullptr)
    {
        const int b = id.index1;
        if (b <= 0 || b >= (int)m_hot.size() || b >= (int)m_bodySerial.size() || id.world0 != worldIndex)
            return nullptr;
        const BodyHot &h = m_hot[b];
        if (!h.movable || h.gen != id.generation || m_bodySerial[b] >= m_vk.snapshotSerial())
            return nullptr;
        const VkWorld::SnapshotView &v = view();
        if (v.serial == 0 || v.serial < m_bodyHold[b] || b >= v.count)
            return nullptr;
        if (plain)
            *plain = h.plain != 0;
        return &v;
    }
    // Every movable body (world-level reads: queries, sensors, joint creation).
    void syncBodies();
    void setBodyUser(int b, void *u)
    {
        bodies[b].user = u;
        if (b < (int)m_hot.size())
            m_hot[b].user = u;
    }
    const VkWorld::StepResult &stepResult() const { return m_result; }

    // ---- errors (sticky; see avbd3d_Error)
    avbd3d_Error error() const { return m_error; }
    const std::string &errorText() const { return m_errorText; }
    // Records the exception being handled (call from a catch block). The first error wins.
    void noteException();

    // ---- contact events (docs/SOLVER_3D_ROADMAP.md Phase 4b-1)
    // Sets a shape's event flags and keeps its device REPORT bit in step with them.
    void setShapeReport(int shapeSlot);
    // The id of contact slot c (its current generation).
    avbd3d_ContactId contactId(int c) const;
    bool contactValid(avbd3d_ContactId id) const;
    // Live contact slots touching a body (isBody) or a shape, and the public data of one slot.
    std::vector<int> contactsOf(int slot, bool isBody) const;
    avbd3d_ContactData contactData(int c) const;
    // Diffs the device report against the contact table; fills the event arrays.
    void collectContacts();
    std::vector<avbd3d_ContactBeginTouchEvent> contactBegin;
    std::vector<avbd3d_ContactEndTouchEvent> contactEnd;
    std::vector<avbd3d_ContactHitEvent> contactHits;
    bool contactTruncated = false; // the last step reported more contacts than the device buffer held
    float hitThreshold = 1.0f;     // m/s

    // ---- sensors and explosions (Phase 4b-2)
    // Diffs every sensor's query result against its overlap list and fills the sensor event
    // arrays. Runs at the end of StepEnd, after the moved bodies' proxies are refreshed.
    void collectSensors();
    // Applies an explosion: a dynamic-tree query, then an impulse on every shape in range.
    void explode(const avbd3d_ExplosionDef &def);
    std::vector<avbd3d_SensorBeginTouchEvent> sensorBegin;
    std::vector<avbd3d_SensorEndTouchEvent> sensorEnd;

    // ---- grab (avbd3d_interact.cpp)
    int grabGroup(int seed, vec3 center, float radius, bool followConnections, avbd3d_BodyId *out, int maxCount);
    int grabSurvivors(int seed, const avbd3d_BodyId *group, int count, uint8_t *keep);
    void grabDrag(const avbd3d_GrabHandle *handles, int count);
    void grabRelease();

    SolverParams params;
    avbd3d_Vec3 gravity{0, -10, 0};
    bool sleepEnabled = true;
    void *user = nullptr;

    // Contact mixing rules (avbd3d_MixingRule values), applied by every body's BodyProps.frictionMix:
    // bits 0..7 the friction rule (FrictionMix), bits 8..15 the restitution rule + 1, 0 = the larger.
    int frictionMixing = 0;    // avbd3d_mixGeometric
    int restitutionMixing = 3; // avbd3d_mixMax
    unsigned mixCode() const;
    void setMixing(int friction, int restitution);

    std::vector<BodyRec> bodies; // index = device slot; slot 0 is the graveyard
    std::vector<ShapeRec> shapes;
    std::vector<JointRec> joints;
    std::vector<avbd3d_BodyMoveEvent> events; // see moveEvents()
    std::vector<avbd3d_BodyMoveEvent> &moveEvents();
    SceneDesc desc;
    int worldIndex = 0; // index in the world table
    void *gpu = nullptr; // the avbd3d_Gpu the world was created on, null when it owns its device
    // The solver, for the render-data and stats extensions (avbd3d_vulkan.cpp).
    VkWorld &vk() { return m_vk; }
    // A body's render group changed: the per-shape group table is recomputed before the next
    // render write or step (syncRenderGroups).
    bool renderGroupsDirty = false;
    void syncRenderGroups();
    // Host-only numeric properties set by the world API (contact tuning, thresholds, flags).
    std::unordered_map<std::string, double> props;
    std::vector<unsigned char> awake;

private:
    friend struct JointBuild;
    int allocBody();
    int allocShape();
    void markBody(int slot);
    void markShape(int slot);
    void applyMass(int slot);
    void setPrimary(int bodySlot, int shapeSlot);
    void queue(int body, BodyWrite type, const void *data, size_t bytes);
    void flush();
    void rebuildLists();
    void appendRows(const SceneDesc &sd, int a, int b, const std::vector<unsigned long long> &ignore);
    // The rows a joint's def builds, with the bodies at their creation poses. jointCount is 0 when
    // the def does not build (a preset refused it).
    SceneDesc buildJointRows(const JointRec &rec);
    void rebuildJoint(int rec);
    std::vector<unsigned long long> ignoreOf(const SceneDesc &sd, int a, int b) const;

    VkWorld m_vk;
    b3DynamicTree m_tree{};
    // Bodies whose pose was pulled since their proxies were last refit (ensureProxies).
    std::vector<int> m_proxyDirty;
    std::vector<unsigned char> m_proxyFlag;
    // Alive, enabled, non-static bodies with a shape: the ones the device moves. Rebuilt with
    // the solver lists.
    std::vector<int> m_movable;
    // The read model (syncBody): per body, the snapshot serial its mirror holds and the first
    // serial it may be pulled from (a host write holds it until its step has finished).
    std::vector<uint64_t> m_bodySerial, m_bodyHold;
    // What the per-step loops (pullBody, moveEvents) need of a BodyRec, packed: a BodyRec is
    // ~200 bytes, and walking 100k of them cost ~4 ms a step in cache misses. Rebuilt with the
    // solver lists; a body changed since is held (holdBody) until the flush that rebuilds them.
    struct BodyHot
    {
        void *user = nullptr;
        uint16_t gen = 0;
        uint8_t movable = 0; // in m_movable
        uint8_t plain = 0;   // origin == centre of mass, no principal rotation: pose == snapshot pose
    };
    std::vector<BodyHot> m_hot;
    VkWorld::SnapshotView m_view; // the host copy of the newest snapshot (view())
    VkWorld::JointSnapshotView m_jview; // the joint multipliers of the same snapshot
    std::vector<avbd3d_JointEvent> m_jevents;
    uint64_t m_allSerial = 0;     // syncBodies() ran for this serial
    uint64_t m_eventsSerial = 0;  // the move events are of this serial
    bool m_wantSnapshot = false;  // something read the mirror since the last StepEnd
    const VkWorld::SnapshotView &view();
    void pullBody(int b);
    // A host write to body b: its mirror holds until the next step that applies it.
    void holdBody(int b)
    {
        if (b < (int)m_bodyHold.size())
            m_bodyHold[b] = m_vk.stepSerial() + 1;
    }
    std::vector<int> m_sensors; // live sensor shapes, ascending
    int m_vkBodies = 0, m_vkShapes = 0, m_vkJoints = 0, m_vkSprings = 0;
    bool m_springsDirty = false; // a destroyed spring's stiffness went to 0: re-upload the springs
    bool m_listsDirty = true;
    bool m_inFlight = false;
    VkWorld::StepResult m_result;
    avbd3d_Error m_error = avbd3d_errorNone;
    std::string m_errorText;
    // Copies VkWorld's sticky device error into m_error once it is set.
    void syncDeviceError();
    std::vector<int> m_freeBodies, m_pendingBodies, m_quarBodies;
    std::vector<int> m_freeShapes, m_pendingShapes, m_quarShapes;
    std::vector<int> m_breakRows;
    std::vector<Op> m_ops;
    std::vector<int> m_refreshBodies, m_refreshShapes, m_newJoints;
    std::vector<int> m_dirtyJoints;
    std::vector<int> m_scratchBodies, m_scratchShapes, m_opIdx;
    std::vector<ShapeMassInput> m_massScratch; // updateMass
    std::vector<unsigned char> m_opData;

    // Contact table: one record per touching shape pair that has a report shape. Indexed by the
    // slot in m_contacts; a freed slot bumps its generation so old ids go invalid.
    struct ContactRec
    {
        bool alive = false;
        bool events = false; // a begin/end event is due (either shape has enableContactEvents)
        uint32_t gen = 1;
        uint32_t stamp = 0;  // the step that last reported this pair
        int shapeA = 0, shapeB = 0, bodyA = 0, bodyB = 0;
        uint16_t genA = 0, genB = 0; // the shapes' generations when the pair began (end events name them)
        uint64_t key = 0;
        avbd3d_Manifold manifold{};
    };
    // End events for overlaps that a destroyed sensor or visitor left behind. Built at destroy
    // time (the ids are only known then) and moved into sensorEnd at the next StepEnd.
    std::vector<avbd3d_SensorEndTouchEvent> m_pendingSensorEnd;
    std::vector<ContactRec> m_contacts;
    std::vector<int> m_freeContacts;
    std::unordered_map<uint64_t, int> m_pairs;
    int m_reportShapes = 0; // shapes with `reporting` set; the device report runs while > 0
    uint32_t m_step = 0;
    int allocContact();
    void releaseContact(int c);
};

inline vec3 toV(avbd3d_Vec3 v) { return vec3{v.x, v.y, v.z}; }
inline avbd3d_Vec3 fromV(vec3 v) { return avbd3d_Vec3{v.x, v.y, v.z}; }
inline quat toQ(avbd3d_Quat q) { return quat{q.v.x, q.v.y, q.v.z, q.s}; }
inline avbd3d_Quat fromQ(quat q) { return avbd3d_Quat{avbd3d_Vec3{q.x, q.y, q.z}, q.w}; }
// R diag(moment) R^T: the body-frame inertia about the centre of mass from the principal moments
// and the principal rotation (principal frame -> body frame).
inline avbd3d_Matrix3 principalToMatrix(vec3 moment, quat rot)
{
    const vec3 c[3] = {rotate(rot, vec3{1, 0, 0}), rotate(rot, vec3{0, 1, 0}), rotate(rot, vec3{0, 0, 1})};
    const float k[3] = {moment.x, moment.y, moment.z};
    float a[3][3] = {};
    for (int i = 0; i < 3; i++)
    {
        const float v[3] = {c[i].x, c[i].y, c[i].z};
        for (int r = 0; r < 3; r++)
            for (int col = 0; col < 3; col++)
                a[r][col] += k[i] * v[r] * v[col];
    }
    avbd3d_Matrix3 m;
    m.cx = avbd3d_Vec3{a[0][0], a[1][0], a[2][0]};
    m.cy = avbd3d_Vec3{a[0][1], a[1][1], a[2][1]};
    m.cz = avbd3d_Vec3{a[0][2], a[1][2], a[2][2]};
    return m;
}

// The avbd3d error for an exception (call from a catch block); `text` gets its message.
avbd3d_Error errorOfCurrentException(std::string &text);

// World table.
World *lookupWorld(avbd3d_WorldId id);
// CreateWorld on a borrowed device (null: the world owns one); `gpu` is recorded on the world.
avbd3d_WorldId createWorldOn(const avbd3d_WorldDef *def, Device *device, void *gpu, avbd3d_Error &err);
// The world in table slot w0, or null.
World *byIndex(int w0);
// True when `p` is a live world in the table (the public Avbd3dWorld pointer).
bool validWorld(const void *p);
// Destroys every world created on `gpu` (avbd3d_DestroyGpu).
void destroyWorldsOn(void *gpu);
// Looks up a world for a world-level call. It does not wait for a step in flight, and does not
// sync the bodies either: a call that reads many of them syncs them itself (syncBodies; the
// queries do it through queryTree()), one that reads a few pulls those (syncBody).
World *settled(avbd3d_WorldId id);
// settled(), after ending a step in flight: for the calls that use the device directly.
World *settledDevice(avbd3d_WorldId id);
World *worldOfBody(avbd3d_BodyId id, int &slot);
World *worldOfShape(avbd3d_ShapeId id, int &slot);
World *worldOfJoint(avbd3d_JointId id, int &rec);
// The public id of a live world (its table slot and current generation).
avbd3d_WorldId worldIdOf(const World *w);

} // namespace avbd3d_impl
