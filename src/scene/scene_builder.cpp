/*
 * Copyright (c) 2026 Chris Giles
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies.
 * Chris Giles makes no representations about the suitability
 * of this software for any purpose.
 * It is provided "as is" without express or implied warranty.
 */

#include "scene_builder.h"
#include "shape_mass.h"
#include "shape_mass_b3.h"

#include "box3d/collision.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace
{

// Same packing as makePairKey in gpu_types.cuh, which the scene layer cannot include:
// low index in the high 32 bits, so a sort gives a canonical duplicate-free order.
unsigned long long pairKey(int a, int b)
{
    const int lo = a < b ? a : b;
    const int hi = a < b ? b : a;
    return ((unsigned long long)(unsigned int)lo << 32) | (unsigned int)hi;
}

} // namespace

void SceneBuilder::clear()
{
    m_pos.clear();
    m_vel.clear();
    m_velAng.clear();
    m_half.clear();
    m_moment.clear();
    m_rot.clear();
    m_mass.clear();
    m_friction.clear();
    m_shape.clear();
    m_tier.clear();
    m_asleep.clear();
    m_group.clear();
    m_jointA.clear();
    m_jointB.clear();
    m_jointRA.clear();
    m_jointRB.clear();
    m_jointKLin.clear();
    m_jointKAng.clear();
    m_jointFrac.clear();
    m_jointFracAng.clear();
    m_jointFrameA.clear();
    m_jointFrameB.clear();
    m_jointAxisMode.clear();
    m_jointGeneric.clear();
    m_jointAxisLo.clear();
    m_jointAxisHi.clear();
    m_springA.clear();
    m_springB.clear();
    m_springRA.clear();
    m_springRB.clear();
    m_springRest.clear();
    m_springK.clear();
    m_ignore.clear();
    m_shapeBody.clear();
    m_bodyPrimary.clear();
    m_shapeLocalPos.clear();
    m_shapeHalf.clear();
    m_shapeLocalRot.clear();
    m_shapeKind.clear();
    m_shapeFriction.clear();
    m_shapeCategory.clear();
    m_shapeMask.clear();
    m_shapeFlags.clear();
    m_shapeGroup.clear();
    m_props.clear();
    m_shapeGeom.clear();
    m_geomBlob.clear();
    m_geomHash.clear();
    m_bodyLocalCenter.clear();
    m_bodyPrincipalRot.clear();
}

BodyProps &SceneBuilder::propsOf(int body)
{
    if ((int)m_props.size() <= body)
        m_props.resize(std::max((size_t)body + 1, m_pos.size()), BodyProps{});
    return m_props[body];
}

BodyProps SceneBuilder::bodyProps(int body) const
{
    return body >= 0 && body < (int)m_props.size() ? m_props[body] : BodyProps{};
}

void SceneBuilder::setShapeFilter(int shape, uint64_t category, uint64_t mask, int group)
{
    if (shape < 0 || shape >= (int)m_shapeBody.size())
        return;
    m_shapeCategory[shape] = category;
    m_shapeMask[shape] = mask;
    m_shapeGroup[shape] = group;
}

void SceneBuilder::setBodyFilter(int body, uint64_t category, uint64_t mask, int group)
{
    for (int s = 0; s < (int)m_shapeBody.size(); s++)
        if (m_shapeBody[s] == body)
            setShapeFilter(s, category, mask, group);
}

#define AVBD_BODY_SETTER(name, params, stmt)                                                                           \
    void SceneBuilder::name params                                                                                     \
    {                                                                                                                  \
        if (body < 0 || body >= (int)m_pos.size())                                                                     \
            return;                                                                                                    \
        BodyProps &p = propsOf(body);                                                                                  \
        stmt;                                                                                                          \
    }

void SceneBuilder::setKinematic(int body, bool kinematic)
{
    if (body < 0 || body >= (int)m_pos.size())
        return;
    BodyProps &p = propsOf(body);
    if (kinematic)
    {
        p.flags |= BODY_KINEMATIC;
        m_mass[body] = 0.0f;
        m_moment[body] = vec3{0, 0, 0};
    }
    else
        p.flags &= ~(unsigned)BODY_KINEMATIC;
}
AVBD_BODY_SETTER(setBodyDamping, (int body, float linear, float angular), p.linearDamping = linear;
                 p.angularDamping = angular)
AVBD_BODY_SETTER(setGravityScale, (int body, float scale), p.gravityScale = scale)
AVBD_BODY_SETTER(setMaxLinearSpeed, (int body, float speed), p.maxLinearSpeed = speed)
AVBD_BODY_SETTER(setSleepThreshold, (int body, float speed), p.sleepThreshold = speed)
AVBD_BODY_SETTER(setEnableSleep, (int body, bool enable), if (enable) p.flags &= ~(unsigned)BODY_NO_SLEEP;
                 else p.flags |= BODY_NO_SLEEP)
AVBD_BODY_SETTER(setMotionLocks, (int body, unsigned lockFlags),
                 p.flags = (p.flags & ~(unsigned)BODY_LOCK_ANY) | (lockFlags & BODY_LOCK_ANY))
AVBD_BODY_SETTER(setTangentSpeed, (int body, float speed), p.tangentSpeed = speed)
AVBD_BODY_SETTER(setRollingResistance, (int body, float coefficient), p.rollingResistance = coefficient)
AVBD_BODY_SETTER(setFrictionMix, (int body, FrictionMix mix), p.frictionMix = (unsigned)mix)
AVBD_BODY_SETTER(setUserMaterialId, (int body, unsigned id), p.userMaterialId = id)
AVBD_BODY_SETTER(setRestitution, (int body, float restitution), p.restitution = restitution)
#undef AVBD_BODY_SETTER

int SceneBuilder::pushShape(int body, int type, vec3 localPos, quat localRot, vec3 half, float friction)
{
    const bool identity = localPos.x == 0.0f && localPos.y == 0.0f && localPos.z == 0.0f && localRot.x == 0.0f &&
                          localRot.y == 0.0f && localRot.z == 0.0f && localRot.w == 1.0f;
    m_shapeBody.push_back(body);
    m_shapeLocalPos.push_back(localPos);
    m_shapeLocalRot.push_back(localRot);
    m_shapeKind.push_back((unsigned char)type);
    m_shapeHalf.push_back(half);
    m_shapeFriction.push_back(friction);
    m_shapeCategory.push_back(1u);
    m_shapeMask.push_back(~0ull);
    m_shapeGroup.push_back(0);
    m_shapeFlags.push_back(identity ? AVBD_SHAPE_FLAG_IDENTITY : 0u);
    m_shapeGeom.push_back(-1);
    return (int)m_shapeBody.size() - 1;
}

int SceneBuilder::addShape(int body, int type, vec3 localPos, quat localRot, vec3 size, float friction)
{
    if (body < 0 || body >= (int)m_pos.size() || type < AVBD_SHAPE_BOX || type > AVBD_SHAPE_CAPSULE)
        return -1;
    vec3 half;
    if (type == AVBD_SHAPE_BOX)
        half = size * 0.5f;
    else if (type == AVBD_SHAPE_SPHERE)
        half = vec3{size.x, size.x, size.x};
    else
        half = vec3{size.x, size.x, size.x + (size.z > 0.0f ? size.z : 0.0f)};
    return pushShape(body, type, localPos, localRot, half, friction);
}

// FNV-1a over the blob, used only when the hull carries no hash (Box3D hashes every hull it
// builds; this covers a hand-assembled blob) so identical hulls still share one upload.
static unsigned long long blobHash(const unsigned char *bytes, size_t n)
{
    unsigned long long h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++)
        h = (h ^ bytes[i]) * 1099511628211ull;
    return h ? h : 1ull;
}

int SceneBuilder::internBlob(const unsigned char *src, size_t n, unsigned long long hash)
{
    const unsigned long long h = hash != 0 ? hash : blobHash(src, n);
    for (size_t i = 0; i < m_geomHash.size(); i++)
        if (m_geomHash[i] == h && m_geomBlob[i].size() == n)
            return (int)i;
    m_geomBlob.emplace_back(src, src + n);
    m_geomHash.push_back(h);
    return (int)m_geomBlob.size() - 1;
}

int SceneBuilder::internGeometry(const b3HullData *hull)
{
    if (hull == nullptr || hull->byteCount <= (int)sizeof(b3HullData) || hull->version != B3_HULL_VERSION)
        return -1;
    return internBlob(reinterpret_cast<const unsigned char *>(hull), (size_t)hull->byteCount,
                      (unsigned long long)hull->hash);
}

// Box3D mesh blob layout, pinned to the vendored headers. The GPU midphase reads these offsets
// directly from the geometry heap, so a Box3D upgrade that moves them must fail the build here.
static_assert(offsetof(b3MeshData, nodeOffset) == 56, "b3MeshData layout changed");
static_assert(offsetof(b3MeshData, vertexOffset) == 64, "b3MeshData layout changed");
static_assert(offsetof(b3MeshData, triangleOffset) == 72, "b3MeshData layout changed");
static_assert(offsetof(b3MeshData, flagsOffset) == 88, "b3MeshData layout changed");
static_assert(sizeof(b3MeshData) == 96, "b3MeshData layout changed");
static_assert(sizeof(b3MeshNode) == 32, "b3MeshNode layout changed");

// The mesh child manifold key packs shape and triangle indices into 21 bits each
// (see narrowphase_mesh.slang makeChildKey). Anything at or above this limit is refused.
static constexpr int kMeshKeyIndexLimit = (1 << 21) - 1;

int SceneBuilder::cookMesh(const vec3 *vertices, int vertexCount, const int *indices, int triangleCount, vec3 &half)
{
    if (vertices == nullptr || indices == nullptr || vertexCount < 3 || triangleCount < 1)
        return -1;
    if (triangleCount >= kMeshKeyIndexLimit)
        return -1;
    std::vector<b3Vec3> verts((size_t)vertexCount);
    for (int i = 0; i < vertexCount; i++)
        verts[(size_t)i] = b3Vec3{vertices[i].x, vertices[i].y, vertices[i].z};
    std::vector<int32_t> tris((size_t)triangleCount * 3);
    for (size_t i = 0; i < tris.size(); i++)
    {
        const int idx = indices[i];
        if (idx < 0 || idx >= vertexCount)
            return -1;
        tris[i] = (int32_t)idx;
    }
    b3MeshDef def = {};
    def.vertices = verts.data();
    def.vertexCount = vertexCount;
    def.indices = tris.data();
    def.triangleCount = triangleCount;
    def.identifyEdges = true; // per-triangle edge flags: internal edges between triangles
    def.weldVertices = false; // keep the caller's indices (a seam must share its vertices)
    b3MeshData *mesh = b3CreateMesh(&def, nullptr, 0);
    if (mesh == nullptr)
        return -1;
    int blob = -1;
    if (mesh->version == B3_MESH_VERSION && mesh->byteCount > (int)sizeof(b3MeshData))
    {
        const unsigned char *src = reinterpret_cast<const unsigned char *>(mesh);
        blob = internBlob(src, (size_t)mesh->byteCount, (unsigned long long)mesh->hash);
        const auto amax = [](float a, float b) { return std::max(std::fabs(a), std::fabs(b)); };
        half = vec3{amax(mesh->bounds.lowerBound.x, mesh->bounds.upperBound.x),
                    amax(mesh->bounds.lowerBound.y, mesh->bounds.upperBound.y),
                    amax(mesh->bounds.lowerBound.z, mesh->bounds.upperBound.z)};
    }
    b3DestroyMesh(mesh);
    return blob;
}

int SceneBuilder::addMeshBody(const vec3 *vertices, int vertexCount, const int *indices, int triangleCount,
                              float friction, vec3 position, quat rotation)
{
    if ((int)m_shape.size() >= kMeshKeyIndexLimit)
        return -1;
    vec3 half{0, 0, 0};
    const int blob = cookMesh(vertices, vertexCount, indices, triangleCount, half);
    if (blob < 0)
        return -1;
    // Static: density 0, no velocity. The mesh is the primary shape.
    const int body =
        addShapedBody(AVBD_SHAPE_MESH, half, 0.0f, friction, position, vec3{0, 0, 0}, vec3{0, 0, 0}, rotation);
    m_shapeGeom[m_bodyPrimary[body]] = blob;
    return body;
}

int SceneBuilder::addMeshShape(int body, const vec3 *vertices, int vertexCount, const int *indices, int triangleCount,
                               vec3 localPos, quat localRot, float friction)
{
    if (body < 0 || body >= (int)m_pos.size())
        return -1;
    if ((int)m_shape.size() >= kMeshKeyIndexLimit)
        return -1;
    vec3 half{0, 0, 0};
    const int blob = cookMesh(vertices, vertexCount, indices, triangleCount, half);
    if (blob < 0)
        return -1;
    // A dynamic body needs a closed mesh: its mass is the enclosed volume.
    if (m_mass[body] > 0.0f && !meshIsClosed(reinterpret_cast<const b3MeshData *>(m_geomBlob[blob].data())))
        return -1;
    const int s = pushShape(body, AVBD_SHAPE_MESH, localPos, localRot, half, friction);
    m_shapeGeom[s] = blob;
    return s;
}

static vec3 hullHalf(const b3HullData *hull)
{
    auto amax = [](float a, float b) { return std::max(std::fabs(a), std::fabs(b)); };
    return vec3{amax(hull->aabb.lowerBound.x, hull->aabb.upperBound.x),
                amax(hull->aabb.lowerBound.y, hull->aabb.upperBound.y),
                amax(hull->aabb.lowerBound.z, hull->aabb.upperBound.z)};
}

int SceneBuilder::addHullShape(int body, const b3HullData *hull, vec3 localPos, quat localRot, float friction)
{
    if (body < 0 || body >= (int)m_pos.size())
        return -1;
    const int blob = internGeometry(hull);
    if (blob < 0)
        return -1;
    const int s = pushShape(body, AVBD_SHAPE_HULL, localPos, localRot, hullHalf(hull), friction);
    m_shapeGeom[s] = blob;
    return s;
}

int SceneBuilder::addHullBody(const b3HullData *hull, float density, float friction, vec3 position, vec3 velocity,
                              vec3 angularVelocity, quat rotation)
{
    const int blob = internGeometry(hull);
    if (blob < 0)
        return -1;
    // Created static; the mass pass below makes it dynamic.
    const int body =
        addShapedBody(AVBD_SHAPE_HULL, hullHalf(hull), 0.0f, friction, position, velocity, angularVelocity, rotation);
    m_shapeGeom[m_bodyPrimary[body]] = blob;
    if (density > 0.0f)
        computeMassFromShapes(body, density);
    return body;
}

bool SceneBuilder::computeMassFromShapes(int body, float density)
{
    if (body < 0 || body >= (int)m_pos.size())
        return false;
    if (!(density > 0.0f))
    {
        m_mass[body] = 0.0f;
        m_moment[body] = vec3{0, 0, 0};
        return true;
    }
    std::vector<int> mine;
    std::vector<ShapeMassInput> in;
    for (int s = 0; s < (int)m_shapeBody.size(); s++)
    {
        if (m_shapeBody[s] != body)
            continue;
        ShapeMassInput si;
        si.kind = m_shapeKind[s];
        si.half = m_shapeHalf[s];
        si.localPos = m_shapeLocalPos[s];
        si.localRot = m_shapeLocalRot[s];
        if (m_shapeGeom[s] >= 0)
        {
            if (si.kind == AVBD_SHAPE_MESH)
                si.mesh = reinterpret_cast<const b3MeshData *>(m_geomBlob[m_shapeGeom[s]].data());
            else
                si.hull = reinterpret_cast<const b3HullData *>(m_geomBlob[m_shapeGeom[s]].data());
        }
        mine.push_back(s);
        in.push_back(si);
    }
    CompoundMass cm;
    if (!computeCompoundMass(in.data(), (int)in.size(), density, cm))
        return false;

    // Solver pose = centre of mass in the principal frame. c and Rp are relative to the
    // CURRENT frame, so a second call composes with the first.
    const quat rp = cm.principalRot;
    const quat rpInv = conjugate(rp);
    const vec3 cWorld = rotate(m_rot[body], cm.center);
    m_pos[body] = m_pos[body] + cWorld;
    m_vel[body] = m_vel[body] + cross(m_velAng[body], cWorld);
    m_rot[body] = normalize(m_rot[body] * rp);
    for (size_t k = 0; k < mine.size(); k++)
    {
        const int s = mine[k];
        const vec3 p = rotate(rpInv, m_shapeLocalPos[s] - cm.center);
        const quat q = rpInv * m_shapeLocalRot[s];
        m_shapeLocalPos[s] = p;
        m_shapeLocalRot[s] = q;
        const bool identity =
            p.x == 0.0f && p.y == 0.0f && p.z == 0.0f && q.x == 0.0f && q.y == 0.0f && q.z == 0.0f && q.w == 1.0f;
        m_shapeFlags[s] = (m_shapeFlags[s] & ~AVBD_SHAPE_FLAG_IDENTITY) | (identity ? AVBD_SHAPE_FLAG_IDENTITY : 0u);
    }
    m_bodyLocalCenter[body] = m_bodyLocalCenter[body] + rotate(m_bodyPrincipalRot[body], cm.center);
    m_bodyPrincipalRot[body] = normalize(m_bodyPrincipalRot[body] * rp);
    m_mass[body] = cm.mass;
    m_moment[body] = cm.moment;
    return true;
}

// The one place a body is appended. addBody/addSphere/addCapsule differ only in the
// shape tag and in how the caller spells the tight OBB.
int SceneBuilder::addShapedBody(int shape, vec3 half, float density, float friction, vec3 position, vec3 velocity,
                                vec3 angularVelocity, quat rotation)
{
    float mass;
    vec3 moment;
    shapeMassProperties(shape, half, density, mass, moment);

    m_pos.push_back(position);
    m_rot.push_back(rotation);
    m_vel.push_back(velocity);
    m_velAng.push_back(angularVelocity);
    m_half.push_back(half);
    m_shape.push_back((unsigned char)shape);
    m_moment.push_back(moment);
    m_mass.push_back(mass);
    m_friction.push_back(friction);
    m_tier.push_back((unsigned char)BodyTier::Interactive);
    m_asleep.push_back(0);
    m_group.push_back(-1);
    m_bodyLocalCenter.push_back(vec3{0, 0, 0});
    m_bodyPrincipalRot.push_back(quat{0, 0, 0, 1});
    const int body = (int)m_pos.size() - 1;
    m_bodyPrimary.push_back(pushShape(body, shape, vec3{0, 0, 0}, quat{0, 0, 0, 1}, half, friction));
    return body;
}

int SceneBuilder::addBody(vec3 size, float density, float friction, vec3 position, vec3 velocity, vec3 angularVelocity,
                          quat rotation)
{
    return addShapedBody(AVBD_SHAPE_BOX, size * 0.5f, density, friction, position, velocity, angularVelocity, rotation);
}

int SceneBuilder::addSphere(float radius, float density, float friction, vec3 position, vec3 velocity,
                            vec3 angularVelocity, quat rotation)
{
    return addShapedBody(AVBD_SHAPE_SPHERE, vec3{radius, radius, radius}, density, friction, position, velocity,
                         angularVelocity, rotation);
}

int SceneBuilder::addCapsule(float radius, float halfSegment, float density, float friction, vec3 position,
                             vec3 velocity, vec3 angularVelocity, quat rotation)
{
    const float hs = halfSegment > 0.0f ? halfSegment : 0.0f;
    return addShapedBody(AVBD_SHAPE_CAPSULE, vec3{radius, radius, radius + hs}, density, friction, position, velocity,
                         angularVelocity, rotation);
}

int SceneBuilder::addJoint(int bodyA, int bodyB, vec3 rA, vec3 rB, float stiffnessLin, float stiffnessAng,
                           float fracture)
{
    m_jointA.push_back(bodyA);
    m_jointB.push_back(bodyB);
    m_jointRA.push_back(rA);
    m_jointRB.push_back(rB);
    m_jointKLin.push_back(stiffnessLin);
    m_jointKAng.push_back(stiffnessAng);
    m_jointFrac.push_back(fracture);
    m_jointFracAng.push_back(-1.0f); // legacy angular test; setJointFractureAng sets a real capacity

    // Weld defaults for the 6-DOF side arrays. generic == 0 keeps this joint on the
    // original code path; the rest is padding so the arrays stay parallel.
    m_jointGeneric.push_back(0);
    m_jointFrameA.push_back(quat{0, 0, 0, 1});
    m_jointFrameB.push_back(quat{0, 0, 0, 1});
    m_jointAxisMode.insert(m_jointAxisMode.end(), 6, (unsigned char)0);
    m_jointAxisLo.insert(m_jointAxisLo.end(), 6, 0.0f);
    m_jointAxisHi.insert(m_jointAxisHi.end(), 6, 0.0f);

    // Constrained bodies do not collide, matching Rigid::constrainedTo.
    if (bodyA >= 0 && bodyB >= 0)
        m_ignore.push_back(pairKey(bodyA, bodyB));

    return (int)m_jointA.size() - 1;
}

int SceneBuilder::addSpring(int bodyA, int bodyB, vec3 rA, vec3 rB, float stiffness, float rest)
{
    m_springA.push_back(bodyA);
    m_springB.push_back(bodyB);
    m_springRA.push_back(rA);
    m_springRB.push_back(rB);
    m_springK.push_back(stiffness);
    m_springRest.push_back(rest); // resolved at finalize, where the poses are known

    if (bodyA >= 0 && bodyB >= 0)
        m_ignore.push_back(pairKey(bodyA, bodyB));

    return (int)m_springA.size() - 1;
}

vec3 SceneBuilder::bodyPos(int i) const
{
    return i >= 0 && i < (int)m_pos.size() ? m_pos[i] : vec3{0, 0, 0};
}

quat SceneBuilder::bodyRot(int i) const
{
    return i >= 0 && i < (int)m_rot.size() ? m_rot[i] : quat{0, 0, 0, 1};
}

int SceneBuilder::addJointWorld(int bodyA, int bodyB, vec3 worldAnchor, float stiffnessLin, float stiffnessAng,
                                float fracture)
{
    const vec3 pA = bodyPos(bodyA), pB = bodyPos(bodyB);
    const quat qA = bodyRot(bodyA), qB = bodyRot(bodyB);
    const vec3 rA = (bodyA >= 0) ? rotate(conjugate(qA), worldAnchor - pA) : worldAnchor;
    const vec3 rB = (bodyB >= 0) ? rotate(conjugate(qB), worldAnchor - pB) : worldAnchor;
    return addJoint(bodyA, bodyB, rA, rB, stiffnessLin, stiffnessAng, fracture);
}

int SceneBuilder::addJoint6(int bodyA, int bodyB, const JointDesc &desc)
{
    // Reuse addJoint for the shared half of the record -- bodies, anchors, fracture and
    // the collision-ignore pair -- then overwrite the 6-DOF side it padded as a weld.
    // The scalar stiffnesses it stages are unread on the generic path: stiffness is per
    // axis there, and lives in JointAxis::hi.
    const int i = addJoint(bodyA, bodyB, desc.rA, desc.rB, INFINITY, INFINITY, desc.fracture);

    m_jointGeneric[i] = 1;
    m_jointFrameA[i] = desc.frameA;
    m_jointFrameB[i] = desc.frameB;
    for (int k = 0; k < 6; k++)
    {
        // The damping ratio rides in the mode byte's upper 5 bits (springs only; forces.slang axisZeta).
        const int md = desc.axis[k].mode;
        int zc = (md == JointAxisSpring || md == JointAxisSpringF) ? (int)(desc.axis[k].damping * 16.0f + 0.5f) : 0;
        zc = zc < 0 ? 0 : (zc > 31 ? 31 : zc);
        m_jointAxisMode[i * 6 + k] = (unsigned char)(md | (zc << 3));
        m_jointAxisLo[i * 6 + k] = desc.axis[k].lo;
        m_jointAxisHi[i * 6 + k] = desc.axis[k].hi;
    }
    return i;
}

int SceneBuilder::addJoint6World(int bodyA, int bodyB, vec3 worldAnchor, quat worldFrame, const JointAxis axes[6],
                                 float fracture)
{
    const vec3 pA = bodyPos(bodyA), pB = bodyPos(bodyB);
    const quat qA = bodyRot(bodyA), qB = bodyRot(bodyB);

    JointDesc d;
    d.rA = (bodyA >= 0) ? rotate(conjugate(qA), worldAnchor - pA) : worldAnchor;
    d.rB = rotate(conjugate(qB), worldAnchor - pB);

    // Both frames are the same world frame seen from each body, so at the authored pose
    // the relative rotation is identity and every angular axis reads zero. A hinge
    // authored at rest therefore starts at angle 0, which is what "limit -30..30" has
    // to mean for the numbers to be usable.
    d.frameA = (bodyA >= 0) ? conjugate(qA) * worldFrame : worldFrame;
    d.frameB = conjugate(qB) * worldFrame;
    d.fracture = fracture;
    for (int k = 0; k < 6; k++)
        d.axis[k] = axes[k];

    return addJoint6(bodyA, bodyB, d);
}

int SceneBuilder::addSpringWorld(int bodyA, int bodyB, vec3 worldAnchorA, vec3 worldAnchorB, float stiffness,
                                 float rest)
{
    const vec3 pA = bodyPos(bodyA), pB = bodyPos(bodyB);
    const quat qA = bodyRot(bodyA), qB = bodyRot(bodyB);
    const vec3 rA = (bodyA >= 0) ? rotate(conjugate(qA), worldAnchorA - pA) : worldAnchorA;
    const vec3 rB = (bodyB >= 0) ? rotate(conjugate(qB), worldAnchorB - pB) : worldAnchorB;
    return addSpring(bodyA, bodyB, rA, rB, stiffness, rest);
}

void SceneBuilder::ignoreCollision(int bodyA, int bodyB)
{
    if (bodyA >= 0 && bodyB >= 0 && bodyA != bodyB)
        m_ignore.push_back(pairKey(bodyA, bodyB));
}

void SceneBuilder::allowCollision(int bodyA, int bodyB)
{
    if (bodyA < 0 || bodyB < 0)
        return;
    const auto it = std::find(m_ignore.begin(), m_ignore.end(), pairKey(bodyA, bodyB));
    if (it != m_ignore.end())
        m_ignore.erase(it);
}

float SceneBuilder::jointTorqueArm(int bodyA, int bodyB) const
{
    const vec3 sizeA = bodyA >= 0 ? m_half[bodyA] * 2.0f : vec3{0, 0, 0};
    const vec3 sizeB = bodyB >= 0 ? m_half[bodyB] * 2.0f : vec3{0, 0, 0};
    return lengthSq(sizeA + sizeB);
}

void SceneBuilder::setBodyTier(int i, BodyTier tier)
{
    if (i >= 0 && i < (int)m_tier.size())
        m_tier[i] = (unsigned char)tier;
}

void SceneBuilder::setBodyRenderGroup(int i, int group)
{
    if (i >= 0 && i < (int)m_group.size())
        m_group[i] = group < 0 ? -1 : group;
}

void SceneBuilder::setAwake(int i, bool awake)
{
    if (i >= 0 && i < (int)m_asleep.size())
        m_asleep[i] = awake ? 0 : 1;
}

void SceneBuilder::rotateZUpToYUp()
{
    const auto mapV = [](vec3 v) { return vec3{v.x, v.z, -v.y}; };
    const quat r = kZUpToYUp; // the same map as mapV on vectors
    for (size_t i = 0; i < m_pos.size(); i++)
    {
        m_pos[i] = mapV(m_pos[i]);
        m_vel[i] = mapV(m_vel[i]);
        m_velAng[i] = mapV(m_velAng[i]);
        m_rot[i] = normalize(r * m_rot[i]);
    }
}

SceneDesc SceneBuilder::finalize() const
{
    SceneDesc d;
    d.params = params;

    const int n = (int)m_pos.size();
    d.count = n;
    d.pos = m_pos;
    d.velLin = m_vel;
    d.velAng = m_velAng;
    d.halfExtents = m_half;
    d.moment = m_moment;
    d.rot = m_rot;
    d.mass = m_mass;
    d.friction = m_friction;
    d.shapeType = m_shape;
    d.renderGroup = m_group;
    d.tier = m_tier;
    d.asleep = m_asleep;
    d.bodyProps.assign(n, BodyProps{});
    for (int i = 0; i < n && i < (int)m_props.size(); i++)
        d.bodyProps[i] = m_props[i];

    // Partition by mass. Static bodies are excluded from the uniform grid: the ground
    // plane alone spans thousands of cells, and binning it would either force an absurd
    // cell size or put it in every bucket.
    float maxHalfLen = 0.0f;
    std::vector<unsigned char> isDyn(n, 0);
    for (int i = 0; i < n; i++)
    {
        if (m_mass[i] > 0)
        {
            d.dynamicIdx.push_back(i);
            isDyn[i] = 1;
        }
        else
        {
            d.staticIdx.push_back(i);
        }
    }
    // The cell is sized from every shape of a dynamic body (an offset shape's AABB is still
    // centred on the shape, so its extent does not depend on the offset).
    for (size_t s = 0; s < m_shapeBody.size(); s++)
        if (isDyn[m_shapeBody[s]])
            maxHalfLen = std::max(maxHalfLen, length(m_shapeHalf[s]));

    d.shapeCount = (int)m_shapeBody.size();
    d.shapeBody = m_shapeBody;
    d.shapeLocalPos = m_shapeLocalPos;
    d.shapeLocalRot = m_shapeLocalRot;
    d.shapeKind = m_shapeKind;
    d.shapeHalf = m_shapeHalf;
    d.shapeFriction = m_shapeFriction;
    d.shapeCategory = m_shapeCategory;
    d.shapeMask = m_shapeMask;
    d.shapeGroup = m_shapeGroup;
    d.shapeFlags = m_shapeFlags;
    d.bodyPrimaryShape = m_bodyPrimary;
    d.shapeGeom = m_shapeGeom;
    d.geomBlob = m_geomBlob;
    d.geomHash = m_geomHash;
    d.bodyLocalCenter = m_bodyLocalCenter;
    d.bodyPrincipalRot = m_bodyPrincipalRot;
    // The 27-cell neighbourhood only finds every overlap while no body's AABB is wider
    // than a cell. A rotated box's AABB half-extent is at most the length of its
    // half-extents vector, so twice the largest of those is a safe cell size.
    d.cellSize = std::max(2.0f * maxHalfLen + 2.0f * COLLISION_MARGIN, 1e-3f);

    d.jointCount = (int)m_jointA.size();
    d.jointA = m_jointA;
    d.jointB = m_jointB;
    d.jointRA = m_jointRA;
    d.jointRB = m_jointRB;
    d.jointKLin = m_jointKLin;
    d.jointKAng = m_jointKAng;
    d.jointFrac = m_jointFrac;
    d.jointFracAng = m_jointFracAng;
    d.jointFrameA = m_jointFrameA;
    d.jointFrameB = m_jointFrameB;
    d.jointAxisMode = m_jointAxisMode;
    d.jointAxisLo = m_jointAxisLo;
    d.jointAxisHi = m_jointAxisHi;
    d.jointGeneric = m_jointGeneric;

    d.jointArm.assign(d.jointCount, 0.0f);
    d.jointRestAng.assign(d.jointCount, quat{0, 0, 0, 1});
    for (int i = 0; i < d.jointCount; i++)
    {
        // torqueArm, from the Joint constructor: sizes are full widths there, and the
        // staged half extents are half of those.
        const vec3 sizeA = m_jointA[i] >= 0 ? m_half[m_jointA[i]] * 2.0f : vec3{0, 0, 0};
        const vec3 sizeB = m_half[m_jointB[i]] * 2.0f;
        d.jointArm[i] = lengthSq(sizeA + sizeB);

        // restAng, from the Joint constructor: the relative orientation the joint is
        // authored at, so the angular constraint holds that difference rather than
        // driving both bodies to a common orientation.
        const quat qA = m_jointA[i] >= 0 ? m_rot[m_jointA[i]] : quat{0, 0, 0, 1};
        d.jointRestAng[i] = inverse(qA) * m_rot[m_jointB[i]];
    }

    d.springCount = (int)m_springA.size();
    d.springA = m_springA;
    d.springB = m_springB;
    d.springRA = m_springRA;
    d.springRB = m_springRB;
    d.springK = m_springK;
    d.springRest = m_springRest;
    for (int i = 0; i < d.springCount; i++)
    {
        // Resolve rest < 0 to the current anchor separation, as the Spring constructor does.
        if (d.springRest[i] >= 0.0f)
            continue;
        const int a = m_springA[i], b = m_springB[i];
        const vec3 pA = a >= 0 ? transform(m_pos[a], m_rot[a], m_springRA[i]) : m_springRA[i];
        const vec3 pB = transform(m_pos[b], m_rot[b], m_springRB[i]);
        d.springRest[i] = length(pA - pB);
    }

    // Sorted, deduplicated suppression list, searched per candidate pair by the
    // broadphase. Sorting is what makes that a binary search rather than a scan.
    d.ignorePairs = m_ignore;
    std::sort(d.ignorePairs.begin(), d.ignorePairs.end());
    d.ignorePairs.erase(std::unique(d.ignorePairs.begin(), d.ignorePairs.end()), d.ignorePairs.end());
    return d;
}

int SceneBuilder::spawnBody(vec3 size, float density, float friction, vec3 position, vec3 velocity,
                            vec3 angularVelocity, quat rotation)
{
    return addBody(size, density, friction, position, velocity, angularVelocity, rotation);
}

int SceneBuilder::spawnGroup(const SpawnBody *bodies, int nb, const SpawnJoint *joints, int nj)
{
    if (!bodies || nb <= 0)
        return -1;
    const int base = bodyCount();
    for (int i = 0; i < nb; i++)
    {
        const SpawnBody &b = bodies[i];
        addBody(b.size, b.density, b.friction, b.position, b.velocity, b.angularVelocity, b.rotation);
    }
    const int total = bodyCount();
    for (int i = 0; i < nj && joints; i++)
    {
        const SpawnJoint &j = joints[i];
        if (j.bodyB < 0 || j.bodyB >= total || j.bodyA >= total)
            continue;
        addJoint(j.bodyA, j.bodyB, j.rA, j.rB, j.stiffnessLin, j.stiffnessAng, j.fracture);
    }
    return base;
}

bool SceneBuilder::findShadowPlane(vec3 &planePoint, vec3 &planeNormal) const
{
    const vec3 up = {0, 1, 0};
    float bestScore = 0.0f;
    bool found = false;

    for (size_t i = 0; i < m_mass.size(); ++i)
    {
        if (m_mass[i] > 0.0f)
            continue;

        const vec3 pos = m_pos[i];
        const vec3 half = m_half[i];
        const vec3 axes[3] = {rotate(m_rot[i], vec3{1, 0, 0}), rotate(m_rot[i], vec3{0, 1, 0}),
                              rotate(m_rot[i], vec3{0, 0, 1})};
        const float h[3] = {half.x, half.y, half.z};

        for (int axis = 0; axis < 3; ++axis)
        {
            const float area = 4.0f * h[(axis + 1) % 3] * h[(axis + 2) % 3];
            if (area <= 0.0f)
                continue;
            for (int s = 0; s < 2; ++s)
            {
                const vec3 n = axes[axis] * (s == 0 ? -1.0f : 1.0f);
                const float upness = dot(n, up);
                if (upness <= 0.15f)
                    continue;
                const float score = area * upness;
                if (!found || score > bestScore)
                {
                    found = true;
                    bestScore = score;
                    planeNormal = n;
                    planePoint = pos + n * h[axis];
                }
            }
        }
    }

    if (!found)
    {
        planePoint = vec3{0, 0, 0};
        planeNormal = vec3{0, 1, 0};
    }
    return found;
}
