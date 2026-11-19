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

#pragma once

// SceneBuilder: the authoring API for scenes. No solver sits behind it -- a scene is described here and
// handed to a backend as a SceneDesc via finalize().

#include "scene_desc.h"

struct b3HullData; // Box3D hull blob (external/box3d); only used through pointers here

// The rotation that takes z-up content into the solver's y-up frame: -90 degrees about +x, which
// maps (x, y, z) to (x, z, -y).
static const quat kZUpToYUp = quat{-0.70710678118654752f, 0.0f, 0.0f, 0.70710678118654752f};

class SceneBuilder
{
public:
    SceneBuilder() = default;

    void clear();

    // Re-expresses everything authored so far from a z-up frame in the solver's y-up frame:
    // (x, y, z) -> (x, z, -y), a proper rotation. Positions, orientations and velocities move;
    // body-local data (shapes, joint frames, anchors, spring data) is frame-independent, and
    // params.gravity is left alone. Lets a z-up scene source (the catalogue) run in a y-up world.
    void rotateZUpToYUp();

    // Bodies. Return the body index. Size is full width for boxes; sphere `radius`;
    // capsule `radius` + `halfSegment` about local +z. Density 0 makes a static body.
    int addBody(vec3 size, float density, float friction, vec3 position, vec3 velocity = vec3{0, 0, 0},
                vec3 angularVelocity = vec3{0, 0, 0}, quat rotation = quat{0, 0, 0, 1});
    int addSphere(float radius, float density, float friction, vec3 position, vec3 velocity = vec3{0, 0, 0},
                  vec3 angularVelocity = vec3{0, 0, 0}, quat rotation = quat{0, 0, 0, 1});
    int addCapsule(float radius, float halfSegment, float density, float friction, vec3 position,
                   vec3 velocity = vec3{0, 0, 0}, vec3 angularVelocity = vec3{0, 0, 0},
                   quat rotation = quat{0, 0, 0, 1});

    // An extra collision shape on an existing body. `type` is AVBD_SHAPE_BOX, _SPHERE or
    // _CAPSULE and `size` reads as the single-shape calls read theirs: a box takes its full
    // extents (x, y, z); a sphere takes size.x as the radius; a capsule takes size.x as the
    // radius and size.z as the half segment length along the shape's local +z.
    // localPos/localRot place the shape in the BODY frame. Returns the shape index (shapes
    // number in authoring order across all bodies; every body's primary shape already
    // exists), or -1 for a bad body or type.
    //
    // An added shape changes only what collides, renders and is hit by queries. It does NOT
    // change the body's mass, inertia, centre of mass or sleep radius: those stay those of
    // the body's primary shape (addBody/addSphere/addCapsule), so author the density to
    // match what the compound should weigh.
    int addShape(int body, int type, vec3 localPos, quat localRot, vec3 size, float friction);

    // Convex hull shapes. The hull is a Box3D b3HullData blob (b3CreateHull,
    // b3MakeBoxHull().base, ...); its bytes are copied into the scene, deduplicated by hash,
    // and uploaded to the device geometry heap. A hull has no collider yet: the narrowphase
    // emits no contacts for it and the renderer does not draw it.
    //   addHullBody: a new body whose primary shape is the hull (mass from the hull at
    //                `density` via computeMassFromShapes; 0 = static).
    //   addHullShape: an extra hull on an existing body, placed in the body frame; returns the
    //                shape index or -1 (bad body or null/invalid hull).
    // The caller keeps ownership of `hull`.
    int addHullBody(const b3HullData *hull, float density, float friction, vec3 position, vec3 velocity = vec3{0, 0, 0},
                    vec3 angularVelocity = vec3{0, 0, 0}, quat rotation = quat{0, 0, 0, 1});
    int addHullShape(int body, const b3HullData *hull, vec3 localPos, quat localRot, float friction);

    // Static triangle mesh shapes. The triangles are cooked by Box3D's b3CreateMesh
    // (identifyEdges on), the resulting b3MeshData blob is copied into the scene, deduplicated by
    // hash and uploaded to the geometry heap like a hull. The narrowphase collides a mesh with
    // convex shapes only; mesh against mesh is not collided.
    //   vertices: vertexCount points; indices: 3 per triangle, counter-clockwise seen from the
    //             side the convex shapes sit on (the normal points toward them).
    //   addMeshBody: a new static body (density 0) whose primary shape is the mesh.
    //   addMeshShape: an extra mesh on an existing STATIC body, in the body frame.
    // Both return -1 on bad input: empty or out-of-range arrays, a failed cook, or a dynamic body
    // (addMeshShape). The caller keeps its arrays.
    int addMeshBody(const vec3 *vertices, int vertexCount, const int *indices, int triangleCount, float friction,
                    vec3 position, quat rotation = quat{0, 0, 0, 1});
    int addMeshShape(int body, const vec3 *vertices, int vertexCount, const int *indices, int triangleCount,
                     vec3 localPos, quat localRot, float friction);

    // Mass from shapes: replaces the body's mass and moment with the sum over ALL
    // its shapes at `density`, using Box3D's b3Compute*Mass, then moves the body into its
    // principal frame. The solver pose becomes the centre of mass in the principal frame
    // (pos, rot, linear velocity are converted), every shape's local transform is re-expressed
    // there, and SceneDesc::bodyLocalCenter / bodyPrincipalRot record the way back to the
    // authored origin frame. Call it AFTER the body's shapes are added and BEFORE joints or
    // springs reference the body: their anchors are in the solver (principal) frame. A body
    // that never calls this is untouched. density <= 0 makes the body static (mass 0).
    // Returns false for a bad body or a body with no massive shape.
    bool computeMassFromShapes(int body, float density);

    // Joints and springs. bodyA < 0 anchors to the world. Return the index.
    int addJoint(int bodyA, int bodyB, vec3 rA, vec3 rB, float stiffnessLin = INFINITY, float stiffnessAng = 0.0f,
                 float fracture = INFINITY);
    int addJointWorld(int bodyA, int bodyB, vec3 worldAnchor, float stiffnessLin = INFINITY, float stiffnessAng = 0.0f,
                      float fracture = INFINITY);
    int addJoint6(int bodyA, int bodyB, const JointDesc &desc);
    int addJoint6World(int bodyA, int bodyB, vec3 worldAnchor, quat worldFrame, const JointAxis axes[6],
                       float fracture = INFINITY);
    int addSpring(int bodyA, int bodyB, vec3 rA, vec3 rB, float stiffness, float rest = -1.0f);
    int addSpringWorld(int bodyA, int bodyB, vec3 worldAnchorA, vec3 worldAnchorB, float stiffness, float rest = -1.0f);
    void ignoreCollision(int bodyA, int bodyB);
    // Undo the ignore pair addJoint staged (Box3D collideConnected = true). A joint is the
    // usual reason a pair is ignored, so this removes one occurrence; the other joints between
    // the same two bodies, if any, keep theirs.
    void allowCollision(int bodyA, int bodyB);

    // Phase 3 joint kinds (forces.slang JOINT_GENERIC_*): 1 = 6-DOF with rotation-vector angles
    // (addJoint6's default), 2 = distance (the three linear rows act along the anchor-to-anchor
    // direction), 3 = 6-DOF with swing-twist angles (cone limit, axis alignment).
    void setJointGeneric(int joint, int kind) { m_jointGeneric[joint] = (unsigned char)kind; }
    // The bound of a SPRING_F axis, in solver force units (angular: N*m / torque arm). A distance
    // joint reads [-ang, +lin]. Both are otherwise unread on the generic path.
    void setJointForceBounds(int joint, float lin, float ang)
    {
        m_jointKLin[joint] = lin;
        m_jointKAng[joint] = ang;
    }
    // Torque arm of a joint exactly as finalize() computes it (squared diagonal of the two
    // bodies' full sizes), so a preset can convert N*m to the solver's force units.
    float jointTorqueArm(int bodyA, int bodyB) const;
    // Authoring-side body properties, for presets that turn a hertz into a stiffness.
    float bodyMass(int i) const { return i >= 0 && i < (int)m_mass.size() ? m_mass[i] : 0.0f; }
    vec3 bodyMoment(int i) const { return i >= 0 && i < (int)m_moment.size() ? m_moment[i] : vec3{0, 0, 0}; }

    // Bending capacity (N*m) of an authored joint; see SceneDesc::jointFracAng. Needs a finite
    // `fracture` on the joint to take effect (an infinite one skips the whole test).
    void setJointFractureAng(int joint, float newtonMetres) { m_jointFracAng[joint] = newtonMetres; }

    // Per-body attributes.
    void setBodyTier(int bodyIdx, BodyTier tier);
    void setBodyRenderGroup(int bodyIdx, int group);
    void setAwake(int bodyIdx, bool awake); // initial sleep state only

    // Collision filter. A pair collides when each shape's category intersects the other's mask;
    // a shared nonzero group overrides that (positive always collides, negative never).
    // Defaults: category 1, mask all 64 bits, group 0. The body form sets every shape the body
    // has so far (shapes added later start at the defaults).
    void setShapeFilter(int shape, uint64_t category, uint64_t mask, int group);
    void setBodyFilter(int body, uint64_t category, uint64_t mask, int group);

    // Per-body settings (BodyProps). Defaults are bit-identical to a body never touched.
    // A kinematic body takes mass 0 (so it is never solved) and moves by its own velocity
    // (VkWorld::setLinearVelocity / setTargetTransform), pushing and carrying dynamic bodies.
    void setKinematic(int body, bool kinematic);
    void setBodyDamping(int body, float linear, float angular);
    void setGravityScale(int body, float scale);
    void setMaxLinearSpeed(int body, float speed); // 0 = unlimited
    void setSleepThreshold(int body, float speed); // 0 = the world's thresholds
    void setEnableSleep(int body, bool enable);
    void setMotionLocks(int body, unsigned lockFlags); // BODY_LOCK_* bits
    // Material (per body; a contact mixes the two bodies' values).
    void setTangentSpeed(int body, float speed);            // conveyor, body-local +X, m/s
    void setRollingResistance(int body, float coefficient); // sphere / capsule only
    void setFrictionMix(int body, FrictionMix mix);         // the larger enum of a pair wins
    void setUserMaterialId(int body, unsigned id);
    void setRestitution(int body, float restitution); // stored, NOT simulated
    BodyProps bodyProps(int body) const;

    // Runtime spawning: the same as addBody()/addJoint() with the spawn structs, so a
    // SpawnEmitter can commit into a builder as it does into a live solver. Return the
    // first new body index (-1 for an empty group).
    int spawnBody(vec3 size, float density, float friction, vec3 position, vec3 velocity = vec3{0, 0, 0},
                  vec3 angularVelocity = vec3{0, 0, 0}, quat rotation = quat{0, 0, 0, 1});
    int spawnGroup(const SpawnBody *bodies, int nb, const SpawnJoint *joints, int nj);

    // The largest up-facing static face: where a shadow lands. False (and a +y plane
    // through the origin) if the scene has no static face facing up. Up is +y.
    bool findShadowPlane(vec3 &planePoint, vec3 &planeNormal) const;

    int bodyCount() const { return (int)m_pos.size(); }
    int jointCount() const { return (int)m_jointA.size(); }
    vec3 bodyPos(int bodyIdx) const;
    quat bodyRot(int bodyIdx) const;
    // AVBD_SHAPE_* of an authored body; AVBD_SHAPE_BOX (0) for an index out of range.
    unsigned char bodyShape(int bodyIdx) const
    {
        return bodyIdx >= 0 && bodyIdx < (int)m_shape.size() ? m_shape[bodyIdx] : 0;
    }

    int shapeCount() const { return (int)m_shapeBody.size(); }

    SolverParams params = defaultSolverParams();

    // The catalogue (scenes.h) rotates its z-up sources into the y-up solver frame. A gate that
    // compares against frozen z-up outputs (tools/golden) or checks z heights sets this to keep
    // the authored frame; it must then also set a z-up params.gravity.
    bool keepZUpSource = false;

    // Produces the import record: derived values computed, ignore pairs sorted, spring
    // rest lengths resolved against the authored poses. Does not consume the builder.
    SceneDesc finalize() const;

private:
    int addShapedBody(int shape, vec3 half, float density, float friction, vec3 position, vec3 velocity,
                      vec3 angularVelocity, quat rotation);
    int pushShape(int body, int type, vec3 localPos, quat localRot, vec3 half, float friction);
    int internGeometry(const b3HullData *hull); // blob index, deduplicated by hash; -1: invalid
    int internBlob(const unsigned char *src, size_t n, unsigned long long hash); // hash 0: computed
    // Cooks the triangles with Box3D and returns the blob index (-1 on failure); `half` is the
    // mesh's local AABB half size.
    int cookMesh(const vec3 *vertices, int vertexCount, const int *indices, int triangleCount, vec3 &half);
    std::vector<int> m_shapeGeom;
    std::vector<std::vector<unsigned char>> m_geomBlob;
    std::vector<unsigned long long> m_geomHash;
    std::vector<vec3> m_bodyLocalCenter;
    std::vector<quat> m_bodyPrincipalRot;

    // The shape list (SceneDesc::shape*), in authoring order; one entry per body is its
    // primary shape.
    std::vector<int> m_shapeBody, m_bodyPrimary;
    std::vector<vec3> m_shapeLocalPos, m_shapeHalf;
    std::vector<quat> m_shapeLocalRot;
    std::vector<unsigned char> m_shapeKind;
    std::vector<float> m_shapeFriction;
    std::vector<uint64_t> m_shapeCategory, m_shapeMask;
    std::vector<unsigned> m_shapeFlags;
    std::vector<int> m_shapeGroup;

    std::vector<vec3> m_pos, m_vel, m_velAng, m_half, m_moment;
    std::vector<quat> m_rot;
    std::vector<float> m_mass, m_friction;
    std::vector<unsigned char> m_shape, m_tier, m_asleep;
    std::vector<int> m_group;
    mutable std::vector<BodyProps> m_props;
    BodyProps &propsOf(int body);

    std::vector<int> m_jointA, m_jointB;
    std::vector<vec3> m_jointRA, m_jointRB;
    std::vector<float> m_jointKLin, m_jointKAng, m_jointFrac, m_jointFracAng;
    std::vector<quat> m_jointFrameA, m_jointFrameB;
    std::vector<unsigned char> m_jointAxisMode, m_jointGeneric;
    std::vector<float> m_jointAxisLo, m_jointAxisHi;

    std::vector<int> m_springA, m_springB;
    std::vector<vec3> m_springRA, m_springRB;
    std::vector<float> m_springRest, m_springK;

    std::vector<unsigned long long> m_ignore;
};
