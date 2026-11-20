#pragma once

// SceneDesc: the plain import record for a scene. Everything a backend needs to build its
// device state, in host arrays, with the derived quantities (broadphase cell size, joint
// torque arm and rest orientation, resolved spring rest length, the sorted collision-ignore
// list) already computed -- so no backend re-derives them, and none needs to know how a
// scene was authored. SceneBuilder produces one; VkWorld consumes it.
//
// In-process only and unversioned. It is not the ABI (include/avbd3d/ is).

#include "scene_types.h"
#include "solver_params.h"

#include <cstdint>
#include <vector>

struct SceneDesc
{
    SolverParams params = defaultSolverParams();

    int count = 0;
    // Per body. halfExtents, friction and shapeType describe the body's PRIMARY shape (the
    // one addBody/addSphere/addCapsule created) and are derived from the shape list below:
    // they are what mass, moment and the sleep gyration radius read. Collision, rendering
    // and queries go through the shape list.
    std::vector<vec3> pos, velLin, velAng, halfExtents, moment;
    std::vector<quat> rot;
    std::vector<float> mass, friction;
    std::vector<unsigned char> shapeType;
    std::vector<int> renderGroup;      // -1: the body's shape class
    std::vector<unsigned char> tier;   // BodyTier
    std::vector<unsigned char> asleep; // authored initial sleep state (1: starts asleep)
    std::vector<int> dynamicIdx, staticIdx;
    float cellSize = 1.0f;

    // Shapes (SceneBuilder::addShape). A body owns one or more collision shapes, each with a
    // pose in the body frame; the world pose of shape s is body pose o local pose. The shape
    // list is in authoring order: every body's primary shape is created with the body, so a
    // scene that never calls addShape has shapeCount == count and shape i belongs to body i.
    // Shapes are only ever appended to, so a grown scene (VkWorld::append) keeps every
    // earlier shape index (they key the persistent contact cache). Mass and inertia stay on
    // the body: an added shape contributes no mass.
    int shapeCount = 0;
    std::vector<int> shapeBody;      // owning body
    std::vector<vec3> shapeLocalPos; // in the body frame
    std::vector<quat> shapeLocalRot;
    std::vector<unsigned char> shapeKind; // AVBD_SHAPE_*
    std::vector<vec3> shapeHalf;          // half extents as halfExtents above
    std::vector<float> shapeFriction;
    // Per-shape bounce and rolling resistance (the API layer's surface material); may be empty or
    // shorter than shapeCount, which reads as 0. Rolling is 0 for a shape that is not a sphere or
    // capsule. Mixed per contact on the device (restitution.slang, primal.slang).
    std::vector<float> shapeRestitution, shapeRolling;
    // Collision filter, stored on device but not consulted yet (a later phase): a pair
    // collides when each shape's category is in the other's mask, and a shared nonzero
    // group index forces (positive) or forbids (negative) it. Defaults: category 1, mask
    // all 64 bits, group 0.
    std::vector<uint64_t> shapeCategory, shapeMask;
    std::vector<int> shapeGroup;
    std::vector<unsigned> shapeFlags; // SHAPE_FLAG_*
    // Per-body settings (scene_types.h BodyProps), one entry per body; may be empty or shorter
    // than `count`, which reads as all defaults.
    std::vector<BodyProps> bodyProps;
    std::vector<int> bodyPrimaryShape; // per body: index of its primary shape

    // Geometry blobs (Box3D b3HullData, verbatim bytes: relocatable, so a byte copy is valid).
    // shapeGeom[s] indexes geomBlob, or -1 for a shape with no blob. Blobs are deduplicated by
    // geomHash at authoring time, so one hull shared by many shapes is stored once. Only ever
    // appended to. VkWorld uploads them into its geometry heap (VkWorld::addGeometry).
    std::vector<int> shapeGeom;
    std::vector<std::vector<unsigned char>> geomBlob;
    std::vector<unsigned long long> geomHash;

    // Origin vs centre of mass (SceneBuilder::computeMassFromShapes). pos/rot above are the
    // SOLVER pose: the centre of mass in the principal frame. The pose the author gave is
    //   originRot = rot * conj(bodyPrincipalRot),  originPos = pos - rotate(originRot, bodyLocalCenter)
    // with bodyLocalCenter in the origin (user) frame and bodyPrincipalRot the principal-to-user
    // rotation. Shape local poses are already expressed in the solver frame. Identity (0, id)
    // for any body that never called computeMassFromShapes.
    std::vector<vec3> bodyLocalCenter;
    std::vector<quat> bodyPrincipalRot;

    int jointCount = 0;
    std::vector<int> jointA, jointB;
    std::vector<vec3> jointRA, jointRB;
    std::vector<float> jointKLin, jointKAng, jointFrac, jointArm;
    std::vector<float>
        jointFracAng; // N*m bending capacity per joint; < 0: legacy test on |lambdaAng| (see jointFracture)
    std::vector<quat> jointRestAng, jointFrameA, jointFrameB;
    std::vector<unsigned char> jointAxisMode, jointGeneric;
    std::vector<float> jointAxisLo, jointAxisHi;

    int springCount = 0;
    std::vector<int> springA, springB;
    std::vector<vec3> springRA, springRB;
    std::vector<float> springRest, springK; // rest resolved: never negative

    // Sorted, unique pair keys the broadphase must never report (jointed neighbours,
    // ignoreCollision). Low index in the high 32 bits, high index in the low 32.
    std::vector<unsigned long long> ignorePairs;
};
