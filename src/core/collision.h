#pragma once

// Narrowphase types and entry point.
//
// Deliberately free of solver.h, and therefore of windows.h and GL: this header plus
// maths.h is everything the SAT routine needs, so the SAT collider compiles in any
// translation unit without dragging in the host-only scaffolding.

#include "maths.h"

#define MAX_MANIFOLD_CONTACTS 8 // Contacts a single box-box manifold can produce
#ifndef COLLISION_MARGIN
#define COLLISION_MARGIN 0.01f // Margin for collision detection to avoid flickering contacts
#endif

// Contact constraint bounds. These live here rather than in solver.h because both the
// CPU solver and the GPU narrowphase apply them, and the GPU side must not pull in
// solver.h (which drags windows.h and GL along with it).
#define PENALTY_MIN 1.0f           // Minimum penalty parameter
#define PENALTY_MAX 10000000000.0f // Maximum penalty parameter
#define STICK_THRESH 0.00001f      // Position threshold for sticking contacts (ie static friction)

// Circuit breaker for a body whose position has diverged, in metres. Not a physical
// quantity -- every demo scene here lives within a few hundred units of the origin --
// it exists only to give k_explosion_guard (the integrate stage) a bound to catch a runaway
// body before its AABB reaches the broadphase. A NaN or astronomically large AABB
// there does not just misbehave locally: aabbOverlap's comparisons against NaN are all
// false, and the overlap test is the negation of them, so a corrupted body reads as
// overlapping *every* body it shares a hash bucket with, which can turn one bad body
// into a broadphase pair count the retry-and-grow path then tries to allocate for --
// see the cap next to that retry in VkWorld::step().
#define WORLD_BOUND 100000.0f

// Used to track contact features between frames
union FeaturePair
{
    struct
    {
        char inR;
        char outR;
        char inI;
        char outI;
    };

    int key;
};

// Contact point information for a single contact
struct Contact
{
    FeaturePair feature;
    vec3 rA; // contact offset in A's local space (relative to center)
    vec3 rB; // contact offset in B's local space (relative to center)
    vec3 C0;
    vec3 penalty;
    vec3 lambda;
    bool stick;
};

// ---------------------------------------------------------------------------
// Shapes (UNITY_DOTS_PLAN.md §4, Tier 1)
//
// BOX is 0 so that every zero-initialised BodyRef and every array that predates
// shapes is still a box. That is not a stylistic choice: it is what lets the
// determinism oracle prove this change numerics-neutral on the existing scenes.
//
// All three shapes share `halfExtents`, and they share it in a specific way --
// **halfExtents is always the tight OBB of the shape**:
//
//   BOX      (hx, hy, hz)            half extents, as always
//   SPHERE   (r,  r,  r)             radius in every slot
//   CAPSULE  (r,  r,  r + halfSeg)   axis is local +z; the segment half-length is
//                                    recovered as halfExtents.z - halfExtents.x
//
// Because that invariant holds, *nothing outside this header changes*: the AABBs the
// broadphase builds, the gyration radius the sleep test uses, and the bounds the
// render culler tests are all already correct for a sphere and a capsule. Only the
// narrowphase needs to know the difference, which is the whole point of putting the
// tag here rather than threading a shape struct through the pipeline.
#define AVBD_SHAPE_BOX 0
#define AVBD_SHAPE_SPHERE 1
#define AVBD_SHAPE_CAPSULE 2
// Id 3 is unused; HULL and MESH are hard-coded in the kernels.
// A convex hull whose Box3D blob (b3HullData) lives in the device geometry heap at the shape's
// geomOffset. halfExtents is the hull's local AABB half size about the shape origin. Not
// collidable yet: the narrowphase emits no contacts for a pair with a hull in it, and the
// renderer skips it (a later phase adds the hull SAT and a mesh draw).
#define AVBD_SHAPE_HULL 4
// A static triangle mesh: a Box3D b3MeshData blob (BVH + vertices + triangles) in the
// device geometry heap at the shape's geomOffset. halfExtents is the mesh's local AABB half size.
// Meshes are static (density 0) and collide only with convex shapes (box, sphere, capsule, hull).
// The narrowphase divert sends those pairs to narrowphase_mesh.slang, which runs a
// midphase over the mesh BVH and then a per-triangle narrowphase. The renderer skips it.
#define AVBD_SHAPE_MESH 5

// Shape flags (the `flags` word of a device shape, SceneDesc::shapeFlags).
//   bit 0: the shape's local pose is the identity (position 0, rotation identity), so its
//          world pose IS the body pose. The kernels skip the composition for such a shape,
//          which keeps a single-shape body bit-identical to the pre-shape solver. Set by
//          SceneBuilder; never set by hand.
//   bit 1: REPORT. A touching manifold with a REPORT shape on either side is written to the
//          contact-report buffer each step (csContactReport). Set by the API layer when the
//          shape has contact or hit events enabled; the solver never reads it otherwise.
// The remaining bits are reserved.
#define AVBD_SHAPE_FLAG_IDENTITY 1u
#define AVBD_SHAPE_FLAG_REPORT 2u

// The minimal body state the narrowphase reads. Decoupling it from Rigid lets the
// SAT routine run unchanged against SoA arrays on the GPU, where there is no Rigid.
struct BodyRef
{
    vec3 pos;
    quat rot;
    vec3 halfExtents;
    int shape; // AVBD_SHAPE_*; 0 (= BOX) is the pre-shapes behaviour
};

// Sphere radius / capsule radius and segment half-length, from the OBB invariant above.
AVBD_HD inline float shapeRadius(const BodyRef &b)
{
    return b.halfExtents.x;
}
AVBD_HD inline float capsuleHalfSegment(const BodyRef &b)
{
    float h = b.halfExtents.z - b.halfExtents.x;
    return h > 0.0f ? h : 0.0f;
}

// Box-box SAT narrowphase. Pure leaf function: no allocation, no global state, writes
// at most MAX_MANIFOLD_CONTACTS contacts. Returns the contact count.
AVBD_HD int collideBoxes(const BodyRef &a, const BodyRef &b, Contact *contacts, float3x3 &basisOut);

// Shape-dispatching narrowphase: the entry point the pipeline calls. Dispatches on
// (a.shape, b.shape) and is exactly collideBoxes when both are boxes -- same code
// path, not an equivalent one.
//
// Contract, identical to collideBoxes and relied on by gpu_constraints: `rA` and `rB`
// are the witness points on each surface expressed in that body's local frame, and
// `basisOut` is orthonormal(-n) where n runs from A toward B. Penetration is never
// stored; the solver recovers it from the witness pair, which is what lets a contact
// stay meaningful after the bodies have moved within a step.
AVBD_HD int collideShapes(const BodyRef &a, const BodyRef &b, Contact *contacts, float3x3 &basisOut);
