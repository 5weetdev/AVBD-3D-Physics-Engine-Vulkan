// Hand-written half of the Box3D device port. See tools/box3d_to_slang.py (which writes
// engine/shaders/b3/b3_gen.slang, the mechanical half) and docs/SOLVER_3D_ROADMAP.md Phase 0.2.
//
// This file holds only what the generator cannot derive from the C sources:
//   * the POD structs, with the same field order as the Box3D C structs (the generator reads the
//     field order from here to rewrite C initializers into Slang constructors),
//   * the hull byte-blob accessors (on device a hull is a device address into a byte buffer),
//   * the pieces of simd.h's scalar path that use memcpy or pointer-to-local,
//   * the shape proxy, which cannot point at a local point on device,
//   * constants that are not macros in the C sources (INFINITY, FLT_*).
//
// Device deviations from the C structs, all invisible to the generated algorithm:
//   * b3LocalManifold holds its (at most 4) points inline instead of through a pointer. Every
//     Box3D collide function is called with capacity <= 4 on this path.
//   * b3ShapeProxy holds up to two inline points beside the optional hull point pointer.
//   * Pointers to locals do not exist: the generator turns C out-params into inout params.
#ifndef B3_SHIM_SLANG_INCLUDED
#define B3_SHIM_SLANG_INCLUDED

#define B3_SIMD_NONE 1

// algorithm.h's B3_SWAP is memcpy based.
#define B3_SWAP(x, y) { let b3swapT = (x); (x) = (y); (y) = b3swapT; }

// Enums (manifold.h, convex_manifold.c). The type names are plain ints on device.
typedef int b3FeatureOwner;
typedef int b3SeparatingFeature;
#define b3_featureShapeA 0
#define b3_featureShapeB 1
#define b3_invalidAxis 0
#define b3_backsideAxis 1
#define b3_faceAxisA 2
#define b3_faceAxisB 3
#define b3_edgePairAxis 4
#define b3_closestPointsAxis 5
#define b3_manualFaceAxisA 6
#define b3_manualFaceAxisB 7
#define b3_manualEdgePairAxis 8

// Time of impact enums (types.h b3TOIState, distance.c b3SeparationType).
typedef int b3TOIState;
#define b3_toiStateUnknown 0
#define b3_toiStateFailed 1
#define b3_toiStateOverlapped 2
#define b3_toiStateHit 3
#define b3_toiStateSeparated 4
typedef int b3SeparationType;
#define b3_separationUnknown 0
#define b3_separationVertices 1
#define b3_separationEdges 2
#define b3_separationFaceA 3
#define b3_separationFaceB 4

// Triangle feature enum (types.h b3TriangleFeature); values must match the C enum order.
typedef int b3TriangleFeature;
#define b3_featureNone 0
#define b3_featureTriangleFace 1
#define b3_featureHullFace 2
#define b3_featureEdge1 3
#define b3_featureEdge2 4
#define b3_featureEdge3 5
#define b3_featureVertex1 6
#define b3_featureVertex2 7
#define b3_featureVertex3 8

// ---------------------------------------------------------------------------------------------
// math_functions.h types
// ---------------------------------------------------------------------------------------------
struct b3Vec3
{
    float x;
    float y;
    float z;
};

struct b3Quat
{
    b3Vec3 v;
    float s;
};

struct b3Transform
{
    b3Vec3 p;
    b3Quat q;
};

struct b3Matrix3
{
    b3Vec3 cx;
    b3Vec3 cy;
    b3Vec3 cz;
};

struct b3AABB
{
    b3Vec3 lowerBound;
    b3Vec3 upperBound;
};

struct b3Plane
{
    b3Vec3 normal;
    float offset;
};

struct b3SegmentDistanceResult
{
    b3Vec3 point1;
    float fraction1;
    b3Vec3 point2;
    float fraction2;
};

struct b3TrianglePoint
{
    b3Vec3 point;
    b3TriangleFeature feature;
};

// Indexed by the 3-bit vertex mask (triangle_manifold.c s_triangleFeatures).
static const b3TriangleFeature s_triangleFeatures[8] = { b3_featureNone, b3_featureVertex1, b3_featureVertex2, b3_featureEdge1,
                                                         b3_featureVertex3, b3_featureEdge3, b3_featureEdge2, b3_featureTriangleFace };

struct b3TriangleData
{
    b3Vec3 v1;
    b3Vec3 v2;
    b3Vec3 v3;
    b3Vec3 e1;
    b3Vec3 e2;
    b3Vec3 e3;
    b3Plane plane;
    int flags;
};

static const b3Vec3 b3Vec3_zero = { 0.0f, 0.0f, 0.0f };
static const b3Vec3 b3Vec3_one = { 1.0f, 1.0f, 1.0f };
static const b3Quat b3Quat_identity = { { 0.0f, 0.0f, 0.0f }, 1.0f };
static const b3Transform b3Transform_identity = { { 0.0f, 0.0f, 0.0f }, { { 0.0f, 0.0f, 0.0f }, 1.0f } };
static const b3Matrix3 b3Mat3_zero = { { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f } };
static const b3Matrix3 b3Mat3_identity = { { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } };

// ---------------------------------------------------------------------------------------------
// C library constants
// ---------------------------------------------------------------------------------------------
#define FLT_MAX 3.402823466e+38f
#define FLT_MIN 1.175494351e-38f
#define FLT_EPSILON 1.192092896e-07f
#define UINT8_MAX 255
static const float INFINITY = asfloat(0x7f800000);

// ---------------------------------------------------------------------------------------------
// types.h / manifold.h / distance.h structs
// ---------------------------------------------------------------------------------------------
struct b3Capsule
{
    b3Vec3 center1;
    b3Vec3 center2;
    float radius;
};

struct b3Sphere
{
    b3Vec3 center;
    float radius;
};

struct b3FeaturePair
{
    uint8_t owner1;
    uint8_t index1;
    uint8_t owner2;
    uint8_t index2;
};

static const b3FeaturePair b3FeaturePair_single = { 0, 0, 0, 0 };

struct b3LocalManifoldPoint
{
    b3Vec3 point;
    float separation;
    b3FeaturePair pair;
    int triangleIndex;
};

// Device deviation: inline point storage (see the file comment).
struct b3LocalManifold
{
    b3Vec3 normal;
    b3Vec3 triangleNormal;
    b3LocalManifoldPoint points[4];
    int pointCount;
    int triangleIndex;
    int i1;
    int i2;
    int i3;
    float squaredDistance;
    int feature;
    int triangleFlags;
};

struct b3SATCache
{
    float separation;
    uint8_t type;
    uint8_t indexA;
    uint8_t indexB;
    uint8_t hit;
};

struct b3SimplexCache
{
    float metric;
    int count; // uint16_t in C; no 16-bit device storage is enabled
    uint8_t indexA[4];
    uint8_t indexB[4];
};

struct b3SeparatingAxis
{
    b3Vec3 normal;
    float separation;
    int indexA;
    int indexB;
    int type;
};

struct b3AxisQuery
{
    b3SeparatingAxis faceA;
    b3SeparatingAxis faceB;
    b3SeparatingAxis edge;
    int separatedFeature;
};

struct b3ClipVertex
{
    b3Vec3 position;
    float separation;
    b3FeaturePair pair;
};

// Device deviation: the C proxy is {const b3Vec3* points; int count; float radius}. A pointer to
// a local point cannot exist on device, so the one- and two-point proxies (sphere, capsule) keep
// their points in p0/p1 and `points` is null. Every `proxy->points[i]` in the generated code goes
// through b3ProxyPoint (see the generator's patch table).
struct b3ShapeProxy
{
    b3Vec3* points;
    int count;
    float radius;
    b3Vec3 p0;
    b3Vec3 p1;
};

struct b3DistanceInput
{
    b3ShapeProxy proxyA;
    b3ShapeProxy proxyB;
    b3Transform transform;
    bool useRadii;
};

struct b3DistanceOutput
{
    b3Vec3 pointA;
    b3Vec3 pointB;
    b3Vec3 normal;
    float distance;
    int iterations;
    int simplexCount;
};

struct b3SimplexVertex
{
    b3Vec3 wA;
    b3Vec3 wB;
    b3Vec3 w;
    float a;
    int indexA;
    int indexB;
};

struct b3Simplex
{
    b3SimplexVertex vertices[4];
    int count;
};

// Time of impact (types.h b3Sweep, b3TOIInput, b3TOIOutput; distance.c b3SeparationFunction).
struct b3Sweep
{
    b3Vec3 localCenter;
    b3Vec3 c1;
    b3Vec3 c2;
    b3Quat q1;
    b3Quat q2;
};

struct b3TOIInput
{
    b3ShapeProxy proxyA;
    b3ShapeProxy proxyB;
    b3Sweep sweepA;
    b3Sweep sweepB;
    float maxFraction;
};

struct b3TOIOutput
{
    b3TOIState state;
    b3Vec3 point;
    b3Vec3 normal;
    float fraction;
    float distance;
    int distanceIterations;
    int pushBackIterations;
    int rootIterations;
    bool usedFallback;
};

struct b3SeparationFunction
{
    b3ShapeProxy* proxyA;
    b3ShapeProxy* proxyB;
    b3Sweep sweepA;
    b3Sweep sweepB;
    b3Vec3 witness1;
    b3Vec3 witness2;
    b3SeparationType type;
};

b3Vec3 b3ProxyPoint(b3ShapeProxy proxy, int i)
{
    if (proxy.points != nullptr)
        return proxy.points[i];
    return i == 0 ? proxy.p0 : proxy.p1;
}

// ---------------------------------------------------------------------------------------------
// Hull blob. A hull is a device address of a b3HullData (144 bytes, scalar layout) followed by
// its arrays; the offsets are in bytes from the start of the b3HullData, exactly as on the host.
// ---------------------------------------------------------------------------------------------
struct b3HullData
{
    uint64_t version;
    uint64_t hash;
    b3AABB aabb;
    float surfaceArea;
    float volume;
    float innerRadius;
    b3Vec3 center;
    b3Matrix3 centralInertia;
    int vertexCount;
    int vertexOffset;
    int pointOffset;
    int edgeCount;
    int edgeOffset;
    int faceCount;
    int planeOffset;
    int faceOffset;
    int soaVertexOffset;
    int soaNormalOffset;
    int byteCount;
};

struct b3HullVertex
{
    uint8_t edge;
};

struct b3HullHalfEdge
{
    uint8_t next;
    uint8_t twin;
    uint8_t origin;
    uint8_t face;
};

struct b3HullFace
{
    uint8_t edge;
};

b3HullVertex* b3GetHullVertices(b3HullData* hull)
{
    return (b3HullVertex*)((uint8_t*)hull + hull->vertexOffset);
}
b3Vec3* b3GetHullPoints(b3HullData* hull)
{
    return (b3Vec3*)((uint8_t*)hull + hull->pointOffset);
}
b3HullHalfEdge* b3GetHullEdges(b3HullData* hull)
{
    return (b3HullHalfEdge*)((uint8_t*)hull + hull->edgeOffset);
}
b3Plane* b3GetHullPlanes(b3HullData* hull)
{
    return (b3Plane*)((uint8_t*)hull + hull->planeOffset);
}
b3HullFace* b3GetHullFaces(b3HullData* hull)
{
    return (b3HullFace*)((uint8_t*)hull + hull->faceOffset);
}
float* b3GetHullSoaVertices(b3HullData* hull)
{
    return (float*)((uint8_t*)hull + hull->soaVertexOffset);
}
float* b3GetHullSoaNormals(b3HullData* hull)
{
    return (float*)((uint8_t*)hull + hull->soaNormalOffset);
}

// ---------------------------------------------------------------------------------------------
// simd.h scalar path pieces that cannot be generated: b3FloatW is the scalar {x,y,z,w} struct;
// the generator emits everything else in that block. memcpy bit casts become asuint/asfloat.
// b3LoadW / b3StoreW take (array-or-pointer, index) because C passes `ptr + i`, which Slang
// cannot do for a local array; the generator rewrites `b3LoadW( p + i )` to `b3LoadW( p, i )`.
// ---------------------------------------------------------------------------------------------
struct b3FloatW
{
    float x;
    float y;
    float z;
    float w;
};

b3FloatW b3EmbedIndexW(b3FloatW value, int baseIndex, int bitCount)
{
    uint mask = (1u << bitCount) - 1;
    float lanes[4] = { value.x, value.y, value.z, value.w };
    for (int i = 0; i < 4; ++i)
    {
        uint bits = asuint(lanes[i]);
        bits = (bits & ~mask) | (uint)(baseIndex + i);
        lanes[i] = asfloat(bits);
    }
    return b3FloatW(lanes[0], lanes[1], lanes[2], lanes[3]);
}

int b3MinIndexW(b3FloatW a, int bitCount)
{
    float m = a.x;
    m = a.y < m ? a.y : m;
    m = a.z < m ? a.z : m;
    m = a.w < m ? a.w : m;
    return (int)(asuint(m) & ((1u << bitCount) - 1));
}

#endif // B3_SHIM_SLANG_INCLUDED
