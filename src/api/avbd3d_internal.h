// Internal types, constants and helpers shared by the avbd3d implementation files.
// Not installed and not part of the public API (that is include/avbd3d/).
// Names here are implementation details and carry no stability promise.

#pragma once

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
#define AVBD3D_LEGACY_API extern "C"
#define AVBD3D_INLINE inline
#else
#define AVBD3D_LEGACY_API
#define AVBD3D_INLINE static inline
#endif
#ifndef AVBD3D_MAX_WORLDS
#define AVBD3D_MAX_WORLDS 128
#endif
#define AVBD3D_MAX_MANIFOLD_POINTS 4
#define AVBD3D_MAX_HULL_VERTICES 128
#define AVBD3D_MAX_HULL_FACES 128
#define AVBD3D_MAX_HULL_EDGES 128
#define AVBD3D_MAX_SHAPE_CAST_POINTS 128
#define AVBD3D_SHAPE_POWER 22
#define AVBD3D_CHILD_POWER (64 - 2 * AVBD3D_SHAPE_POWER)
#define AVBD3D_MAX_CHILD_SHAPES (1 << AVBD3D_CHILD_POWER)

typedef struct avbd3d_WorldId
{
    uint16_t index1;
    uint16_t generation;
} avbd3d_WorldId;

typedef struct avbd3d_BodyId
{
    int32_t index1;
    uint16_t world0;
    uint16_t generation;
} avbd3d_BodyId;

typedef struct avbd3d_ShapeId
{
    int32_t index1;
    uint16_t world0;
    uint16_t generation;
} avbd3d_ShapeId;

typedef struct avbd3d_JointId
{
    int32_t index1;
    uint16_t world0;
    uint16_t generation;
} avbd3d_JointId;

typedef struct avbd3d_ContactId
{
    int32_t index1;
    uint16_t world0;
    int16_t padding;
    uint32_t generation;
} avbd3d_ContactId;

typedef enum avbd3d_Error
{
    avbd3d_errorNone = 0,
    avbd3d_errorOutOfMemory,
    avbd3d_errorDevice,
    avbd3d_errorDeviceLost,
} avbd3d_Error;

#ifdef __cplusplus
#define AVBD3D_NULL_ID                                                                                                 \
    {                                                                                                                  \
    }
#else
#define AVBD3D_NULL_ID {0}
#endif
static const avbd3d_WorldId avbd3d_nullWorldId = AVBD3D_NULL_ID;
static const avbd3d_BodyId avbd3d_nullBodyId = AVBD3D_NULL_ID;
static const avbd3d_ShapeId avbd3d_nullShapeId = AVBD3D_NULL_ID;
static const avbd3d_JointId avbd3d_nullJointId = AVBD3D_NULL_ID;

typedef struct avbd3d_Vec3
{
    float x;
    float y;
    float z;
} avbd3d_Vec3;

typedef struct avbd3d_Quat
{
    avbd3d_Vec3 v;
    float s;
} avbd3d_Quat;

typedef struct avbd3d_Transform
{
    avbd3d_Vec3 p;
    avbd3d_Quat q;
} avbd3d_Transform;

typedef avbd3d_Vec3 avbd3d_Pos;
typedef avbd3d_Transform avbd3d_WorldTransform;

typedef struct avbd3d_Matrix3
{
    avbd3d_Vec3 cx, cy, cz;
} avbd3d_Matrix3;

typedef struct avbd3d_AABB
{
    avbd3d_Vec3 lowerBound;
    avbd3d_Vec3 upperBound;
} avbd3d_AABB;

typedef struct avbd3d_Plane
{
    avbd3d_Vec3 normal;
    float offset;
} avbd3d_Plane;

typedef struct avbd3d_DebugShape avbd3d_DebugShape;
typedef void *avbd3d_CreateDebugShapeCallback(const avbd3d_DebugShape *debugShape, void *userContext);
typedef void avbd3d_DestroyDebugShapeCallback(void *userShape, void *userContext);
typedef bool avbd3d_OverlapResultFcn(avbd3d_ShapeId shapeId, void *context);

typedef float avbd3d_CastResultFcn(avbd3d_ShapeId shapeId, avbd3d_Pos point, avbd3d_Vec3 normal, float fraction,
                                   uint64_t userMaterialId, int triangleIndex, int childIndex, void *context);

typedef struct avbd3d_Capacity
{
    int staticShapeCount;

    int dynamicShapeCount;
    int staticBodyCount;
    int dynamicBodyCount;

    int contactCount;
} avbd3d_Capacity;

typedef struct avbd3d_WorldDef
{
    avbd3d_Vec3 gravity;

    float restitutionThreshold;
    float hitEventThreshold;
    bool enableSleep;
    bool enableContinuous;
    void *userData;
    avbd3d_CreateDebugShapeCallback *createDebugShape;
    avbd3d_DestroyDebugShapeCallback *destroyDebugShape;
    void *userDebugShapeContext;
    avbd3d_Capacity capacity;

    int internalValue;
} avbd3d_WorldDef;

AVBD3D_LEGACY_API avbd3d_WorldDef avbd3d_DefaultWorldDef(void);

typedef enum avbd3d_BodyType
{
    avbd3d_staticBody = 0,

    avbd3d_kinematicBody = 1,
    avbd3d_dynamicBody = 2,

    avbd3d_bodyTypeCount,
} avbd3d_BodyType;

typedef struct avbd3d_MotionLocks
{
    bool linearX;

    bool linearY;
    bool linearZ;
    bool angularX;
    bool angularY;

    bool angularZ;
} avbd3d_MotionLocks;

typedef struct avbd3d_BodyDef
{
    avbd3d_BodyType type;

    avbd3d_Pos position;
    avbd3d_Quat rotation;
    avbd3d_Vec3 linearVelocity;
    avbd3d_Vec3 angularVelocity;
    float linearDamping;
    float angularDamping;
    float gravityScale;
    float sleepThreshold;
    const char *name;
    void *userData;
    avbd3d_MotionLocks motionLocks;
    bool enableSleep;
    bool isAwake;
    bool isBullet;
    bool isEnabled;

    int internalValue;
} avbd3d_BodyDef;

AVBD3D_LEGACY_API avbd3d_BodyDef avbd3d_DefaultBodyDef(void);

typedef struct avbd3d_Filter
{
    uint64_t categoryBits;

    uint64_t maskBits;

    int groupIndex;
} avbd3d_Filter;

AVBD3D_LEGACY_API avbd3d_Filter avbd3d_DefaultFilter(void);

typedef struct avbd3d_SurfaceMaterial
{
    float friction;

    float restitution;
    float rollingResistance;
    avbd3d_Vec3 tangentVelocity;
    uint64_t userMaterialId;
    uint32_t customColor;

    uint32_t padding;
} avbd3d_SurfaceMaterial;

AVBD3D_LEGACY_API avbd3d_SurfaceMaterial avbd3d_DefaultSurfaceMaterial(void);

typedef enum avbd3d_ShapeType
{
    avbd3d_capsuleShape,

    avbd3d_compoundShape,
    avbd3d_heightShape,
    avbd3d_hullShape,
    avbd3d_meshShape,
    avbd3d_sphereShape,

    avbd3d_shapeTypeCount
} avbd3d_ShapeType;

typedef struct avbd3d_ShapeDef
{
    const char *name;

    void *userData;
    avbd3d_SurfaceMaterial *materials;
    int materialCount;
    avbd3d_SurfaceMaterial baseMaterial;
    float density;
    float explosionScale;
    avbd3d_Filter filter;
    bool isSensor;
    bool enableSensorEvents;
    bool enableContactEvents;
    bool enableHitEvents;
    bool invokeContactCreation;
    bool updateBodyMass;
    bool enableSpeculativeContact;
    int internalValue;
} avbd3d_ShapeDef;
AVBD3D_LEGACY_API avbd3d_ShapeDef avbd3d_DefaultShapeDef(void);

typedef enum avbd3d_JointType
{
    avbd3d_parallelJoint,
    avbd3d_distanceJoint,
    avbd3d_filterJoint,
    avbd3d_motorJoint,
    avbd3d_prismaticJoint,
    avbd3d_revoluteJoint,
    avbd3d_sphericalJoint,
    avbd3d_weldJoint,
    avbd3d_wheelJoint,
} avbd3d_JointType;

typedef struct avbd3d_JointDef
{
    void *userData;

    avbd3d_BodyId bodyIdA;
    avbd3d_BodyId bodyIdB;
    avbd3d_Transform localFrameA;
    avbd3d_Transform localFrameB;
    float forceThreshold;
    float torqueThreshold;
    float drawScale;
    bool collideConnected;

    int internalValue;
} avbd3d_JointDef;

typedef struct avbd3d_DistanceJointDef
{
    avbd3d_JointDef base;

    float length;
    bool enableSpring;
    float lowerSpringForce;
    float upperSpringForce;
    float hertz;
    float dampingRatio;
    bool enableLimit;
    float minLength;
    float maxLength;
    bool enableMotor;
    float maxMotorForce;

    float motorSpeed;
} avbd3d_DistanceJointDef;

AVBD3D_LEGACY_API avbd3d_DistanceJointDef avbd3d_DefaultDistanceJointDef(void);

typedef struct avbd3d_MotorJointDef
{
    avbd3d_JointDef base;

    avbd3d_Vec3 linearVelocity;
    float maxVelocityForce;
    avbd3d_Vec3 angularVelocity;
    float maxVelocityTorque;
    float linearHertz;
    float linearDampingRatio;
    float maxSpringForce;
    float angularHertz;
    float angularDampingRatio;

    float maxSpringTorque;
} avbd3d_MotorJointDef;

AVBD3D_LEGACY_API avbd3d_MotorJointDef avbd3d_DefaultMotorJointDef(void);

typedef struct avbd3d_FilterJointDef
{
    avbd3d_JointDef base;
} avbd3d_FilterJointDef;

AVBD3D_LEGACY_API avbd3d_FilterJointDef avbd3d_DefaultFilterJointDef(void);

typedef struct avbd3d_ParallelJointDef
{
    avbd3d_JointDef base;

    float hertz;
    float dampingRatio;
    float maxTorque;
} avbd3d_ParallelJointDef;
AVBD3D_LEGACY_API avbd3d_ParallelJointDef avbd3d_DefaultParallelJointDef(void);

typedef struct avbd3d_PrismaticJointDef
{
    avbd3d_JointDef base;

    bool enableSpring;
    float hertz;
    float dampingRatio;
    float targetTranslation;
    bool enableLimit;
    float lowerTranslation;
    float upperTranslation;
    bool enableMotor;
    float maxMotorForce;

    float motorSpeed;
} avbd3d_PrismaticJointDef;

AVBD3D_LEGACY_API avbd3d_PrismaticJointDef avbd3d_DefaultPrismaticJointDef(void);

typedef struct avbd3d_RevoluteJointDef
{
    avbd3d_JointDef base;

    float targetAngle;
    bool enableSpring;
    float hertz;
    float dampingRatio;
    bool enableLimit;
    float lowerAngle;
    float upperAngle;
    bool enableMotor;
    float maxMotorTorque;

    float motorSpeed;
} avbd3d_RevoluteJointDef;

AVBD3D_LEGACY_API avbd3d_RevoluteJointDef avbd3d_DefaultRevoluteJointDef(void);

typedef struct avbd3d_SphericalJointDef
{
    avbd3d_JointDef base;

    bool enableSpring;
    float hertz;
    float dampingRatio;
    avbd3d_Quat targetRotation;
    bool enableConeLimit;
    float coneAngle;
    bool enableTwistLimit;
    float lowerTwistAngle;
    float upperTwistAngle;
    bool enableMotor;
    float maxMotorTorque;

    avbd3d_Vec3 motorVelocity;
} avbd3d_SphericalJointDef;

AVBD3D_LEGACY_API avbd3d_SphericalJointDef avbd3d_DefaultSphericalJointDef(void);

typedef struct avbd3d_WeldJointDef
{
    avbd3d_JointDef base;

    float linearHertz;
    float angularHertz;
    float linearDampingRatio;

    float angularDampingRatio;
} avbd3d_WeldJointDef;

AVBD3D_LEGACY_API avbd3d_WeldJointDef avbd3d_DefaultWeldJointDef(void);

typedef struct avbd3d_WheelJointDef
{
    avbd3d_JointDef base;

    bool enableSuspensionSpring;
    float suspensionHertz;
    float suspensionDampingRatio;
    bool enableSuspensionLimit;
    float lowerSuspensionLimit;
    float upperSuspensionLimit;
    bool enableSpinMotor;
    float maxSpinTorque;
    float spinSpeed;
    bool enableSteering;
    float steeringHertz;
    float steeringDampingRatio;
    float targetSteeringAngle;
    float maxSteeringTorque;
    bool enableSteeringLimit;
    float lowerSteeringLimit;

    float upperSteeringLimit;
} avbd3d_WheelJointDef;

AVBD3D_LEGACY_API avbd3d_WheelJointDef avbd3d_DefaultWheelJointDef(void);

typedef struct avbd3d_ExplosionDef
{
    uint64_t maskBits;

    avbd3d_Pos position;
    float radius;
    float falloff;

    float impulsePerArea;
} avbd3d_ExplosionDef;

AVBD3D_LEGACY_API avbd3d_ExplosionDef avbd3d_DefaultExplosionDef(void);

typedef struct avbd3d_SensorBeginTouchEvent
{
    avbd3d_ShapeId sensorShapeId;

    avbd3d_ShapeId visitorShapeId;
} avbd3d_SensorBeginTouchEvent;

typedef struct avbd3d_SensorEndTouchEvent
{
    avbd3d_ShapeId sensorShapeId;

    avbd3d_ShapeId visitorShapeId;
} avbd3d_SensorEndTouchEvent;

typedef struct avbd3d_SensorEvents
{
    avbd3d_SensorBeginTouchEvent *beginEvents;

    avbd3d_SensorEndTouchEvent *endEvents;
    int beginCount;

    int endCount;
} avbd3d_SensorEvents;

typedef struct avbd3d_ContactBeginTouchEvent
{
    avbd3d_ShapeId shapeIdA;

    avbd3d_ShapeId shapeIdB;

    avbd3d_ContactId contactId;
} avbd3d_ContactBeginTouchEvent;

typedef struct avbd3d_ContactEndTouchEvent
{
    avbd3d_ShapeId shapeIdA;

    avbd3d_ShapeId shapeIdB;

    avbd3d_ContactId contactId;
} avbd3d_ContactEndTouchEvent;

typedef struct avbd3d_ContactHitEvent
{
    avbd3d_ShapeId shapeIdA;

    avbd3d_ShapeId shapeIdB;
    avbd3d_ContactId contactId;
    avbd3d_Pos point;
    avbd3d_Vec3 normal;
    float approachSpeed;
    uint64_t userMaterialIdA;
    uint64_t userMaterialIdB;
} avbd3d_ContactHitEvent;

typedef struct avbd3d_ContactEvents
{
    avbd3d_ContactBeginTouchEvent *beginEvents;

    avbd3d_ContactEndTouchEvent *endEvents;
    avbd3d_ContactHitEvent *hitEvents;
    int beginCount;
    int endCount;

    int hitCount;
} avbd3d_ContactEvents;

typedef struct avbd3d_BodyMoveEvent
{
    void *userData;

    avbd3d_WorldTransform transform;
    avbd3d_BodyId bodyId;

    int32_t fellAsleep;
} avbd3d_BodyMoveEvent;

typedef struct avbd3d_BodyEvents
{
    avbd3d_BodyMoveEvent *moveEvents;

    int moveCount;
} avbd3d_BodyEvents;

typedef struct avbd3d_JointEvent
{
    avbd3d_JointId jointId;

    void *userData;
} avbd3d_JointEvent;

typedef struct avbd3d_JointEvents
{
    avbd3d_JointEvent *jointEvents;

    int count;
} avbd3d_JointEvents;

typedef struct avbd3d_ContactData
{
    avbd3d_ContactId contactId;

    avbd3d_ShapeId shapeIdA;
    avbd3d_ShapeId shapeIdB;
    const struct avbd3d_Manifold *manifolds;

    int manifoldCount;
} avbd3d_ContactData;

typedef struct avbd3d_QueryFilter
{
    uint64_t categoryBits;

    uint64_t maskBits;
    uint64_t id;

    const char *name;
} avbd3d_QueryFilter;

AVBD3D_LEGACY_API avbd3d_QueryFilter avbd3d_DefaultQueryFilter(void);

typedef struct avbd3d_RayResult
{
    avbd3d_ShapeId shapeId;

    avbd3d_Pos point;
    avbd3d_Vec3 normal;
    uint64_t userMaterialId;
    float fraction;
    int triangleIndex;
    int childIndex;
    int nodeVisits;
    int leafVisits;

    bool hit;
} avbd3d_RayResult;

typedef struct avbd3d_ShapeProxy
{
    const avbd3d_Vec3 *points;

    int count;

    float radius;
} avbd3d_ShapeProxy;

typedef struct avbd3d_CastOutput
{
    avbd3d_Vec3 normal;

    avbd3d_Vec3 point;
    float fraction;
    int iterations;
    int triangleIndex;
    int childIndex;
    int materialIndex;

    bool hit;
} avbd3d_CastOutput;

typedef avbd3d_CastOutput avbd3d_WorldCastOutput;

typedef struct avbd3d_BodyCastResult
{
    avbd3d_ShapeId shapeId;

    avbd3d_Pos point;
    avbd3d_Vec3 normal;
    float fraction;
    int triangleIndex;
    uint64_t userMaterialId;
    int iterations;

    bool hit;
} avbd3d_BodyCastResult;

typedef struct avbd3d_TreeNodeChildren
{
    int child1; ///< child node index 1
    int child2; ///< child node index 2
} avbd3d_TreeNodeChildren;

typedef struct avbd3d_TreeNode
{
    avbd3d_AABB aabb; // 24

    uint64_t categoryBits; // 8

    union
    {
        avbd3d_TreeNodeChildren children;

        uint64_t userData;
    }; // 8

    union
    {
        int parent;

        int next;
    }; // 4

    uint16_t height; // 2

    uint16_t flags; // 2
} avbd3d_TreeNode;

#define AVBD3D_DYNAMIC_TREE_VERSION 0x93EDAF889FD30B4Aull

typedef struct avbd3d_DynamicTree
{
    uint64_t version;

    avbd3d_TreeNode *nodes;
    int root;
    int nodeCount;
    int nodeCapacity;
    int proxyCount;
    int freeList;
    int *leafIndices;
    avbd3d_AABB *leafBoxes;
    avbd3d_Vec3 *leafCenters;
    int *binIndices;

    int rebuildCapacity;
} avbd3d_DynamicTree;

typedef struct avbd3d_TreeStats
{
    int nodeVisits;

    int leafVisits;
} avbd3d_TreeStats;

typedef struct avbd3d_PlaneResult
{
    avbd3d_Plane plane;

    avbd3d_Vec3 point;
    int triangleIndex;
    int childIndex;
    int materialIndex;
} avbd3d_PlaneResult;

typedef struct avbd3d_CollisionPlane
{
    avbd3d_Plane plane;

    float pushLimit;
    float push;

    bool clipVelocity;
} avbd3d_CollisionPlane;

typedef struct avbd3d_PlaneSolverResult
{
    avbd3d_Vec3 delta;

    int iterationCount;
} avbd3d_PlaneSolverResult;

typedef struct avbd3d_BodyPlaneResult
{
    avbd3d_ShapeId shapeId;

    avbd3d_PlaneResult result;
} avbd3d_BodyPlaneResult;

typedef bool avbd3d_PlaneResultFcn(avbd3d_ShapeId shapeId, const avbd3d_PlaneResult *plane, int planeCount,
                                   void *context);
typedef bool avbd3d_MoverFilterFcn(avbd3d_ShapeId shapeId, void *context);

typedef struct avbd3d_MassData
{
    float mass;

    avbd3d_Vec3 center;

    avbd3d_Matrix3 inertia;
} avbd3d_MassData;

typedef struct avbd3d_Sphere
{
    avbd3d_Vec3 center;

    float radius;
} avbd3d_Sphere;

typedef struct avbd3d_Capsule
{
    avbd3d_Vec3 center1;

    avbd3d_Vec3 center2;

    float radius;
} avbd3d_Capsule;

typedef struct avbd3d_HullVertex
{
    uint8_t edge;
} avbd3d_HullVertex;

typedef struct avbd3d_HullHalfEdge
{
    uint8_t next;

    uint8_t twin;
    uint8_t origin;

    uint8_t face;
} avbd3d_HullHalfEdge;

typedef struct avbd3d_HullFace
{
    uint8_t edge;
} avbd3d_HullFace;

#define AVBD3D_HULL_VERSION 0x4A4C9587DE57485Cull

typedef struct avbd3d_HullData
{
    uint64_t version;

    uint64_t hash;
    avbd3d_AABB aabb;
    float surfaceArea;
    float volume;
    float innerRadius;
    avbd3d_Vec3 center;
    avbd3d_Matrix3 centralInertia;
    int32_t vertexCount;
    int32_t vertexOffset;
    int32_t pointOffset;
    int32_t edgeCount;
    int32_t edgeOffset;
    int32_t faceCount;
    int32_t planeOffset;
    int32_t faceOffset;
    int32_t soaVertexOffset;
    int32_t soaNormalOffset;
    int32_t byteCount;
} avbd3d_HullData;

typedef struct avbd3d_BoxHull
{
    avbd3d_HullData base;
    avbd3d_HullVertex boxVertices[8]; ///< Box vertices.
    avbd3d_Vec3 boxPoints[8];         ///< Box points.
    avbd3d_HullHalfEdge boxEdges[24]; ///< Box half-edges.
    avbd3d_Plane boxPlanes[6];        ///< Box face planes.
    avbd3d_HullFace boxFaces[6];      ///< Box faces.
    uint8_t padding[2];               ///< Explicit padding.
    float vx[8];                      ///< vertex x
    float vy[8];                      ///< vertex y
    float vz[8];                      ///< vertex z
    float nx[8];                      ///< normal x, padded to multiple of 4
    float ny[8];                      ///< normal y, padded to multiple of 4
    float nz[8];                      ///< normal z, padded to multiple of 4
} avbd3d_BoxHull;

typedef struct avbd3d_MeshDef
{
    avbd3d_Vec3 *vertices;

    size_t stride;
    int32_t *indices;
    uint8_t *materialIndices;
    float weldTolerance;
    int vertexCount;
    int triangleCount;
    bool weldVertices;
    bool useMedianSplit;
    bool identifyEdges;

    bool clockWiseWinding;
} avbd3d_MeshDef;

#define AVBD3D_MESH_VERSION 0xAAAB9A00F1A8AAF7ull

typedef struct avbd3d_MeshTriangle
{
    int32_t index1; ///< Index of vertex 1.
    int32_t index2; ///< Index of vertex 2.
    int32_t index3; ///< Index of vertex 3.
} avbd3d_MeshTriangle;

typedef struct avbd3d_MeshData
{
    uint64_t version;

    uint64_t hash;
    int32_t byteCount;
    avbd3d_AABB bounds;
    float surfaceArea;
    int32_t treeHeight;
    int32_t degenerateCount;
    int32_t nodeOffset;
    int32_t nodeCount;
    int32_t vertexOffset;
    int32_t vertexCount;
    int32_t triangleOffset;
    int32_t triangleCount;
    int32_t materialOffset;
    int32_t materialCount;
    int32_t flagsOffset;

    int32_t padding;
} avbd3d_MeshData;

typedef struct avbd3d_Mesh
{
    const avbd3d_MeshData *data;

    avbd3d_Vec3 scale;
} avbd3d_Mesh;

typedef struct avbd3d_HeightFieldDef
{
    float *heights;

    uint8_t *materialIndices;
    avbd3d_Vec3 scale;
    int countX;
    int countZ;
    float globalMinimumHeight;
    float globalMaximumHeight;

    bool clockwiseWinding;
} avbd3d_HeightFieldDef;

typedef struct avbd3d_HeightFieldData
{
    uint64_t version;

    uint64_t hash;
    int byteCount;
    avbd3d_AABB aabb;
    float minHeight;
    float maxHeight;
    float heightScale;
    avbd3d_Vec3 scale;
    int32_t columnCount;
    int32_t rowCount;
    int32_t heightsOffset;
    int32_t materialOffset;
    int32_t flagsOffset;
    uint8_t clockwise;

    uint8_t padding[7];
} avbd3d_HeightFieldData;

typedef struct avbd3d_CompoundCapsuleDef
{
    avbd3d_Capsule capsule;

    avbd3d_SurfaceMaterial material;
} avbd3d_CompoundCapsuleDef;

typedef struct avbd3d_CompoundHullDef
{
    const avbd3d_HullData *hull;

    avbd3d_Transform transform;

    avbd3d_SurfaceMaterial material;
} avbd3d_CompoundHullDef;

typedef struct avbd3d_CompoundMeshDef
{
    const avbd3d_MeshData *meshData;

    avbd3d_Transform transform;
    avbd3d_Vec3 scale;
    const avbd3d_SurfaceMaterial *materials;

    int materialCount;
} avbd3d_CompoundMeshDef;

typedef struct avbd3d_CompoundSphereDef
{
    avbd3d_Sphere sphere;

    avbd3d_SurfaceMaterial material;
} avbd3d_CompoundSphereDef;

typedef struct avbd3d_CompoundDef
{
    avbd3d_CompoundCapsuleDef *capsules;

    int capsuleCount;
    avbd3d_CompoundHullDef *hulls;
    int hullCount;
    avbd3d_CompoundMeshDef *meshes;
    int meshCount;
    avbd3d_CompoundSphereDef *spheres;

    int sphereCount;
} avbd3d_CompoundDef;

#define AVBD3D_COMPOUND_VERSION                                                                                        \
    (0xB11DCE70FAD5622Bull ^ AVBD3D_DYNAMIC_TREE_VERSION ^ AVBD3D_MESH_VERSION ^ AVBD3D_HULL_VERSION)
#define AVBD3D_MAX_COMPOUND_MESH_MATERIALS 4

typedef struct avbd3d_CompoundData
{
    uint64_t version;

    int byteCount;
    int nodeOffset;
    avbd3d_DynamicTree tree;
    int materialOffset;
    int materialCount;
    int capsuleOffset;
    int capsuleCount;
    int hullOffset;
    int hullCount;
    int sharedHullCount;
    int meshOffset;
    int meshCount;
    int sharedMeshCount;
    int sphereOffset;

    int sphereCount;
} avbd3d_CompoundData;

typedef struct avbd3d_ChildShape
{
    union
    {
        avbd3d_Capsule capsule;      ///< Capsule.
        const avbd3d_HullData *hull; ///< Hull.
        avbd3d_Mesh mesh;            ///< Mesh.
        avbd3d_Sphere sphere;        ///< Sphere.
    };

    avbd3d_Transform transform;
    int materialIndices[AVBD3D_MAX_COMPOUND_MESH_MATERIALS];

    avbd3d_ShapeType type;
} avbd3d_ChildShape;

typedef struct avbd3d_ManifoldPoint
{
    avbd3d_Vec3 anchorA;

    avbd3d_Vec3 anchorB;
    float separation;
    float baseSeparation;
    float normalImpulse;
    float totalNormalImpulse;
    float normalVelocity;
    uint32_t featureId;
    int triangleIndex;

    int32_t persisted;
} avbd3d_ManifoldPoint;

typedef struct avbd3d_Manifold
{
    avbd3d_ManifoldPoint points[AVBD3D_MAX_MANIFOLD_POINTS];

    avbd3d_Vec3 normal;
    float twistImpulse;
    avbd3d_Vec3 frictionImpulse;
    avbd3d_Vec3 rollingImpulse;
    int pointCount;
} avbd3d_Manifold;

typedef enum avbd3d_HexColor
{
    avbd3d_colorAliceBlue = 0xF0F8FF,
    avbd3d_colorAntiqueWhite = 0xFAEBD7,
    avbd3d_colorAqua = 0x00FFFF,
    avbd3d_colorAquamarine = 0x7FFFD4,
    avbd3d_colorAzure = 0xF0FFFF,
    avbd3d_colorBeige = 0xF5F5DC,
    avbd3d_colorBisque = 0xFFE4C4,
    avbd3d_colorBlack = 0x000000,
    avbd3d_colorBlanchedAlmond = 0xFFEBCD,
    avbd3d_colorBlue = 0x0000FF,
    avbd3d_colorBlueViolet = 0x8A2BE2,
    avbd3d_colorBrown = 0xA52A2A,
    avbd3d_colorBurlywood = 0xDEB887,
    avbd3d_colorCadetBlue = 0x5F9EA0,
    avbd3d_colorChartreuse = 0x7FFF00,
    avbd3d_colorChocolate = 0xD2691E,
    avbd3d_colorCoral = 0xFF7F50,
    avbd3d_colorCornflowerBlue = 0x6495ED,
    avbd3d_colorCornsilk = 0xFFF8DC,
    avbd3d_colorCrimson = 0xDC143C,
    avbd3d_colorCyan = 0x00FFFF,
    avbd3d_colorDarkBlue = 0x00008B,
    avbd3d_colorDarkCyan = 0x008B8B,
    avbd3d_colorDarkGoldenRod = 0xB8860B,
    avbd3d_colorDarkGray = 0xA9A9A9,
    avbd3d_colorDarkGreen = 0x006400,
    avbd3d_colorDarkKhaki = 0xBDB76B,
    avbd3d_colorDarkMagenta = 0x8B008B,
    avbd3d_colorDarkOliveGreen = 0x556B2F,
    avbd3d_colorDarkOrange = 0xFF8C00,
    avbd3d_colorDarkOrchid = 0x9932CC,
    avbd3d_colorDarkRed = 0x8B0000,
    avbd3d_colorDarkSalmon = 0xE9967A,
    avbd3d_colorDarkSeaGreen = 0x8FBC8F,
    avbd3d_colorDarkSlateBlue = 0x483D8B,
    avbd3d_colorDarkSlateGray = 0x2F4F4F,
    avbd3d_colorDarkTurquoise = 0x00CED1,
    avbd3d_colorDarkViolet = 0x9400D3,
    avbd3d_colorDeepPink = 0xFF1493,
    avbd3d_colorDeepSkyBlue = 0x00BFFF,
    avbd3d_colorDimGray = 0x696969,
    avbd3d_colorDodgerBlue = 0x1E90FF,
    avbd3d_colorFireBrick = 0xB22222,
    avbd3d_colorFloralWhite = 0xFFFAF0,
    avbd3d_colorForestGreen = 0x228B22,
    avbd3d_colorFuchsia = 0xFF00FF,
    avbd3d_colorGainsboro = 0xDCDCDC,
    avbd3d_colorGhostWhite = 0xF8F8FF,
    avbd3d_colorGold = 0xFFD700,
    avbd3d_colorGoldenRod = 0xDAA520,
    avbd3d_colorGray = 0x808080,
    avbd3d_colorGreen = 0x008000,
    avbd3d_colorGreenYellow = 0xADFF2F,
    avbd3d_colorHoneyDew = 0xF0FFF0,
    avbd3d_colorHotPink = 0xFF69B4,
    avbd3d_colorIndianRed = 0xCD5C5C,
    avbd3d_colorIndigo = 0x4B0082,
    avbd3d_colorIvory = 0xFFFFF0,
    avbd3d_colorKhaki = 0xF0E68C,
    avbd3d_colorLavender = 0xE6E6FA,
    avbd3d_colorLavenderBlush = 0xFFF0F5,
    avbd3d_colorLawnGreen = 0x7CFC00,
    avbd3d_colorLemonChiffon = 0xFFFACD,
    avbd3d_colorLightBlue = 0xADD8E6,
    avbd3d_colorLightCoral = 0xF08080,
    avbd3d_colorLightCyan = 0xE0FFFF,
    avbd3d_colorLightGoldenRodYellow = 0xFAFAD2,
    avbd3d_colorLightGray = 0xD3D3D3,
    avbd3d_colorLightGreen = 0x90EE90,
    avbd3d_colorLightPink = 0xFFB6C1,
    avbd3d_colorLightSalmon = 0xFFA07A,
    avbd3d_colorLightSeaGreen = 0x20B2AA,
    avbd3d_colorLightSkyBlue = 0x87CEFA,
    avbd3d_colorLightSlateGray = 0x778899,
    avbd3d_colorLightSteelBlue = 0xB0C4DE,
    avbd3d_colorLightYellow = 0xFFFFE0,
    avbd3d_colorLime = 0x00FF00,
    avbd3d_colorLimeGreen = 0x32CD32,
    avbd3d_colorLinen = 0xFAF0E6,
    avbd3d_colorMagenta = 0xFF00FF,
    avbd3d_colorMaroon = 0x800000,
    avbd3d_colorMediumAquaMarine = 0x66CDAA,
    avbd3d_colorMediumBlue = 0x0000CD,
    avbd3d_colorMediumOrchid = 0xBA55D3,
    avbd3d_colorMediumPurple = 0x9370DB,
    avbd3d_colorMediumSeaGreen = 0x3CB371,
    avbd3d_colorMediumSlateBlue = 0x7B68EE,
    avbd3d_colorMediumSpringGreen = 0x00FA9A,
    avbd3d_colorMediumTurquoise = 0x48D1CC,
    avbd3d_colorMediumVioletRed = 0xC71585,
    avbd3d_colorMidnightBlue = 0x191970,
    avbd3d_colorMintCream = 0xF5FFFA,
    avbd3d_colorMistyRose = 0xFFE4E1,
    avbd3d_colorMoccasin = 0xFFE4B5,
    avbd3d_colorNavajoWhite = 0xFFDEAD,
    avbd3d_colorNavy = 0x000080,
    avbd3d_colorOldLace = 0xFDF5E6,
    avbd3d_colorOlive = 0x808000,
    avbd3d_colorOliveDrab = 0x6B8E23,
    avbd3d_colorOrange = 0xFFA500,
    avbd3d_colorOrangeRed = 0xFF4500,
    avbd3d_colorOrchid = 0xDA70D6,
    avbd3d_colorPaleGoldenRod = 0xEEE8AA,
    avbd3d_colorPaleGreen = 0x98FB98,
    avbd3d_colorPaleTurquoise = 0xAFEEEE,
    avbd3d_colorPaleVioletRed = 0xDB7093,
    avbd3d_colorPapayaWhip = 0xFFEFD5,
    avbd3d_colorPeachPuff = 0xFFDAB9,
    avbd3d_colorPeru = 0xCD853F,
    avbd3d_colorPink = 0xFFC0CB,
    avbd3d_colorPlum = 0xDDA0DD,
    avbd3d_colorPowderBlue = 0xB0E0E6,
    avbd3d_colorPurple = 0x800080,
    avbd3d_colorRebeccaPurple = 0x663399,
    avbd3d_colorRed = 0xFF0000,
    avbd3d_colorRosyBrown = 0xBC8F8F,
    avbd3d_colorRoyalBlue = 0x4169E1,
    avbd3d_colorSaddleBrown = 0x8B4513,
    avbd3d_colorSalmon = 0xFA8072,
    avbd3d_colorSandyBrown = 0xF4A460,
    avbd3d_colorSeaGreen = 0x2E8B57,
    avbd3d_colorSeaShell = 0xFFF5EE,
    avbd3d_colorSienna = 0xA0522D,
    avbd3d_colorSilver = 0xC0C0C0,
    avbd3d_colorSkyBlue = 0x87CEEB,
    avbd3d_colorSlateBlue = 0x6A5ACD,
    avbd3d_colorSlateGray = 0x708090,
    avbd3d_colorSnow = 0xFFFAFA,
    avbd3d_colorSpringGreen = 0x00FF7F,
    avbd3d_colorSteelBlue = 0x4682B4,
    avbd3d_colorTan = 0xD2B48C,
    avbd3d_colorTeal = 0x008080,
    avbd3d_colorThistle = 0xD8BFD8,
    avbd3d_colorTomato = 0xFF6347,
    avbd3d_colorTurquoise = 0x40E0D0,
    avbd3d_colorViolet = 0xEE82EE,
    avbd3d_colorWheat = 0xF5DEB3,
    avbd3d_colorWhite = 0xFFFFFF,
    avbd3d_colorWhiteSmoke = 0xF5F5F5,
    avbd3d_colorYellow = 0xFFFF00,
    avbd3d_colorYellowGreen = 0x9ACD32,

    avbd3d_colorBox2DRed = 0xDC3132,
    avbd3d_colorBox2DBlue = 0x30AEBF,
    avbd3d_colorBox2DGreen = 0x8CC924,
    avbd3d_colorBox2DYellow = 0xFFEE8C
} avbd3d_HexColor;

typedef struct avbd3d_DebugShape
{
    avbd3d_ShapeId shapeId;

    avbd3d_ShapeType type;

    union
    {
        const avbd3d_Capsule *capsule;             ///< Capsule shape.
        const avbd3d_CompoundData *compound;       ///< Compound shape.
        const avbd3d_HeightFieldData *heightField; ///< Height-field shape.
        const avbd3d_HullData *hull;               ///< Convex hull shape.
        const avbd3d_Mesh *mesh;                   ///< Mesh shape with scale.
        const avbd3d_Sphere *sphere;               ///< Sphere shape.
    };
} avbd3d_DebugShape;

typedef struct avbd3d_DebugDraw
{
    void (*DrawShapeFcn)(void *userShape, avbd3d_WorldTransform transform, avbd3d_HexColor color, void *context);

    void (*DrawSegmentFcn)(avbd3d_Pos p1, avbd3d_Pos p2, avbd3d_HexColor color, void *context);
    void (*DrawTransformFcn)(avbd3d_WorldTransform transform, void *context);
    void (*DrawPointFcn)(avbd3d_Pos p, float size, avbd3d_HexColor color, void *context);
    void (*DrawSphereFcn)(avbd3d_Pos p, float radius, avbd3d_HexColor color, float alpha, void *context);
    void (*DrawCapsuleFcn)(avbd3d_Pos p1, avbd3d_Pos p2, float radius, avbd3d_HexColor color, float alpha,
                           void *context);
    void (*DrawBoundsFcn)(avbd3d_AABB aabb, avbd3d_HexColor color, void *context);
    void (*DrawBoxFcn)(avbd3d_Vec3 extents, avbd3d_WorldTransform transform, avbd3d_HexColor color, void *context);
    void (*DrawStringFcn)(avbd3d_Pos p, const char *s, avbd3d_HexColor color, void *context);
    avbd3d_AABB drawingBounds;
    float forceScale;
    float jointScale;
    bool drawShapes;
    bool drawJoints;
    bool drawJointExtras;
    bool drawBounds;
    bool drawMass;
    bool drawSleep;
    bool drawBodyNames;
    bool drawContacts;
    bool drawAnchorA;
    bool drawGraphColors;
    bool drawContactFeatures;
    bool drawContactNormals;
    bool drawContactForces;
    bool drawIslands;

    void *context;
} avbd3d_DebugDraw;

AVBD3D_LEGACY_API avbd3d_DebugDraw avbd3d_DefaultDebugDraw(void);
AVBD3D_LEGACY_API avbd3d_HullData *avbd3d_CreateHull(const avbd3d_Vec3 *points, int pointCount, int maxVertexCount);
AVBD3D_LEGACY_API void avbd3d_DestroyHull(avbd3d_HullData *hull);
AVBD3D_LEGACY_API avbd3d_BoxHull avbd3d_MakeBoxHull(float hx, float hy, float hz);

AVBD3D_INLINE const avbd3d_Vec3 *avbd3d_GetMeshVertices(const avbd3d_MeshData *mesh)
{
    if (mesh->vertexOffset == 0)
    {
        return NULL;
    }

    return (const avbd3d_Vec3 *)((intptr_t)mesh + mesh->vertexOffset);
}

AVBD3D_INLINE const avbd3d_MeshTriangle *avbd3d_GetMeshTriangles(const avbd3d_MeshData *mesh)
{
    if (mesh->triangleOffset == 0)
    {
        return NULL;
    }

    return (const avbd3d_MeshTriangle *)((intptr_t)mesh + mesh->triangleOffset);
}

AVBD3D_LEGACY_API avbd3d_MeshData *avbd3d_CreateMesh(const avbd3d_MeshDef *def, int *degenerateTriangleIndices,
                                                     int degenerateCapacity);
AVBD3D_LEGACY_API void avbd3d_DestroyMesh(avbd3d_MeshData *mesh);
AVBD3D_LEGACY_API avbd3d_HeightFieldData *avbd3d_CreateHeightField(const avbd3d_HeightFieldDef *data);
AVBD3D_LEGACY_API void avbd3d_DestroyHeightField(avbd3d_HeightFieldData *heightField);
AVBD3D_LEGACY_API avbd3d_ChildShape avbd3d_GetCompoundChild(const avbd3d_CompoundData *compound, int childIndex);
AVBD3D_LEGACY_API const avbd3d_SurfaceMaterial *avbd3d_GetCompoundMaterials(const avbd3d_CompoundData *compound);
AVBD3D_LEGACY_API avbd3d_CompoundData *avbd3d_CreateCompound(const avbd3d_CompoundDef *def);
AVBD3D_LEGACY_API void avbd3d_DestroyCompound(avbd3d_CompoundData *compound);
AVBD3D_LEGACY_API avbd3d_PlaneSolverResult avbd3d_SolvePlanes(avbd3d_Vec3 targetDelta, avbd3d_CollisionPlane *planes,
                                                              int count);
AVBD3D_LEGACY_API avbd3d_Vec3 avbd3d_ClipVector(avbd3d_Vec3 vector, const avbd3d_CollisionPlane *planes, int count);
AVBD3D_LEGACY_API avbd3d_WorldId avbd3d_CreateWorld(const avbd3d_WorldDef *def, avbd3d_Error *outErr);
AVBD3D_LEGACY_API void avbd3d_DestroyWorld(avbd3d_WorldId worldId);
AVBD3D_LEGACY_API void avbd3d_World_Step(avbd3d_WorldId worldId, float timeStep);
AVBD3D_LEGACY_API void avbd3d_World_Draw(avbd3d_WorldId worldId, avbd3d_DebugDraw *draw, uint64_t maskBits);
AVBD3D_LEGACY_API avbd3d_AABB avbd3d_World_GetBounds(avbd3d_WorldId worldId);
AVBD3D_LEGACY_API avbd3d_BodyEvents avbd3d_World_GetBodyEvents(avbd3d_WorldId worldId);
AVBD3D_LEGACY_API avbd3d_SensorEvents avbd3d_World_GetSensorEvents(avbd3d_WorldId worldId);
AVBD3D_LEGACY_API avbd3d_ContactEvents avbd3d_World_GetContactEvents(avbd3d_WorldId worldId);
AVBD3D_LEGACY_API avbd3d_JointEvents avbd3d_World_GetJointEvents(avbd3d_WorldId worldId);

AVBD3D_LEGACY_API avbd3d_TreeStats avbd3d_World_OverlapAABB(avbd3d_WorldId worldId, avbd3d_AABB aabb,
                                                            avbd3d_QueryFilter filter, avbd3d_OverlapResultFcn *fcn,
                                                            void *context);

AVBD3D_LEGACY_API avbd3d_TreeStats avbd3d_World_OverlapShape(avbd3d_WorldId worldId, avbd3d_Pos origin,
                                                             const avbd3d_ShapeProxy *proxy, avbd3d_QueryFilter filter,
                                                             avbd3d_OverlapResultFcn *fcn, void *context);

AVBD3D_LEGACY_API avbd3d_TreeStats avbd3d_World_CastRay(avbd3d_WorldId worldId, avbd3d_Pos origin,
                                                        avbd3d_Vec3 translation, avbd3d_QueryFilter filter,
                                                        avbd3d_CastResultFcn *fcn, void *context);

AVBD3D_LEGACY_API avbd3d_RayResult avbd3d_World_CastRayClosest(avbd3d_WorldId worldId, avbd3d_Pos origin,
                                                               avbd3d_Vec3 translation, avbd3d_QueryFilter filter);

AVBD3D_LEGACY_API avbd3d_TreeStats avbd3d_World_CastShape(avbd3d_WorldId worldId, avbd3d_Pos origin,
                                                          const avbd3d_ShapeProxy *proxy, avbd3d_Vec3 translation,
                                                          avbd3d_QueryFilter filter, avbd3d_CastResultFcn *fcn,
                                                          void *context);

AVBD3D_LEGACY_API float avbd3d_World_CastMover(avbd3d_WorldId worldId, avbd3d_Pos origin, const avbd3d_Capsule *mover,
                                               avbd3d_Vec3 translation, avbd3d_QueryFilter filter,
                                               avbd3d_MoverFilterFcn *fcn, void *context);

AVBD3D_LEGACY_API void avbd3d_World_CollideMover(avbd3d_WorldId worldId, avbd3d_Pos origin, const avbd3d_Capsule *mover,
                                                 avbd3d_QueryFilter filter, avbd3d_PlaneResultFcn *fcn, void *context);

AVBD3D_LEGACY_API void avbd3d_World_EnableSleeping(avbd3d_WorldId worldId, bool flag);
AVBD3D_LEGACY_API void avbd3d_World_EnableContinuous(avbd3d_WorldId worldId, bool flag);
AVBD3D_LEGACY_API void avbd3d_World_SetRestitutionThreshold(avbd3d_WorldId worldId, float value);
AVBD3D_LEGACY_API void avbd3d_World_SetHitEventThreshold(avbd3d_WorldId worldId, float value);
AVBD3D_LEGACY_API void avbd3d_World_SetGravity(avbd3d_WorldId worldId, avbd3d_Vec3 gravity);
AVBD3D_LEGACY_API avbd3d_Vec3 avbd3d_World_GetGravity(avbd3d_WorldId worldId);
AVBD3D_LEGACY_API void avbd3d_World_Explode(avbd3d_WorldId worldId, const avbd3d_ExplosionDef *explosionDef);
AVBD3D_LEGACY_API avbd3d_BodyId avbd3d_CreateBody(avbd3d_WorldId worldId, const avbd3d_BodyDef *def);
AVBD3D_LEGACY_API void avbd3d_DestroyBody(avbd3d_BodyId bodyId);
AVBD3D_LEGACY_API avbd3d_BodyType avbd3d_Body_GetType(avbd3d_BodyId bodyId);
AVBD3D_LEGACY_API void avbd3d_Body_SetType(avbd3d_BodyId bodyId, avbd3d_BodyType type);
AVBD3D_LEGACY_API void avbd3d_Body_SetUserData(avbd3d_BodyId bodyId, void *userData);
AVBD3D_LEGACY_API void *avbd3d_Body_GetUserData(avbd3d_BodyId bodyId);
AVBD3D_LEGACY_API avbd3d_Pos avbd3d_Body_GetPosition(avbd3d_BodyId bodyId);
AVBD3D_LEGACY_API avbd3d_Quat avbd3d_Body_GetRotation(avbd3d_BodyId bodyId);
AVBD3D_LEGACY_API avbd3d_WorldTransform avbd3d_Body_GetTransform(avbd3d_BodyId bodyId);
AVBD3D_LEGACY_API void avbd3d_Body_SetTransform(avbd3d_BodyId bodyId, avbd3d_Pos position, avbd3d_Quat rotation);
AVBD3D_LEGACY_API avbd3d_Vec3 avbd3d_Body_GetLocalPoint(avbd3d_BodyId bodyId, avbd3d_Pos worldPoint);
AVBD3D_LEGACY_API avbd3d_Pos avbd3d_Body_GetWorldPoint(avbd3d_BodyId bodyId, avbd3d_Vec3 localPoint);
AVBD3D_LEGACY_API avbd3d_Vec3 avbd3d_Body_GetLocalVector(avbd3d_BodyId bodyId, avbd3d_Vec3 worldVector);
AVBD3D_LEGACY_API avbd3d_Vec3 avbd3d_Body_GetWorldVector(avbd3d_BodyId bodyId, avbd3d_Vec3 localVector);
AVBD3D_LEGACY_API avbd3d_Vec3 avbd3d_Body_GetLinearVelocity(avbd3d_BodyId bodyId);
AVBD3D_LEGACY_API avbd3d_Vec3 avbd3d_Body_GetAngularVelocity(avbd3d_BodyId bodyId);
AVBD3D_LEGACY_API void avbd3d_Body_SetLinearVelocity(avbd3d_BodyId bodyId, avbd3d_Vec3 linearVelocity);
AVBD3D_LEGACY_API void avbd3d_Body_SetAngularVelocity(avbd3d_BodyId bodyId, avbd3d_Vec3 angularVelocity);
AVBD3D_LEGACY_API void avbd3d_Body_SetTargetTransform(avbd3d_BodyId bodyId, avbd3d_WorldTransform target,
                                                      float timeStep, bool wake);
AVBD3D_LEGACY_API avbd3d_Vec3 avbd3d_Body_GetLocalPointVelocity(avbd3d_BodyId bodyId, avbd3d_Vec3 localPoint);
AVBD3D_LEGACY_API avbd3d_Vec3 avbd3d_Body_GetWorldPointVelocity(avbd3d_BodyId bodyId, avbd3d_Pos worldPoint);
AVBD3D_LEGACY_API void avbd3d_Body_ApplyForce(avbd3d_BodyId bodyId, avbd3d_Vec3 force, avbd3d_Pos point, bool wake);
AVBD3D_LEGACY_API void avbd3d_Body_ApplyForceToCenter(avbd3d_BodyId bodyId, avbd3d_Vec3 force, bool wake);
AVBD3D_LEGACY_API void avbd3d_Body_ApplyTorque(avbd3d_BodyId bodyId, avbd3d_Vec3 torque, bool wake);
AVBD3D_LEGACY_API void avbd3d_Body_ApplyLinearImpulse(avbd3d_BodyId bodyId, avbd3d_Vec3 impulse, avbd3d_Pos point,
                                                      bool wake);
AVBD3D_LEGACY_API void avbd3d_Body_ApplyLinearImpulseToCenter(avbd3d_BodyId bodyId, avbd3d_Vec3 impulse, bool wake);
AVBD3D_LEGACY_API void avbd3d_Body_ApplyAngularImpulse(avbd3d_BodyId bodyId, avbd3d_Vec3 impulse, bool wake);
AVBD3D_LEGACY_API float avbd3d_Body_GetMass(avbd3d_BodyId bodyId);
AVBD3D_LEGACY_API avbd3d_Matrix3 avbd3d_Body_GetLocalRotationalInertia(avbd3d_BodyId bodyId);
AVBD3D_LEGACY_API float avbd3d_Body_GetInverseMass(avbd3d_BodyId bodyId);
AVBD3D_LEGACY_API avbd3d_Matrix3 avbd3d_Body_GetWorldInverseRotationalInertia(avbd3d_BodyId bodyId);
AVBD3D_LEGACY_API avbd3d_Vec3 avbd3d_Body_GetLocalCenter(avbd3d_BodyId bodyId);
AVBD3D_LEGACY_API avbd3d_Pos avbd3d_Body_GetWorldCenter(avbd3d_BodyId bodyId);
AVBD3D_LEGACY_API void avbd3d_Body_SetMassData(avbd3d_BodyId bodyId, avbd3d_MassData massData);
AVBD3D_LEGACY_API avbd3d_MassData avbd3d_Body_GetMassData(avbd3d_BodyId bodyId);
AVBD3D_LEGACY_API void avbd3d_Body_ApplyMassFromShapes(avbd3d_BodyId bodyId);
AVBD3D_LEGACY_API void avbd3d_Body_SetLinearDamping(avbd3d_BodyId bodyId, float linearDamping);
AVBD3D_LEGACY_API void avbd3d_Body_SetAngularDamping(avbd3d_BodyId bodyId, float angularDamping);
AVBD3D_LEGACY_API void avbd3d_Body_SetGravityScale(avbd3d_BodyId bodyId, float gravityScale);
AVBD3D_LEGACY_API bool avbd3d_Body_IsAwake(avbd3d_BodyId bodyId);
AVBD3D_LEGACY_API void avbd3d_Body_SetAwake(avbd3d_BodyId bodyId, bool awake);
AVBD3D_LEGACY_API bool avbd3d_Body_IsEnabled(avbd3d_BodyId bodyId);
AVBD3D_LEGACY_API void avbd3d_Body_Disable(avbd3d_BodyId bodyId);
AVBD3D_LEGACY_API void avbd3d_Body_Enable(avbd3d_BodyId bodyId);
AVBD3D_LEGACY_API void avbd3d_Body_SetMotionLocks(avbd3d_BodyId bodyId, avbd3d_MotionLocks locks);
AVBD3D_LEGACY_API void avbd3d_Body_SetBullet(avbd3d_BodyId bodyId, bool flag);
AVBD3D_LEGACY_API void avbd3d_Body_EnableHitEvents(avbd3d_BodyId bodyId, bool flag);
AVBD3D_LEGACY_API avbd3d_WorldId avbd3d_Body_GetWorld(avbd3d_BodyId bodyId);
AVBD3D_LEGACY_API int avbd3d_Body_GetShapeCount(avbd3d_BodyId bodyId);
AVBD3D_LEGACY_API int avbd3d_Body_GetShapes(avbd3d_BodyId bodyId, avbd3d_ShapeId *shapeArray, int capacity);
AVBD3D_LEGACY_API int avbd3d_Body_GetJointCount(avbd3d_BodyId bodyId);
AVBD3D_LEGACY_API int avbd3d_Body_GetJoints(avbd3d_BodyId bodyId, avbd3d_JointId *jointArray, int capacity);
AVBD3D_LEGACY_API int avbd3d_Body_GetContactCapacity(avbd3d_BodyId bodyId);
AVBD3D_LEGACY_API int avbd3d_Body_GetContactData(avbd3d_BodyId bodyId, avbd3d_ContactData *contactData, int capacity);
AVBD3D_LEGACY_API avbd3d_AABB avbd3d_Body_ComputeAABB(avbd3d_BodyId bodyId);
AVBD3D_LEGACY_API float avbd3d_Body_GetClosestPoint(avbd3d_BodyId bodyId, avbd3d_Vec3 *result, avbd3d_Vec3 target);

AVBD3D_LEGACY_API avbd3d_BodyCastResult avbd3d_Body_CastRay(avbd3d_BodyId bodyId, avbd3d_Pos origin,
                                                            avbd3d_Vec3 translation, avbd3d_QueryFilter filter,
                                                            float maxFraction, avbd3d_WorldTransform bodyTransform);

AVBD3D_LEGACY_API avbd3d_BodyCastResult avbd3d_Body_CastShape(avbd3d_BodyId bodyId, avbd3d_Pos origin,
                                                              const avbd3d_ShapeProxy *proxy, avbd3d_Vec3 translation,
                                                              avbd3d_QueryFilter filter, float maxFraction,
                                                              bool canEncroach, avbd3d_WorldTransform bodyTransform);

AVBD3D_LEGACY_API bool avbd3d_Body_OverlapShape(avbd3d_BodyId bodyId, avbd3d_Pos origin, const avbd3d_ShapeProxy *proxy,
                                                avbd3d_QueryFilter filter, avbd3d_WorldTransform bodyTransform);

AVBD3D_LEGACY_API int avbd3d_Body_CollideMover(avbd3d_BodyId bodyId, avbd3d_BodyPlaneResult *bodyPlanes,
                                               int planeCapacity, avbd3d_Pos origin, const avbd3d_Capsule *mover,
                                               avbd3d_QueryFilter filter, avbd3d_WorldTransform bodyTransform);

AVBD3D_LEGACY_API avbd3d_ShapeId avbd3d_CreateSphereShape(avbd3d_BodyId bodyId, const avbd3d_ShapeDef *def,
                                                          const avbd3d_Sphere *sphere);
AVBD3D_LEGACY_API avbd3d_ShapeId avbd3d_CreateCapsuleShape(avbd3d_BodyId bodyId, const avbd3d_ShapeDef *def,
                                                           const avbd3d_Capsule *capsule);
AVBD3D_LEGACY_API avbd3d_ShapeId avbd3d_CreateHullShape(avbd3d_BodyId bodyId, const avbd3d_ShapeDef *def,
                                                        const avbd3d_HullData *hull);

AVBD3D_LEGACY_API avbd3d_ShapeId avbd3d_CreateTransformedHullShape(avbd3d_BodyId bodyId, const avbd3d_ShapeDef *def,
                                                                   const avbd3d_HullData *hull,
                                                                   avbd3d_Transform transform, avbd3d_Vec3 scale);

AVBD3D_LEGACY_API avbd3d_ShapeId avbd3d_CreateMeshShape(avbd3d_BodyId bodyId, const avbd3d_ShapeDef *def,
                                                        const avbd3d_MeshData *mesh, avbd3d_Vec3 scale);
AVBD3D_LEGACY_API avbd3d_ShapeId avbd3d_CreateHeightFieldShape(avbd3d_BodyId bodyId, const avbd3d_ShapeDef *def,
                                                               const avbd3d_HeightFieldData *heightField);
AVBD3D_LEGACY_API avbd3d_ShapeId avbd3d_CreateBakedCompoundShape(avbd3d_BodyId bodyId, avbd3d_ShapeDef *def,
                                                                 const avbd3d_CompoundData *compound);
AVBD3D_LEGACY_API void avbd3d_DestroyShape(avbd3d_ShapeId shapeId, bool updateBodyMass);
AVBD3D_LEGACY_API avbd3d_ShapeType avbd3d_Shape_GetType(avbd3d_ShapeId shapeId);
AVBD3D_LEGACY_API avbd3d_BodyId avbd3d_Shape_GetBody(avbd3d_ShapeId shapeId);
AVBD3D_LEGACY_API avbd3d_WorldId avbd3d_Shape_GetWorld(avbd3d_ShapeId shapeId);
AVBD3D_LEGACY_API bool avbd3d_Shape_IsSensor(avbd3d_ShapeId shapeId);
AVBD3D_LEGACY_API void avbd3d_Shape_SetUserData(avbd3d_ShapeId shapeId, void *userData);
AVBD3D_LEGACY_API void *avbd3d_Shape_GetUserData(avbd3d_ShapeId shapeId);
AVBD3D_LEGACY_API void avbd3d_Shape_SetDensity(avbd3d_ShapeId shapeId, float density, bool updateBodyMass);
AVBD3D_LEGACY_API void avbd3d_Shape_SetFriction(avbd3d_ShapeId shapeId, float friction);
AVBD3D_LEGACY_API void avbd3d_Shape_SetRestitution(avbd3d_ShapeId shapeId, float restitution);
AVBD3D_LEGACY_API void avbd3d_Shape_SetSurfaceMaterial(avbd3d_ShapeId shapeId, avbd3d_SurfaceMaterial surfaceMaterial);
AVBD3D_LEGACY_API void avbd3d_Shape_SetFilter(avbd3d_ShapeId shapeId, avbd3d_Filter filter);
AVBD3D_LEGACY_API void avbd3d_Shape_EnableSensorEvents(avbd3d_ShapeId shapeId, bool flag);
AVBD3D_LEGACY_API void avbd3d_Shape_EnableContactEvents(avbd3d_ShapeId shapeId, bool flag);
AVBD3D_LEGACY_API void avbd3d_Shape_EnableHitEvents(avbd3d_ShapeId shapeId, bool flag);
AVBD3D_LEGACY_API avbd3d_WorldCastOutput avbd3d_Shape_RayCast(avbd3d_ShapeId shapeId, avbd3d_Pos origin,
                                                              avbd3d_Vec3 translation);
AVBD3D_LEGACY_API int avbd3d_Shape_GetContactCapacity(avbd3d_ShapeId shapeId);
AVBD3D_LEGACY_API int avbd3d_Shape_GetContactData(avbd3d_ShapeId shapeId, avbd3d_ContactData *contactData,
                                                  int capacity);
AVBD3D_LEGACY_API int avbd3d_Shape_GetSensorCapacity(avbd3d_ShapeId shapeId);
AVBD3D_LEGACY_API int avbd3d_Shape_GetSensorData(avbd3d_ShapeId shapeId, avbd3d_ShapeId *visitorIds, int capacity);
AVBD3D_LEGACY_API avbd3d_AABB avbd3d_Shape_GetAABB(avbd3d_ShapeId shapeId);
AVBD3D_LEGACY_API avbd3d_MassData avbd3d_Shape_ComputeMassData(avbd3d_ShapeId shapeId);
AVBD3D_LEGACY_API void avbd3d_DestroyJoint(avbd3d_JointId jointId, bool wakeAttached);
AVBD3D_LEGACY_API avbd3d_JointType avbd3d_Joint_GetType(avbd3d_JointId jointId);
AVBD3D_LEGACY_API avbd3d_BodyId avbd3d_Joint_GetBodyA(avbd3d_JointId jointId);
AVBD3D_LEGACY_API avbd3d_BodyId avbd3d_Joint_GetBodyB(avbd3d_JointId jointId);
AVBD3D_LEGACY_API avbd3d_WorldId avbd3d_Joint_GetWorld(avbd3d_JointId jointId);
AVBD3D_LEGACY_API void avbd3d_Joint_SetLocalFrameA(avbd3d_JointId jointId, avbd3d_Transform localFrame);
AVBD3D_LEGACY_API void avbd3d_Joint_SetLocalFrameB(avbd3d_JointId jointId, avbd3d_Transform localFrame);
AVBD3D_LEGACY_API void avbd3d_Joint_SetCollideConnected(avbd3d_JointId jointId, bool shouldCollide);
AVBD3D_LEGACY_API void avbd3d_Joint_SetUserData(avbd3d_JointId jointId, void *userData);
AVBD3D_LEGACY_API void *avbd3d_Joint_GetUserData(avbd3d_JointId jointId);
AVBD3D_LEGACY_API void avbd3d_Joint_WakeBodies(avbd3d_JointId jointId);
AVBD3D_LEGACY_API avbd3d_Vec3 avbd3d_Joint_GetConstraintForce(avbd3d_JointId jointId);
AVBD3D_LEGACY_API avbd3d_Vec3 avbd3d_Joint_GetConstraintTorque(avbd3d_JointId jointId);
AVBD3D_LEGACY_API void avbd3d_Joint_SetForceThreshold(avbd3d_JointId jointId, float threshold);
AVBD3D_LEGACY_API void avbd3d_Joint_SetTorqueThreshold(avbd3d_JointId jointId, float threshold);
AVBD3D_LEGACY_API avbd3d_JointId avbd3d_CreateParallelJoint(avbd3d_WorldId worldId, const avbd3d_ParallelJointDef *def);
AVBD3D_LEGACY_API void avbd3d_ParallelJoint_SetSpringHertz(avbd3d_JointId jointId, float hertz);
AVBD3D_LEGACY_API void avbd3d_ParallelJoint_SetSpringDampingRatio(avbd3d_JointId jointId, float dampingRatio);
AVBD3D_LEGACY_API void avbd3d_ParallelJoint_SetMaxTorque(avbd3d_JointId jointId, float force);
AVBD3D_LEGACY_API avbd3d_JointId avbd3d_CreateDistanceJoint(avbd3d_WorldId worldId, const avbd3d_DistanceJointDef *def);
AVBD3D_LEGACY_API void avbd3d_DistanceJoint_SetLength(avbd3d_JointId jointId, float length);
AVBD3D_LEGACY_API void avbd3d_DistanceJoint_EnableSpring(avbd3d_JointId jointId, bool enableSpring);
AVBD3D_LEGACY_API void avbd3d_DistanceJoint_SetSpringForceRange(avbd3d_JointId jointId, float lowerForce,
                                                                float upperForce);
AVBD3D_LEGACY_API void avbd3d_DistanceJoint_SetSpringHertz(avbd3d_JointId jointId, float hertz);
AVBD3D_LEGACY_API void avbd3d_DistanceJoint_SetSpringDampingRatio(avbd3d_JointId jointId, float dampingRatio);
AVBD3D_LEGACY_API void avbd3d_DistanceJoint_EnableLimit(avbd3d_JointId jointId, bool enableLimit);
AVBD3D_LEGACY_API void avbd3d_DistanceJoint_SetLengthRange(avbd3d_JointId jointId, float minLength, float maxLength);
AVBD3D_LEGACY_API void avbd3d_DistanceJoint_EnableMotor(avbd3d_JointId jointId, bool enableMotor);
AVBD3D_LEGACY_API void avbd3d_DistanceJoint_SetMotorSpeed(avbd3d_JointId jointId, float motorSpeed);
AVBD3D_LEGACY_API void avbd3d_DistanceJoint_SetMaxMotorForce(avbd3d_JointId jointId, float force);
AVBD3D_LEGACY_API float avbd3d_DistanceJoint_GetCurrentLength(avbd3d_JointId jointId);
AVBD3D_LEGACY_API float avbd3d_DistanceJoint_GetMotorForce(avbd3d_JointId jointId);
AVBD3D_LEGACY_API avbd3d_JointId avbd3d_CreateMotorJoint(avbd3d_WorldId worldId, const avbd3d_MotorJointDef *def);
AVBD3D_LEGACY_API void avbd3d_MotorJoint_SetLinearVelocity(avbd3d_JointId jointId, avbd3d_Vec3 velocity);
AVBD3D_LEGACY_API void avbd3d_MotorJoint_SetAngularVelocity(avbd3d_JointId jointId, avbd3d_Vec3 velocity);
AVBD3D_LEGACY_API void avbd3d_MotorJoint_SetMaxVelocityForce(avbd3d_JointId jointId, float maxForce);
AVBD3D_LEGACY_API void avbd3d_MotorJoint_SetMaxVelocityTorque(avbd3d_JointId jointId, float maxTorque);
AVBD3D_LEGACY_API void avbd3d_MotorJoint_SetLinearHertz(avbd3d_JointId jointId, float hertz);
AVBD3D_LEGACY_API void avbd3d_MotorJoint_SetLinearDampingRatio(avbd3d_JointId jointId, float damping);
AVBD3D_LEGACY_API void avbd3d_MotorJoint_SetAngularHertz(avbd3d_JointId jointId, float hertz);
AVBD3D_LEGACY_API void avbd3d_MotorJoint_SetAngularDampingRatio(avbd3d_JointId jointId, float damping);
AVBD3D_LEGACY_API void avbd3d_MotorJoint_SetMaxSpringForce(avbd3d_JointId jointId, float maxForce);
AVBD3D_LEGACY_API void avbd3d_MotorJoint_SetMaxSpringTorque(avbd3d_JointId jointId, float maxTorque);
AVBD3D_LEGACY_API avbd3d_JointId avbd3d_CreateFilterJoint(avbd3d_WorldId worldId, const avbd3d_FilterJointDef *def);
AVBD3D_LEGACY_API avbd3d_JointId avbd3d_CreatePrismaticJoint(avbd3d_WorldId worldId,
                                                             const avbd3d_PrismaticJointDef *def);
AVBD3D_LEGACY_API void avbd3d_PrismaticJoint_EnableSpring(avbd3d_JointId jointId, bool enableSpring);
AVBD3D_LEGACY_API void avbd3d_PrismaticJoint_SetSpringHertz(avbd3d_JointId jointId, float hertz);
AVBD3D_LEGACY_API void avbd3d_PrismaticJoint_SetSpringDampingRatio(avbd3d_JointId jointId, float dampingRatio);
AVBD3D_LEGACY_API void avbd3d_PrismaticJoint_SetTargetTranslation(avbd3d_JointId jointId, float targetTranslation);
AVBD3D_LEGACY_API void avbd3d_PrismaticJoint_EnableLimit(avbd3d_JointId jointId, bool enableLimit);
AVBD3D_LEGACY_API void avbd3d_PrismaticJoint_SetLimits(avbd3d_JointId jointId, float lower, float upper);
AVBD3D_LEGACY_API void avbd3d_PrismaticJoint_EnableMotor(avbd3d_JointId jointId, bool enableMotor);
AVBD3D_LEGACY_API void avbd3d_PrismaticJoint_SetMotorSpeed(avbd3d_JointId jointId, float motorSpeed);
AVBD3D_LEGACY_API void avbd3d_PrismaticJoint_SetMaxMotorForce(avbd3d_JointId jointId, float force);
AVBD3D_LEGACY_API float avbd3d_PrismaticJoint_GetMotorForce(avbd3d_JointId jointId);
AVBD3D_LEGACY_API float avbd3d_PrismaticJoint_GetTranslation(avbd3d_JointId jointId);
AVBD3D_LEGACY_API float avbd3d_PrismaticJoint_GetSpeed(avbd3d_JointId jointId);
AVBD3D_LEGACY_API avbd3d_JointId avbd3d_CreateRevoluteJoint(avbd3d_WorldId worldId, const avbd3d_RevoluteJointDef *def);
AVBD3D_LEGACY_API void avbd3d_RevoluteJoint_EnableSpring(avbd3d_JointId jointId, bool enableSpring);
AVBD3D_LEGACY_API void avbd3d_RevoluteJoint_SetSpringHertz(avbd3d_JointId jointId, float hertz);
AVBD3D_LEGACY_API void avbd3d_RevoluteJoint_SetSpringDampingRatio(avbd3d_JointId jointId, float dampingRatio);
AVBD3D_LEGACY_API void avbd3d_RevoluteJoint_SetTargetAngle(avbd3d_JointId jointId, float targetRadians);
AVBD3D_LEGACY_API void avbd3d_RevoluteJoint_EnableLimit(avbd3d_JointId jointId, bool enableLimit);
AVBD3D_LEGACY_API void avbd3d_RevoluteJoint_SetLimits(avbd3d_JointId jointId, float lowerLimitRadians,
                                                      float upperLimitRadians);
AVBD3D_LEGACY_API void avbd3d_RevoluteJoint_EnableMotor(avbd3d_JointId jointId, bool enableMotor);
AVBD3D_LEGACY_API void avbd3d_RevoluteJoint_SetMotorSpeed(avbd3d_JointId jointId, float motorSpeed);
AVBD3D_LEGACY_API void avbd3d_RevoluteJoint_SetMaxMotorTorque(avbd3d_JointId jointId, float torque);
AVBD3D_LEGACY_API float avbd3d_RevoluteJoint_GetMotorTorque(avbd3d_JointId jointId);
AVBD3D_LEGACY_API float avbd3d_RevoluteJoint_GetAngle(avbd3d_JointId jointId);
AVBD3D_LEGACY_API float avbd3d_RevoluteJoint_GetSpeed(avbd3d_JointId jointId);
AVBD3D_LEGACY_API avbd3d_JointId avbd3d_CreateSphericalJoint(avbd3d_WorldId worldId,
                                                             const avbd3d_SphericalJointDef *def);
AVBD3D_LEGACY_API void avbd3d_SphericalJoint_EnableConeLimit(avbd3d_JointId jointId, bool enableLimit);
AVBD3D_LEGACY_API void avbd3d_SphericalJoint_SetConeLimit(avbd3d_JointId jointId, float angleRadians);
AVBD3D_LEGACY_API void avbd3d_SphericalJoint_EnableTwistLimit(avbd3d_JointId jointId, bool enableLimit);
AVBD3D_LEGACY_API void avbd3d_SphericalJoint_SetTwistLimits(avbd3d_JointId jointId, float lowerLimitRadians,
                                                            float upperLimitRadians);
AVBD3D_LEGACY_API void avbd3d_SphericalJoint_EnableSpring(avbd3d_JointId jointId, bool enableSpring);
AVBD3D_LEGACY_API void avbd3d_SphericalJoint_SetSpringHertz(avbd3d_JointId jointId, float hertz);
AVBD3D_LEGACY_API void avbd3d_SphericalJoint_SetSpringDampingRatio(avbd3d_JointId jointId, float dampingRatio);
AVBD3D_LEGACY_API void avbd3d_SphericalJoint_EnableMotor(avbd3d_JointId jointId, bool enableMotor);
AVBD3D_LEGACY_API void avbd3d_SphericalJoint_SetMotorVelocity(avbd3d_JointId jointId, avbd3d_Vec3 motorVelocity);
AVBD3D_LEGACY_API void avbd3d_SphericalJoint_SetMaxMotorTorque(avbd3d_JointId jointId, float torque);
AVBD3D_LEGACY_API avbd3d_Vec3 avbd3d_SphericalJoint_GetMotorTorque(avbd3d_JointId jointId);
AVBD3D_LEGACY_API float avbd3d_SphericalJoint_GetConeAngle(avbd3d_JointId jointId);
AVBD3D_LEGACY_API float avbd3d_SphericalJoint_GetTwistAngle(avbd3d_JointId jointId);
AVBD3D_LEGACY_API void avbd3d_SphericalJoint_SetTargetRotation(avbd3d_JointId jointId, avbd3d_Quat targetRotation);
AVBD3D_LEGACY_API avbd3d_JointId avbd3d_CreateWeldJoint(avbd3d_WorldId worldId, const avbd3d_WeldJointDef *def);
AVBD3D_LEGACY_API void avbd3d_WeldJoint_SetLinearHertz(avbd3d_JointId jointId, float hertz);
AVBD3D_LEGACY_API void avbd3d_WeldJoint_SetLinearDampingRatio(avbd3d_JointId jointId, float dampingRatio);
AVBD3D_LEGACY_API void avbd3d_WeldJoint_SetAngularHertz(avbd3d_JointId jointId, float hertz);
AVBD3D_LEGACY_API void avbd3d_WeldJoint_SetAngularDampingRatio(avbd3d_JointId jointId, float dampingRatio);
AVBD3D_LEGACY_API avbd3d_JointId avbd3d_CreateWheelJoint(avbd3d_WorldId worldId, const avbd3d_WheelJointDef *def);
AVBD3D_LEGACY_API void avbd3d_WheelJoint_EnableSuspension(avbd3d_JointId jointId, bool flag);
AVBD3D_LEGACY_API void avbd3d_WheelJoint_SetSuspensionHertz(avbd3d_JointId jointId, float hertz);
AVBD3D_LEGACY_API void avbd3d_WheelJoint_SetSuspensionDampingRatio(avbd3d_JointId jointId, float dampingRatio);
AVBD3D_LEGACY_API void avbd3d_WheelJoint_EnableSuspensionLimit(avbd3d_JointId jointId, bool flag);
AVBD3D_LEGACY_API void avbd3d_WheelJoint_SetSuspensionLimits(avbd3d_JointId jointId, float lower, float upper);
AVBD3D_LEGACY_API void avbd3d_WheelJoint_EnableSpinMotor(avbd3d_JointId jointId, bool flag);
AVBD3D_LEGACY_API void avbd3d_WheelJoint_SetSpinMotorSpeed(avbd3d_JointId jointId, float speed);
AVBD3D_LEGACY_API void avbd3d_WheelJoint_SetMaxSpinTorque(avbd3d_JointId jointId, float torque);
AVBD3D_LEGACY_API float avbd3d_WheelJoint_GetSpinTorque(avbd3d_JointId jointId);
AVBD3D_LEGACY_API float avbd3d_WheelJoint_GetSpinSpeed(avbd3d_JointId jointId);
AVBD3D_LEGACY_API float avbd3d_WheelJoint_GetSteeringAngle(avbd3d_JointId jointId);
AVBD3D_LEGACY_API void avbd3d_WheelJoint_EnableSteering(avbd3d_JointId jointId, bool flag);
AVBD3D_LEGACY_API void avbd3d_WheelJoint_SetSteeringHertz(avbd3d_JointId jointId, float hertz);
AVBD3D_LEGACY_API void avbd3d_WheelJoint_SetSteeringDampingRatio(avbd3d_JointId jointId, float dampingRatio);
AVBD3D_LEGACY_API void avbd3d_WheelJoint_SetMaxSteeringTorque(avbd3d_JointId jointId, float torque);
AVBD3D_LEGACY_API void avbd3d_WheelJoint_EnableSteeringLimit(avbd3d_JointId jointId, bool flag);
AVBD3D_LEGACY_API void avbd3d_WheelJoint_SetSteeringLimits(avbd3d_JointId jointId, float lowerRadians,
                                                           float upperRadians);
AVBD3D_LEGACY_API void avbd3d_WheelJoint_SetTargetSteeringAngle(avbd3d_JointId jointId, float radians);
AVBD3D_LEGACY_API avbd3d_ContactData avbd3d_Contact_GetData(avbd3d_ContactId contactId);

typedef enum avbd3d_Feature
{
    avbd3d_featureSphereShape,
    avbd3d_featureCapsuleShape,
    avbd3d_featureHullShape,
    avbd3d_featureMeshShape,
    avbd3d_featureHeightFieldShape,
    avbd3d_featureCompoundShape,

    avbd3d_featureKinematicBody,
    avbd3d_featureMotionLocks,
    avbd3d_featureBodyDamping,
    avbd3d_featureGravityScale,
    avbd3d_featureWorldFrameInertia,

    avbd3d_featureWeldJoint,
    avbd3d_featureRevoluteJoint,
    avbd3d_featurePrismaticJoint,
    avbd3d_featureWheelJoint,
    avbd3d_featureSphericalJoint,
    avbd3d_featureDistanceJoint,
    avbd3d_featureMotorJoint,
    avbd3d_featureParallelJoint,
    avbd3d_featureFilterJoint,
    avbd3d_featureSpringDamping,

    avbd3d_featureRestitution,
    avbd3d_featureRollingResistance,
    avbd3d_featureTangentVelocity,
    avbd3d_featureFrictionMixing,

    avbd3d_featureSensors,
    avbd3d_featureSensorEvents,
    avbd3d_featureContactEvents,
    avbd3d_featureHitEvents,
    avbd3d_featureJointEvents,
    avbd3d_featureBodyEvents,
    avbd3d_featureContactData,

    avbd3d_featureRayCast,
    avbd3d_featureShapeCast,
    avbd3d_featureOverlap,
    avbd3d_featureCharacterMover,
    avbd3d_featureExplosions,
    avbd3d_featureDebugDraw,

    avbd3d_featureSpeculativeContacts,
    avbd3d_featureContinuous,
    avbd3d_featureBullets,

    avbd3d_featureDoublePrecision,
    avbd3d_featureDeterminism,
    avbd3d_featureRecording,
    avbd3d_featureSoftStepSolver,

    avbd3d_featureAsyncStep,
    avbd3d_featureSolverDef,
    avbd3d_featureBatchedBodyAccess,
    avbd3d_featureGrab,
    avbd3d_featureJointFracture,
    avbd3d_featureOriginShift,
    avbd3d_featureAvbdJoint,

    avbd3d_featureCount
} avbd3d_Feature;

typedef enum avbd3d_FeatureStatus
{
    avbd3d_featureUnsupported,

    avbd3d_featureSupported
} avbd3d_FeatureStatus;

AVBD3D_LEGACY_API void avbd3d_World_StepBegin(avbd3d_WorldId worldId, float timeStep);
AVBD3D_LEGACY_API bool avbd3d_World_IsStepInFlight(avbd3d_WorldId worldId);

typedef struct avbd3d_SolverDef
{
    int iterations;

    int primalLanes;
    float alpha;
    float betaLin;
    float betaAng;
    float gamma;
    bool autoKStart;
    float dualDamping;
    float dualDampVel;
    int sleepFrames;

    float sleepLinearVelocity;
    float sleepAngularVelocity;

    float sleepDisplacement;
    float sleepFracture;
    float sleepLeak;

    bool enableKillFallen;
    float killHeight;

    float grabStrength;
    int contactCap;

    int coarseIterations;
    float coarseCell;
    float coarseDamping;

    bool worldFrameInertia;

    int internalValue;
} avbd3d_SolverDef;

AVBD3D_LEGACY_API avbd3d_SolverDef avbd3d_DefaultSolverDef(void);
AVBD3D_LEGACY_API void avbd3d_World_SetSolverDef(avbd3d_WorldId worldId, const avbd3d_SolverDef *def);

typedef enum avbd3d_MixingRule
{
    avbd3d_mixGeometric = 0,
    avbd3d_mixAverage,
    avbd3d_mixMin,
    avbd3d_mixMax,
    avbd3d_mixMultiply,
} avbd3d_MixingRule;

AVBD3D_LEGACY_API void avbd3d_World_SetFrictionMixing(avbd3d_WorldId worldId, avbd3d_MixingRule rule);
AVBD3D_LEGACY_API void avbd3d_World_SetRestitutionMixing(avbd3d_WorldId worldId, avbd3d_MixingRule rule);

typedef enum avbd3d_BodyDataType
{
    avbd3d_bodyDataPosition,

    avbd3d_bodyDataRotation,
    avbd3d_bodyDataTransform,
    avbd3d_bodyDataLinearVelocity,
    avbd3d_bodyDataAngularVelocity,
    avbd3d_bodyDataAwake,

    avbd3d_bodyDataTypeCount
} avbd3d_BodyDataType;

AVBD3D_LEGACY_API int avbd3d_World_GetBodyData(avbd3d_WorldId worldId, const avbd3d_BodyId *ids, int count,
                                               avbd3d_BodyDataType type, void *data);

AVBD3D_LEGACY_API int avbd3d_World_SetBodyData(avbd3d_WorldId worldId, const avbd3d_BodyId *ids, int count,
                                               avbd3d_BodyDataType type, const void *data);

#define avbd3d_avbdJoint ((avbd3d_JointType)100)
#define avbd3d_avbdSpring ((avbd3d_JointType)101)

typedef enum avbd3d_AvbdJointKind
{
    avbd3d_avbdJointPoint = 0,
    avbd3d_avbdJointSixDof = 1,
    avbd3d_avbdJointDistance = 2,
    avbd3d_avbdJointSwingTwist = 3,
} avbd3d_AvbdJointKind;

typedef enum avbd3d_AvbdAxisMode
{
    avbd3d_avbdAxisFree = 0,
    avbd3d_avbdAxisLocked,
    avbd3d_avbdAxisLimited,
    avbd3d_avbdAxisMotor,
    avbd3d_avbdAxisSpring,
    avbd3d_avbdAxisSpringForce,
    avbd3d_avbdAxisVelocityMotor,
    avbd3d_avbdAxisCone,
} avbd3d_AvbdAxisMode;

typedef struct avbd3d_AvbdAxis
{
    avbd3d_AvbdAxisMode mode;
    float lo;
    float hi;
} avbd3d_AvbdAxis;

typedef struct avbd3d_AvbdJointDef
{
    avbd3d_JointDef base;
    avbd3d_AvbdJointKind kind;
    float linearStiffness;
    float angularStiffness;
    avbd3d_AvbdAxis axes[6];
    float fractureForce;
    float fractureTorque;
} avbd3d_AvbdJointDef;

AVBD3D_LEGACY_API avbd3d_AvbdJointDef avbd3d_DefaultAvbdJointDef(void);
AVBD3D_LEGACY_API avbd3d_JointId avbd3d_CreateAvbdJoint(avbd3d_WorldId worldId, const avbd3d_AvbdJointDef *def);

typedef struct avbd3d_AvbdSpringDef
{
    avbd3d_JointDef base;
    float stiffness;
    float restLength;
} avbd3d_AvbdSpringDef;

AVBD3D_LEGACY_API avbd3d_AvbdSpringDef avbd3d_DefaultAvbdSpringDef(void);
AVBD3D_LEGACY_API avbd3d_JointId avbd3d_CreateAvbdSpring(avbd3d_WorldId worldId, const avbd3d_AvbdSpringDef *def);
#define AVBD3D_MAX_GRAB_BODIES 256

typedef struct avbd3d_GrabHandle
{
    avbd3d_BodyId body;

    avbd3d_Vec3 localAnchor;
    avbd3d_Pos target;

    float weight;
} avbd3d_GrabHandle;

AVBD3D_LEGACY_API int avbd3d_World_GrabGroup(avbd3d_WorldId worldId, avbd3d_BodyId seed, avbd3d_Pos center,
                                             float radius, bool followConnections, avbd3d_BodyId *outBodies,
                                             int maxCount);

AVBD3D_LEGACY_API int avbd3d_World_GrabSurvivors(avbd3d_WorldId worldId, avbd3d_BodyId seed, const avbd3d_BodyId *group,
                                                 int groupCount, uint8_t *outKeep);

AVBD3D_LEGACY_API void avbd3d_World_GrabDrag(avbd3d_WorldId worldId, const avbd3d_GrabHandle *handles, int count);
AVBD3D_LEGACY_API void avbd3d_World_GrabRelease(avbd3d_WorldId worldId);
typedef struct avbd3d_Gpu avbd3d_Gpu;
typedef uint64_t avbd3d_MakeSurfaceFcn(void *vkInstance, void *context);

typedef struct avbd3d_GpuDef
{
    const char *const *instanceExtensions;
    int instanceExtensionCount;

    avbd3d_MakeSurfaceFcn *makeSurface;
    void *surfaceContext;

    bool enableValidation;
    bool verbose;

    int internalValue;
} avbd3d_GpuDef;

typedef struct avbd3d_VulkanContext
{
    void *instance;       ///< VkInstance
    void *physicalDevice; ///< VkPhysicalDevice
    void *device;         ///< VkDevice
    void *queue;          ///< VkQueue: the solver's queue, which also presents on a presentable GPU
    uint32_t queueFamily;
    uint32_t apiVersion;
    uint64_t commandPool;   ///< VkCommandPool of the solver (reset only by the solver)
    uint64_t pipelineCache; ///< VkPipelineCache
    uint64_t surface;       ///< VkSurfaceKHR, 0 when the GPU is not presentable
    uint32_t subgroupSize;
    const char *deviceName;
} avbd3d_VulkanContext;

AVBD3D_LEGACY_API avbd3d_GpuDef avbd3d_DefaultGpuDef(void);
AVBD3D_LEGACY_API avbd3d_Gpu *avbd3d_CreateGpu(const avbd3d_GpuDef *def, avbd3d_Error *outErr);
AVBD3D_LEGACY_API void avbd3d_DestroyGpu(avbd3d_Gpu *gpu);
AVBD3D_LEGACY_API avbd3d_WorldId avbd3d_CreateWorldOnGpu(const avbd3d_WorldDef *def, avbd3d_Gpu *gpu,
                                                         avbd3d_Error *outErr);
AVBD3D_LEGACY_API avbd3d_VulkanContext avbd3d_Gpu_GetVulkanContext(const avbd3d_Gpu *gpu);
AVBD3D_LEGACY_API uint64_t avbd3d_Gpu_GetAllocatedBytes(const avbd3d_Gpu *gpu);
AVBD3D_LEGACY_API int avbd3d_GetValidationMessageCount(void);

typedef struct avbd3d_RenderInstance
{
    float position[3];
    float rotation[4]; ///< x, y, z, w
    float halfExtents[3];
    uint32_t rgba; ///< 0xAABBGGRR
} avbd3d_RenderInstance;

typedef enum avbd3d_Overlay
{
    avbd3d_overlayNone = 0,
    avbd3d_overlaySleep,       ///< asleep vs awake, with the settling countdown as a gradient
    avbd3d_overlayGrid,        ///< broadphase hash grid: one hue per occupied cell
    avbd3d_overlayGraphColor,  ///< graph colouring: the parallel batches of the solve
    avbd3d_overlayLoad,        ///< peak contact normal force on the body
    avbd3d_overlaySpeed,       ///< linear speed
    avbd3d_overlayDegree,      ///< number of bodies touching this one
    avbd3d_overlayJointDegree, ///< number of unbroken joints and springs on this body
    avbd3d_overlayJointLoad,   ///< worst connected joint load, compression vs tension
    avbd3d_overlayJointStress, ///< worst connected joint load as a fraction of its fracture limit
    avbd3d_overlayAxialLoad,   ///< contacts pushing vs joints pulling
    avbd3d_overlayFriction,    ///< fraction of the friction cone in use
    avbd3d_overlayPenetration, ///< deepest contact penetration
    avbd3d_overlayCount
} avbd3d_Overlay;

typedef struct avbd3d_RenderDef
{
    const float *frustumPlanes;

    avbd3d_Overlay overlay;
    float speedMax, loadMax, jointLoadMax, penetrationMax;
    int degreeMax, jointDegreeMax;
    bool colorStatics;

    int internalValue;
} avbd3d_RenderDef;

typedef struct avbd3d_RenderRange
{
    int first;
    int count;
} avbd3d_RenderRange;

AVBD3D_LEGACY_API avbd3d_RenderDef avbd3d_DefaultRenderDef(void);
AVBD3D_LEGACY_API void avbd3d_World_WriteInstances(avbd3d_WorldId worldId, const avbd3d_RenderDef *def);
AVBD3D_LEGACY_API uint64_t avbd3d_World_GetInstanceBuffer(avbd3d_WorldId worldId);
AVBD3D_LEGACY_API int avbd3d_World_GetRenderGroupCount(avbd3d_WorldId worldId);
AVBD3D_LEGACY_API avbd3d_RenderRange avbd3d_World_GetRenderGroupRange(avbd3d_WorldId worldId, int group);
AVBD3D_LEGACY_API void avbd3d_Body_SetRenderGroup(avbd3d_BodyId bodyId, int group);
AVBD3D_LEGACY_API int avbd3d_Body_GetRenderGroup(avbd3d_BodyId bodyId);

typedef enum avbd3d_ContactTint
{
    avbd3d_contactTintForce = 0, ///< normal force, on the load overlay's ramp
    avbd3d_contactTintFriction,  ///< fraction of the friction cone in use
    avbd3d_contactTintStick,     ///< static friction holding vs slipping
    avbd3d_contactTintFlat,      ///< one colour
} avbd3d_ContactTint;

typedef struct avbd3d_ContactMarkerDef
{
    bool points;        ///< a small cube at each contact point
    bool normals;       ///< a needle along the contact normal, scaled by force
    bool manifoldLinks; ///< a rod joining the centres of each touching pair
    avbd3d_ContactTint tint;
    float pointSize;    ///< half extent of a point marker, metres
    float normalLength; ///< needle length at full force, metres
    float forceMax;     ///< force at which the tint saturates
    int maxMarkers;     ///< markers past this are not written

    int internalValue;
} avbd3d_ContactMarkerDef;

AVBD3D_LEGACY_API avbd3d_ContactMarkerDef avbd3d_DefaultContactMarkerDef(void);
AVBD3D_LEGACY_API int avbd3d_World_WriteContactMarkers(avbd3d_WorldId worldId, const avbd3d_ContactMarkerDef *def);
AVBD3D_LEGACY_API uint64_t avbd3d_World_GetContactMarkerBuffer(avbd3d_WorldId worldId);
AVBD3D_LEGACY_API int avbd3d_World_GetContactMarkersRequested(avbd3d_WorldId worldId);
