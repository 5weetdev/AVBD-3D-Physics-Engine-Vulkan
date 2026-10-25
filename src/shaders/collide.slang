// Phase 3 spike: box-box SAT narrowphase
// (collideBoxes, the namespace-local helpers it calls, and the box half of
// collideShapes's dispatch). Same operation order as the reference source, function for
// function, so the two are easy to diff by eye -- see tools/test_vk_collide.cpp for the
// numerical comparison against the host-compiled SAT collider (avbd_core, linked directly,
// no device launch needed since collideBoxes is AVBD_HD).
//
// Phase 3b adds the round-shape narrowphase (collideRoundRound / collideBoxSphere /
// collideBoxCapsule) and the collideShapes dispatcher, ported the same way: function
// for function against the SAT collider's namespace-local helpers, same operation order, same
// epsilons. See csCollideShapes below and tools/test_vk_collide.cpp for the comparison
// against the host-compiled collideShapes (collision.h / the SAT collider).

// Named module so engine/shaders/narrowphase.slang can `import collide;` and call
// collideShapesImpl directly instead of duplicating ~500 lines of SAT/round-shape code
// -- the manifold builder's collision call is exactly this function (see narrowphase.slang's kernel, which calls collideShapes the same way).
module collide;

import maths;

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

public static const int MAX_CONTACTS = 8;      // MAX_MANIFOLD_CONTACTS, collision.h
static const int MAX_POLY_VERTS = 16;
static const float SAT_AXIS_EPSILON = 1.0e-6f;
static const float PLANE_EPSILON = 1.0e-5f;
static const float CONTACT_MERGE_DIST_SQ = 1.0e-6f;
static const float COLLISION_MARGIN_C = 0.01f;   // collision.h: COLLISION_MARGIN

static const int AXIS_FACE_A = 0;
static const int AXIS_FACE_B = 1;
static const int AXIS_EDGE = 2;

// ---------------------------------------------------------------------------
// Input/output buffers. One BodyRef per side (collision.h's BodyRef, boxes only: no
// shape tag needed since this kernel is collideBoxes specifically, not the dispatcher).
// ---------------------------------------------------------------------------
struct BodyBoxSoA
{
    vec3 *pos;
    quat *rot;
    vec3 *halfExtents;
    uint count;
    uint _pad0;
};

// One manifold's worth of output, flattened: fixed MAX_CONTACTS slots per pair (unused
// slots have key -1), matching tools/test_vk_collide.cpp's host-side Result layout.
public struct ContactOut
{
    public int featureKey;
    public vec3 rA;
    public vec3 rB;
};

struct CollideOut
{
    int *count;         // per-pair contact count
    float3x3 *basis;    // per-pair contact basis
    ContactOut *contacts; // MAX_CONTACTS per pair, flat
};

struct CollideArgs
{
    BodyBoxSoA a;
    BodyBoxSoA b;
    CollideOut out;
};

// ---------------------------------------------------------------------------
// OBB + SAT axis bookkeeping, mirroring the SAT collider's anonymous-namespace structs.
// ---------------------------------------------------------------------------
struct OBB
{
    vec3 center;
    quat rotation;
    vec3 half;
    vec3 axis0, axis1, axis2;
};

float axisAt(OBB box, int i, int comp)
{
    vec3 a = (i == 0) ? box.axis0 : (i == 1 ? box.axis1 : box.axis2);
    return compAt(a, comp);
}
vec3 axisVec(OBB box, int i) { return (i == 0) ? box.axis0 : (i == 1 ? box.axis1 : box.axis2); }
float halfAt(OBB box, int i) { return compAt(box.half, i); }

OBB makeOBB(vec3 pos, quat rot, vec3 half)
{
    OBB box;
    box.center = pos;
    box.rotation = rot;
    box.half = half;
    box.axis0 = rotateQ(rot, vec3(1.0f, 0.0f, 0.0f));
    box.axis1 = rotateQ(rot, vec3(0.0f, 1.0f, 0.0f));
    box.axis2 = rotateQ(rot, vec3(0.0f, 0.0f, 1.0f));
    return box;
}

float absDot(vec3 a, vec3 b) { return abs(dot3(a, b)); }

vec3 supportPoint(OBB box, vec3 dir)
{
    float sx = dot3(dir, box.axis0) >= 0.0f ? 1.0f : -1.0f;
    float sy = dot3(dir, box.axis1) >= 0.0f ? 1.0f : -1.0f;
    float sz = dot3(dir, box.axis2) >= 0.0f ? 1.0f : -1.0f;
    vec3 r = box.center;
    r = add(r, scale(box.axis0, box.half.x * sx));
    r = add(r, scale(box.axis1, box.half.y * sy));
    r = add(r, scale(box.axis2, box.half.z * sz));
    return r;
}

void getFaceAxes(OBB box, int axisIndex, out vec3 u, out vec3 v, out float extentU, out float extentV)
{
    if (axisIndex == 0)
    {
        u = box.axis1; v = box.axis2; extentU = box.half.y; extentV = box.half.z;
    }
    else if (axisIndex == 1)
    {
        u = box.axis0; v = box.axis2; extentU = box.half.x; extentV = box.half.z;
    }
    else
    {
        u = box.axis0; v = box.axis1; extentU = box.half.x; extentV = box.half.y;
    }
}

struct FaceFrame
{
    int axisIndex;
    vec3 normal, center, u, v;
    float extentU, extentV;
};

FaceFrame buildFaceFrame(OBB box, int axisIndex, vec3 outwardNormal)
{
    float sign = dot3(outwardNormal, axisVec(box, axisIndex)) >= 0.0f ? 1.0f : -1.0f;
    FaceFrame frame;
    frame.axisIndex = axisIndex;
    frame.normal = scale(axisVec(box, axisIndex), sign);
    frame.center = add(box.center, scale(frame.normal, halfAt(box, axisIndex)));
    getFaceAxes(box, axisIndex, frame.u, frame.v, frame.extentU, frame.extentV);
    return frame;
}

int chooseIncidentFaceAxis(OBB box, vec3 referenceNormal)
{
    int axis = 0;
    float best = -3.402823e38f;
    for (int i = 0; i < 3; ++i)
    {
        float d = absDot(axisVec(box, i), referenceNormal);
        if (d > best) { best = d; axis = i; }
    }
    return axis;
}

void buildIncidentFace(OBB box, int axisIndex, vec3 referenceNormal, out vec3 outVerts[4])
{
    float sign = dot3(axisVec(box, axisIndex), referenceNormal) > 0.0f ? -1.0f : 1.0f;
    vec3 faceNormal = scale(axisVec(box, axisIndex), sign);
    vec3 faceCenter = add(box.center, scale(faceNormal, halfAt(box, axisIndex)));

    vec3 u, v; float extentU, extentV;
    getFaceAxes(box, axisIndex, u, v, extentU, extentV);

    outVerts[0] = add(add(faceCenter, scale(u, extentU)), scale(v, extentV));
    outVerts[1] = add(sub(faceCenter, scale(u, extentU)), scale(v, extentV));
    outVerts[2] = sub(sub(faceCenter, scale(u, extentU)), scale(v, extentV));
    outVerts[3] = sub(add(faceCenter, scale(u, extentU)), scale(v, extentV));
}

// clipPolygonAgainstPlane, transliterated with fixed MAX_POLY_VERTS arrays in place of
// pointer arguments (Slang has no raw pointer arithmetic over local arrays the way the
// the reference version does; the loop structure is identical).
int clipPolygonAgainstPlane(vec3 inVerts[MAX_POLY_VERTS], int inCount, vec3 planeNormal,
                            float planeOffset, out vec3 outVerts[MAX_POLY_VERTS])
{
    if (inCount <= 0)
        return 0;

    int outCount = 0;
    vec3 a = inVerts[inCount - 1];
    float da = dot3(planeNormal, a) - planeOffset;

    for (int i = 0; i < inCount; ++i)
    {
        vec3 b = inVerts[i];
        float db = dot3(planeNormal, b) - planeOffset;

        bool aInside = da <= PLANE_EPSILON;
        bool bInside = db <= PLANE_EPSILON;

        if (aInside != bInside)
        {
            float t = 0.0f;
            float denom = da - db;
            if (abs(denom) > SAT_AXIS_EPSILON)
                t = clampf01(da / denom, 0.0f, 1.0f);
            if (outCount < MAX_POLY_VERTS)
                outVerts[outCount++] = add(a, scale(sub(b, a), t));
        }
        if (bInside && outCount < MAX_POLY_VERTS)
            outVerts[outCount++] = b;

        a = b;
        da = db;
    }
    return outCount;
}

// addContact: mirrors the SAT collider, writing into fixed-size local arrays instead of
// pointer + running counter (the counter is still threaded through explicitly).
bool addContact(inout ContactOut contacts[MAX_CONTACTS], inout int contactCount,
                inout vec3 contactMidpoints[MAX_CONTACTS], OBB boxA, OBB boxB,
                vec3 xA, vec3 xB, int featureKey)
{
    vec3 midpoint = scale(add(xA, xB), 0.5f);
    for (int i = 0; i < contactCount; ++i)
    {
        vec3 d = sub(midpoint, contactMidpoints[i]);
        if (lengthSq3(d) < CONTACT_MERGE_DIST_SQ)
            return false;
    }
    if (contactCount >= MAX_CONTACTS)
        return false;

    ContactOut c;
    c.featureKey = featureKey;
    c.rA = rotateQ(conjugateQ(boxA.rotation), sub(xA, boxA.center));
    c.rB = rotateQ(conjugateQ(boxB.rotation), sub(xB, boxB.center));
    contacts[contactCount] = c;
    contactMidpoints[contactCount] = midpoint;
    ++contactCount;
    return true;
}

struct SatAxis
{
    int type;
    int indexA, indexB;
    float separation;
    vec3 normalAB;
    bool valid;
};

bool testAxis(OBB boxA, OBB boxB, vec3 delta, vec3 axis, int type, int indexA, int indexB,
             inout SatAxis best)
{
    float lenSq = lengthSq3(axis);
    if (lenSq < SAT_AXIS_EPSILON)
        return true;

    float invLen = 1.0f / sqrt(lenSq);
    vec3 n = scale(axis, invLen);
    if (dot3(n, delta) < 0.0f)
        n = neg(n);

    float distance = abs(dot3(delta, n));

    float rA = boxA.half.x * absDot(n, boxA.axis0) + boxA.half.y * absDot(n, boxA.axis1) +
              boxA.half.z * absDot(n, boxA.axis2);
    float rB = boxB.half.x * absDot(n, boxB.axis0) + boxB.half.y * absDot(n, boxB.axis1) +
              boxB.half.z * absDot(n, boxB.axis2);

    float separation = distance - (rA + rB);
    if (separation > 0.0f)
        return false;

    if (!best.valid || separation > best.separation)
    {
        best.valid = true;
        best.type = type;
        best.indexA = indexA;
        best.indexB = indexB;
        best.separation = separation;
        best.normalAB = n;
    }
    return true;
}

void supportEdge(OBB box, int axisIndex, vec3 dir, out vec3 edgeA, out vec3 edgeB)
{
    int axis1 = (axisIndex + 1) % 3;
    int axis2 = (axisIndex + 2) % 3;

    float sign1 = dot3(dir, axisVec(box, axis1)) >= 0.0f ? 1.0f : -1.0f;
    float sign2 = dot3(dir, axisVec(box, axis2)) >= 0.0f ? 1.0f : -1.0f;

    vec3 edgeCenter = add(box.center,
        add(scale(axisVec(box, axis1), halfAt(box, axis1) * sign1),
            scale(axisVec(box, axis2), halfAt(box, axis2) * sign2)));

    edgeA = sub(edgeCenter, scale(axisVec(box, axisIndex), halfAt(box, axisIndex)));
    edgeB = add(edgeCenter, scale(axisVec(box, axisIndex), halfAt(box, axisIndex)));
}

void closestPointsOnSegments(vec3 p0, vec3 p1, vec3 q0, vec3 q1, out vec3 c0, out vec3 c1)
{
    vec3 d1 = sub(p1, p0);
    vec3 d2 = sub(q1, q0);
    vec3 r = sub(p0, q0);
    float a = dot3(d1, d1);
    float e = dot3(d2, d2);
    float f = dot3(d2, r);

    float s = 0.0f, t = 0.0f;

    if (a <= SAT_AXIS_EPSILON && e <= SAT_AXIS_EPSILON)
    {
        c0 = p0; c1 = q0; return;
    }

    if (a <= SAT_AXIS_EPSILON)
    {
        t = clampf01(f / e, 0.0f, 1.0f);
    }
    else
    {
        float c = dot3(d1, r);
        if (e <= SAT_AXIS_EPSILON)
        {
            s = clampf01(-c / a, 0.0f, 1.0f);
        }
        else
        {
            float b = dot3(d1, d2);
            float denom = a * e - b * b;
            if (abs(denom) > SAT_AXIS_EPSILON)
                s = clampf01((b * f - c * e) / denom, 0.0f, 1.0f);
            t = (b * s + f) / e;
            if (t < 0.0f) { t = 0.0f; s = clampf01(-c / a, 0.0f, 1.0f); }
            else if (t > 1.0f) { t = 1.0f; s = clampf01((b - c) / a, 0.0f, 1.0f); }
        }
    }
    c0 = add(p0, scale(d1, s));
    c1 = add(q0, scale(d2, t));
}

int buildFaceManifold(OBB boxA, OBB boxB, bool referenceIsA, int referenceAxis, vec3 normalAB,
                      out ContactOut contacts[MAX_CONTACTS])
{
    OBB referenceBox = referenceIsA ? boxA : boxB;
    OBB incidentBox = referenceIsA ? boxB : boxA;
    vec3 referenceOutward = referenceIsA ? normalAB : neg(normalAB);

    FaceFrame referenceFace = buildFaceFrame(referenceBox, referenceAxis, referenceOutward);
    int incidentAxis = chooseIncidentFaceAxis(incidentBox, referenceFace.normal);

    vec3 clip0[MAX_POLY_VERTS], clip1[MAX_POLY_VERTS];
    vec3 face4[4];
    buildIncidentFace(incidentBox, incidentAxis, referenceFace.normal, face4);
    for (int i = 0; i < 4; ++i) clip0[i] = face4[i];
    int count = 4;

    vec3 n0 = referenceFace.u;
    float o0 = dot3(n0, referenceFace.center) + referenceFace.extentU;
    count = clipPolygonAgainstPlane(clip0, count, n0, o0, clip1);
    if (count == 0) { contacts[0].featureKey = -1; return 0; }

    vec3 n1 = neg(referenceFace.u);
    float o1 = dot3(n1, referenceFace.center) + referenceFace.extentU;
    count = clipPolygonAgainstPlane(clip1, count, n1, o1, clip0);
    if (count == 0) { contacts[0].featureKey = -1; return 0; }

    vec3 n2 = referenceFace.v;
    float o2 = dot3(n2, referenceFace.center) + referenceFace.extentV;
    count = clipPolygonAgainstPlane(clip0, count, n2, o2, clip1);
    if (count == 0) { contacts[0].featureKey = -1; return 0; }

    vec3 n3 = neg(referenceFace.v);
    float o3 = dot3(n3, referenceFace.center) + referenceFace.extentV;
    count = clipPolygonAgainstPlane(clip1, count, n3, o3, clip0);
    if (count == 0) { contacts[0].featureKey = -1; return 0; }

    int contactCount = 0;
    vec3 contactMidpoints[MAX_CONTACTS];
    int featurePrefix = (referenceIsA ? AXIS_FACE_A : AXIS_FACE_B) << 24;
    featurePrefix |= (referenceAxis & 0xFF) << 16;
    featurePrefix |= (incidentAxis & 0xFF) << 8;

    for (int i = 0; i < count && contactCount < MAX_CONTACTS; ++i)
    {
        vec3 pIncident = clip0[i];
        float distance = dot3(sub(pIncident, referenceFace.center), referenceFace.normal);
        if (distance > PLANE_EPSILON)
            continue;

        vec3 pReference = sub(pIncident, scale(referenceFace.normal, distance));
        vec3 xA = referenceIsA ? pReference : pIncident;
        vec3 xB = referenceIsA ? pIncident : pReference;

        addContact(contacts, contactCount, contactMidpoints, boxA, boxB, xA, xB, featurePrefix | (i & 0xFF));
    }

    if (contactCount == 0)
    {
        vec3 xA = supportPoint(boxA, normalAB);
        vec3 xB = supportPoint(boxB, neg(normalAB));
        addContact(contacts, contactCount, contactMidpoints, boxA, boxB, xA, xB, featurePrefix);
    }
    return contactCount;
}

int buildEdgeContact(OBB boxA, OBB boxB, int axisA, int axisB, vec3 normalAB,
                     out ContactOut contacts[MAX_CONTACTS])
{
    vec3 a0, a1, b0, b1;
    supportEdge(boxA, axisA, normalAB, a0, a1);
    supportEdge(boxB, axisB, neg(normalAB), b0, b1);

    vec3 xA, xB;
    closestPointsOnSegments(a0, a1, b0, b1, xA, xB);

    int contactCount = 0;
    vec3 contactMidpoints[MAX_CONTACTS];
    int featureKey = (AXIS_EDGE << 24) | ((axisA & 0xFF) << 8) | (axisB & 0xFF);
    addContact(contacts, contactCount, contactMidpoints, boxA, boxB, xA, xB, featureKey);

    if (contactCount == 0)
    {
        xA = supportPoint(boxA, normalAB);
        xB = supportPoint(boxB, neg(normalAB));
        addContact(contacts, contactCount, contactMidpoints, boxA, boxB, xA, xB, featureKey);
    }
    return contactCount;
}

// collideBoxes: the public entry point, same structure as the SAT collider's.
int collideBoxesImpl(vec3 posA, quat rotA, vec3 halfA, vec3 posB, quat rotB, vec3 halfB,
                     out ContactOut contacts[MAX_CONTACTS], out float3x3 basisOut)
{
    OBB boxA = makeOBB(posA, rotA, halfA);
    OBB boxB = makeOBB(posB, rotB, halfB);
    vec3 delta = sub(boxB.center, boxA.center);

    SatAxis bestFace;
    bestFace.separation = -3.402823e38f;
    bestFace.valid = false;

    SatAxis bestEdge;
    bestEdge.separation = -3.402823e38f;
    bestEdge.valid = false;

    for (int i = 0; i < 3; ++i)
        if (!testAxis(boxA, boxB, delta, axisVec(boxA, i), AXIS_FACE_A, i, -1, bestFace))
        { contacts[0].featureKey = -1; return 0; }

    for (int i = 0; i < 3; ++i)
        if (!testAxis(boxA, boxB, delta, axisVec(boxB, i), AXIS_FACE_B, -1, i, bestFace))
        { contacts[0].featureKey = -1; return 0; }

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            vec3 axis = cross3(axisVec(boxA, i), axisVec(boxB, j));
            if (!testAxis(boxA, boxB, delta, axis, AXIS_EDGE, i, j, bestEdge))
            { contacts[0].featureKey = -1; return 0; }
        }

    if (!bestFace.valid) { contacts[0].featureKey = -1; return 0; }

    SatAxis best = bestFace;
    if (bestEdge.valid)
    {
        const float edgeRelTol = 0.95f;
        const float edgeAbsTol = 0.01f;
        if (edgeRelTol * bestEdge.separation > bestFace.separation + edgeAbsTol)
            best = bestEdge;
    }

    basisOut = orthonormalF(neg(best.normalAB));

    if (best.type == AXIS_EDGE)
        return buildEdgeContact(boxA, boxB, best.indexA, best.indexB, best.normalAB, contacts);
    if (best.type == AXIS_FACE_A)
        return buildFaceManifold(boxA, boxB, true, best.indexA, best.normalAB, contacts);
    return buildFaceManifold(boxA, boxB, false, best.indexB, best.normalAB, contacts);
}

// ---------------------------------------------------------------------------
// Round-shape narrowphase: collideRoundRound / collideBoxSphere / collideBoxCapsule,
//. See collision.h for AVBD_SHAPE_*, shapeRadius,
// capsuleHalfSegment -- mirrored here as constants/helpers since Slang can't include
// the C++ header.
// ---------------------------------------------------------------------------
public static const int AVBD_SHAPE_BOX = 0;
public static const int AVBD_SHAPE_SPHERE = 1;
public static const int AVBD_SHAPE_CAPSULE = 2;
// Shape id 3 is unused; HULL and MESH keep their numbers.
public static const int AVBD_SHAPE_HULL = 4;     // geometry-heap hull (narrowphase_hull.slang)
public static const int AVBD_SHAPE_MESH = 5;     // static triangle mesh (narrowphase_mesh.slang)
static const int SHAPE_FEATURE_TAG = 0x10 << 24;
static const float OBB_DEGENERATE_DIST = 1.0e-4f;

float shapeRadiusF(vec3 halfExtents) { return halfExtents.x; }
float capsuleHalfSegmentF(vec3 halfExtents)
{
    float h = halfExtents.z - halfExtents.x;
    return h > 0.0f ? h : 0.0f;
}

void capsuleSegment(vec3 pos, quat rot, vec3 halfExtents, out vec3 p0, out vec3 p1)
{
    vec3 axis = scale(rotateQ(rot, vec3(0.0f, 0.0f, 1.0f)), capsuleHalfSegmentF(halfExtents));
    p0 = sub(pos, axis);
    p1 = add(pos, axis);
}

void roundSegment(int shape, vec3 pos, quat rot, vec3 halfExtents, out vec3 p0, out vec3 p1)
{
    if (shape == AVBD_SHAPE_CAPSULE)
        capsuleSegment(pos, rot, halfExtents, p0, p1);
    else
    {
        p0 = pos;
        p1 = pos;
    }
}

vec3 safeNormal(vec3 d)
{
    float lenSq = lengthSq3(d);
    if (lenSq < SAT_AXIS_EPSILON * SAT_AXIS_EPSILON)
        return vec3(0.0f, 0.0f, 1.0f);
    return scale(d, 1.0f / sqrt(lenSq));
}

struct ShapeManifold
{
    int count;
    vec3 normalAB;
    vec3 xA[MAX_CONTACTS];
    vec3 xB[MAX_CONTACTS];
    int key[MAX_CONTACTS];
};

void addRoundContact(inout ShapeManifold m, vec3 pA, vec3 pB, float radiusA, float radiusB,
                     vec3 n, int key)
{
    if (m.count >= MAX_CONTACTS)
        return;
    m.xA[m.count] = add(pA, scale(n, radiusA));
    m.xB[m.count] = sub(pB, scale(n, radiusB));
    m.key[m.count] = key;
    ++m.count;
}

int collideRoundRound(vec3 posA, quat rotA, vec3 halfA, int shapeA,
                      vec3 posB, quat rotB, vec3 halfB, int shapeB, inout ShapeManifold m)
{
    vec3 a0, a1, b0, b1;
    roundSegment(shapeA, posA, rotA, halfA, a0, a1);
    roundSegment(shapeB, posB, rotB, halfB, b0, b1);

    float rA = shapeRadiusF(halfA);
    float rB = shapeRadiusF(halfB);
    float reach = rA + rB + COLLISION_MARGIN_C;

    vec3 cA, cB;
    closestPointsOnSegments(a0, a1, b0, b1, cA, cB);

    vec3 delta = sub(cB, cA);
    if (lengthSq3(delta) > reach * reach)
    {
        m.count = 0;
        return 0;
    }

    m.count = 0;
    m.normalAB = safeNormal(delta);

    vec3 dA = sub(a1, a0);
    vec3 dB = sub(b1, b0);
    float lenA = length3(dA);
    float lenB = length3(dB);
    if (lenA > SAT_AXIS_EPSILON && lenB > SAT_AXIS_EPSILON)
    {
        vec3 uA = scale(dA, 1.0f / lenA);
        vec3 uB = scale(dB, 1.0f / lenB);
        const float kParallel = 0.999f;
        if (abs(dot3(uA, uB)) > kParallel)
        {
            float tb0 = dot3(sub(b0, a0), uA);
            float tb1 = dot3(sub(b1, a0), uA);
            float lo = max(0.0f, min(tb0, tb1));
            float hi = min(lenA, max(tb0, tb1));
            if (hi - lo > SAT_AXIS_EPSILON)
            {
                for (int i = 0; i < 2; ++i)
                {
                    float t = (i == 0) ? lo : hi;
                    vec3 pA = add(a0, scale(uA, t));
                    float tB = clampf01(dot3(sub(pA, b0), uB), 0.0f, lenB);
                    vec3 pB = add(b0, scale(uB, tB));
                    addRoundContact(m, pA, pB, rA, rB, m.normalAB, SHAPE_FEATURE_TAG | i);
                }
                return m.count;
            }
        }
    }

    addRoundContact(m, cA, cB, rA, rB, m.normalAB, SHAPE_FEATURE_TAG);
    return m.count;
}

vec3 closestPointOnOBB(OBB box, vec3 p, out bool inside, out vec3 faceNormal)
{
    vec3 d = sub(p, box.center);
    float local[3];
    float clamped[3];
    inside = true;
    for (int i = 0; i < 3; ++i)
    {
        local[i] = dot3(d, axisVec(box, i));
        clamped[i] = clampf01(local[i], -halfAt(box, i), halfAt(box, i));
        if (clamped[i] != local[i])
            inside = false;
    }

    vec3 result = box.center;
    for (int i = 0; i < 3; ++i)
        result = add(result, scale(axisVec(box, i), clamped[i]));

    int best = 0;
    float bestDepth = halfAt(box, 0) - abs(local[0]);
    for (int i = 1; i < 3; ++i)
    {
        float depth = halfAt(box, i) - abs(local[i]);
        if (depth < bestDepth)
        {
            bestDepth = depth;
            best = i;
        }
    }
    float s = local[best] >= 0.0f ? 1.0f : -1.0f;
    faceNormal = scale(axisVec(box, best), s);

    if (inside)
        result = add(p, scale(faceNormal, bestDepth));

    return result;
}

vec3 boxPointNormal(vec3 toPoint, bool inside, vec3 faceNormal)
{
    if (inside || lengthSq3(toPoint) < OBB_DEGENERATE_DIST * OBB_DEGENERATE_DIST)
        return faceNormal;
    return safeNormal(toPoint);
}

int collideBoxSphere(vec3 posA, quat rotA, vec3 halfA, vec3 posB, vec3 halfB, inout ShapeManifold m)
{
    OBB box = makeOBB(posA, rotA, halfA);
    float r = shapeRadiusF(halfB);

    bool inside = false;
    vec3 faceNormal = vec3(0.0f, 0.0f, 1.0f);
    vec3 onBox = closestPointOnOBB(box, posB, inside, faceNormal);

    vec3 delta = sub(posB, onBox);
    float reach = r + COLLISION_MARGIN_C;
    if (!inside && lengthSq3(delta) > reach * reach)
    {
        m.count = 0;
        return 0;
    }

    m.normalAB = boxPointNormal(delta, inside, faceNormal);
    m.xA[0] = onBox;
    m.xB[0] = sub(posB, scale(m.normalAB, r));
    m.key[0] = SHAPE_FEATURE_TAG;
    m.count = 1;
    return 1;
}

bool testCapsuleAxis(OBB box, vec3 segDir, float halfSeg, float radius, vec3 delta, vec3 axis,
                     int index, inout SatAxis best)
{
    float lenSq = lengthSq3(axis);
    if (lenSq < SAT_AXIS_EPSILON)
        return true;

    vec3 n = scale(axis, 1.0f / sqrt(lenSq));
    if (dot3(n, delta) < 0.0f)
        n = neg(n);

    float projBox = absDot(n, box.axis0) * box.half.x + absDot(n, box.axis1) * box.half.y +
                    absDot(n, box.axis2) * box.half.z;
    float projSeg = abs(dot3(n, segDir)) * halfSeg;

    float separation = abs(dot3(delta, n)) - projBox - projSeg - radius;
    if (separation > COLLISION_MARGIN_C)
        return false;

    if (separation > best.separation)
    {
        best.separation = separation;
        best.normalAB = n;
        best.indexA = index;
        best.valid = true;
    }
    return true;
}

int collideBoxCapsule(vec3 posA, quat rotA, vec3 halfA, vec3 posB, quat rotB, vec3 halfB,
                      inout ShapeManifold m)
{
    OBB box = makeOBB(posA, rotA, halfA);
    float radius = shapeRadiusF(halfB);
    float halfSeg = capsuleHalfSegmentF(halfB);

    vec3 p0, p1;
    capsuleSegment(posB, rotB, halfB, p0, p1);
    vec3 segDir = rotateQ(rotB, vec3(0.0f, 0.0f, 1.0f));
    vec3 delta = sub(posB, box.center);

    SatAxis bestFace;
    bestFace.separation = -3.402823e38f;
    bestFace.valid = false;
    SatAxis bestEdge;
    bestEdge.separation = -3.402823e38f;
    bestEdge.valid = false;

    for (int i = 0; i < 3; ++i)
        if (!testCapsuleAxis(box, segDir, halfSeg, radius, delta, axisVec(box, i), i, bestFace))
        { m.count = 0; return 0; }

    for (int i = 0; i < 3; ++i)
        if (!testCapsuleAxis(box, segDir, halfSeg, radius, delta, cross3(axisVec(box, i), segDir), -1, bestEdge))
        { m.count = 0; return 0; }

    {
        vec3 cA, cB;
        closestPointsOnSegments(box.center, box.center, p0, p1, cA, cB);
        bool inside = false;
        vec3 faceNormal = vec3(0.0f, 0.0f, 1.0f);
        vec3 onBox = closestPointOnOBB(box, cB, inside, faceNormal);
        if (!inside && lengthSq3(sub(cB, onBox)) >= OBB_DEGENERATE_DIST * OBB_DEGENERATE_DIST)
            if (!testCapsuleAxis(box, segDir, halfSeg, radius, delta, sub(cB, onBox), -1, bestEdge))
            { m.count = 0; return 0; }
    }

    if (!bestFace.valid) { m.count = 0; return 0; }

    SatAxis best = bestFace;
    if (bestEdge.valid)
    {
        const float edgeRelTol = 0.95f;
        const float edgeAbsTol = 0.01f;
        if (edgeRelTol * bestEdge.separation > bestFace.separation + edgeAbsTol)
            best = bestEdge;
    }

    m.count = 0;
    m.normalAB = best.normalAB;

    if (best.indexA >= 0 && halfSeg > SAT_AXIS_EPSILON)
    {
        FaceFrame face = buildFaceFrame(box, best.indexA, best.normalAB);

        float t0 = 0.0f;
        float t1 = 1.0f;
        vec3 d = sub(p1, p0);
        bool clipped = true;

        for (int side = 0; side < 4 && clipped; ++side)
        {
            vec3 axis = (side < 2) ? face.u : face.v;
            float extent = (side < 2) ? face.extentU : face.extentV;
            float s = (side & 1) != 0 ? -1.0f : 1.0f;

            float num = extent - dot3(scale(axis, s), sub(p0, face.center));
            float den = dot3(scale(axis, s), d);

            if (abs(den) < SAT_AXIS_EPSILON)
            {
                if (num < 0.0f)
                    clipped = false;
                continue;
            }

            float t = num / den;
            if (den > 0.0f)
                t1 = min(t1, t);
            else
                t0 = max(t0, t);
        }

        if (clipped && t1 - t0 > 1.0e-4f)
        {
            for (int i = 0; i < 2; ++i)
            {
                float t = (i == 0) ? t0 : t1;
                vec3 onSeg = add(p0, scale(d, t));
                float distance = dot3(sub(onSeg, face.center), face.normal);
                if (distance - radius > COLLISION_MARGIN_C)
                    continue;
                vec3 onFace = sub(onSeg, scale(face.normal, distance));
                if (m.count >= MAX_CONTACTS)
                    break;
                m.xA[m.count] = onFace;
                m.xB[m.count] = sub(onSeg, scale(m.normalAB, radius));
                m.key[m.count] = SHAPE_FEATURE_TAG | (2 + i);
                ++m.count;
            }
            if (m.count > 0)
                return m.count;
        }
    }

    vec3 deepest = (dot3(p0, m.normalAB) <= dot3(p1, m.normalAB)) ? p0 : p1;
    if (halfSeg > SAT_AXIS_EPSILON && abs(dot3(m.normalAB, segDir)) < 0.999f)
    {
        vec3 cA, cB;
        closestPointsOnSegments(box.center, box.center, p0, p1, cA, cB);
        if (dot3(cB, m.normalAB) < dot3(deepest, m.normalAB) + SAT_AXIS_EPSILON)
            deepest = cB;
    }

    bool inside = false;
    vec3 faceNormal = vec3(0.0f, 0.0f, 1.0f);
    vec3 onBox = closestPointOnOBB(box, deepest, inside, faceNormal);

    m.xA[0] = onBox;
    m.xB[0] = sub(deepest, scale(m.normalAB, radius));
    m.key[0] = SHAPE_FEATURE_TAG | 4;
    m.count = 1;
    return 1;
}

// collideShapes: the public dispatcher, mirroring the SAT collider's collideShapes. Box-box
// is delegated to collideBoxesImpl (never reimplemented -- see the SAT collider's comment on
// why a second box-box path would be a determinism-oracle trap). Mixed pairs are
// canonicalised (box first) and un-swapped on the way out.
public int collideShapesImpl(vec3 posA, quat rotA, vec3 halfA, int shapeA,
                      vec3 posB, quat rotB, vec3 halfB, int shapeB,
                      out ContactOut contacts[MAX_CONTACTS], out float3x3 basisOut)
{
    bool roundA = shapeA != AVBD_SHAPE_BOX;
    bool roundB = shapeB != AVBD_SHAPE_BOX;

    if (!roundA && !roundB)
        return collideBoxesImpl(posA, rotA, halfA, posB, rotB, halfB, contacts, basisOut);

    ShapeManifold m;
    m.count = 0;
    bool swapped = false;
    int count = 0;

    if (roundA && roundB)
    {
        count = collideRoundRound(posA, rotA, halfA, shapeA, posB, rotB, halfB, shapeB, m);
    }
    else
    {
        swapped = roundA;
        vec3 boxPos = swapped ? posB : posA;
        quat boxRot = swapped ? rotB : rotA;
        vec3 boxHalf = swapped ? halfB : halfA;
        vec3 rndPos = swapped ? posA : posB;
        quat rndRot = swapped ? rotA : rotB;
        vec3 rndHalf = swapped ? halfA : halfB;
        int rndShape = swapped ? shapeA : shapeB;

        count = (rndShape != AVBD_SHAPE_CAPSULE)
                    ? collideBoxSphere(boxPos, boxRot, boxHalf, rndPos, rndHalf, m)
                    : collideBoxCapsule(boxPos, boxRot, boxHalf, rndPos, rndRot, rndHalf, m);
    }

    for (int k = 0; k < MAX_CONTACTS; ++k)
    {
        contacts[k].featureKey = -1;
        contacts[k].rA = vec3(0, 0, 0);
        contacts[k].rB = vec3(0, 0, 0);
    }

    if (count <= 0)
    {
        basisOut = orthonormalF(vec3(0, 0, 1));
        return 0;
    }

    if (swapped)
    {
        m.normalAB = neg(m.normalAB);
        for (int i = 0; i < count; ++i)
        {
            vec3 t = m.xA[i];
            m.xA[i] = m.xB[i];
            m.xB[i] = t;
        }
    }

    basisOut = orthonormalF(neg(m.normalAB));

    quat invA = conjugateQ(rotA);
    quat invB = conjugateQ(rotB);
    for (int i = 0; i < count; ++i)
    {
        contacts[i].featureKey = m.key[i];
        contacts[i].rA = rotateQ(invA, sub(m.xA[i], posA));
        contacts[i].rB = rotateQ(invB, sub(m.xB[i], posB));
    }

    return count;
}

// ---------------------------------------------------------------------------
// Shape-aware buffers/kernel: like BodyBoxSoA but with a per-body shape tag, so the
// same dispatch csCollideShapes drives box-box, round-round and mixed pairs -- unlike
// csCollideBoxes above, which is box-only (kept for the phase-3 spike's exact
// box-box comparison).
// ---------------------------------------------------------------------------
struct BodyShapeSoA
{
    vec3 *pos;
    quat *rot;
    vec3 *halfExtents;
    int *shape;
    uint count;
    uint _pad0;
};

struct CollideShapesArgs
{
    BodyShapeSoA a;
    BodyShapeSoA b;
    CollideOut out;
};

[shader("compute")]
[numthreads(64, 1, 1)]
void csCollideShapes(uint3 tid: SV_DispatchThreadID)
{
    CollideShapesArgs *args = (CollideShapesArgs *)pc.argsAddr;
    uint i = tid.x;
    if (i >= args->a.count)
        return;

    vec3 posA = args->a.pos[i];
    quat rotA = args->a.rot[i];
    vec3 halfA = args->a.halfExtents[i];
    int shapeA = args->a.shape[i];
    vec3 posB = args->b.pos[i];
    quat rotB = args->b.rot[i];
    vec3 halfB = args->b.halfExtents[i];
    int shapeB = args->b.shape[i];

    ContactOut contacts[MAX_CONTACTS];
    float3x3 basis;
    int count = collideShapesImpl(posA, rotA, halfA, shapeA, posB, rotB, halfB, shapeB, contacts, basis);

    args->out.count[i] = count;
    args->out.basis[i] = basis;
    for (int k = 0; k < MAX_CONTACTS; ++k)
        args->out.contacts[i * MAX_CONTACTS + k] = contacts[k];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csCollideBoxes(uint3 tid: SV_DispatchThreadID)
{
    CollideArgs *args = (CollideArgs *)pc.argsAddr;
    uint i = tid.x;
    if (i >= args->a.count)
        return;

    vec3 posA = args->a.pos[i];
    quat rotA = args->a.rot[i];
    vec3 halfA = args->a.halfExtents[i];
    vec3 posB = args->b.pos[i];
    quat rotB = args->b.rot[i];
    vec3 halfB = args->b.halfExtents[i];

    ContactOut contacts[MAX_CONTACTS];
    for (int k = 0; k < MAX_CONTACTS; ++k) { contacts[k].featureKey = -1; contacts[k].rA = vec3(0,0,0); contacts[k].rB = vec3(0,0,0); }
    float3x3 basis;
    int count = collideBoxesImpl(posA, rotA, halfA, posB, rotB, halfB, contacts, basis);

    args->out.count[i] = count;
    args->out.basis[i] = basis;
    for (int k = 0; k < MAX_CONTACTS; ++k)
        args->out.contacts[i * MAX_CONTACTS + k] = contacts[k];
}
