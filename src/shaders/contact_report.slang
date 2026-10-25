// Contact report (api3d contact and hit events). One kernel, csContactReport, run once per
// step after the velocity reconstruction. Thread t looks at manifold slot t (the shape-pair
// cache, the same indexing the narrowphase writes). A slot is reported when it is touching
// (contactCount > 0) and at least one of its two shapes carries AVBD_SHAPE_FLAG_REPORT.
//
// Each reported manifold writes one ContactReportRecord (192 bytes) at an index taken from an
// atomic counter. The counter counts every reported manifold, even past the capacity: the
// host reads `required` back, and a step whose required exceeds the capacity is marked
// truncated (the excess records are dropped, never written past the buffer). Nothing here
// reruns the step.
//
// Record fields (world space, A -> B along the normal):
//   normal          -n_AB, from shape A toward shape B (manifold basis row 0 is -n_AB)
//   approachSpeed   -(vB - vA) . n using the LINEAR velocities at the start of the step
//                   (prevVelLin). Positive while the pair closes. Angular velocity is ignored.
//   normalImpulse   sum over all contacts of max(0, -lambda.x): the compressive impulse
//   tangentImpulse  sum over all contacts of |lambda.yz|
//   separation      min over all contacts of C0.x, the normal constraint error (margin
//                   included): negative while penetrating
//   pointCount      contacts in the manifold (all of them; points[] holds the first 4)
//   points[i]       anchorA/anchorB: contact point relative to the body centre of mass, in
//                   world orientation (rotate(rot, rA)); separation and normalImpulse as above.

import maths;

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

struct ContactReportPoint
{
    float ax;
    float ay;
    float az;
    float bx;
    float by;
    float bz;
    float separation;
    float normalImpulse;
};

struct ContactReportRecord
{
    int shapeA;
    int shapeB;
    int bodyA;
    int bodyB;
    float nx;
    float ny;
    float nz;
    float approachSpeed;
    float normalImpulse;
    float tangentImpulse;
    float separation;
    float _f0;
    uint pointCount;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    ContactReportPoint points[4];
};

struct ContactReportArgs
{
    // body state (read only)
    vec3 *pos;
    quat *rot;
    vec3 *prevVelLin;
    // manifold slots (read only)
    int *bodyA;
    int *bodyB;
    int *shapeA;
    int *shapeB;
    float3x3 *basis;
    int *contactOffset;
    int *contactCount;
    vec3 *rA;
    vec3 *rB;
    vec3 *C0;
    vec3 *lambda;
    // shape flags (read only). Shape ids index this array.
    uint *shapeFlags;
    // output
    int *counter;                 // [0]: reported manifolds this step (may exceed capacity)
    ContactReportRecord *records; // capacity entries
    uint capacity;
    uint slots;                   // manifold slots to scan
};

static const uint SHAPE_FLAG_REPORT = 2u;

[shader("compute")]
[numthreads(128, 1, 1)]
void csContactReport(uint3 tid: SV_DispatchThreadID)
{
    ContactReportArgs *a = (ContactReportArgs *)pc.argsAddr;
    uint t = tid.x;
    if (t >= a->slots)
        return;
    int n = a->contactCount[t];
    if (n <= 0)
        return;
    int sa = a->shapeA[t], sb = a->shapeB[t];
    if (((a->shapeFlags[sa] | a->shapeFlags[sb]) & SHAPE_FLAG_REPORT) == 0u)
        return;

    int ia = a->bodyA[t], ib = a->bodyB[t];
    vec3 nAB = neg(a->basis[t].row0);
    vec3 posA = a->pos[ia], posB = a->pos[ib];
    quat rotA = a->rot[ia], rotB = a->rot[ib];
    float approach = -dot3(sub(a->prevVelLin[ib], a->prevVelLin[ia]), nAB);

    float sep = 3.402823e38f;
    float normalImpulse = 0.0f, tangentImpulse = 0.0f;
    int off = a->contactOffset[t];
    int stored = min(n, 4);

    int idx;
    InterlockedAdd(a->counter[0], 1, idx);
    // Reads of the manifold are done either way; only the write is gated on the capacity.
    ContactReportRecord r;
    r.shapeA = sa;
    r.shapeB = sb;
    r.bodyA = ia;
    r.bodyB = ib;
    r.nx = nAB.x;
    r.ny = nAB.y;
    r.nz = nAB.z;
    r.approachSpeed = approach;
    r._f0 = 0.0f;
    r.pointCount = (uint)n;
    r._pad0 = 0u;
    r._pad1 = 0u;
    r._pad2 = 0u;

    for (int i = 0; i < n; ++i)
    {
        int o = off + i;
        vec3 c0 = a->C0[o];
        vec3 lam = a->lambda[o];
        vec3 wa = rotateQ(rotA, a->rA[o]);
        vec3 wb = rotateQ(rotB, a->rB[o]);
        float pn = max(0.0f, -lam.x);
        sep = min(sep, c0.x);
        normalImpulse += pn;
        tangentImpulse += sqrt(lam.y * lam.y + lam.z * lam.z);
        if (i < stored)
        {
            ContactReportPoint p;
            p.ax = wa.x; p.ay = wa.y; p.az = wa.z;
            p.bx = wb.x; p.by = wb.y; p.bz = wb.z;
            p.separation = c0.x;
            p.normalImpulse = pn;
            r.points[i] = p;
        }
    }
    r.normalImpulse = normalImpulse;
    r.tangentImpulse = tangentImpulse;
    r.separation = sep;
    if ((uint)idx < a->capacity)
        a->records[idx] = r;
}
