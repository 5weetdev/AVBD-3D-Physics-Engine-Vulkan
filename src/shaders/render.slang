// Ported from engine/render/instances/overlays/debug_draw (Phase 5C, completed).
//
// csWriteInstances now covers the reference k_write_instances in full: frustum culling
// (bounding-sphere test against 6 planes, compacted via an atomic `visible` counter),
// every OverlayMode (overlayColorFor below), and per-shape/group segmentation
// (shapeBase/shapeVisible, one atomic-compacted region per group).
//
// The overlay structural aggregates (the overlay reductions' k_overlay_stats_clear/
// _contacts/_forces and k_overlay_load_clear/_accum) are ported as their own kernels
// below: the host
// clears+accumulates into OverlayStats/an unsigned int load buffer, then passes the
// results into WriteInstancesArgs.ov exactly as the reference passes them into k_write_instances'
// OverlayView parameter.
//
// the debug-draw writer's k_write_debug_instances and k_write_force_instances are
// ported as csWriteDebugInstances, sharing rodRotation/
// basisRotation/debugRgba/emitInstance/jointAxialLoad with the same term order as the reference.
//
// csPackPose ports k_pack_pose (the instance writer) unchanged from the Phase 5C
// gate -- a plain per-index scatter, and now also the target of the real
// persistently-mapped pose buffer (see engine/vk/vk_buffer.h:MappedBuffer):
// PackPoseArgs.out can point directly at a MappedBuffer's device address, and the host
// reads the result back through its mapped pointer after the fence wait, with no
// readback copy -- mirroring the pose buffer's pinned-memory contract.

import maths;
import types;

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// Mirrors engine/vk/vk_args.h::BodyInstanceGpu / the device structs'
// BodyInstance, byte for byte (44 bytes: 3 float pos + 4 float rot + 3 float half +
// 1 uint rgba, no padding needed).
struct BodyInstance
{
    float3 pos;
    float4 rot;
    float3 half;
    uint rgba;
};

// Mirrors render_types's OverlayView, field for field. Every pointer may be null,
// in which case the overlay that wanted it falls back to the default hue (mirroring the
// the reference `break` in overlayColorFor's switch, which drops to the no-overlay-colour return).
struct OverlayView
{
    int mode;
    float speedMax;
    float loadMax;
    float degreeMax;
    float invCellSize;
    float sleepFrames;
    int colorCount;
    int colorStatics;
    int *graphColor;
    uint *load;
    int *degree;
    float jointLoadMax;
    float degreeJointMax;
    float penetrationMax;
    int *jointDegree;
    int *push;
    int *pull;
    int *stress;
    int *cone;
    int *penetration;
};

// OverlayMode, transcribed verbatim.
static const int OVERLAY_NONE = 0;
static const int OVERLAY_SLEEP = 1;
static const int OVERLAY_GRID = 2;
static const int OVERLAY_GRAPH_COLOR = 3;
static const int OVERLAY_LOAD = 4;
static const int OVERLAY_SPEED = 5;
static const int OVERLAY_DEGREE = 6;
static const int OVERLAY_JOINT_DEGREE = 7;
static const int OVERLAY_JOINT_LOAD = 8;
static const int OVERLAY_JOINT_STRESS = 9;
static const int OVERLAY_AXIAL_LOAD = 10;
static const int OVERLAY_FRICTION = 11;
static const int OVERLAY_PENETRATION = 12;

// ---------------------------------------------------------------------------
// Cell math -- local copy of broadphase.slang's packCellS/hashCellS/cellCoordS (bit
// for bit the same as the broadphase's packCell/hashCell/cellCoord), needed here
// only for OVERLAY_GRID. Duplicated rather than imported: broadphase.slang is a
// self-contained kernel file, not a shared module, matching how every other .slang
// file in this tree only imports maths/types.
// ---------------------------------------------------------------------------
uint64_t packCellR(int x, int y, int z)
{
    const uint64_t bias = uint64_t(1) << 20;
    uint64_t ux = (uint64_t(x) + bias) & uint64_t(0x1FFFFF);
    uint64_t uy = (uint64_t(y) + bias) & uint64_t(0x1FFFFF);
    uint64_t uz = (uint64_t(z) + bias) & uint64_t(0x1FFFFF);
    return (ux << 42) | (uy << 21) | uz;
}

uint hashCellR(uint64_t c, uint mask)
{
    c ^= c >> 33;
    c *= (uint64_t(0xff51afd7u) << 32) | uint64_t(0xed558ccdu);
    c ^= c >> 33;
    c *= (uint64_t(0xc4ceb9feu) << 32) | uint64_t(0x1a85ec53u);
    c ^= c >> 33;
    return (uint)c & mask;
}

int cellCoordR(float x, float invCellSize)
{
    return (int)floor(x * invCellSize);
}

// Same 32-bit mix the reference k_write_instances uses to turn a body index into a stable
// per-body hue -- murmur3-style finalizer, identical constants and shift amounts.
uint hashIndex(uint i)
{
    uint h = i;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

// HSV->RGB with the same sector/f/p/q/t decomposition as the reference source.
float3 hsvR(float h, float s, float v)
{
    h = h - floor(h);
    float sixth = h * 6.0f;
    int sector = (int)sixth;
    float f = sixth - (float)sector;
    float p = v * (1.0f - s), q = v * (1.0f - s * f), t = v * (1.0f - s * (1.0f - f));
    switch (sector % 6)
    {
    case 0: return float3(v, t, p);
    case 1: return float3(q, v, p);
    case 2: return float3(p, v, t);
    case 3: return float3(p, q, v);
    case 4: return float3(t, p, v);
    default: return float3(v, p, q);
    }
}

// The per-body hue used with no overlay on -- same 0.55/0.95 saturation/value as
// render_types's overlayHsv callers use for the graph-color and grid overlays, and
// the same decomposition csWriteInstances' pre-Phase-5C-completion default hue used.
float3 hueColor(uint h32)
{
    float hue = (float)(h32 & 0xFFFFu) / 65535.0f * 6.0f;
    int sector = (int)hue;
    float f = hue - (float)sector;
    const float s = 0.55f, v = 0.95f;
    float pv = v * (1.0f - s);
    float qv = v * (1.0f - s * f);
    float tv = v * (1.0f - s * (1.0f - f));
    switch (sector % 6)
    {
    case 0: return float3(v, tv, pv);
    case 1: return float3(qv, v, pv);
    case 2: return float3(pv, v, tv);
    case 3: return float3(pv, qv, v);
    case 4: return float3(tv, pv, v);
    default: return float3(v, pv, qv);
    }
}

// Black-body heat ramp: render_types, term for term.
float3 overlayHeat(float t)
{
    t = min(max(t, 0.0f), 1.0f);
    if (t < 0.25f)
    {
        float u = t * 4.0f;
        return float3(0.10f + 0.20f * u, 0.12f + 0.06f * u, 0.45f + 0.35f * u);
    }
    if (t < 0.5f)
    {
        float u = (t - 0.25f) * 4.0f;
        return float3(0.30f + 0.65f * u, 0.18f - 0.06f * u, 0.80f - 0.62f * u);
    }
    if (t < 0.75f)
    {
        float u = (t - 0.5f) * 4.0f;
        return float3(0.95f, 0.12f + 0.63f * u, 0.18f - 0.10f * u);
    }
    float u = (t - 0.75f) * 4.0f;
    return float3(0.95f + 0.05f * u, 0.75f + 0.25f * u, 0.08f + 0.87f * u);
}

// Diverging ramp: render_types, term for term.
float3 overlayDiverge(float t)
{
    t = min(max(t, -1.0f), 1.0f);
    float3 mid = float3(0.82f, 0.82f, 0.84f);
    if (t >= 0.0f)
    {
        float u = sqrt(t);
        return float3(mid.x + 0.13f * u, mid.y - 0.62f * u, mid.z - 0.60f * u);
    }
    float u = sqrt(-t);
    return float3(mid.x - 0.70f * u, mid.y - 0.36f * u, mid.z + 0.10f * u);
}

// render_types -- golden-ratio hue spacing.
float3 overlayIndexColor(uint i, float sat, float val)
{
    return hsvR((float)i * 0.61803399f, sat, val);
}

// render_types, same derivation and term order.
float jointAxialLoadR(vec3 lambdaLin, vec3 pA, vec3 pB)
{
    vec3 d = sub(pA, pB);
    float len = length3(d);
    if (len < 1e-6f)
        return 0.0f;
    return dot3(lambdaLin, divs(d, len));
}

// render_types, ported switch-for-switch, break-for-break (a Slang
// `break` inside a `switch` falls through to the same "no overlay colour" sentinel
// return the reference version's `break` falls through to).
float3 overlayColorFor(OverlayView ov, BodySoA b, int i, out bool has)
{
    has = true;
    switch (ov.mode)
    {
    case OVERLAY_SLEEP:
    {
        if (!b.awake[i])
            return float3(0.16f, 0.38f, 0.98f);
        float t = ov.sleepFrames > 0.0f
                      ? min((float)b.sleepTimer[i] / ov.sleepFrames, 1.0f)
                      : 0.0f;
        return float3(0.98f - 0.55f * t, 0.55f + 0.15f * t, 0.12f + 0.62f * t);
    }
    case OVERLAY_GRID:
    {
        vec3 p = b.pos[i];
        uint64_t cell = packCellR(cellCoordR(p.x, ov.invCellSize),
                                  cellCoordR(p.y, ov.invCellSize),
                                  cellCoordR(p.z, ov.invCellSize));
        uint h = hashCellR(cell, 0xFFFFFFFFu);
        return overlayIndexColor(h & 0xFFFFu, 0.72f, 0.72f + 0.28f * (float)((h >> 16) & 1));
    }
    case OVERLAY_GRAPH_COLOR:
    {
        if (ov.graphColor == nullptr)
            break;
        int c = ov.graphColor[i];
        if (c < 0)
            return float3(0.35f, 0.36f, 0.40f);
        return overlayIndexColor((uint)c, 0.70f, 0.95f);
    }
    case OVERLAY_LOAD:
    {
        if (ov.load == nullptr)
            break;
        float f = asfloat(ov.load[i]);
        return overlayHeat(sqrt(min(f / max(ov.loadMax, 1e-6f), 1.0f)));
    }
    case OVERLAY_SPEED:
        return overlayHeat(length3(b.velLin[i]) / max(ov.speedMax, 1e-6f));
    case OVERLAY_DEGREE:
    {
        if (ov.degree == nullptr)
            break;
        return overlayHeat((float)ov.degree[i] / max(ov.degreeMax, 1e-6f));
    }
    case OVERLAY_JOINT_DEGREE:
    {
        if (ov.jointDegree == nullptr)
            break;
        int d = ov.jointDegree[i];
        if (d == 0)
            return float3(0.35f, 0.36f, 0.40f);
        return overlayHeat((float)d / max(ov.degreeJointMax, 1e-6f));
    }
    case OVERLAY_JOINT_LOAD:
    case OVERLAY_AXIAL_LOAD:
    {
        if (ov.push == nullptr || ov.pull == nullptr)
            break;
        float push = asfloat(ov.push[i]);
        float pull = asfloat(ov.pull[i]);
        float m = max(ov.jointLoadMax, 1e-6f);
        float signed_ = (pull >= push ? pull : -push) / m;
        return overlayDiverge(signed_);
    }
    case OVERLAY_JOINT_STRESS:
    {
        if (ov.stress == nullptr)
            break;
        float s = asfloat(ov.stress[i]);
        if (s < 0.0f)
            return float3(0.35f, 0.36f, 0.40f);
        return overlayHeat(s);
    }
    case OVERLAY_FRICTION:
    {
        if (ov.cone == nullptr)
            break;
        float c = asfloat(ov.cone[i]);
        if (c < 0.0f)
            return float3(0.35f, 0.36f, 0.40f);
        return overlayHeat(c);
    }
    case OVERLAY_PENETRATION:
    {
        if (ov.penetration == nullptr)
            break;
        float d = asfloat(ov.penetration[i]);
        return overlayHeat(sqrt(min(d / max(ov.penetrationMax, 1e-9f), 1.0f)));
    }
    default:
        break;
    }
    has = false;
    return float3(0.0f, 0.0f, 0.0f);
}

struct WriteInstancesArgs
{
    BodySoA b;
    BodyInstance *out;
    int maxInstances;
    int _pad0;
    float *frustum;   // 24 floats (6 planes * float4), or null
    int *visible;     // atomic compaction counter, or null
    OverlayView ov;
    int *groupOf;     // per-SHAPE render group, or null (falls back to the shape's type)
    int *shapeBase;   // per-group base slot, or null (disables segmentation)
    int *shapeVisible; // per-group atomic counter, or null
    ShapeSoA s;       // one instance is written per shape
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csWriteInstances(uint3 tid: SV_DispatchThreadID)
{
    WriteInstancesArgs *args = (WriteInstancesArgs *)pc.argsAddr;
    BodySoA b = args->b;
    ShapeSoA sh = args->s;

    // One thread per SHAPE. `si` is the shape (and the dense output slot), `i` its body:
    // colour, overlay and mass are the body's, geometry the shape's.
    int si = (int)tid.x;
    if ((uint)si >= sh.count)
        return;
    int i = sh.body[si];

    vec3 p = b.pos[i];
    quat q = b.rot[i];
    if ((sh.flags[si] & 1u) == 0u)
    {
        p = add(p, rotateQ(q, sh.localPos[si]));
        q = mulQQ(q, sh.localRot[si]);
    }
    vec3 h = sh.halfExtents[si];

    // Hulls are not drawn yet (type 4). Only the dense, uncompacted layout has a slot to fill.
    if (sh.shapeType[si] == (uint8_t)4 || sh.shapeType[si] == (uint8_t)5)
    {
        if (args->shapeBase == nullptr && args->frustum == nullptr && si < args->maxInstances)
        {
            BodyInstance hidden;
            hidden.pos = float3(p.x, p.y, p.z);
            hidden.rot = float4(0.0f, 0.0f, 0.0f, 1.0f);
            hidden.half = float3(0.0f, 0.0f, 0.0f);
            hidden.rgba = 0u;
            args->out[si] = hidden;
        }
        return;
    }

    if (args->frustum != nullptr)
    {
        // Bounding-sphere test against the 6 view planes, same over-estimate the reference takes
        // (radius = half-extent diagonal, rotation-invariant).
        float r = sqrt(h.x * h.x + h.y * h.y + h.z * h.z);
        for (int pl = 0; pl < 6; pl++)
        {
            float ex = args->frustum[pl * 4 + 0];
            float ey = args->frustum[pl * 4 + 1];
            float ez = args->frustum[pl * 4 + 2];
            float ew = args->frustum[pl * 4 + 3];
            if (ex * p.x + ey * p.y + ez * p.z + ew < -r)
                return;
        }
    }

    BodyInstance inst;
    inst.pos = float3(p.x, p.y, p.z);
    inst.rot = float4(q.x, q.y, q.z, q.w);
    inst.half = float3(h.x, h.y, h.z);

    int shape = (int)sh.shapeType[si];
    uint alphaDyn = (uint)(0xFF - 2 * shape) << 24;
    uint alphaSta = (uint)(0xFE - 2 * shape) << 24;

    float3 ovColor = float3(-1.0f, 0.0f, 0.0f);
    bool hasOv = false;
    if (args->ov.mode != OVERLAY_NONE && (b.mass[i] > 0.0f || args->ov.colorStatics != 0))
        ovColor = overlayColorFor(args->ov, b, i, hasOv);

    if (hasOv)
    {
        inst.rgba = (b.mass[i] > 0.0f ? alphaDyn : alphaSta) |
                    ((uint)(min(ovColor.z, 1.0f) * 255.0f) << 16) |
                    ((uint)(min(ovColor.y, 1.0f) * 255.0f) << 8) |
                    (uint)(min(ovColor.x, 1.0f) * 255.0f);
    }
    else if (b.mass[i] <= 0.0f)
    {
        inst.rgba = alphaSta | 0x006E6E64u;
    }
    else
    {
        uint h32 = hashIndex((uint)i);
        float3 col = hueColor(h32);
        inst.rgba = alphaDyn |
                    ((uint)(col.z * 255.0f) << 16) |
                    ((uint)(col.y * 255.0f) << 8) |
                    (uint)(col.x * 255.0f);
    }

    int slot;
    if (args->shapeBase != nullptr)
    {
        int grp = args->groupOf != nullptr ? args->groupOf[si] : shape;
        int prev;
        InterlockedAdd(args->shapeVisible[grp], 1, prev);
        slot = args->shapeBase[grp] + prev;
    }
    else if (args->frustum != nullptr)
    {
        int prev;
        InterlockedAdd(*args->visible, 1, prev);
        slot = prev;
    }
    else
    {
        slot = si;
    }

    if (slot < args->maxInstances)
        args->out[slot] = inst;
}

// ---------------------------------------------------------------------------
// Overlay structural aggregates (the overlay reductions)
// ---------------------------------------------------------------------------

struct OverlayStats
{
    int *push;
    int *pull;
    int *stress;
    int *cone;
    int *penetration;
    int *jointDegree;
};

void statMax(int *dst, int i, float v)
{
    if (dst != nullptr)
    {
        int old;
        InterlockedMax(dst[i], asint(v), old);
    }
}

struct OverlayStatsClearArgs
{
    OverlayStats s;
    int n;
    int _pad0;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csOverlayStatsClear(uint3 tid: SV_DispatchThreadID)
{
    OverlayStatsClearArgs *args = (OverlayStatsClearArgs *)pc.argsAddr;
    int i = (int)tid.x;
    if (i >= args->n)
        return;
    const int zero = asint(0.0f);
    const int none = asint(-1.0f);
    if (args->s.push != nullptr) args->s.push[i] = zero;
    if (args->s.pull != nullptr) args->s.pull[i] = zero;
    if (args->s.penetration != nullptr) args->s.penetration[i] = zero;
    if (args->s.stress != nullptr) args->s.stress[i] = none;
    if (args->s.cone != nullptr) args->s.cone[i] = none;
    if (args->s.jointDegree != nullptr) args->s.jointDegree[i] = 0;
}

struct OverlayStatsContactsArgs
{
    ManifoldSoA m;
    int slots;
    int _pad0;
    OverlayStats s;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csOverlayStatsContacts(uint3 tid: SV_DispatchThreadID)
{
    OverlayStatsContactsArgs *args = (OverlayStatsContactsArgs *)pc.argsAddr;
    ManifoldSoA m = args->m;
    int t = (int)tid.x;
    if (t >= args->slots || m.contactCount[t] <= 0)
        return;

    int off = m.contactOffset[t];
    float friction = m.friction[t];

    float peakForce = 0.0f;
    float peakCone = 0.0f;
    float peakPen = 0.0f;

    for (int c = 0; c < m.contactCount[t]; c++)
    {
        vec3 lam = m.lambda[off + c];
        float fn = abs(lam.x);
        peakForce = max(peakForce, fn);

        if (args->s.cone != nullptr)
        {
            float bounds = fn * friction;
            float ft = sqrt(lam.y * lam.y + lam.z * lam.z);
            if (bounds > 1e-6f)
                peakCone = max(peakCone, min(ft / bounds, 1.0f));
        }

        if (args->s.penetration != nullptr)
            peakPen = max(peakPen, -m.C0[off + c].x);
    }

    int ia = m.bodyA[t], ib = m.bodyB[t];
    statMax(args->s.push, ia, peakForce);
    statMax(args->s.push, ib, peakForce);
    statMax(args->s.cone, ia, peakCone);
    statMax(args->s.cone, ib, peakCone);
    statMax(args->s.penetration, ia, peakPen);
    statMax(args->s.penetration, ib, peakPen);
}

struct OverlayStatsForcesArgs
{
    BodySoA b;
    JointSoA j;
    SpringSoA sp;
    int jointsAuthored;
    int _pad0;
    OverlayStats s;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csOverlayStatsForces(uint3 tid: SV_DispatchThreadID)
{
    OverlayStatsForcesArgs *args = (OverlayStatsForcesArgs *)pc.argsAddr;
    BodySoA b = args->b;
    JointSoA j = args->j;
    SpringSoA sp = args->sp;

    int t = (int)tid.x;
    int total = args->jointsAuthored + (int)sp.count;
    if (t >= total)
        return;

    bool isJoint = t < args->jointsAuthored;
    int i = isJoint ? t : t - args->jointsAuthored;

    int ia, ib;
    vec3 rA, rB;
    if (isJoint)
    {
        if (j.broken[i])
            return;
        ia = j.bodyA[i];
        ib = j.bodyB[i];
        rA = j.rA[i];
        rB = j.rB[i];
    }
    else
    {
        ia = sp.bodyA[i];
        ib = sp.bodyB[i];
        rA = sp.rA[i];
        rB = sp.rB[i];
    }
    if (ib < 0 || (uint)ib >= b.count)
        return;

    if (args->s.jointDegree != nullptr)
    {
        if (ia >= 0)
        {
            int old;
            InterlockedAdd(args->s.jointDegree[ia], 1, old);
        }
        int old2;
        InterlockedAdd(args->s.jointDegree[ib], 1, old2);
    }

    vec3 pA = ia >= 0 ? transformQ(b.pos[ia], b.rot[ia], rA) : rA;
    vec3 pB = transformQ(b.pos[ib], b.rot[ib], rB);

    float axial = 0.0f;
    if (isJoint)
        axial = jointAxialLoadR(j.lambdaLin[i], pA, pB);
    else
    {
        float len = length3(sub(pB, pA));
        axial = sp.stiffness[i] * (len - sp.rest[i]);
    }

    float tension = max(axial, 0.0f);
    float compression = max(-axial, 0.0f);
    if (ia >= 0)
    {
        statMax(args->s.pull, ia, tension);
        statMax(args->s.push, ia, compression);
    }
    statMax(args->s.pull, ib, tension);
    statMax(args->s.push, ib, compression);

    if (args->s.stress != nullptr && isJoint)
    {
        float frac = j.fracture[i];
        if (!isinf(frac) && frac > 0.0f && !isnan(frac))
        {
            float f = min(max(length3(j.lambdaAng[i]), length3(j.lambdaLin[i])) / frac, 1.0f);
            if (ia >= 0)
                statMax(args->s.stress, ia, f);
            statMax(args->s.stress, ib, f);
        }
    }
}

struct OverlayLoadClearArgs
{
    uint *load;
    int n;
    int _pad0;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csOverlayLoadClear(uint3 tid: SV_DispatchThreadID)
{
    OverlayLoadClearArgs *args = (OverlayLoadClearArgs *)pc.argsAddr;
    int i = (int)tid.x;
    if (i < args->n)
        args->load[i] = 0u;
}

struct OverlayLoadAccumArgs
{
    BodySoA b;
    ManifoldSoA m;
    int slots;
    int _pad0;
    uint *load;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csOverlayLoadAccum(uint3 tid: SV_DispatchThreadID)
{
    OverlayLoadAccumArgs *args = (OverlayLoadAccumArgs *)pc.argsAddr;
    ManifoldSoA m = args->m;
    int t = (int)tid.x;
    if (t >= args->slots || m.contactCount[t] <= 0)
        return;

    int off = m.contactOffset[t];
    float peak = 0.0f;
    for (int c = 0; c < m.contactCount[t]; c++)
        peak = max(peak, abs(m.lambda[off + c].x));

    uint bits = asuint(peak);
    uint old;
    InterlockedMax(args->load[m.bodyA[t]], bits, old);
    InterlockedMax(args->load[m.bodyB[t]], bits, old);
}

// ---------------------------------------------------------------------------
// Debug markers and force rods (the debug-draw writer)
// ---------------------------------------------------------------------------

// Shortest arc rotating +x onto unit vector d (the debug-draw writer).
quat rodRotation(vec3 d)
{
    float w = 1.0f + d.x;
    if (w < 1.0e-6f)
        return quat(0.0f, 0.0f, 1.0f, 0.0f);
    return normalizeQ(quat(0.0f, -d.z, d.y, w));
}

// Shepperd's method, pivoting on the largest diagonal term (the debug-draw writer).
quat basisRotation(float3x3 m)
{
    float m00 = m.row0.x, m01 = m.row1.x, m02 = m.row2.x;
    float m10 = m.row0.y, m11 = m.row1.y, m12 = m.row2.y;
    float m20 = m.row0.z, m21 = m.row1.z, m22 = m.row2.z;

    float tr = m00 + m11 + m22;
    quat q;
    if (tr > 0.0f)
    {
        float s = sqrt(tr + 1.0f) * 2.0f;
        q = quat((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25f * s);
    }
    else if (m00 > m11 && m00 > m22)
    {
        float s = sqrt(1.0f + m00 - m11 - m22) * 2.0f;
        q = quat(0.25f * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s);
    }
    else if (m11 > m22)
    {
        float s = sqrt(1.0f + m11 - m00 - m22) * 2.0f;
        q = quat((m01 + m10) / s, 0.25f * s, (m12 + m21) / s, (m02 - m20) / s);
    }
    else
    {
        float s = sqrt(1.0f + m22 - m00 - m11) * 2.0f;
        q = quat((m02 + m20) / s, (m12 + m21) / s, 0.25f * s, (m10 - m01) / s);
    }
    return normalizeQ(q);
}

uint debugRgba(vec3 c)
{
    return 0xFC000000u |
           ((uint)(min(max(c.z, 0.0f), 1.0f) * 255.0f) << 16) |
           ((uint)(min(max(c.y, 0.0f), 1.0f) * 255.0f) << 8) |
           (uint)(min(max(c.x, 0.0f), 1.0f) * 255.0f);
}

void emitInstance(BodyInstance *out, int maxInstances, int *counter,
                  vec3 p, quat q, vec3 half, uint rgba)
{
    int slot;
    InterlockedAdd(*counter, 1, slot);
    if (slot >= maxInstances)
        return;

    BodyInstance inst;
    inst.pos = float3(p.x, p.y, p.z);
    inst.rot = float4(q.x, q.y, q.z, q.w);
    inst.half = float3(half.x, half.y, half.z);
    inst.rgba = rgba;
    out[slot] = inst;
}

struct DebugDrawView
{
    int contactPoints;
    int contactNormals;
    int manifoldLinks;
    int tint;
    float pointSize;
    float normalLength;
    float forceMax;
};

struct WriteDebugInstancesArgs
{
    BodySoA b;
    ManifoldSoA m;
    int slots;
    int _pad0;
    BodyInstance *out;
    int maxInstances;
    int _pad1;
    int *counter;
    DebugDrawView d;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csWriteDebugInstances(uint3 tid: SV_DispatchThreadID)
{
    WriteDebugInstancesArgs *args = (WriteDebugInstancesArgs *)pc.argsAddr;
    BodySoA b = args->b;
    ManifoldSoA m = args->m;
    DebugDrawView d = args->d;

    int t = (int)tid.x;
    if (t >= args->slots || m.contactCount[t] <= 0)
        return;

    int ia = m.bodyA[t], ib = m.bodyB[t];
    int off = m.contactOffset[t];
    int n = m.contactCount[t];
    float3x3 basis = m.basis[t];
    float friction = m.friction[t];

    vec3 posA = b.pos[ia], posB = b.pos[ib];
    quat rotA = b.rot[ia], rotB = b.rot[ib];

    float manifoldForce = 0.0f;

    if (d.contactPoints != 0 || d.contactNormals != 0)
    {
        quat plate = basisRotation(basis);
        for (int c = 0; c < n; c++)
        {
            vec3 lam = m.lambda[off + c];
            float fn = abs(lam.x);
            manifoldForce += fn;

            vec3 pA = transformQ(posA, rotA, m.rA[off + c]);
            vec3 pB = transformQ(posB, rotB, m.rB[off + c]);
            vec3 mid = scale(add(pA, pB), 0.5f);

            vec3 col;
            if (d.tint == 1)
            {
                float bounds = fn * friction;
                if (bounds <= 1e-6f)
                    col = vec3(0.45f, 0.46f, 0.50f);
                else
                {
                    float ft = sqrt(lam.y * lam.y + lam.z * lam.z);
                    float3 hc = overlayHeat(min(ft / bounds, 1.0f));
                    col = vec3(hc.x, hc.y, hc.z);
                }
            }
            else if (d.tint == 2)
            {
                col = (m.stick[off + c] & 1) != 0 ? vec3(0.20f, 0.85f, 0.35f)
                                             : vec3(1.00f, 0.45f, 0.10f);
            }
            else if (d.tint == 3)
            {
                col = vec3(1.0f, 0.95f, 0.25f);
            }
            else
            {
                float3 hc = overlayHeat(sqrt(min(fn / max(d.forceMax, 1e-6f), 1.0f)));
                col = vec3(hc.x, hc.y, hc.z);
            }
            uint rgba = debugRgba(col);

            if (d.contactPoints != 0)
            {
                float s = d.pointSize;
                emitInstance(args->out, args->maxInstances, args->counter, mid, plate,
                            vec3(s * 0.25f, s, s), rgba);
            }

            if (d.contactNormals != 0)
            {
                float len = d.normalLength * min(fn / max(d.forceMax, 1e-6f), 1.0f);
                if (len > 1e-4f)
                {
                    vec3 nrm = basis.row0;
                    float r = d.pointSize * 0.35f;
                    emitInstance(args->out, args->maxInstances, args->counter,
                                add(mid, scale(nrm, len * 0.5f)), rodRotation(nrm),
                                vec3(len * 0.5f, r, r), rgba);
                }
            }
        }
    }

    if (d.manifoldLinks != 0)
    {
        if (d.contactPoints == 0 && d.contactNormals == 0)
            for (int c = 0; c < n; c++)
                manifoldForce += abs(m.lambda[off + c].x);

        vec3 dir = sub(posB, posA);
        float len = length3(dir);
        if (len > 1e-5f)
        {
            dir = divs(dir, len);
            float3 hc = overlayHeat(sqrt(min(manifoldForce / max(d.forceMax, 1e-6f), 1.0f)));
            vec3 col = vec3(hc.x, hc.y, hc.z);
            float r = d.pointSize * 0.5f;
            emitInstance(args->out, args->maxInstances, args->counter,
                        scale(add(posA, posB), 0.5f), rodRotation(dir),
                        vec3(len * 0.5f, r, r), debugRgba(col));
        }
    }
}
