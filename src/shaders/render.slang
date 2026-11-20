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
// ported as csWriteDebugInstances / csWriteForceInstances, sharing rodRotation/
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

public struct PushConstants
{
    public uint64_t argsAddr;
};

[[vk::push_constant]]
public ConstantBuffer<PushConstants> pc;

// Mirrors engine/vk/vk_args.h::BodyInstanceGpu / the device structs'
// BodyInstance, byte for byte (44 bytes: 3 float pos + 4 float rot + 3 float half +
// 1 uint rgba, no padding needed).
public struct BodyInstance
{
    public float3 pos;
    public float4 rot;
    public float3 half;
    public uint rgba;
};

// Mirrors render_types's OverlayView, field for field. Every pointer may be null,
// in which case the overlay that wanted it falls back to the default hue (mirroring the
// the reference `break` in overlayColorFor's switch, which drops to the no-overlay-colour return).
public struct OverlayView
{
    public int mode;
    public float speedMax;
    public float loadMax;
    public float degreeMax;
    public float invCellSize;
    public float sleepFrames;
    public int colorCount;
    public int colorStatics;
    public int *graphColor;
    public uint *load;
    public int *degree;
    public float jointLoadMax;
    public float degreeJointMax;
    public float penetrationMax;
    public int *jointDegree;
    public int *push;
    public int *pull;
    public int *stress;
    public int *cone;
    public int *penetration;
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
public float3 overlayHeat(float t)
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
public float3 overlayDiverge(float t)
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
public float jointAxialLoadR(vec3 lambdaLin, vec3 pA, vec3 pB)
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

public struct WriteInstancesArgs
{
    public BodySoA b;
    public BodyInstance *out;
    public int maxInstances;
    public int _pad0;
    public float *frustum;   // 24 floats (6 planes * float4), or null
    public int *visible;     // atomic compaction counter, or null
    public OverlayView ov;
    public int *groupOf;     // per-SHAPE render group, or null (falls back to the shape's type)
    public int *shapeBase;   // per-group base slot, or null (disables segmentation)
    public int *shapeVisible; // per-group atomic counter, or null
    public ShapeSoA s;       // one instance is written per shape
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
