// Debug and force marker instance writers (split from render.slang).

import maths;
import types;
import render;

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
