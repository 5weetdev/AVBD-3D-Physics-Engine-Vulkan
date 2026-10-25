// Vulkan physics demo: every graphics pass the renderer draws, one module: flat or per-instance colour, fwidth outlines, unlit, a 2048^2
// PCF shadow map, planar ground shadows and the 8-tap screen-space AO.
//
// Two vertex formats feed the same passes:
//   vsMesh    -- pos + normal (box, sphere, rods, debug markers): world = iPos + R(aPos*iHalf).
//   vsCapsule -- aDir + aSign: addCapsule()'s half-extent {r,r,r+halfSegment} is
//                non-uniform, so the caps are rebuilt as aDir*r + (0,0,aSign*halfSegment)
//                rather than scaled (see vk_renderer.h).
//
// PushConstants::pass picks the transform: 0 camera (main), 1 light (shadow depth),
// 2 camera (SSAO depth prepass), 3 planar projection. Passes 1-3 draw only dynamic
// bodies; the rest collapse to one point
// outside the clip volume.
//
// "Dynamic" is the instance alpha byte, as render.slang's csWriteInstances writes it: bodies
// get 0xFF - 2*shape (dynamic) or 0xFE - 2*shape (static), so a body is dynamic exactly
// when the byte is odd. Rods (0xFD) and markers (0xFC) share that byte space, which is
// why the draw says whether it is a body draw (PushConstants::bodies) -- a dynamic
// sphere is 0xFD too.

// Scalar layout (-fvk-use-scalar-layout). Mirrors VkRenderer's FrameParams exactly.
// Matrices are stored as four column float4s -- column-major, as DemoApp builds them --
// and applied by hand, so no matrix-layout convention is involved at all.
struct FrameParams
{
    float4 viewProj[4];
    float4 lightViewProj[4];
    float4 planarViewProj[4];
    float3 lightDir;
    uint shadowMode; // 0 off, 1 ground planar, 2 shadow map
    float3 blockBaseColor;
    uint useBlockColoring;
    float3 outlineColor;
    float outlineThickness;
    uint blockOutlines;
    uint unlit;
    float shadowStrength;
    uint ssaoEnabled;
    float ssaoRadius;
    float ssaoBias;
    float ssaoStrength;
    float zNear;
    float zFar;
    float2 screenSize;
    float shadowTexel;
};

struct PushConstants
{
    uint64_t frameAddr; // FrameParams for this frame in flight
    uint pass;
    uint bodies; // 1 when the draw is solver bodies, 0 for rods / debug markers
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] Texture2D<float> shadowMap;
[[vk::binding(1, 0)]] SamplerState shadowSampler;
[[vk::binding(2, 0)]] Texture2D<float> cameraDepth;
[[vk::binding(3, 0)]] SamplerState pointSampler;

static const uint PASS_MAIN = 0;
static const uint PASS_LIGHT_DEPTH = 1;
static const uint PASS_CAMERA_DEPTH = 2;
static const uint PASS_PLANAR = 3;

struct VsOut
{
    float4 clip : SV_Position;
    float3 normal : NORMAL0;
    float3 localNormal : NORMAL1;
    float3 localPos : TEXCOORD0;
    float4 color : COLOR0;
    float4 shadowCoord : TEXCOORD1;
    nointerpolation uint dyn : TEXCOORD2;
};

float3 qrot(float4 q, float3 v)
{
    float3 qv = float3(q.x, q.y, q.z);
    return v + 2.0 * cross(qv, cross(qv, v) + q.w * v);
}

float4 xform(float4 m[4], float3 p)
{
    return m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3];
}

VsOut shade(float3 local, float3 localNormal, float3 localPosForOutline, float3 iPos, float4 iRot, float4 iColor)
{
    FrameParams *f = (FrameParams *)pc.frameAddr;
    VsOut o;
    uint alphaByte = uint(iColor.a * 255.0 + 0.5);
    o.dyn = (pc.bodies != 0 && (alphaByte & 1u) != 0) ? 1u : 0u;

    float3 world = iPos + qrot(iRot, local);
    o.normal = qrot(iRot, localNormal);
    o.localNormal = localNormal;
    o.localPos = localPosForOutline;
    o.color = iColor;
    o.shadowCoord = float4(0, 0, 0, 1);

    if (pc.pass != PASS_MAIN && o.dyn == 0)
    {
        // Statics, rods and markers cast nothing: one point outside the clip volume.
        o.clip = float4(2.0, 2.0, 2.0, 1.0);
        return o;
    }

    if (pc.pass == PASS_LIGHT_DEPTH)
        o.clip = xform(f.lightViewProj, world);
    else if (pc.pass == PASS_PLANAR)
        o.clip = xform(f.planarViewProj, world);
    else
    {
        o.clip = xform(f.viewProj, world);
        o.shadowCoord = xform(f.lightViewProj, world);
    }
    return o;
}

struct MeshIn
{
    float3 aPos : POSITION;
    float3 aNormal : NORMAL;
    float3 iPos : IPOS;
    float4 iRot : IROT;
    float3 iHalf : IHALF;
    float4 iColor : ICOLOR;
};

[shader("vertex")]
VsOut vsMesh(MeshIn v)
{
    return shade(v.aPos * v.iHalf, v.aNormal, v.aPos, v.iPos, v.iRot, v.iColor);
}

struct CapsuleIn
{
    float3 aDir : POSITION;
    float aSign : TEXCOORD0;
    float3 iPos : IPOS;
    float4 iRot : IROT;
    float3 iHalf : IHALF;
    float4 iColor : ICOLOR;
};

[shader("vertex")]
VsOut vsCapsule(CapsuleIn v)
{
    float radius = v.iHalf.x;
    float halfSegment = v.iHalf.z - v.iHalf.x;
    float3 local = v.aDir * radius + float3(0, 0, v.aSign * halfSegment);
    // localPos == localNormal == aDir: a unit direction, so the outline distance
    // (1 - |p|) + |n| is 1 on every axis and a capsule, like a sphere, gets no edge lines.
    return shade(local, v.aDir, v.aDir, v.iPos, v.iRot, v.iColor);
}

float sampleShadowPCF(FrameParams *f, float4 sc, float3 n, float3 l)
{
    // Orthographic light, so w == 1; divide anyway. The light matrix is
    // built for Vulkan clip space (depth 0..1), so the coordinates only need x,y moved
    // into 0..1; z is already on the depth scale.
    float3 p = sc.xyz / sc.w;
    float2 uv = p.xy * 0.5 + 0.5;
    if (p.z > 1.0 || uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
        return 1.0;

    float cosTheta = clamp(dot(n, -l), 0.0, 1.0);
    float bias = max(0.003 * (1.0 - cosTheta), 0.0008);
    float current = p.z - bias;

    float lit = 0.0;
    for (int x = -1; x <= 1; ++x)
        for (int y = -1; y <= 1; ++y)
        {
            float stored = shadowMap.SampleLevel(shadowSampler, uv + float2(x, y) * f.shadowTexel, 0);
            lit += (current > stored) ? 0.0 : 1.0;
        }
    return lit / 9.0;
}

float linearizeDepth(FrameParams *f, float d)
{
    // Inverse of DemoApp::perspective(vulkanClip): depth 0..1 maps near..far.
    return f.zNear * f.zFar / (f.zFar - d * (f.zFar - f.zNear));
}

float computeSSAO(FrameParams *f, float2 fragCoord)
{
    if (f.ssaoEnabled == 0)
        return 1.0;

    float2 uv = fragCoord / f.screenSize;
    float current = linearizeDepth(f, cameraDepth.SampleLevel(pointSampler, uv, 0));

    const float2 offsets[8] = {
        float2(1.5, 0.5), float2(-1.5, -0.5), float2(-0.5, 1.5), float2(0.5, -1.5),
        float2(2.5, -1.5), float2(-2.5, 1.5), float2(-1.5, -2.5), float2(1.5, 2.5)};

    float2 scaledTexel = (1.0 / f.screenSize) * f.ssaoRadius / current;
    float ao = 0.0;
    for (int i = 0; i < 8; ++i)
    {
        float s = linearizeDepth(f, cameraDepth.SampleLevel(pointSampler, uv + offsets[i] * scaledTexel, 0));
        // Occluded when the neighbour is nearer than this point, weighted down when it
        // is far nearer (a foreground silhouette, not a crease) to avoid halos.
        float rangeCheck = smoothstep(0.0, 1.0, f.ssaoRadius / abs(current - s));
        if (s < current - f.ssaoBias)
            ao += rangeCheck;
    }
    return 1.0 - (ao / 8.0) * f.ssaoStrength;
}

[shader("fragment")]
float4 fsMain(VsOut i) : SV_Target
{
    FrameParams *f = (FrameParams *)pc.frameAddr;

    float3 baseColor = i.color.rgb;
    if (f.useBlockColoring == 0 && i.dyn != 0)
        baseColor = f.blockBaseColor;

    if (f.blockOutlines != 0)
    {
        float3 d = 1.0 - abs(i.localPos);
        float3 faceDist = d + abs(i.localNormal);
        float minEdgeDist = min(faceDist.x, min(faceDist.y, faceDist.z));
        float pixelDist = minEdgeDist / max(fwidth(minEdgeDist), 1e-5);
        if (pixelDist < f.outlineThickness)
            baseColor = lerp(baseColor, f.outlineColor, clamp(f.outlineThickness - pixelDist, 0.0, 1.0));
    }

    return lit(i, baseColor, normalize(i.normal));
}

float4 lit(VsOut i, float3 baseColor, float3 n)
{
    FrameParams *f = (FrameParams *)pc.frameAddr;
    if (f.unlit != 0)
        return float4(baseColor, 1.0);

    float diffuse = max(dot(n, -f.lightDir), 0.0);
    // A little bounce from below keeps the underside of a pile readable.
    float bounce = max(n.y, 0.0) * 0.15;

    float shadowFactor = 1.0;
    if (f.shadowMode == 2)
        shadowFactor = lerp(1.0 - f.shadowStrength, 1.0, sampleShadowPCF(f, i.shadowCoord, n, f.lightDir));

    float ssao = computeSSAO(f, i.clip.xy);

    float3 shaded = baseColor * ((0.30 + bounce) * ssao + 0.75 * diffuse * shadowFactor * ssao);
    return float4(shaded, 1.0);
}

[shader("fragment")]
float4 fsPlanar(VsOut i) : SV_Target
{
    FrameParams *f = (FrameParams *)pc.frameAddr;
    // shade() already collapses everything that is not a dynamic body in this pass, so
    // this never fires; it states the rule here too, and it is what makes the stage
    // read the flat `dyn` varying vsMesh/vsCapsule write (an interface with an
    // unconsumed output is what the validation layer flags on this pipeline).
    if (i.dyn == 0)
        discard;
    return float4(0.0, 0.0, 0.0, f.shadowStrength);
}
