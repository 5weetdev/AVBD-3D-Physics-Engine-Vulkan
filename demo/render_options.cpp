/*
 * Copyright (c) 2026 Chris Giles
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies.
 * Chris Giles makes no representations about the suitability
 * of this software for any purpose.
 * It is provided "as is" without express or implied warranty.
 */

#include "render_options.h"

#include "demo_app.h" // DemoApp::matMul

#include <imgui.h>
#include <math.h>

void backgroundColor(BackgroundTheme theme, float out[4])
{
    const float light[4] = {0.92f, 0.93f, 0.95f, 1.0f};
    const float dark[4] = {0.10f, 0.11f, 0.13f, 1.0f};
    const float *c = theme == BACKGROUND_LIGHT ? light : dark;
    for (int i = 0; i < 4; i++)
        out[i] = c[i];
}

void applyOriginalLook(RenderOptions &o)
{
    o.background = BACKGROUND_LIGHT;
    o.shadowMode = SHADOW_GROUND_PLANAR;
    o.shadowStrength = 0.35f;
    o.blockColoring = false;
    o.blockBaseColor = vec3{0.80f, 0.84f, 0.90f};
    o.blockOutlines = true;
    o.outlineThickness = 2.5f;
    o.outlineColor = vec3{0.10f, 0.12f, 0.14f};
    o.unlit = true;
    o.ssaoEnabled = false;
}

void applyModernLook(RenderOptions &o)
{
    o.background = BACKGROUND_DARK;
    o.shadowMode = SHADOW_MAP;
    o.shadowStrength = 0.50f;
    o.blockColoring = true;
    o.blockOutlines = true;
    o.outlineThickness = 1.8f;
    o.outlineColor = vec3{0.08f, 0.08f, 0.10f};
    o.unlit = false;
    o.ssaoEnabled = true;
}

void renderOptionsUI(RenderOptions &o)
{
    if (ImGui::Button("Original Demo Look"))
        applyOriginalLook(o);
    ImGui::SameLine();
    if (ImGui::Button("Modern Look"))
        applyModernLook(o);

    const char *shadowModes[] = {"Off", "Ground Planar (Original)", "Shadow Map (Full)"};
    int shadowIdx = (int)o.shadowMode;
    if (ImGui::Combo("Shadows", &shadowIdx, shadowModes, 3))
        o.shadowMode = (ShadowMode)shadowIdx;
    if (o.shadowMode != SHADOW_OFF)
        ImGui::SliderFloat("Shadow strength", &o.shadowStrength, 0.0f, 1.0f, "%.2f");

    ImGui::Checkbox("Block coloring", &o.blockColoring);
    if (!o.blockColoring)
        ImGui::ColorEdit3("Block color", &o.blockBaseColor.x);

    ImGui::Checkbox("Block outlines", &o.blockOutlines);
    if (o.blockOutlines)
    {
        ImGui::SliderFloat("Outline width", &o.outlineThickness, 1.0f, 6.0f, "%.1f px");
        ImGui::ColorEdit3("Outline color", &o.outlineColor.x);
    }

    ImGui::Checkbox("Unlit / Flat shading", &o.unlit);

    ImGui::Checkbox("Ambient Occlusion (SSAO)", &o.ssaoEnabled);
    if (o.ssaoEnabled)
    {
        ImGui::SliderFloat("SSAO Radius", &o.ssaoRadius, 0.1f, 50.0f, "%.1f");
        ImGui::SliderFloat("SSAO Bias", &o.ssaoBias, 0.0f, 1.0f, "%.3f");
        ImGui::SliderFloat("SSAO Strength", &o.ssaoStrength, 0.0f, 3.0f, "%.2f");
    }

    const char *bgThemes[] = {"Dark Theme", "Light Theme (Original)"};
    int bgIdx = (int)o.background;
    if (ImGui::Combo("Background", &bgIdx, bgThemes, 2))
        o.background = (BackgroundTheme)bgIdx;
}

namespace
{
void matOrtho(float out[16], float l, float r, float b, float t, float n, float f, bool vulkanClip)
{
    for (int i = 0; i < 16; i++)
        out[i] = 0.0f;
    out[0] = 2.0f / (r - l);
    out[12] = -(r + l) / (r - l);
    out[15] = 1.0f;
    if (vulkanClip)
    {
        // y flipped so the winding (and so which faces cull) flipped because Vulkan clip y points down, which keeps the winding; depth 0..1.
        out[5] = -2.0f / (t - b);
        out[13] = (t + b) / (t - b);
        out[10] = -1.0f / (f - n);
        out[14] = -n / (f - n);
    }
    else
    {
        out[5] = 2.0f / (t - b);
        out[13] = -(t + b) / (t - b);
        out[10] = -2.0f / (f - n);
        out[14] = -(f + n) / (f - n);
    }
}

void matLookAt(float out[16], vec3 eye, vec3 target, vec3 up)
{
    vec3 f = normalize(target - eye);
    vec3 s = cross(f, up);
    if (lengthSq(s) < 1e-6f)
        s = cross(f, vec3{0, 0, 1});
    s = normalize(s);
    vec3 u = cross(s, f);

    out[0] = s.x;  out[4] = s.y;  out[8] = s.z;   out[12] = -dot(s, eye);
    out[1] = u.x;  out[5] = u.y;  out[9] = u.z;   out[13] = -dot(u, eye);
    out[2] = -f.x; out[6] = -f.y; out[10] = -f.z; out[14] = dot(f, eye);
    out[3] = 0;    out[7] = 0;    out[11] = 0;    out[15] = 1;
}
} // namespace

void shadowLightViewProj(const RenderOptions &o, bool vulkanClip, float out[16])
{
    vec3 ldir = normalize(o.lightDir);
    vec3 eyePos = o.camTarget - ldir * (o.camDistance * 1.5f + 50.0f);
    float orthoSize = fmaxf(o.camDistance * 1.3f, 25.0f);
    float lightProj[16], lightView[16];
    matOrtho(lightProj, -orthoSize, orthoSize, -orthoSize, orthoSize, 1.0f, orthoSize * 4.0f + 100.0f, vulkanClip);
    matLookAt(lightView, eyePos, o.camTarget, vec3{0, 1, 0});
    DemoApp::matMul(out, lightProj, lightView);
}

void planarShadowMatrix(const RenderOptions &o, float out[16])
{
    vec3 n = normalize(o.groundPlaneNormal);
    vec3 p = o.groundPlanePoint;
    // Plane Ax + By + Cz + D = 0 with slight offset above ground
    const float plane[4] = {n.x, n.y, n.z, -(dot(n, p) + 0.05f)};
    vec3 lvec = -normalize(o.lightDir); // vector pointing towards light
    const float light[4] = {lvec.x, lvec.y, lvec.z, 0.0f};

    const float d = plane[0] * light[0] + plane[1] * light[1] + plane[2] * light[2] + plane[3] * light[3];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            out[c * 4 + r] = (r == c ? d : 0.0f) - light[r] * plane[c];
}
