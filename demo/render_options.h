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

#pragma once

// The renderer's look, shared by both demos: the options, the "Graphics" panel that
// edits them, and the two presets. demo/vk_renderer.cpp
// implements every field here, so the look is a fact about one struct, not something to
// re-check control by control.

#include "maths.h"

enum ShadowMode
{
    SHADOW_OFF = 0,
    SHADOW_GROUND_PLANAR = 1,
    SHADOW_MAP = 2
};

enum BackgroundTheme
{
    BACKGROUND_DARK = 0,
    BACKGROUND_LIGHT = 1 // the original demo's
};

struct RenderOptions
{
    bool cull = true;
    ShadowMode shadowMode = SHADOW_GROUND_PLANAR;
    float shadowStrength = 0.40f;
    bool blockColoring = false;
    // Forces the per-instance colour through regardless of blockColoring. The debug
    // overlays write their result into the instance colour, so they need that channel
    // to reach the fragment shader without the user's flat-colour preference stomping
    // it -- and without silently rewriting that preference behind their back.
    bool forceInstanceColor = false;
    vec3 blockBaseColor = {0.80f, 0.84f, 0.90f};
    bool blockOutlines = true;
    float outlineThickness = 1.0f;
    vec3 outlineColor = {0.10f, 0.12f, 0.14f};
    bool unlit = false;
    vec3 lightDir = {-0.45f, -1.0f, 0.95f};
    vec3 camTarget = {0, 0, 5.0f};
    float camDistance = 40.0f;
    vec3 groundPlanePoint = {0, 0, 0};
    vec3 groundPlaneNormal = {0, 1, 0};

    bool ssaoEnabled = false;
    float ssaoRadius = 10.0f;
    float ssaoBias = 0.1f;
    float ssaoStrength = 1.0f;

    BackgroundTheme background = BACKGROUND_DARK;
};

// The light both demos use. Every frame sets it into RenderOptions::lightDir.
inline vec3 demoLightDir() { return normalize(vec3{-0.4f, -0.75f, 0.5f}); }

// Clear colour for the theme, RGBA.
void backgroundColor(BackgroundTheme theme, float out[4]);

void applyOriginalLook(RenderOptions &o);
void applyModernLook(RenderOptions &o);

// The "Graphics" panel section: every RenderOptions control, shared by both demos.
void renderOptionsUI(RenderOptions &o);

// Shadow-map light view-projection: an orthographic light that follows the camera
// target, sized by the orbit distance. vulkanClip gives Vulkan's clip space (depth
// 0..1, y down) for a render target sampled at uv = ndc.xy * 0.5 + 0.5; otherwise the -1..1 (OpenGL-style) convention.
void shadowLightViewProj(const RenderOptions &o, bool vulkanClip, float out[16]);

// Planar ground-shadow projection: flattens world geometry onto the ground plane
// (lifted 5 cm against z-fighting) along the light. Multiply by viewProj to draw.
void planarShadowMatrix(const RenderOptions &o, float out[16]);
