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

// Scenes for the GPU solver.
//
// These are the scenes from the reference scenes.h, rebuilt against SceneBuilder::addBody
// instead of `new Rigid(solver, ...)` -- same shapes, same parameters. The jointed and
// sprung scenes use addJoint / addSpring / ignoreCollision, which stand in for the CPU's
// Joint, Spring and IgnoreCollision force objects.
//
// Concrete scenes are modularized into separate headers under `scenes/` to make
// inspecting and modifying individual scenes straightforward.

#include "scene_builder.h"

// Scene definition modules
#include "scenes/scene_common.h"
#include "scenes/scene_basic.h"
#include "scenes/scene_massive.h"
#include "scenes/scene_joints.h"
#include "scenes/scene_towers.h"
#include "scenes/scene_panel_house.h"
#include "scenes/scene_skyscraper.h"
#include "scenes/scene_suspension_bridge.h"
#include "scenes/scene_passenger_plane.h"

// The scene sources below are authored z-up (the paper's frame). The solver is y-up, so each
// catalogue entry builds its scene and then rotates it into the solver's frame; gpuScenes[]
// therefore yields y-up content. camTargetHeight stays in the source frame, which is the
// height along the up axis in both.
template <SceneView (*Build)(SceneBuilder *)> static SceneView yUpScene(SceneBuilder *s)
{
    const SceneView v = Build(s);
    if (!s->keepZUpSource)
        s->rotateZUpToYUp();
    return v;
}

static SceneView (*gpuScenes[])(SceneBuilder *) = {
    yUpScene<sceneEmptyGpu>,          yUpScene<sceneGroundGpu>,         yUpScene<sceneDynamicFrictionGpu>,
    yUpScene<sceneStaticFrictionGpu>, yUpScene<scenePyramidGpu>,        yUpScene<sceneStackGpu>,
    yUpScene<sceneStackRatioGpu>,     yUpScene<sceneStackRatioDeepGpu>, yUpScene<sceneFreshContactsGpu>,
    yUpScene<sceneMassive10kGpu>,     yUpScene<sceneMassive100kGpu>,    yUpScene<sceneMassive200kGpu>,
    yUpScene<sceneMassive400kGpu>,    yUpScene<sceneMassive600kGpu>,    yUpScene<scenePyramids10kGpu>,
    yUpScene<scenePyramids100kGpu>,   yUpScene<scenePyramids200kGpu>,   yUpScene<scenePyramids400kGpu>,
    yUpScene<sceneRopeGpu>,           yUpScene<sceneHeavyRopeGpu>,      yUpScene<sceneChainRatioGpu>,
    yUpScene<sceneSpringGpu>,         yUpScene<sceneSpringsRatioGpu>,   yUpScene<sceneSoftBodyGpu>,
    yUpScene<sceneBridgeGpu>,         yUpScene<sceneRoundTowers20kGpu>, yUpScene<sceneSquareTowers20kGpu>,
    yUpScene<scenePanelHouseGpu>,     yUpScene<sceneSkyscraperGpu>,     yUpScene<sceneSuspensionBridgeGpu>,
    yUpScene<scenePassengerPlaneGpu>};

static const char *gpuSceneNames[] = {
    "Empty",          "Ground",           "Dynamic Friction",  "Static Friction", "Pyramid",       "Stack",
    "Stack Ratio",    "Stack Ratio Deep", "Fresh Contacts",    "Massive 10k",     "Massive 100k",  "Massive 200k",
    "Massive 400k",   "Massive 600k",     "Pyramids 10k",      "Pyramids 100k",   "Pyramids 200k", "Pyramids 400k",
    "Rope",           "Heavy Rope",       "Chain Ratio",       "Spring",          "Spring Ratio",  "Soft Body",
    "Bridge",         "Round Towers 20k", "Square Towers 20k", "Panel House",     "Skyscraper",    "Suspension Bridge",
    "Passenger Plane"};

static const int gpuSceneCount = sizeof(gpuScenes) / sizeof(gpuScenes[0]);
