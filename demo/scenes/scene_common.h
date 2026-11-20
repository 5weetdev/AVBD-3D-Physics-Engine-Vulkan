#pragma once

#include "scene_builder.h"

#include <cmath>

struct SceneView
{
    float camDistance;
    float camTargetHeight; // look-at height in the scene source's own (z-up) frame
};
