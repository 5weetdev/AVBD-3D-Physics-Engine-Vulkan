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

// What the 2D and 3D worlds' rerun loops share. A step that overflowed a capacity or recorded a
// bound that was too small is discarded, the resource grown or the bound widened, and the pass
// recorded again; these are the counters that report why, the pass ceiling, and the rules that
// widen a bound. Each world keeps its own record and grow code.

#include <algorithm>

namespace avbdvk
{

// A pass that is rerun more often than this has a bug, not a growing scene.
inline constexpr int kMaxRerunPasses = 64;

// Rerun counters common to StepResult (3D) and StepResult2D.
struct RerunCounters
{
    int capacityReruns = 0; // pair/contact capacity overflow: grow + rerun
    int jpReruns = 0;       // Jones-Plassmann round bound too small
    int colorReruns = 0;    // recorded colour bound too small
    int submits = 0;        // vkQueueSubmit calls the step made (1 unless it reran)
};

// Jones-Plassmann round bound too small: the coloring did not converge within it.
inline bool jpBoundShort(bool jpConverged, int rounds, int maxRounds)
{
    return !jpConverged && rounds < maxRounds;
}
inline int widenJpRounds(int rounds, int maxRounds)
{
    return std::min(2 * rounds, maxRounds);
}

// Recorded colour bound too small: the build produced more colours than the primal sweep was
// recorded for.
inline bool colorBoundShort(int bound, int builtColors, int maxColors)
{
    return bound < maxColors && builtColors > bound;
}
inline int widenColorBound(int builtColors, int maxColors)
{
    return std::min(builtColors + 2, maxColors);
}

} // namespace avbdvk
