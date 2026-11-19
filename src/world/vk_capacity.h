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

// Capacity growth shared by the 2D and 3D worlds. Capacities only affect how often a step reruns,
// never the numerics.

#include <algorithm>

namespace avbdvk
{

// Ceiling used when a world has no hard cap of its own.
constexpr int kDefaultCapacityCeiling = 20 * 1000 * 1000;

// need + 25% + slack, clamped to a ceiling so a runaway count cannot ask for the moon.
// A negative need (an overflowed counter) asks for the ceiling.
inline int cappedGrow(int need, int slack, int ceiling)
{
    if (need < 0)
        return ceiling;
    const long long grown = (long long)need + need / 4 + slack;
    return (int)std::min<long long>(grown, ceiling);
}

inline int nextPow2i(int v)
{
    int p = 1;
    while (p < v)
        p <<= 1;
    return p;
}

} // namespace avbdvk
