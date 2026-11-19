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

// Conversions between the public avbd3d_* value types and the engine's maths types.

#include "avbd3d/avbd3d.h"
#include "maths.h"

#include <cstring>

namespace avbd3d_impl
{

// Same bytes, two type names: the avbd3d and Box3D structs are layout-identical (static_asserts in
// avbd3d_api.cpp).
template <class To, class From> To avbd3d_bit(const From &from)
{
    static_assert(sizeof(To) == sizeof(From), "avbd3d and Box3D layouts differ");
    To to;
    std::memcpy(&to, &from, sizeof(To));
    return to;
}

inline vec3 toV(avbd3d_Vec3 v)
{
    return vec3{v.x, v.y, v.z};
}
inline avbd3d_Vec3 fromV(vec3 v)
{
    return avbd3d_Vec3{v.x, v.y, v.z};
}
inline quat toQ(avbd3d_Quat q)
{
    return quat{q.v.x, q.v.y, q.v.z, q.s};
}
inline avbd3d_Quat fromQ(quat q)
{
    return avbd3d_Quat{avbd3d_Vec3{q.x, q.y, q.z}, q.w};
}

} // namespace avbd3d_impl
