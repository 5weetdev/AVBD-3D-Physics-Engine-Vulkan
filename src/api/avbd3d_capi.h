// Internal helpers shared by the public C API translation units (avbd3d_capi_gen.cpp,
// avbd3d_capi_special.cpp, avbd3d_geometry.cpp). The public API (include/avbd3d) is a thin layer over
// the world implementation, which keeps its own internal names (avbd3d_Xxx, avbd3d_internal.h).
// The two header sets have different names for everything, so they can be included together.
#pragma once

#include "avbd3d/avbd3d.h"
#include "avbd3d/avbd3d_geometry.h"
#include "avbd3d/avbd3d_vulkan.h"

#include "avbd3d_world.h"
#include "avbd3d_internal.h"

#include <cstring>

using avbd3d_impl::World;

static constexpr int kInternalValue = 1152023; // the legacy defs' internalValue cookie

#include <type_traits>

// Scalars pass through the conversions unchanged.
template <class T, class = typename std::enable_if<std::is_arithmetic<T>::value>::type> static inline T lg(T v) { return v; }

#include "avbd3d_capi_conv.h"

// The public result for a world's recorded error.
static inline Avbd3dResult mapError(avbd3d_Error e)
{
    switch (e)
    {
    case avbd3d_errorNone:
        return AVBD3D_OK;
    case avbd3d_errorDeviceLost:
        return AVBD3D_ERR_DEVICE_LOST;
    case avbd3d_errorOutOfMemory:
        return AVBD3D_ERR_OUT_OF_MEMORY;
    default:
        return AVBD3D_ERR_DEVICE;
    }
}

// Entry checks. A NULL or unknown world is NULL_HANDLE; a stale id is INVALID_ARG; a dead world
// answers every call with its error.
static inline Avbd3dResult enterWorld(Avbd3dWorld *p, World *&w, avbd3d_WorldId &lw)
{
    if (!p || !avbd3d_impl::validWorld(p))
        return AVBD3D_ERR_NULL_HANDLE;
    w = reinterpret_cast<World *>(p);
    if (w->error() != avbd3d_errorNone)
        return mapError(w->error());
    lw = avbd3d_impl::worldIdOf(w);
    return AVBD3D_OK;
}

static inline Avbd3dResult enterBody(Avbd3dBodyId id, World *&w, avbd3d_BodyId &lid)
{
    lid = lg(id);
    w = avbd3d_impl::byIndex(lid.world0);
    if (!w)
        return AVBD3D_ERR_INVALID_ARG;
    if (w->error() != avbd3d_errorNone)
        return mapError(w->error());
    return w->bodyValid(lid) ? AVBD3D_OK : AVBD3D_ERR_INVALID_ARG;
}

static inline Avbd3dResult enterShape(Avbd3dShapeId id, World *&w, avbd3d_ShapeId &lid)
{
    lid = lg(id);
    w = avbd3d_impl::byIndex(lid.world0);
    if (!w)
        return AVBD3D_ERR_INVALID_ARG;
    if (w->error() != avbd3d_errorNone)
        return mapError(w->error());
    return w->shapeValid(lid) ? AVBD3D_OK : AVBD3D_ERR_INVALID_ARG;
}

static inline Avbd3dResult enterJoint(Avbd3dJointId id, World *&w, avbd3d_JointId &lid)
{
    lid = lg(id);
    w = avbd3d_impl::byIndex(lid.world0);
    if (!w)
        return AVBD3D_ERR_INVALID_ARG;
    if (w->error() != avbd3d_errorNone)
        return mapError(w->error());
    return w->jointValid(lid) ? AVBD3D_OK : AVBD3D_ERR_INVALID_ARG;
}

static inline Avbd3dResult enterContact(Avbd3dContactId id, World *&w, avbd3d_ContactId &lid)
{
    lid = lg(id);
    w = avbd3d_impl::byIndex(lid.world0);
    if (!w)
        return AVBD3D_ERR_INVALID_ARG;
    if (w->error() != avbd3d_errorNone)
        return mapError(w->error());
    return w->contactValid(lid) ? AVBD3D_OK : AVBD3D_ERR_INVALID_ARG;
}

// Exit check: a call that killed the world reports why.
static inline Avbd3dResult leave(World *w)
{
    return w->error() != avbd3d_errorNone ? mapError(w->error()) : AVBD3D_OK;
}

// The public pointer of a live world, or NULL.
static inline Avbd3dWorld *worldHandle(avbd3d_WorldId id)
{
    return reinterpret_cast<Avbd3dWorld *>(avbd3d_impl::lookupWorld(id));
}
