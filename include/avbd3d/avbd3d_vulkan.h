/* SPDX-FileCopyrightText: 2026 avbd3d contributors
   SPDX-License-Identifier: MIT */

/* avbd3d Vulkan: a Vulkan device shared between worlds and the host renderer, the per-body
   render instances the solver packs on the GPU, and contact markers. Handles are passed as
   opaque values so this header needs no vulkan.h. */
#ifndef AVBD3D_VULKAN_H
#define AVBD3D_VULKAN_H

#include "avbd3d.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A Vulkan device and queue. Opaque. Create one, then create worlds on it with
   avbd3d_create_world_on_gpu: worlds on one Gpu share its device and the host renderer's
   resources. Destroy every world before the Gpu. */
typedef struct Avbd3dGpu Avbd3dGpu;

/* Makes the presentation surface against the GPU's instance, e.g.
   SDL_Vulkan_CreateSurface(window, (VkInstance)instance, &surface). Returns the VkSurfaceKHR,
   or 0 on failure. The GPU destroys the surface when it is destroyed. */
typedef uint64_t Avbd3dMakeSurfaceFcn( void* vkInstance, void* context );

/* Use avbd3d_default_gpu_def() to initialize this structure. */
typedef struct Avbd3dGpuDef
{
	/* Set to sizeof(Avbd3dGpuDef); avbd3d_default_gpu_def does. */
	uint32_t struct_size;

	/* Extra instance extensions, e.g. what SDL_Vulkan_GetInstanceExtensions reports. */
	const char* const* instanceExtensions;
	int32_t instanceExtensionCount;

	/* Non-null: the GPU is presentable. Called once, right after the instance is created; the
	   device is then chosen from those whose queue can present to the surface, and
	   VK_KHR_swapchain is enabled. */
	Avbd3dMakeSurfaceFcn* makeSurface;
	void* surfaceContext;

	/* Force the Khronos validation layer on (AVBD_VK_VALIDATION=1 also turns it on). */
	int32_t enableValidation;

	/* Print the device selection to stderr. */
	int32_t verbose;
} Avbd3dGpuDef;

/* The Vulkan objects of a GPU, for a renderer that shares it. */
typedef struct Avbd3dVulkanContext
{
	void* instance; /* VkInstance */
	void* physicalDevice; /* VkPhysicalDevice */
	void* device; /* VkDevice */
	void* queue; /* VkQueue: the solver's queue, which also presents on a presentable GPU */
	uint32_t queueFamily;
	uint32_t apiVersion;
	uint64_t commandPool; /* VkCommandPool of the solver (reset only by the solver) */
	uint64_t pipelineCache; /* VkPipelineCache */
	uint64_t surface; /* VkSurfaceKHR, 0 when the GPU is not presentable */
	uint32_t subgroupSize;
	const char* deviceName;
} Avbd3dVulkanContext;

/* How the instance colour is chosen. AVBD3D_OVERLAY_NONE keeps each body's own colour. */
typedef enum Avbd3dOverlay
{
	AVBD3D_OVERLAY_NONE = 0,
	AVBD3D_OVERLAY_SLEEP,  /* asleep vs awake, with the settling countdown as a gradient */
	AVBD3D_OVERLAY_GRID,  /* broadphase hash grid: one hue per occupied cell */
	AVBD3D_OVERLAY_GRAPH_COLOR,  /* graph colouring: the parallel batches of the solve */
	AVBD3D_OVERLAY_LOAD,  /* peak contact normal force on the body */
	AVBD3D_OVERLAY_SPEED,  /* linear speed */
	AVBD3D_OVERLAY_DEGREE,  /* number of bodies touching this one */
	AVBD3D_OVERLAY_JOINT_DEGREE,  /* number of unbroken joints and springs on this body */
	AVBD3D_OVERLAY_JOINT_LOAD,  /* worst connected joint load, compression vs tension */
	AVBD3D_OVERLAY_JOINT_STRESS,  /* worst connected joint load as a fraction of its fracture limit */
	AVBD3D_OVERLAY_AXIAL_LOAD,  /* contacts pushing vs joints pulling */
	AVBD3D_OVERLAY_FRICTION,  /* fraction of the friction cone in use */
	AVBD3D_OVERLAY_PENETRATION,  /* deepest contact penetration */
	AVBD3D_OVERLAY_COUNT,
} Avbd3dOverlay;

/* Use avbd3d_default_render_def() to initialize this structure. */
typedef struct Avbd3dRenderDef
{
	/* Six frustum planes (nx, ny, nz, d), inside where n.p + d >= 0, or null for no culling. */
	const float* frustumPlanes;

	int32_t /* Avbd3dOverlay */ overlay;
	/* Upper end of each overlay's heat ramp. */
	float speedMax, loadMax, jointLoadMax, penetrationMax;
	int32_t degreeMax, jointDegreeMax;
	/* Static bodies join the overlay instead of staying grey. */
	int32_t colorStatics;
} Avbd3dRenderDef;

/* A range of instances in the instance buffer. */
typedef struct Avbd3dRenderRange
{
	int32_t first;
	int32_t count;
} Avbd3dRenderRange;

/* How a contact marker is tinted. */
typedef enum Avbd3dContactTint
{
	AVBD3D_CONTACT_TINT_FORCE = 0,  /* normal force, on the load overlay's ramp */
	AVBD3D_CONTACT_TINT_FRICTION,  /* fraction of the friction cone in use */
	AVBD3D_CONTACT_TINT_STICK,  /* static friction holding vs slipping */
	AVBD3D_CONTACT_TINT_FLAT,  /* one colour */
} Avbd3dContactTint;

/* Use avbd3d_default_contact_marker_def() to initialize this structure. */
typedef struct Avbd3dContactMarkerDef
{
	int32_t points; /* a small cube at each contact point */
	int32_t normals; /* a needle along the contact normal, scaled by force */
	int32_t manifoldLinks; /* a rod joining the centres of each touching pair */
	int32_t /* Avbd3dContactTint */ tint;
	float pointSize; /* half extent of a point marker, metres */
	float normalLength; /* needle length at full force, metres */
	float forceMax; /* force at which the tint saturates */
	int32_t maxMarkers; /* markers past this are not written */
} Avbd3dContactMarkerDef;

/* One drawable body: a box, sphere or capsule of the given half extents (a sphere uses x, a
   capsule x = radius and z = half height along its local z). 44 bytes, tightly packed. */
typedef struct Avbd3dRenderInstance
{
	float position[3];
	float rotation[4]; /* x, y, z, w */
	float halfExtents[3];
	uint32_t rgba; /* 0xAABBGGRR */
} Avbd3dRenderInstance;

/* ---------------------------------------------------------------------------
   Defaults
   --------------------------------------------------------------------------- */

/* struct_size is set; every other field is zero. */
AVBD3D_API Avbd3dGpuDef avbd3d_default_gpu_def(void);

AVBD3D_API Avbd3dRenderDef avbd3d_default_render_def(void);

AVBD3D_API Avbd3dContactMarkerDef avbd3d_default_contact_marker_def(void);

/* ---------------------------------------------------------------------------
   Device
   --------------------------------------------------------------------------- */

/* Creates a Vulkan instance and device. Returns AVBD3D_ERR_DEVICE when no usable device exists
   and AVBD3D_ERR_OUT_OF_MEMORY when memory ran out. */
AVBD3D_API Avbd3dResult avbd3d_create_gpu(const Avbd3dGpuDef *def, Avbd3dGpu **out_gpu);

/* Destroys the device. Every world made on it must be destroyed first. */
AVBD3D_API Avbd3dResult avbd3d_destroy_gpu(Avbd3dGpu *gpu);

/* Creates a world on a shared device. The world is destroyed with avbd3d_destroy_world. */
AVBD3D_API Avbd3dResult avbd3d_create_world_on_gpu(Avbd3dGpu *gpu, const Avbd3dWorldDef *def,
                                                   Avbd3dWorld **out_world);

/* The Vulkan objects of the device, for a renderer that shares it. The pointers inside stay
   valid until the Gpu is destroyed. */
AVBD3D_API Avbd3dResult avbd3d_gpu_get_vulkan_context(const Avbd3dGpu *gpu, Avbd3dVulkanContext *out);

/* Device memory the buffers of every world on the Gpu hold, in bytes. */
AVBD3D_API Avbd3dResult avbd3d_gpu_get_allocated_bytes(const Avbd3dGpu *gpu, uint64_t *out);

/* Messages the Khronos validation layer has reported since the process started. */
AVBD3D_API Avbd3dResult avbd3d_get_validation_message_count(int32_t *out);

/* ---------------------------------------------------------------------------
   Bodies
   --------------------------------------------------------------------------- */

/* Put a body in its own render group (>= 3), or back in its shape's group (-1). */
AVBD3D_API Avbd3dResult avbd3d_body_set_render_group(Avbd3dBodyId id, int32_t group);

AVBD3D_API Avbd3dResult avbd3d_body_get_render_group(Avbd3dBodyId id, int32_t *out);

/* ---------------------------------------------------------------------------
   Render data
   --------------------------------------------------------------------------- */

/* Write the visible bodies' instances, grouped by render group: groups 0, 1 and 2 are boxes
   (and hulls), spheres and capsules; a body given a group >= 3 is written in that group's
   range. */
AVBD3D_API Avbd3dResult avbd3d_world_write_instances(Avbd3dWorld *world, const Avbd3dRenderDef *def);

/* The VkBuffer holding Avbd3dRenderInstance records. */
AVBD3D_API Avbd3dResult avbd3d_world_get_instance_buffer(Avbd3dWorld *world, uint64_t *out);

AVBD3D_API Avbd3dResult avbd3d_world_get_render_group_count(Avbd3dWorld *world, int32_t *out_count);

/* The instances the last WriteInstances wrote for a render group. */
AVBD3D_API Avbd3dResult avbd3d_world_get_render_group_range(Avbd3dWorld *world, int32_t group,
                                                            Avbd3dRenderRange *out);

/* Write contact markers as Avbd3dRenderInstance boxes. Returns how many were written; see
   avbd3d_world_get_contact_markers_requested for how many the markers wanted before the cap. */
AVBD3D_API Avbd3dResult avbd3d_world_write_contact_markers(Avbd3dWorld *world,
                                                           const Avbd3dContactMarkerDef *def, int32_t *out);

AVBD3D_API Avbd3dResult avbd3d_world_get_contact_marker_buffer(Avbd3dWorld *world, uint64_t *out);

AVBD3D_API Avbd3dResult avbd3d_world_get_contact_markers_requested(Avbd3dWorld *world, int32_t *out);

#ifdef __cplusplus
}
#endif

#endif /* AVBD3D_VULKAN_H */
