# Concepts

**What you get:** the rules every call follows. Read this once before the [cheatsheet](api-cheatsheet.md).

## Units and axes

| Item | Value |
| :--- | :--- |
| Length, mass, time, angle | metres, kilograms, seconds, radians |
| Default gravity | `(0, -10, 0)` |
| Up | There is no up vector. Gravity defines down; any direction works |
| Rotation | Unit quaternion `{v, s}` |

## Handles and ids

| Type | What it is |
| :--- | :--- |
| `Avbd3dWorld*` | Opaque pointer. NULL gives `AVBD3D_ERR_NULL_HANDLE` |
| `Avbd3dBodyId`, `ShapeId`, `JointId`, `ContactId` | Value types `{index1, world0, generation}`. Destroying an object makes its id stale |
| `avbd3d_*_is_valid(id)` | Returns 1 for a live id, else 0 |

A stale id passed to any call returns `AVBD3D_ERR_INVALID_ARG`.

## Defs

Every object is created from a plain struct. Call the matching `avbd3d_default_*` function first, edit the fields you need, then pass the struct by pointer. `Avbd3dCapabilities` and `Avbd3dGpuDef` carry a `struct_size` field; set it to the size of the struct (`avbd3d_default_gpu_def` does).

## Results

| Code | Value | Meaning |
| :--- | :--- | :--- |
| `AVBD3D_OK` | 0 | Success |
| `AVBD3D_ERR_NULL_HANDLE` | -1 | World or GPU pointer is NULL or not live |
| `AVBD3D_ERR_INVALID_ARG` | -2 | NULL argument, bad value or stale id |
| `AVBD3D_ERR_UNSUPPORTED` | -3 | Not available on this build or device |
| `AVBD3D_ERR_DEVICE` | -4 | No usable Vulkan device, or a Vulkan call failed. The world is dead |
| `AVBD3D_ERR_CAPACITY` | -5 | A hard capacity ceiling was reached; the step committed clamped results. The world keeps working |
| `AVBD3D_ERR_DEVICE_LOST` | -6 | Driver reset or faulting kernel. The world is dead |
| `AVBD3D_ERR_OUT_OF_MEMORY` | -7 | Host or device memory ran out. The world is dead |

`avbd3d_result_string` returns the name of a code. There is no sticky error: the result of the call that failed says why. After a fatal code every call on that world returns it, and only `avbd3d_destroy_world` works.

Calls return values through the last parameters (`out`, `out_count`, `out_id`, `out_world`). Booleans are `int32_t`. Arrays are passed as pointer and capacity, and a count query uses capacity 0.

## Stepping

| Call | Effect |
| :--- | :--- |
| `avbd3d_step(world, dt)` | `step_begin` then `step_end` |
| `avbd3d_step_begin(world, dt)` | Records and submits the whole step, then returns |
| `avbd3d_step_end(world)` | Waits for the step and reads a small status block |

While a step is in flight, getters, queries and events read the newest finished step. Setters, creates and destroys are queued and applied at the top of the next step. The render-data writes of `avbd3d_vulkan.h`, grab move and end, and `avbd3d_get_stats` end the step first.

```c
avbd3d_step_end(world);        /* finish step N (no-op on the first frame) */
/* read poses and events of step N, write forces and velocities */
avbd3d_step_begin(world, dt);  /* step N + 1 runs while the frame goes on */
```

A body written since the last finished step keeps the written value until the step that applies it has finished. Forces and torques act for the next step only; impulses change the velocity before it.

## Events

Events are buffered after a step and read with `avbd3d_world_get_body_events`, `_contact_events`, `_sensor_events` and `_joint_events`. Each fills a struct that points into world-owned arrays. The pointers become invalid when bodies are destroyed or the next step runs.

## Body origin and centre of mass

Positions and rotations set and read through the body calls are the origin frame. `avbd3d_body_get_world_center` gives the centre of mass computed from the shapes. A body whose centre of mass is not at its origin rotates about the centre of mass. `avbd3d_body_set_mass_data` overrides mass, centre and inertia until `avbd3d_body_apply_mass_from_shapes` restores them.

## Memory

Device buffers grow with headroom and never shrink. If a pass needs more room than it has, the step grows the buffer and reruns that pass, so a step normally takes one queue submission and sometimes more (`Avbd3dStats.submits`, `capacity_reruns`). A hard ceiling gives `AVBD3D_ERR_CAPACITY`.

## Threads

Nothing is thread safe. Use one world from one thread at a time.
