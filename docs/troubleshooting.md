# Troubleshooting

**What you get:** symptoms, causes and fixes, as tables.

## Build

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| `slangc not found` | Vulkan SDK missing or `VULKAN_SDK` not set | Install the SDK, open a new terminal, check `echo %VULKAN_SDK%` |
| `cl.exe` or `vcvars64.bat` not found | MSVC Build Tools missing | Install them as in [requirements](requirements.md). `build.bat` looks for them through `vswhere` |
| `external/SDL` empty | Cloned without submodules | `git submodule update --init` |
| Python not found | Python 3 missing or not on `PATH` | Install it and open a new terminal |
| Stale shader after an edit | `slangc` writes no dependency file, so an edit to an imported module does not rebuild everything | `build.bat clean` |

## Runtime

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| `avbd3d_create_world` returns `AVBD3D_ERR_DEVICE` | No Vulkan 1.3 device, or a required feature is missing | Update the driver. The message names the missing feature on stderr |
| Every call returns `AVBD3D_ERR_DEVICE_LOST` | A driver reset or a faulting kernel. The world is dead | Destroy it and create a new one. See "Device lost" below |
| `avbd3d_step` returns `AVBD3D_ERR_CAPACITY` | A hard capacity ceiling was reached and the step committed clamped results | Read `avbd3d_get_stats` (`pair_capacity`, `contact_capacity`, `manifold_capacity`) and reduce the scene |
| A call returns `AVBD3D_ERR_INVALID_ARG` | NULL output pointer, out-of-range value or a stale id | Check the id was not destroyed |
| A call returns `AVBD3D_ERR_NULL_HANDLE` | The world pointer is NULL | Check the create result |
| Setter has no visible effect | While a step is in flight, setters are queued for the next step | Call `avbd3d_step_end`, then step again |
| Bodies sit still after a hit | They are asleep; wake is velocity-gated | `avbd3d_body_set_awake`, or lower the sleep velocities in `Avbd3dSolverDef` |

## Device lost

The cause is nearly always a shader reading or writing outside a buffer, and it shows up at scale. To find it:

1. Run with `AVBD_VK_VALIDATION=1` and `VK_LAYER_GPUAV_ENABLE=1`. GPU-assisted validation names the dispatch, the invocation and the address. An address near 0 means a null device address was passed.
2. Reproduce with the same scene size. A small scene hides these bugs.
3. Open an issue with the output (see [CONTRIBUTING](../CONTRIBUTING.md)).

## Validation

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| No validation output | Layer not installed or variable not set | Set `AVBD_VK_VALIDATION=1` in the same terminal |
| `avbd3d_demo --smoke` exits non-zero | A step failed or a validation message appeared | Read stderr; `avbd3d_get_validation_message_count` gives the count |
