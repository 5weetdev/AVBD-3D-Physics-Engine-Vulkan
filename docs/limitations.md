# Limitations

**What you get:** what the solver does not do, what happens when you try, and what to do instead.

| Gap | Effect | Workaround |
| :--- | :--- | :--- |
| Mesh shapes colliding with mesh shapes | The pair is skipped: no contact | Use convex hulls or a compound on one of the two bodies |
| Height fields on dynamic bodies | Not supported | Keep height fields on static bodies |
| Meshes on dynamic bodies need a closed mesh | `avbd3d_create_mesh_shape` returns `AVBD3D_ERR_INVALID_ARG` for an open mesh on a dynamic body | Close the mesh, or make the body static |
| Compound children | Dynamic compounds take convex children only (spheres, capsules, hulls) | Split concave parts into hulls |
| `avbd3d_body_set_type` on an open-mesh body | Returns `AVBD3D_OK` and the body stays static | Check the mesh before changing the type |
| Continuous collision against dynamic meshes | Not swept | Keep fast bodies away from dynamic meshes, or use a thicker convex proxy |
| `avbd3d_shape_set_filter` | Does not wake neighbouring sleeping bodies | Call `avbd3d_body_set_awake` on them |
| Conveyor tangent velocity, joint snapshots | Read the primary shape of the body only | Put the conveyor shape first on the body |
| `avbd3d_body_get_closest_point` | Includes shape radii, unlike Box3D | Subtract the radius yourself if you need the core distance |
| Determinism | Not guaranteed. Results can differ between GPUs, drivers and `primalLanes` values | Do not use it for lock-step networking |
| Subgroup size | The cooperative kernels assume 32. The device is not rejected if its size differs, and other sizes are untested | Use an NVIDIA GPU |
| Platforms | Tested on Windows 10 (NVIDIA RTX 4070 SUPER) and Fedora Linux 45 (GCC, AMD Radeon 780M, Mesa RADV). macOS (MoltenVK) has build scripts and guarded code but is **untested**; D3D11 interop (`enableExternalWin32`) is Windows only | Report what breaks |
| Thread safety | None. One world, one thread at a time | Serialise calls |
| Colours | A scene whose constraint graph needs more than 128 colours reports a failed step | Reduce the number of constraints per body |
| Not offered | Recording, double precision, task system, pre-solve and custom-filter callbacks, cloth, level of detail | - |

Status: **experimental**. The C ABI is 1.0 and changes only with a version bump, but the solver has had little use outside its own tests and demo.
