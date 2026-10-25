# Getting started

**What you get:** a walk through [examples/hello.c](../examples/hello.c), then how to link your own project.

## Build and run the example

```bat
build.bat release
build\release\avbd3d_hello.exe
```

It prints the resting height of a box and `OK`.

## hello.c step by step

| Step | Code | What it does |
| :--- | :--- | :--- |
| 1 | `avbd3d_abi_version() >> 16` against `AVBD3D_ABI_VERSION_MAJOR` | Checks the library matches the header |
| 2 | `avbd3d_default_world_def()` and `avbd3d_create_world(&wd, &world)` | Creates a world and its Vulkan device |
| 3 | `avbd3d_default_body_def()`, set `type` and `position`, `avbd3d_create_body` | Creates a static ground body and a dynamic body |
| 4 | `avbd3d_make_box_hull`, `avbd3d_create_hull_shape` | Attaches a box shape to each body |
| 5 | `avbd3d_step(world, 1.0f / 60.0f)` x 120 | Simulates two seconds |
| 6 | `avbd3d_body_get_position` | Reads the pose of the dynamic box |
| 7 | `avbd3d_destroy_world` | Frees the world |

Each call returns an `Avbd3dResult`; the example stops on the first non-zero code and prints `avbd3d_result_string`.

## Link your own project

CMake, after [installing](requirements.md#install-and-consume):

```cmake
find_package(avbd3d REQUIRED)
add_executable(game main.c)
target_link_libraries(game PRIVATE avbd3d::avbd3d)
```

Without CMake: add `include` to the include path, link `avbd3d.lib` (Windows) or `libavbd3d`, and on Windows define `AVBD3D_USE_SHARED` before including the headers. Put `avbd3d.dll` next to your executable.

## Next

| Need | Read |
| :--- | :--- |
| The rules every call follows | [concepts](concepts.md) |
| Which call does what | [API cheatsheet](api-cheatsheet.md) |
| Shapes and joints | [shapes and joints](shapes-and-joints.md) |
| A full application | `demo/` (the SDL and ImGui demo uses only the public headers for the solver) |
