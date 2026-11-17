# Requirements

**What you get:** what to install, then the commands that build the library, the demo and the tests. Nothing is prebuilt.

You need a GPU and driver with Vulkan 1.3. The build is tested on one machine: RTX 4070 SUPER, Windows 10 x64. Linux and macOS (MoltenVK) are supported targets, but they are **untested**.

## 1. Install

### Windows

| Tool | Why | Install |
| :--- | :--- | :--- |
| Git | Clone the repo and its submodules | `winget install Git.Git` |
| Visual Studio 2022 Build Tools | MSVC compiler | `winget install Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive"` |
| CMake 3.20+ | Build system | `winget install Kitware.CMake` |
| Ninja | Generator | `winget install Ninja-build.Ninja` |
| Python 3 | Embeds the compiled shaders | `winget install Python.Python.3.12` |
| Vulkan SDK | Vulkan loader, validation layer, `slangc` | <https://vulkan.lunarg.com/sdk/home#windows>. The installer sets `VULKAN_SDK`. |

### Linux and macOS (untested)

| Tool | Install |
| :--- | :--- |
| Git, CMake 3.20+, Ninja, Python 3, a C++17 compiler | Your package manager (`apt`, `dnf`, `brew`) |
| Vulkan SDK | <https://vulkan.lunarg.com/sdk/home>. Source its `setup-env.sh` so `VULKAN_SDK` is set and `slangc` is on `PATH`. |
| SDL2 build dependencies (demo only) | See SDL's `docs/README-linux.md` (X11 or Wayland headers) |

Fedora package names (use the `-devel` equivalents elsewhere): `gcc-c++ cmake ninja-build python3 vulkan-loader-devel vulkan-headers`, plus the X11/Wayland headers SDL needs (`libX11-devel libXext-devel libXrandr-devel libXcursor-devel libXfixes-devel libXi-devel libXScrnSaver-devel libxkbcommon-devel libXrender-devel`).

The code is written to compile on Linux: Windows-only calls and the `VK_KHR_*_win32` interop extensions are guarded by `_WIN32`, and `enableExternalWin32()` is a no-op elsewhere. A full Linux build and test run has not been done yet.

Open a new terminal after installing so `PATH` picks everything up.

## 2. Get the source

```bat
git clone --recurse-submodules https://github.com/5weetdev/AVBD-3D-Physics-Engine-Vulkan.git
cd AVBD-3D-Physics-Engine-Vulkan
```

If you already cloned without submodules, run `git submodule update --init`. The build scripts also do this when `external/SDL` is empty.

## 3. Build

| Platform | Command |
| :--- | :--- |
| Windows | `build.bat [release\|debug] [clean] [test] [run]` |
| Linux, macOS | `./build.sh [release\|debug] [clean] [test] [run]` |

| Argument | Effect |
| :--- | :--- |
| `release` (default) | Optimised build in `build/release` |
| `debug` | Debug build in `build/debug` |
| `clean` | Delete that build folder first |
| `test` | Run the test suite (`ctest`) after building |
| `run` | Launch the demo after building |

`build.bat` finds MSVC through `vswhere` and sets up its environment itself, so a plain terminal works.

## 4. Outputs

All outputs land in the build folder (`build/release` or `build/debug`).

| Target | Output | Purpose |
| :--- | :--- | :--- |
| `avbd3d` | `avbd3d.dll` + `avbd3d.lib` (Windows), `libavbd3d.so` / `.dylib` | The C library. Link against this. |
| `avbd3d_demo` | `avbd3d_demo(.exe)`, `SDL2.dll` | SDL2 + ImGui demo |
| `avbd3d_hello` | `avbd3d_hello(.exe)` | [examples/hello.c](../examples/hello.c) |
| tests | `avbd3d_test_*` | Run with `ctest --test-dir build/release` |

To build one target, run `cmake --build build/release --target <name>`.

## CMake options

| Option | Default | Effect |
| :--- | :--- | :--- |
| `AVBD3D_BUILD_DEMO` | `ON` | Build the demo (needs the SDL and ImGui submodules) |
| `AVBD3D_BUILD_TESTS` | `ON` | Build the tests and register them with CTest |
| `AVBD3D_BUILD_EXAMPLES` | `ON` | Build `examples/hello.c` |

For example, to build only the library:

```bat
cmake -B build/lib -S . -G Ninja -DCMAKE_BUILD_TYPE=Release -DAVBD3D_BUILD_DEMO=OFF -DAVBD3D_BUILD_TESTS=OFF -DAVBD3D_BUILD_EXAMPLES=OFF
cmake --build build/lib
```

## Install and consume

```bat
cmake --install build/release --prefix <install dir>
```

| Installed | Where |
| :--- | :--- |
| `avbd3d.dll` / `.so` / `.dylib` | `bin` (Windows) or `lib` |
| `avbd3d.lib` | `lib` |
| `avbd3d.h`, `avbd3d_geometry.h`, `avbd3d_vulkan.h` | `include/avbd3d` |
| CMake package | `lib/cmake/avbd3d` |

In your project:

```cmake
find_package(avbd3d REQUIRED)   # pass -DCMAKE_PREFIX_PATH=<install dir>
target_link_libraries(game PRIVATE avbd3d::avbd3d)
```

## Problems

See [troubleshooting](troubleshooting.md).
