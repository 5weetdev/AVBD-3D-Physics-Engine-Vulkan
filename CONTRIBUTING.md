# Contributing to avbd3d

Thanks for helping. This page covers the minimum you need before opening a pull request.

## Build

Follow [docs/requirements.md](docs/requirements.md). Use `build.bat clean` (Windows) or `./build.sh clean` (Linux, macOS) when you change CMake files.

## AI-assisted contributions

AI-assisted pull requests are accepted. Say so in the description, and make sure you have read and tested the change yourself.

## Rules for changes

- Keep the solver's numerics unchanged unless the change is a deliberate fix.
- Do not change the C ABI (`include/avbd3d/*.h`) without a version bump (`AVBD3D_ABI_VERSION_MAJOR` / `_MINOR`).
- Keep `src/` independent of `demo/`.
- Do not edit `external/box3d/`. It is an unmodified upstream copy.
- A step is one queue submission and allocates nothing after warm-up. The `onesubmit` test checks this.

## Reporting GPU or driver bugs

Open an issue with:

- GPU model, driver version and operating system
- Vulkan SDK version
- The exact scene and steps that failed
- The output of `avbd3d_demo --smoke` with `AVBD_VK_VALIDATION=1` set

## Pull request checklist

- [ ] Builds from a clean build folder with no new warnings
- [ ] `build.bat test` (or `./build.sh test`) passes
- [ ] `avbd3d_demo --smoke` exits 0 with `AVBD_VK_VALIDATION=1`
- [ ] Docs updated for any changed behaviour or command
