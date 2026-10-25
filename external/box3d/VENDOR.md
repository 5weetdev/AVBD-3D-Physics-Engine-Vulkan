# Vendored: Box3D

Upstream: https://github.com/erincatto/box3d
Pinned commit: 47d7f7cc7e091142c08d11dc7d2e493c5d34f536

An unmodified copy of Box3D's `src/` and `include/` plus its `LICENSE`. Nothing else from
upstream (tests, samples, benchmarks, docs, build files) is kept. It is not a submodule and has
no nested `.git`. Do not edit these files: adaptations belong in `src/`, so that re-pinning
upstream stays a directory replace.

`avbd3d_box3d` (see the top-level `CMakeLists.txt`) compiles every `src/*.c` into a static library.

To re-pin: delete `src/` and `include/`, copy them from the new commit, update the hash above.
