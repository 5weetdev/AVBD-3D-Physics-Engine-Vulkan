# Third-party notices

avbd3d is released under the MIT License (see [LICENSE](LICENSE)). It includes or is derived from the third-party works below.

## Box3D (vendored and ported code)

- The host geometry code is a subset of Box3D's sources, compiled unmodified from `external/box3d/`: hull, mesh, height-field and compound builders, mass properties, the dynamic tree, casts and the character mover.
- The device collision code in `src/shaders/b3/` is a Slang port generated from the same sources.

- Copyright (c) 2026 Erin Catto
- License: MIT. Full text: [external/box3d/LICENSE](external/box3d/LICENSE)

Source: https://github.com/erincatto/box3d

## SDL2 (demo only)

Vendored as a git submodule at `external/SDL`. Used by `avbd3d_demo` only; not part of the `avbd3d` library.

- Copyright (C) 1997-2025 Sam Lantinga
- License: zlib. Full text: [external/SDL/LICENSE.txt](external/SDL/LICENSE.txt)

Source: https://github.com/libsdl-org/SDL

## Dear ImGui (demo only)

Vendored as a git submodule at `external/imgui`. Used by `avbd3d_demo` only.

- Copyright (c) 2014-2025 Omar Cornut
- License: MIT. Full text: [external/imgui/LICENSE.txt](external/imgui/LICENSE.txt)

Source: https://github.com/ocornut/imgui

## AVBD demo code (Chris Giles)

The solver grew out of Chris Giles's AVBD 3D demo. Source files that carry his header keep it, as its terms require:

- Copyright (c) 2025-2026 Chris Giles

```
Permission to use, copy, modify, distribute and sell this software
and its documentation for any purpose is hereby granted without fee,
provided that the above copyright notice appear in all copies.
Chris Giles makes no representations about the suitability
of this software for any purpose.
It is provided "as is" without express or implied warranty.
```

Source: https://github.com/savant117/avbd-demo3d

## AVBD method

The solver implements Augmented Vertex Block Descent, the method of Chris Giles, Elie Diaz and Cem Yuksel:

> Chris Giles, Elie Diaz, and Cem Yuksel. "Augmented Vertex Block Descent." ACM Transactions on Graphics 44(4), Article 90 (SIGGRAPH 2025).
