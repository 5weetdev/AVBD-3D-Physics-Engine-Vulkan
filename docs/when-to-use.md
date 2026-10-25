# When to use it

**What you get:** the cases where a GPU AVBD solver fits and where a CPU engine is the better choice.

## Fit

| Your case | Fit | Why |
| :--- | :--- | :--- |
| Tens of thousands to 100,000+ bodies in contact | good | Cost is spread over thousands of GPU threads |
| Tall stacks and piles that must stay stable at 8 iterations | good | Augmented Lagrangian penalties ramp per contact |
| Heavy mass ratios, breakable and chained joints | good | Joints use the same implicit solve as contacts |
| A handful of bodies | poor | A step is one GPU submission plus a status read. The fixed cost dominates |
| Bit-for-bit reproducible simulation | no | Float accumulation order on the GPU is not fixed. See [limitations](limitations.md) |
| Network rollback or replays | no | Same reason |
| No Vulkan 1.3 GPU | no | There is no CPU fallback |
| Your renderer needs poses on the CPU every frame | possible | Poses are read through `avbd3d_world_get_body_data`; this costs a transfer |

## Measured cost

These figures were taken on an RTX 4070 SUPER with sleeping off, 8 iterations and 4 lanes per body, as milliseconds per step. They come from the project's earlier measurement of the same solver code on a settled pile, and were not re-run for this release.

| Scene | Step |
| :--- | :--- |
| 10,000 bodies, perturbed | 1.88 ms |
| 100,000 bodies, perturbed | 13.46 ms |
| 100,000 bodies, unperturbed | 13.50 ms |

Re-measure on your own scene. A flat step time on a quiet pile says nothing about a pile that is moving.

## Choosing

| If you need | Use |
| :--- | :--- |
| Many bodies, GPU available, results need not be identical across runs | this library |
| Few bodies, determinism, or no GPU | a CPU engine |
| Box3D's feature set | Box3D itself. This library borrows its API style and collision code and does not match its behaviour |
