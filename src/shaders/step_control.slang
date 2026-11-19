// Step control kernels: the primal sweep and gate dispatch arguments, and the step-flag / colour-bound
// flag kernels. Split from primal.slang, whose library functions they use.

import maths;
import types;
import contact;
import forces;
import dual_body;
import primal;

// ---------------------------------------------------------------------------
// csPrimalDispatchArgs: one thread per colour turns the device colour ranges into
// VkDispatchIndirectCommand {groups, 1, 1}, so the per-colour primal sweep needs no
// host copy of the ranges. An empty colour gets zero groups, which is a legal no-op.
// ---------------------------------------------------------------------------
struct PrimalDispatchArgs
{
    int *colorStart;
    int *colorEnd;
    uint *out;
    uint colorCount;
    uint bodiesPerGroup;
};

[shader("compute")]
[numthreads(128, 1, 1)]
void csPrimalDispatchArgs(uint3 tid: SV_DispatchThreadID)
{
    PrimalDispatchArgs *args = (PrimalDispatchArgs *)pc.argsAddr;
    uint c = tid.x;
    if (c >= args->colorCount)
        return;
    int n = max(args->colorEnd[c] - args->colorStart[c], 0);
    args->out[3 * c + 0] = ((uint)n + args->bodiesPerGroup - 1) / args->bodiesPerGroup;
    args->out[3 * c + 1] = 1;
    args->out[3 * c + 2] = 1;
}

// ---------------------------------------------------------------------------
// csGateDispatchArgs: copies `count` VkDispatchIndirectCommands src -> dst, or zeroes
// them when the step's overflow flag is set. Lets a whole phase be skipped on device.
// ---------------------------------------------------------------------------
struct GateDispatchArgs
{
    uint *src;
    uint *dst;
    int *flag;
    int count;
    int _pad0;
};

[shader("compute")]
[numthreads(128, 1, 1)]
void csGateDispatchArgs(uint3 tid: SV_DispatchThreadID)
{
    GateDispatchArgs *args = (GateDispatchArgs *)pc.argsAddr;
    int t = (int)tid.x;
    if (t >= args->count)
        return;
    bool closed = (args->flag != nullptr) && (args->flag[0] != 0);
    for (int k = 0; k < 3; k++)
        args->dst[3 * t + k] = closed ? 0u : args->src[3 * t + k];
}

// ---------------------------------------------------------------------------
// Step-loop flags: any of these set stepState[0], which closes the gate on phases 6-8
// and makes the host rerun the step's recorded tail with larger bounds.
// csStepFlagIfSet: flag if a device counter/slot is non-zero.
// csStepColorBound: flag if a colour at or past the recorded bound is non-empty.
// ---------------------------------------------------------------------------
struct StepFlagIfSetArgs
{
    int *src;
    int *flag;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void csStepFlagIfSet(uint3 tid: SV_DispatchThreadID)
{
    StepFlagIfSetArgs *args = (StepFlagIfSetArgs *)pc.argsAddr;
    if (tid.x != 0)
        return;
    if (args->src[0] != 0)
        args->flag[0] = 1;
}

// csStepFlagIfGreater: capacity overflow. A step's broadphase pair counter or narrowphase
// contact counter past its capacity sets word 1 (the capacity flag, which closes the gate on
// phases 4-5) and word 0 (which closes the gate on phases 6-8).
struct StepFlagIfGreaterArgs
{
    int *count;
    int *flag;
    int cap;
    int _pad;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void csStepFlagIfGreater(uint3 tid: SV_DispatchThreadID)
{
    StepFlagIfGreaterArgs *args = (StepFlagIfGreaterArgs *)pc.argsAddr;
    if (tid.x != 0)
        return;
    if (args->count[0] > args->cap)
    {
        args->flag[0] = 1;
        args->flag[1] = 1;
    }
}

struct StepColorBoundArgs
{
    int *start;
    int *end;
    int *flag;
    int bound;
    int maxColors;
};

[shader("compute")]
[numthreads(128, 1, 1)]
void csStepColorBound(uint3 tid: SV_DispatchThreadID)
{
    StepColorBoundArgs *args = (StepColorBoundArgs *)pc.argsAddr;
    int c = args->bound + (int)tid.x;
    if (c >= args->maxColors)
        return;
    if (args->end[c] > args->start[c])
        args->flag[0] = 1;
}
