// The sleep stage -- the rest test and the island rule that commits
// it. Kernel-for-kernel: k_sleep_block_fracture -> csSleepBlockFracture,
// k_sleep_block_drag -> csSleepBlockDrag, k_sleep_mark -> csSleepMark,
// k_sleep_commit -> csSleepCommit, k_active_flags -> csActiveFlags. Same term order and
// same guards as the reference source throughout; see that file for the physics commentary
// (the derivation of why this is a net-displacement test and not a velocity test, and
// why a fracture-loaded joint or an active drag blocks its endpoint from sleeping).
//
// Host orchestration (the anyBlock guard, the memset before the block kernels, the
// dyn/solveCount early-outs) is engine/vk/vk_sleep.h/.cpp's launchSleepCommit,
// mirroring the launch seam's launchSleepCommit exactly.

import maths;
import types;

struct PushConstants
{
    uint64_t argsAddr;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// ForceType, matching gpu_forces / constraints.slang.
static const int FORCE_MANIFOLD_SL = 0;
static const int FORCE_JOINT_SL = 1;
static const int FORCE_SPRING_SL = 2;
static const int AVBD_FORCE_INDEX_BITS_SL = 29;
static const uint AVBD_MAX_FORCE_INDEX_SL = (uint(1) << AVBD_FORCE_INDEX_BITS_SL) - 1u;

int graphBodySL(uint64_t k) { return (int)(uint)(k >> 32); }
int graphTypeSL(uint64_t k) { return (int)((uint(k & uint64_t(0xFFFFFFFFu))) >> AVBD_FORCE_INDEX_BITS_SL); }
int graphIndexSL(uint64_t k) { return (int)(uint(k & uint64_t(0xFFFFFFFFu)) & AVBD_MAX_FORCE_INDEX_SL); }

// Mirrors the device structs' ConstraintGraph -- key/isA/start/end, same private
// copy convention primal.slang/dual.slang use for their own graph mirror.
// graphOther, specialized to the three real SoAs (m/j/sp) rather than the raw bodyA/
// bodyB arrays constraints.slang's csJpRound uses -- k_sleep_commit calls the real
// graphOther(g, e, m, j, sp) helper, so this takes the same three SoAs.
int graphOtherSL(ConstraintGraph g, int e, ManifoldSoA m, JointSoA j, SpringSoA sp)
{
    uint64_t key = g.key[e];
    int isA = g.isA[e];
    int type = graphTypeSL(key);
    int idx = graphIndexSL(key);
    if (type == FORCE_JOINT_SL)
        return isA != 0 ? j.bodyB[idx] : j.bodyA[idx];
    if (type == FORCE_SPRING_SL)
        return isA != 0 ? sp.bodyB[idx] : sp.bodyA[idx];
    return isA != 0 ? m.bodyB[idx] : m.bodyA[idx];
}

// ---------------------------------------------------------------------------
// csSleepBlockFracture -- k_sleep_block_fracture
// ---------------------------------------------------------------------------
struct SleepBlockFractureArgs
{
    BodySoA b;
    JointSoA j;
    float ratio;
    float _pad0;
    uint8_t *block; // never null: stored through unguarded
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csSleepBlockFracture(uint3 tid: SV_DispatchThreadID)
{
    SleepBlockFractureArgs *args = (SleepBlockFractureArgs *)pc.argsAddr;
    JointSoA j = args->j;
    int t = (int)tid.x;
    if ((uint)t >= j.count || j.broken[t] != 0)
        return;
    float frac = j.fracture[t];
    if (isinf(frac))
        return;

    float limit = frac * args->ratio;
    if (lengthSq3(j.lambdaLin[t]) <= limit * limit && lengthSq3(j.lambdaAng[t]) <= limit * limit)
        return;

    int ia = j.bodyA[t], ib = j.bodyB[t];
    if (ia >= 0)
        args->block[ia] = 1;
    if (ib >= 0)
        args->block[ib] = 1;
}

// ---------------------------------------------------------------------------
// csSleepBlockDrag -- k_sleep_block_drag
// ---------------------------------------------------------------------------
struct SleepBlockDragArgs
{
    int *dragIdx;
    int dragCount;
    uint8_t *block; // never null: stored through unguarded
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csSleepBlockDrag(uint3 tid: SV_DispatchThreadID)
{
    SleepBlockDragArgs *args = (SleepBlockDragArgs *)pc.argsAddr;
    int t = (int)tid.x;
    if (t >= args->dragCount)
        return;
    int i = args->dragIdx[t];
    if (i >= 0)
        args->block[i] = 1;
}

// ---------------------------------------------------------------------------
// csSleepMark -- k_sleep_mark
// ---------------------------------------------------------------------------
struct SleepMarkArgs
{
    BodySoA b;
    int *dyn;
    int dynCount;
    uint8_t *candidate;
    int frames;
    float linVel;
    float angVel;
    float disp;
    uint8_t *block; // may be null (no fracture joints, no active drags)
    uint hasBlock;
    float leak;     // 0: reset-on-spike (classic); > 0: leaky streak, design 2.7
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csSleepMark(uint3 tid: SV_DispatchThreadID)
{
    SleepMarkArgs *args = (SleepMarkArgs *)pc.argsAddr;
    BodySoA b = args->b;
    int t = (int)tid.x;
    if (t >= args->dynCount)
        return;
    int i = args->dyn[t];

    if (args->hasBlock != 0 && args->block[i] != 0)
    {
        b.sleepTimer[i] = 0;
        args->candidate[i] = 0;
        return;
    }

    // 2 = asleep: a valid neighbour for csSleepCommit, but not a streak of this body's own.
    if (b.awake[i] == 0)
    {
        args->candidate[i] = 2;
        return;
    }

    // Per-body sleep settings (BodyProps; b.props is null in a world with none): enableSleep
    // off keeps the body awake, sleepThreshold > 0 replaces the fast-reject velocities.
    float linV = args->linVel, angV = args->angVel;
    if (b.props != nullptr)
    {
        if ((b.props[i].flags & BODY_NO_SLEEP) != 0u)
        {
            b.sleepTimer[i] = 0;
            args->candidate[i] = 0;
            return;
        }
        if (b.props[i].sleepThreshold > 0.0f)
        {
            linV = b.props[i].sleepThreshold;
            angV = b.props[i].sleepThreshold;
        }
    }

    int timer = b.sleepTimer[i];
    if (args->leak > 0.0f)
    {
        // Leaky streak: a spike drains progress in proportion to its size.
        float eps = args->linVel * 2.0f;
        float d = length3(b.velLin[i]) + b.gyration[i] * length3(b.velAng[i]);
        if (d > eps)
        {
            int keep = timer - (int)ceil(args->leak * d / eps);
            b.sleepTimer[i] = max(keep, 0);
            args->candidate[i] = 0;
            return;
        }
    }
    bool moving = args->leak <= 0.0f && (length3(b.velLin[i]) > linV * 2.0f ||
                                         length3(b.velAng[i]) > angV * 2.0f);
    if (moving)
    {
        b.sleepTimer[i] = 0;
        args->candidate[i] = 0;
        return;
    }

    if (timer == 0)
    {
        b.restPos[i] = b.pos[i];
        b.restRot[i] = b.rot[i];
    }

    timer = timer + 1;
    if (timer < args->frames)
    {
        // Inside a window the flag from the last completed window stands (1 lets an island
        // whose bodies' windows are out of phase commit). A 2 is left over from sleep: this
        // body was woken this step, and csSleepCommit would sleep it again at once (a
        // sleeping box moved by setTransform hung in the air where it was put).
        b.sleepTimer[i] = timer;
        if (args->candidate[i] == 2)
            args->candidate[i] = 0;
        return;
    }

    vec3 dx = sub(b.pos[i], b.restPos[i]);
    vec3 dq = subQQ(b.rot[i], b.restRot[i]);
    float d = length3(dx) + b.gyration[i] * length3(dq) * 2.0f;

    args->candidate[i] = (d <= args->disp) ? 1 : 0;

    b.sleepTimer[i] = 0;
    b.restPos[i] = b.pos[i];
    b.restRot[i] = b.rot[i];
}

// ---------------------------------------------------------------------------
// csSleepCommit -- k_sleep_commit
// ---------------------------------------------------------------------------
struct SleepCommitArgs
{
    BodySoA b;
    ConstraintGraph g;
    ManifoldSoA m;
    JointSoA j;
    SpringSoA sp;
    int *active;
    int activeCount;
    int _pad0;
    uint8_t *candidate;
    int *touchCount;
    int *touchHash;
    int *snapCount;
    int *snapHash;
    // when set, activeCount is a bound and the live count is min(*ptr, activeCount).
    int *activeCountPtr;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csSleepCommit(uint3 tid: SV_DispatchThreadID)
{
    SleepCommitArgs *args = (SleepCommitArgs *)pc.argsAddr;
    BodySoA b = args->b;
    int t = (int)tid.x;
    int liveCount = args->activeCount;
    if (args->activeCountPtr != nullptr)
        liveCount = min(args->activeCountPtr[0], liveCount);
    if (t >= liveCount)
        return;
    int i = args->active[t];
    if (args->candidate[i] == 0)
        return;

    for (int e = args->g.start[i]; e < args->g.end[i]; e++)
    {
        int other = graphOtherSL(args->g, e, args->m, args->j, args->sp);
        if (other < 0 || b.mass[other] <= 0.0f)
            continue;
        if (args->candidate[other] == 0)
            return;
    }

    b.awake[i] = 0;
    b.velLin[i] = vec3(0.0f, 0.0f, 0.0f);
    b.velAng[i] = vec3(0.0f, 0.0f, 0.0f);
    b.prevVelLin[i] = vec3(0.0f, 0.0f, 0.0f);

    args->snapCount[i] = args->touchCount[i];
    args->snapHash[i] = args->touchHash[i];
}

// ---------------------------------------------------------------------------
// csActiveFlags -- k_active_flags
// ---------------------------------------------------------------------------
struct ActiveFlagsArgs
{
    BodySoA b;
    int *dyn;
    int dynCount;
    int _pad0;
    uint8_t *flags;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void csActiveFlags(uint3 tid: SV_DispatchThreadID)
{
    ActiveFlagsArgs *args = (ActiveFlagsArgs *)pc.argsAddr;
    int t = (int)tid.x;
    if (t >= args->dynCount)
        return;
    int i = args->dyn[t];
    args->flags[t] = args->b.awake[i];
}
