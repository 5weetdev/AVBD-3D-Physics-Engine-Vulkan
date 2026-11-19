/*
 * Copyright (c) 2026 Chris Giles
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies.
 * Chris Giles makes no representations about the suitability
 * of this software for any purpose.
 * It is provided "as is" without express or implied warranty.
 */

#pragma once

// Solver tunables and their defaults. Backend-free: the Vulkan world and
// the C ABI all take this by value, and none of them owns its definition.

#include "maths.h"

struct SolverParams
{
    float dt;       // Timestep
    vec3 gravity;   // Gravity vector, m/s^2. Default (0, -10, 0): the solver is y-up
    int iterations; // Solver iterations

    // Cooperative lanes per body in the primal sweep (default `4`).
    //
    // One thread per body cannot fill a modern GPU at these scene sizes. A 50k-body
    // pile splits over 5-7 graph colors, so a dispatch is ~10k threads against 86k
    // thread slots -- 12% of the machine -- while the work it does lives on contacts,
    // of which there are 5.5x the machine. `primalLanes` consecutive threads instead
    // cooperate on one body, striding its CSR row in parallel and reducing the 6x6
    // system through warp shuffles.
    //
    // 4 is the measured optimum on every scene: the mean CSR row is ~9.4 edges, so a
    // full 32-lane warp leaves 23 lanes idle and is markedly *worse* than one thread
    // per body. Colors narrower than the SM count keep the 32-lane fallback, where the
    // grid rather than the lane count is what has to be widened.
    //
    // `1` restores one thread per body everywhere; `0` restores the older heuristic,
    // which is the last bit-exact setting. Any value above 1 changes the order in which
    // a body's CSR row accumulates, so it is not bit-exact.
    int primalLanes;

    float alpha;   // Stabilization parameter
    float betaLin; // Penalty ramping parameter for linear constraints
    float betaAng; // Penalty ramping parameter for angular constraints
    float gamma;   // Warmstarting decay parameter

    // Deactivation ("sleeping"). A dynamic body that stays under both velocity
    // thresholds for sleepFrames consecutive steps, and whose every contact neighbour
    // is equally quiet, is frozen and dropped from the constraint graph until
    // something touches it again. This does not change the solver's update in any way;
    // it changes which bodies are in the set the unmodified update runs over, exactly
    // as mass <= 0 already does for statics.
    //
    // sleepFrames <= 0 disables deactivation entirely and restores the old behaviour
    // of solving every dynamic body every step.
    // sleepEnabled is the master switch, separate from sleepFrames so the UI can flip
    // deactivation off and back on without losing the tuned window length.
    bool sleepEnabled; // Master switch; false solves every dynamic body every step
    int sleepFrames;   // Consecutive quiet steps before a body may sleep
    float sleepLinVel; // Linear speed above which a step is disqualified outright
    float sleepAngVel; // Angular speed above which a step is disqualified outright
    // Net displacement allowed over the whole sleepFrames window. This, not the
    // velocity pair above, is what actually decides rest: the velocity test is only a
    // fast reject for bodies that are unambiguously moving.
    float sleepDisp; // Metres of net movement allowed across the rest window
    // Leaky streak. 0 = classic: one fast step resets the
    // quiet-step count. > 0: the point-speed metric d = |v| + r|w| against e = 2*sleepLinVel;
    // a step with d > e costs leak*d/e steps of progress instead of all of them.
    float sleepLeak;
    // A breakable joint carrying more than this fraction of its fracture load keeps
    // its endpoints awake. Without it a structure can freeze just under the breaking
    // point and never collapse, because lambda keeps ratcheting up even at rest.
    float sleepFracture;

    // Initialize a new contact's penalty from the constraint's effective mass rather
    // than from the global PENALTY_MIN floor -- k = 1 / (J M^-1 J^T dt^2), the XPBD
    // generalized inverse mass. Only affects contacts with no warm start to inherit.
    // See DUAL_STEP_CONTROL_PLAN.md for the measurements behind this being on.
    bool autoKStart;

    // Dual damping strength, in [0, 1] (default `0.1`). Scales back the contact dual
    // warm start in proportion to how far that contact's penalty has been ramped above
    // its effective-mass baseline. 0 disables it and is bit-exact against the pre-0.1
    // behaviour; 1 fully removes the warm start from a maximally ramped contact.
    //
    // It is the energy control for deep piles: without it a 37-layer pile at 8
    // iterations carries 37% more kinetic energy and 12% more penetration at the p99.
    // See the measurements at the assignment in the solver facade.
    float dualDamping;

    // Relative contact speed, in m/s, at which dual damping fades to zero. Below it the
    // damping ramps in linearly. Keeps the damping confined to quasi-static contacts.
    float dualDampVel;

    // A body whose position falls below killHeight (measured along the up axis, -normalize(gravity); +Y if gravity is zero) is force-slept
    // and excluded from further simulation -- the cheap approximation of despawning,
    // since the buffer model has no index-reclaiming removal path (see VkWorld's capacity code).
    // It stays resident (so indices remain stable for anything holding one) but costs
    // nothing more: frozen exactly like a sleeping body, and just as unreachable by the
    // wake paths once nothing is left down there to touch it.
    bool killFallenEnabled; // Master switch
    float killHeight;       // Kill threshold: height along the up axis

    // Continuous collision (ccd.slang, Phase 5): after the solve, a body that moved more than its
    // smallest half extent is swept against statics (bullets also against non-bullet dynamics)
    // and clamped to the earliest time of impact. Box3D's default is on.
    bool enableContinuous;

    // Multiplier on the mouse-drag handle's spring stiffness (see dragStiffnessFor in
    // the drag path). 1 is the tuned default; lower values give a softer grab that a
    // heavy structure can pull out of your hand.
    float grabStrength;

    // Angular inertia in the world frame: the primal uses R diag(moment) R^T (R = the body
    // rotation at the start of the step) and the explicit gyroscopic torque -w x (I w), so a
    // torque-free spinning body conserves its angular momentum. On by default:
    // diag(moment) on the world-frame step drifts a tilted spinning thin box's angular momentum
    // by 138% over 120 steps (2.6% with this on), and the perturbed 100k pile measured no cost
    // (alternated, 3 rounds, within noise).
    bool worldFrameInertia;

    // Angular velocity from the step's rotation (integrate.slang spinStep): the exact inverse of
    // the rotation update, capped at a quarter turn per step. Off is the legacy 2 vec(dq) / dt,
    // which loses (w dt)^2 / 8 of the spin every step (a free sphere at 8 rad/s lost 11% in 1 s,
    // and a rolling one slowed with no rolling resistance). On by default; off
    // exists only so the golden gates can replay the frozen goldens, which used the legacy form.
    bool exactSpinVelocity;

    // ---- Fidelity trade. Off by default, and with it off the step is bit-identical to a
    // build without it. ----

    // Contact cap per manifold. A box-box face contact can produce up to 8 points; a
    // cap of 4 keeps the deepest point and then the points that spread furthest from
    // the ones already kept (the area-preserving reduction game engines use), so a
    // resting face keeps its four corners. 0 (or 8 and above) disables the cap; 1-3
    // run as 4, because a smaller cap drops a different corner of a rocking face each
    // step and a pile never settles (measured; VkWorld's effectiveContactCap).
    int contactCap;

    // Two-level AVBD: after each colour sweep, a coarse
    // rigid-mode correction per spatial aggregate (grid cell). coarseIterations: how many
    // of the iterations get it (0 = off). coarseCell: aggregate edge in metres (0 = 4 x the
    // broadphase cell). coarseDamping: scale on the coarse step.
    int coarseIterations;
    float coarseCell;
    float coarseDamping;
};

// The tuned defaults. Every value here was measured; the comments carry the numbers.
inline SolverParams defaultSolverParams()
{
    SolverParams params;
    params.dt = 1.0f / 60.0f;
    params.gravity = vec3{0.0f, -10.0f, 0.0f};
    params.iterations = 12;

    // Four cooperative lanes per body; see the field comment for why 4.
    params.primalLanes = 4;

    // See the CPU solver for the reasoning behind these values. beta remains split
    // into linear and angular because the constraint functions are not normalised.
    params.betaLin = 10000.0f;
    params.betaAng = 100.0f;
    // T2, corrected. Was 0.95; 0.99 was measured and rejected, see below.
    //
    // alpha is the fraction of the previous step's constraint error carried into this
    // one, so 1 - alpha is error discarded by construction, every step. On a deep pile
    // that discard is a standing energy source no iteration count removes.
    //
    // Measured with sleeping DISABLED, alternated A/B, on the 37-layer pile at 8
    // iterations, in both the quiescent state and the state left behind after a 10 m
    // cube of density 10 has been dropped on it:
    //
    //                    quiescent              after a drop
    //   0.97    step -3.3%  resid -25.8%   step -2.9%  resid -21.4%  energy  +6.7%
    //   0.99    step  -85%  resid -58.8%   step -7.1%  resid -41.5%  energy +115%
    //
    // 0.99 looks transformative on a pile that has never been touched, and is not:
    // the quiescent contact set it reaches (202,612 points, one 4-point vertical
    // manifold per body) is metastable. One dropped cube takes it to ~914,000 and it
    // does not come back within 600 steps. What is left is a pile holding twice the
    // kinetic energy and moving at twice the speed of the 0.95 baseline, for a 7%
    // step-time saving. Under a harder impact the saving inverts.
    //
    // 0.97 is the same effect without the trade: better residual and fewer contacts
    // in both regimes, quieter when quiescent, and energy-neutral after an impact.
    params.alpha = 0.97f;
    params.gamma = 0.99f;

    // On by default: measured a clear win on every scene tested -- lower residual,
    // calmer rest state and 16-27% faster -- and it converges to the same answer as
    // the old constant floor, so it changes the path to the solution and not the
    // solution. See DUAL_STEP_CONTROL_PLAN.md for the numbers.
    params.autoKStart = true;
    // On at 0.1. The pile is the scene that needed it: at 8 iterations the dual warm
    // start is a net energy source, and damping it in proportion to how far the penalty
    // has escaped its effective-mass baseline removes that energy without touching
    // contacts that are merely heavily loaded.
    //
    // Measured on the 37-layer pile (50,653 bodies), sleeping off, 8 iterations, mean
    // over 60 steps after 100 warmup, against dualDamping = 0:
    //
    //     kinetic energy   2848  -> 1797   (-37%)
    //     |C| normal p99   3.93e-3 -> 3.44e-3   (-12%)
    //     |C| friction     6.73e-4 -> 5.98e-4   (-11%)
    //     |C| normal mean  4.48e-4 -> 4.71e-4   (+5%, the one thing that gets worse)
    //     step time        3.24 ms -> 3.32 ms   (inside run-to-run spread)
    //
    // The mean normal residual rises while its p99 falls, which is the shape to expect:
    // the damping is targeted at the deepest, most ratcheted contacts, so it trades a
    // little error spread evenly across the contact set for a lot of error at the tail.
    // `100k` moves the same way (energy 2.26e4 -> 2.04e4, friction -9%, time flat), and
    // `pyramid`, `stack` and `dynamic_friction` are unaffected beyond scene noise.
    //
    // Under a perturbation -- a 3 m cube dropped 1 m onto the settled pile, 240 measured
    // steps -- end-of-window energy falls 5.41e4 -> 3.89e4 and penetration p99 falls 15%,
    // so the win is not an artifact of measuring a quiescent scene.
    //
    // 0.2 and 0.3 keep buying energy (down to 1280 and 1001) but inflate the contact set
    // under perturbation by 21% and 26%, which is the cost this scene can least afford.
    params.dualDamping = 0.1f;
    params.dualDampVel = 0.1f;

    // Sleeping is off by default: every dynamic body is solved every step. The tuning
    // below is what the "Never sleep" switch turns back on, and it is still the value
    // sleeping runs with once enabled.
    //
    // Half a second of settled behaviour before a body may sleep.
    //
    // A conservative tuning: a body must hold still to within a millimetre for just
    // over a second before it is allowed to freeze.
    //
    // The velocity pair is no longer the rest criterion -- k_sleep_mark explains why a
    // velocity threshold cannot work for a loaded joint -- it is a fast reject, applied
    // at 2x, that discards bodies which are plainly in motion without making them wait
    // out the window. It also sets the wake hysteresis, so at 0.4 a sleeper is woken by
    // a neighbour moving at 0.8 m/s or more.
    params.sleepEnabled = false;
    params.sleepFrames = 64;
    params.sleepLinVel = 0.4f;
    params.sleepAngVel = 0.4f;

    // 1mm of NET movement across the whole 64-frame window: about 1mm/s of tolerated
    // drift, which is essentially "not moving at all".
    //
    // This is deliberately far tighter than what the throughput sweep alone would pick.
    // Sweeping displacement on a 10k pile settled for 1500 steps, at a 30-frame window,
    // gives a sharp knee and then a plateau:
    //
    //      5mm ->  6% asleep     20mm -> 77%     50mm -> 84%
    //     10mm -> 32%            30mm -> 85%
    //     15mm -> 77%
    //
    // so 20mm is where the population goes to sleep and the step time drops. 1mm sits
    // well below that knee and will put far fewer bodies to sleep on a pile that is
    // still redistributing -- which is the intended trade: simulation quality is being
    // bought with throughput, on the principle that a body frozen while it was still
    // creeping is a visible artifact whereas a body left awake is merely slower. The
    // longer 64-frame window partly offsets it, since jitter has more time to cancel.
    //
    // Both are exposed in the UI ("Sleep travel", "Sleep frames") next to the "Never
    // sleep" switch, so this is a starting point to tune against a scene rather than a
    // measured optimum. See k_sleep_mark for why net displacement is the right quantity
    // regardless of where the threshold is set.
    params.sleepDisp =
        0.005f; // 2026-10: 0.001 kept 85-45% of Pyramids 10k awake at 900 steps (8/12 it); 0.005 sleeps all, same height
    params.sleepLeak = 0.0f;

    // 80% of the fracture load. Below this a joint has enough headroom that it is not
    // about to break; above it the endpoints stay awake so the break can happen.
    params.sleepFracture = 0.8f;

    // On by default: anything that falls 50m below the origin is almost always a body
    // that missed the scene's floor plane entirely, not one that is still doing
    // something interesting.
    params.killFallenEnabled = true;
    params.enableContinuous = true;
    params.killHeight = -50.0f;

    params.grabStrength = 1.0f;
    params.worldFrameInertia = true;
    params.exactSpinVelocity = true;

    // Fidelity trade: off.
    params.contactCap = 0;

    params.coarseIterations = 0;
    params.coarseCell = 0.0f;
    params.coarseDamping = 0.7f;
    return params;
}
