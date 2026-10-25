// The per-slot bodies of the dual stage, shared by dual.slang's one-thread-per-slot kernels
// (csDual, csJointDual) and the fused small-world iteration kernel in primal.slang
// (csIterateSmall), which loops the same functions over a strided slot range inside one
// workgroup. No entry points and no push constant here: those stay with the importers.
// The arithmetic is exactly what dual.slang held before the split; only the thread-index
// guard moved to the callers.

module dual_body;

import maths;
import types;
import contact;
import forces;

public static const float PENALTY_MAX = 10000000000.0f;
public static const float STICK_THRESH = 0.00001f;

// True when neither end of the edge can move this step (see the dual stage's
// jointDormant). Phase 4b's hand-built test scenes keep every body awake, so this
// matters only for parity with the reference gate rather than for behavior in the test.
public bool jointDormant(BodySoA b, int ia, int ib)
{
    bool stillA = ia < 0 || b.mass[ia] <= 0.0f || b.awake[ia] == 0;
    bool stillB = ib < 0 || b.mass[ib] <= 0.0f || b.awake[ib] == 0;
    return stillA && stillB;
}

public struct JointDualArgs
{
    public BodySoA b;
    public JointSoA j;
    public SolverParamsGpu p;
};

// One joint's dual update; the caller has bounds-checked i against j.count.
public void jointDualSlot(JointDualArgs *args, int i)
{
    BodySoA b = args->b;
    JointSoA j = args->j;
    if ((uint)i >= j.count)
        return;
    if (jointDormant(b, j.bodyA[i], j.bodyB[i]))
        return;
    jointDual(j, i, b, args->p.alpha, args->p.betaLin, args->p.betaAng);
}

public struct DualArgs
{
    public BodySoA b;
    public ManifoldSoA m;
    public int slots;
    public SolverParamsGpu p;
};

// One manifold slot's dual update.
public void dualSlot(DualArgs *args, int t)
{
    BodySoA b = args->b;
    ManifoldSoA m = args->m;
    if (t >= args->slots)
        return;

    int n = m.contactCount[t];
    if (n <= 0)
        return;

    int ia = m.bodyA[t];
    int ib = m.bodyB[t];
    float3x3 basis = m.basis[t];
    float friction = m.friction[t];
    int off = m.contactOffset[t];

    vec3 dqALin = sub(b.pos[ia], b.initialLin[ia]);
    vec3 dqAAng = subQQ(b.rot[ia], b.initialAng[ia]);
    vec3 dqBLin = sub(b.pos[ib], b.initialLin[ib]);
    vec3 dqBAng = subQQ(b.rot[ib], b.initialAng[ib]);

    float alpha = args->p.alpha;
    float betaLin = args->p.betaLin;

    for (int c = 0; c < n; c++)
    {
        int o = off + c;
        vec3 penalty = m.penalty[o];

        ContactEval ce = evalContact(basis, friction, b.rot[ia], b.rot[ib], m.rA[o], m.rB[o],
                                      dqALin, dqAAng, dqBLin, dqBAng, m.C0[o], penalty, m.lambda[o], alpha);

        m.lambda[o] = ce.F;
        uint stk = (uint)m.stick[o];

        if (ce.F.x < 0.0f)
            penalty.x = min(penalty.x + betaLin * abs(ce.C.x), PENALTY_MAX);
        if (ce.frictionScale <= ce.bounds)
        {
            penalty.y = min(penalty.y + betaLin * abs(ce.C.y), PENALTY_MAX);
            penalty.z = min(penalty.z + betaLin * abs(ce.C.z), PENALTY_MAX);
            stk = (sqrt(ce.C.y * ce.C.y + ce.C.z * ce.C.z) < STICK_THRESH ? 1u : 0u);
        }
        m.stick[o] = (uint8_t)stk;
        m.penalty[o] = penalty;
    }
}
