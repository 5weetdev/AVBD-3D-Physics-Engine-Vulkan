// evalContact: the shared prologue of
// Manifold::updatePrimal / updateDual on the CPU -- see that file's header comment.
// Same term order throughout so the two read side by side.

module contact;

import maths;

public struct ContactEval
{
    public float3x3 jALin, jBLin; // dC/dxLin for each body
    public float3x3 jAAng, jBAng; // dC/dxAng for each body
    public vec3 C;                // constraint value, normal in [0], friction in [1],[2]
    public vec3 F;                // force after normal and friction-cone clamping
    public float frictionScale;   // magnitude of the unclamped friction force
    public float bounds;          // friction cone limit, |F[0]| * friction
};

public ContactEval evalContact(float3x3 basis, float friction,
                                quat rotA, quat rotB,
                                vec3 rA, vec3 rB,
                                vec3 dqALin, vec3 dqAAng, vec3 dqBLin, vec3 dqBAng,
                                vec3 C0, vec3 penalty, vec3 lambda, float alpha)
{
    ContactEval e;

    vec3 rAWorld = rotateQ(rotA, rA);
    vec3 rBWorld = rotateQ(rotB, rB);

    e.jALin = basis;
    e.jBLin = negM(basis);
    // maths.h: jAAng = {cross(rA, jALin[0]), ...} where jALin[r] is ROW r of the
    // basis, not column r. The two only agree for a symmetric basis (flat ground), which
    // is why every axis-aligned test passed while tilted contacts got wrong torque arms.
    e.jAAng.row0 = cross3(rAWorld, e.jALin.row0);
    e.jAAng.row1 = cross3(rAWorld, e.jALin.row1);
    e.jAAng.row2 = cross3(rAWorld, e.jALin.row2);
    e.jBAng.row0 = cross3(rBWorld, e.jBLin.row0);
    e.jBAng.row1 = cross3(rBWorld, e.jBLin.row1);
    e.jBAng.row2 = cross3(rBWorld, e.jBLin.row2);

    float3x3 K = diagonalV(penalty);
    vec3 C0s = scale(C0, 1.0f - alpha);
    e.C = add(add(add(add(C0s, mulMV(e.jALin, dqALin)), mulMV(e.jBLin, dqBLin)),
                  mulMV(e.jAAng, dqAAng)), mulMV(e.jBAng, dqBAng));

    // Compute force
    e.F = add(mulMV(K, e.C), lambda);

    // Clamp normal force
    e.F.x = min(e.F.x, 0.0f);

    // Clamp norm of friction forces to achieve a friction cone
    e.bounds = abs(e.F.x) * friction;
    e.frictionScale = sqrt(e.F.y * e.F.y + e.F.z * e.F.z);
    if (e.frictionScale > e.bounds && e.frictionScale > 0.0f)
    {
        e.F.y *= e.bounds / e.frictionScale;
        e.F.z *= e.bounds / e.frictionScale;
    }

    return e;
}
