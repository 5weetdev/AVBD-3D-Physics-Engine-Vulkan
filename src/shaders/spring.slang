// Plain distance springs and cloth edges (SpringSoA mode != 0).

module spring;

import maths;
import types;
import joint_weld;

// ---------------------------------------------------------------------------
// Cloth springs (SpringSoA mode != 0, SceneBuilder::addCloth). Both ends are particles
// (AVBD_SHAPE_PARTICLE, rotation held fixed by the primal solve) or a static anchor, and
// the anchors are the body centres (rA = rB = 0), so only the linear block is stamped.
//
// A stretch edge (mode 1) is the AVBD hard constraint of Eq. 8 on C = |pA - pB| - rest,
// regularised as Eq. 18 (C - alpha*C0), with the force clamped to tension (F >= 0): a
// cloth edge resists stretching and goes slack when compressed, so the sheet folds and
// drapes rather than acting like a plate. The penalty ramps by betaLin*|C| while the
// edge is taut (Eq. 12), up to the edge's own stiffness (INFINITY: CLOTH_PENALTY_MAX).
// lambda > fracture breaks it (cloth.slang csClothSpringDual), and the tear kernel then
// splits the mesh vertices that the break disconnects.
// ---------------------------------------------------------------------------
public static const float CLOTH_PENALTY_MAX = 10000000000.0f;

// A tearable edge also breaks on strain: past this stretch (length / rest - 1) it goes,
// whatever force it carries. The ramped penalty is finite, so a force test alone lets the
// triangles at a tear tip stretch to many times their size before the edge lets go.
public static const float CLOTH_TEAR_STRAIN = 0.15f;

// Transliterated from Spring::updatePrimal. No dual state.
public void stampSpring(SpringSoA s, int i, bool isA, BodySoA b,
                         inout float3x3 lhsLin, inout float3x3 lhsAng, inout float3x3 lhsCross,
                         inout vec3 rhsLin, inout vec3 rhsAng)
{
    stampSpringA(s, i, isA, b, 0.0f, lhsLin, lhsAng, lhsCross, rhsLin, rhsAng);
}

public void stampSpringA(SpringSoA s, int i, bool isA, BodySoA b, float alpha,
                         inout float3x3 lhsLin, inout float3x3 lhsAng, inout float3x3 lhsCross,
                         inout vec3 rhsLin, inout vec3 rhsAng)
{
    int ia = s.bodyA[i], ib = s.bodyB[i];
    vec3 rA = s.rA[i], rB = s.rB[i];

    vec3 pA = jointAnchor(b, ia, rA);
    vec3 pB = transformQ(b.pos[ib], b.rot[ib], rB);
    vec3 d = sub(pA, pB);
    float dLen = length3(d);
    if (dLen <= 1.0e-6f)
        return;

    float stiffness = s.stiffness[i];
    vec3 n = divs(d, dLen);
    float C = dLen - s.rest[i];
    float f = stiffness * C;

    vec3 rWorld;
    vec3 jLin;
    vec3 jAng;
    if (isA)
    {
        rWorld = rotateQ(b.rot[ia], rA);
        jLin = n;
        jAng = cross3(rWorld, n);
    }
    else
    {
        rWorld = rotateQ(b.rot[ib], rB);
        jLin = neg(n);
        jAng = neg(cross3(rWorld, n));
    }

    vec3 F = scale(jLin, f);
    vec3 Tau = scale(jAng, f);
    float3x3 Kll = scaleM(outerM(jLin, jLin), stiffness);
    float3x3 Kla = scaleM(outerM(jAng, jLin), stiffness);
    float3x3 Kaa = scaleM(outerM(jAng, jAng), stiffness);

    lhsLin = addM(lhsLin, Kll);
    lhsAng = addM(lhsAng, Kaa);
    lhsCross = addM(lhsCross, Kla);
    rhsLin = add(rhsLin, F);
    rhsAng = add(rhsAng, Tau);
}
