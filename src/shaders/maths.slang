// Minimal vec3/quat math mirroring engine/core/maths.h, only the subset k_integrate
// needs. Kept as a separate module (not a copy of the whole header) so the parity
// surface with the reference math is easy to audit by eye.
//
// Layout: vec3 is 3 floats (12 bytes) and quat is 4 floats (16 bytes), matching the
// C++ structs exactly under scalar layout -- no padding either side.

module maths;

public struct vec3
{
    public float x, y, z;
};

public struct quat
{
    public float x, y, z, w;
};

// Mirrors engine/core/maths.h:float3x3 -- three row vectors, 36 bytes, no padding
// (row[3] of vec3, alignment 4 throughout; used only as a pointee type here, e.g.
// ManifoldSoA::basis in types.slang, not yet by value in any ported kernel).
public struct float3x3
{
    public vec3 row0, row1, row2;
};

public vec3 add(vec3 a, vec3 b) { return vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
public vec3 sub(vec3 a, vec3 b) { return vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
public vec3 scale(vec3 a, float b) { return vec3(a.x * b, a.y * b, a.z * b); }
public vec3 divs(vec3 a, float b) { return vec3(a.x / b, a.y / b, a.z / b); }

public float dot3(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
public float lengthSq3(vec3 v) { return dot3(v, v); }
public float length3(vec3 v) { return sqrt(lengthSq3(v)); }

public float lengthSqQ(quat q) { return q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w; }
public float lengthQ(quat q) { return sqrt(lengthSqQ(q)); }
public quat divq(quat a, float b) { return quat(a.x / b, a.y / b, a.z / b, a.w / b); }
public quat normalizeQ(quat q) { return divq(q, lengthQ(q)); }

// quat * quat (Hamilton product), same operand/term order as maths.h operator*(quat,quat)
public quat mulQQ(quat a, quat b)
{
    return quat(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

public quat scaleQ(quat a, float b) { return quat(a.x * b, a.y * b, a.z * b, a.w * b); }
public quat addQQ(quat a, quat b) { return quat(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }

// quat + vec3 (maths.h: normalize(a + quat{b,0} * a * 0.5)) -- the orientation update.
public quat addQV(quat a, vec3 b)
{
    quat bq = quat(b.x, b.y, b.z, 0.0f);
    quat delta = scaleQ(mulQQ(bq, a), 0.5f);
    return normalizeQ(addQQ(a, delta));
}

public float signf(float x) { return x < 0.0f ? -1.0f : (x > 0.0f ? 1.0f : 0.0f); }

public float clampf(float x, float a, float b) { return max(a, min(b, x)); }

// ---------------------------------------------------------------------------
// Extensions for engine/shaders/collide.slang (Phase 3 spike): the subset of
// engine/core/maths.h collideBoxes actually calls. Same term order as the reference source
// so the two read side by side.
// ---------------------------------------------------------------------------

public vec3 neg(vec3 a) { return vec3(-a.x, -a.y, -a.z); }
// vec3 is a hand-written struct (not HLSL's builtin float3), so the generic HLSL abs<T,N>
// overload set does not apply to it -- component-wise abs needed by forces.slang's
// fracture test.
public vec3 absv(vec3 a) { return vec3(abs(a.x), abs(a.y), abs(a.z)); }
public vec3 cross3(vec3 a, vec3 b)
{
    return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
public vec3 normalize3(vec3 v) { return divs(v, length3(v)); }
public float compAt(vec3 v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }
public float clampf01(float x, float lo, float hi) { return max(lo, min(hi, x)); }

public quat conjugateQ(quat q) { return quat(-q.x, -q.y, -q.z, q.w); }

// rotate(quat, vec3): v + 2*w*(u x v) + 2*(u x (u x v)), matching maths.h's term order
// (angle.w * t) + cross(u, t) after t = cross(u, v) * 2.
public vec3 rotateQ(quat angle, vec3 v)
{
    vec3 u = vec3(angle.x, angle.y, angle.z);
    vec3 t = scale(cross3(u, v), 2.0f);
    return add(add(v, scale(t, angle.w)), cross3(u, t));
}

public vec3 transformQ(vec3 qLin, quat qAng, vec3 v) { return add(qLin, rotateQ(qAng, v)); }

// float3x3 row access, matching maths.h's operator[]: row 0/1/2.
public vec3 rowAt(float3x3 m, int i) { return i == 0 ? m.row0 : (i == 1 ? m.row1 : m.row2); }
public float compMatAt(float3x3 m, int i, int j) { return compAt(rowAt(m, i), j); }

public float3x3 orthonormalF(vec3 normal)
{
    vec3 t1 = abs(normal.x) > abs(normal.z) ? vec3(-normal.y, normal.x, 0.0f)
                                             : vec3(0.0f, -normal.z, normal.y);
    t1 = normalize3(t1);
    vec3 t2 = cross3(normal, t1);
    float3x3 m;
    m.row0 = normal;
    m.row1 = t1;
    m.row2 = t2;
    return m;
}

// ---------------------------------------------------------------------------
// Extensions for engine/shaders/{primal,dual,contact,forces}.slang (Phase 4b): the
// float3x3 algebra and LDL^T solve engine/core/maths.h uses in the primal/dual step.
// Same operation order as the reference source throughout.
// ---------------------------------------------------------------------------

public float3x3 addM(float3x3 a, float3x3 b)
{
    float3x3 r;
    r.row0 = add(a.row0, b.row0);
    r.row1 = add(a.row1, b.row1);
    r.row2 = add(a.row2, b.row2);
    return r;
}

public float3x3 subM(float3x3 a, float3x3 b)
{
    float3x3 r;
    r.row0 = sub(a.row0, b.row0);
    r.row1 = sub(a.row1, b.row1);
    r.row2 = sub(a.row2, b.row2);
    return r;
}

public float3x3 negM(float3x3 a)
{
    float3x3 r;
    r.row0 = neg(a.row0);
    r.row1 = neg(a.row1);
    r.row2 = neg(a.row2);
    return r;
}

public float3x3 scaleM(float3x3 a, float b)
{
    float3x3 r;
    r.row0 = scale(a.row0, b);
    r.row1 = scale(a.row1, b);
    r.row2 = scale(a.row2, b);
    return r;
}

public float3x3 divM(float3x3 a, float b)
{
    float3x3 r;
    r.row0 = divs(a.row0, b);
    r.row1 = divs(a.row1, b);
    r.row2 = divs(a.row2, b);
    return r;
}

public vec3 colAt(float3x3 m, int i)
{
    return i == 0 ? vec3(m.row0.x, m.row1.x, m.row2.x)
         : i == 1 ? vec3(m.row0.y, m.row1.y, m.row2.y)
                  : vec3(m.row0.z, m.row1.z, m.row2.z);
}

// a * b (matrix product), same term order as maths.h operator*(float3x3,float3x3).
public float3x3 mulMM(float3x3 a, float3x3 b)
{
    float3x3 r;
    r.row0 = vec3(dot3(a.row0, colAt(b, 0)), dot3(a.row0, colAt(b, 1)), dot3(a.row0, colAt(b, 2)));
    r.row1 = vec3(dot3(a.row1, colAt(b, 0)), dot3(a.row1, colAt(b, 1)), dot3(a.row1, colAt(b, 2)));
    r.row2 = vec3(dot3(a.row2, colAt(b, 0)), dot3(a.row2, colAt(b, 1)), dot3(a.row2, colAt(b, 2)));
    return r;
}

// a * v
public vec3 mulMV(float3x3 a, vec3 v)
{
    return vec3(dot3(a.row0, v), dot3(a.row1, v), dot3(a.row2, v));
}

public float3x3 transposeM(float3x3 a)
{
    float3x3 r;
    r.row0 = vec3(a.row0.x, a.row1.x, a.row2.x);
    r.row1 = vec3(a.row0.y, a.row1.y, a.row2.y);
    r.row2 = vec3(a.row0.z, a.row1.z, a.row2.z);
    return r;
}

public float3x3 diagonalM(float m00, float m11, float m22)
{
    float3x3 r;
    r.row0 = vec3(m00, 0.0f, 0.0f);
    r.row1 = vec3(0.0f, m11, 0.0f);
    r.row2 = vec3(0.0f, 0.0f, m22);
    return r;
}

public float3x3 diagonalV(vec3 d) { return diagonalM(d.x, d.y, d.z); }

public float3x3 zeroM()
{
    float3x3 r;
    r.row0 = vec3(0.0f, 0.0f, 0.0f);
    r.row1 = vec3(0.0f, 0.0f, 0.0f);
    r.row2 = vec3(0.0f, 0.0f, 0.0f);
    return r;
}

// a * diagonal(d), without materializing the diagonal matrix (maths.h:mulDiag).
public float3x3 mulDiag(float3x3 a, vec3 d)
{
    float3x3 r;
    r.row0 = vec3(a.row0.x * d.x, a.row0.y * d.y, a.row0.z * d.z);
    r.row1 = vec3(a.row1.x * d.x, a.row1.y * d.y, a.row1.z * d.z);
    r.row2 = vec3(a.row2.x * d.x, a.row2.y * d.y, a.row2.z * d.z);
    return r;
}

public float3x3 outerM(vec3 a, vec3 b)
{
    float3x3 r;
    r.row0 = scale(b, a.x);
    r.row1 = scale(b, a.y);
    r.row2 = scale(b, a.z);
    return r;
}

public float3x3 skewM(vec3 r_)
{
    float3x3 m;
    m.row0 = vec3(0.0f, -r_.z, r_.y);
    m.row1 = vec3(r_.z, 0.0f, -r_.x);
    m.row2 = vec3(-r_.y, r_.x, 0.0f);
    return m;
}

public float3x3 diagonalizeM(float3x3 m)
{
    return diagonalM(length3(colAt(m, 0)), length3(colAt(m, 1)), length3(colAt(m, 2)));
}

// quat - quat (maths.h): (a * inverse(b)).vec() * 2
public vec3 subQQ(quat a, quat b)
{
    float lsq = lengthSqQ(b);
    quat conj = conjugateQ(b);
    quat invB = quat(conj.x / lsq, conj.y / lsq, conj.z / lsq, conj.w / lsq);
    quat d = mulQQ(a, invB);
    return vec3(d.x * 2.0f, d.y * 2.0f, d.z * 2.0f);
}

public quat inverseQ(quat q)
{
    float lsq = lengthSqQ(q);
    quat conj = conjugateQ(q);
    return quat(conj.x / lsq, conj.y / lsq, conj.z / lsq, conj.w / lsq);
}

// Pivot guard for LDL^T (maths.h:ldlPivot).
public float ldlPivot(float d, float ldlEps) { return d > ldlEps ? d : ldlEps; }

// 6x6 SPD LDL^T solve, transliterated term-for-term from maths.h:solve.
public void solve6(float3x3 aLin, float3x3 aAng, float3x3 aCross, vec3 bLin, vec3 bAng,
                    out vec3 xLin, out vec3 xAng)
{
    float A11 = aLin.row0.x;
    float A21 = aLin.row1.x, A22 = aLin.row1.y;
    float A31 = aLin.row2.x, A32 = aLin.row2.y, A33 = aLin.row2.z;
    float A41 = aCross.row0.x, A42 = aCross.row0.y, A43 = aCross.row0.z, A44 = aAng.row0.x;
    float A51 = aCross.row1.x, A52 = aCross.row1.y, A53 = aCross.row1.z, A54 = aAng.row1.x, A55 = aAng.row1.y;
    float A61 = aCross.row2.x, A62 = aCross.row2.y, A63 = aCross.row2.z, A64 = aAng.row2.x, A65 = aAng.row2.y, A66 = aAng.row2.z;

    float maxDiag = max(max(max(A11, A22), max(A33, A44)), max(A55, A66));
    float ldlEps = max(maxDiag * 1.0e-5f, 1.0e-6f);

    float D1 = ldlPivot(A11, ldlEps);

    float L21 = A21 / D1;
    float L31 = A31 / D1;
    float L41 = A41 / D1;
    float L51 = A51 / D1;
    float L61 = A61 / D1;

    float D2raw = A22 - L21 * L21 * D1;
    float D2 = ldlPivot(D2raw, ldlEps);

    float L32 = (A32 - L21 * L31 * D1) / D2;
    float L42 = (A42 - L21 * L41 * D1) / D2;
    float L52 = (A52 - L21 * L51 * D1) / D2;
    float L62 = (A62 - L21 * L61 * D1) / D2;

    float D3raw = A33 - (L31 * L31 * D1 + L32 * L32 * D2);
    float D3 = ldlPivot(D3raw, ldlEps);

    float L43 = (A43 - L31 * L41 * D1 - L32 * L42 * D2) / D3;
    float L53 = (A53 - L31 * L51 * D1 - L32 * L52 * D2) / D3;
    float L63 = (A63 - L31 * L61 * D1 - L32 * L62 * D2) / D3;

    float D4raw = A44 - (L41 * L41 * D1 + L42 * L42 * D2 + L43 * L43 * D3);
    float D4 = ldlPivot(D4raw, ldlEps);

    float L54 = (A54 - L41 * L51 * D1 - L42 * L52 * D2 - L43 * L53 * D3) / D4;
    float L64 = (A64 - L41 * L61 * D1 - L42 * L62 * D2 - L43 * L63 * D3) / D4;

    float D5raw = A55 - (L51 * L51 * D1 + L52 * L52 * D2 + L53 * L53 * D3 + L54 * L54 * D4);
    float D5 = ldlPivot(D5raw, ldlEps);

    float L65 = (A65 - L51 * L61 * D1 - L52 * L62 * D2 - L53 * L63 * D3 - L54 * L64 * D4) / D5;

    float D6raw = A66 - (L61 * L61 * D1 + L62 * L62 * D2 + L63 * L63 * D3 + L64 * L64 * D4 + L65 * L65 * D5);
    float D6 = ldlPivot(D6raw, ldlEps);

    float y1 = bLin.x;
    float y2 = bLin.y - L21 * y1;
    float y3 = bLin.z - L31 * y1 - L32 * y2;
    float y4 = bAng.x - L41 * y1 - L42 * y2 - L43 * y3;
    float y5 = bAng.y - L51 * y1 - L52 * y2 - L53 * y3 - L54 * y4;
    float y6 = bAng.z - L61 * y1 - L62 * y2 - L63 * y3 - L64 * y4 - L65 * y5;

    float z1 = y1 / D1;
    float z2 = y2 / D2;
    float z3 = y3 / D3;
    float z4 = y4 / D4;
    float z5 = y5 / D5;
    float z6 = y6 / D6;

    float xAng2 = z6;
    float xAng1 = z5 - L65 * xAng2;
    float xAng0 = z4 - L54 * xAng1 - L64 * xAng2;
    float xLin2 = z3 - L43 * xAng0 - L53 * xAng1 - L63 * xAng2;
    float xLin1 = z2 - L32 * xLin2 - L42 * xAng0 - L52 * xAng1 - L62 * xAng2;
    float xLin0 = z1 - L21 * xLin1 - L31 * xLin2 - L41 * xAng0 - L51 * xAng1 - L61 * xAng2;

    xLin = vec3(xLin0, xLin1, xLin2);
    xAng = vec3(xAng0, xAng1, xAng2);
}
