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

#include <cmath>

// Function annotation. Every function in this header is a pure leaf with no STL,
// no exceptions and no virtuals. AVBD_HD is kept so declarations stay uniform across the
// tree; it expands to nothing.
#define AVBD_HD

// Note: no `using namespace std;` here. Everything below uses the global-namespace C math
// functions (sqrtf, fabsf, ...), which <cmath> provides directly.

// Math types

struct vec2
{
    float x, y;

    AVBD_HD float &operator[](int i) { return ((float *)this)[i]; }
    AVBD_HD const float &operator[](int i) const { return ((float *)this)[i]; }
};

struct vec3
{
    float x, y, z;

    AVBD_HD vec2 &xy() { return *(vec2 *)this; }
    AVBD_HD const vec2 &xy() const { return *(vec2 *)this; }

    AVBD_HD float &operator[](int i) { return ((float *)this)[i]; }
    AVBD_HD const float &operator[](int i) const { return ((float *)this)[i]; }
};

struct quat
{
    float x, y, z, w;

    AVBD_HD vec3 &vec() { return *(vec3 *)this; }
    AVBD_HD const vec3 &vec() const { return *(vec3 *)this; }
    AVBD_HD float &operator[](int i) { return ((float *)this)[i]; }
    AVBD_HD const float &operator[](int i) const { return ((float *)this)[i]; }
};

struct float2x2
{
    vec2 row[2];

    AVBD_HD vec2 &operator[](int i) { return row[i]; }
    AVBD_HD const vec2 &operator[](int i) const { return row[i]; }

    AVBD_HD vec2 col(int i) const { return vec2{row[0][i], row[1][i]}; }
};

struct float3x3
{
    vec3 row[3];

    AVBD_HD vec3 &operator[](int i) { return row[i]; }
    AVBD_HD const vec3 &operator[](int i) const { return row[i]; }

    AVBD_HD vec3 col(int i) const { return vec3{row[0][i], row[1][i], row[2][i]}; }
};

// vec2 operators

AVBD_HD float dot(vec2 a, vec2 b);

AVBD_HD inline vec2 operator+=(vec2 &a, vec2 b)
{
    a.x += b.x;
    a.y += b.y;
    return a;
}

AVBD_HD inline vec2 operator-=(vec2 &a, vec2 b)
{
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

AVBD_HD inline vec2 operator-(vec2 v)
{
    return {-v.x, -v.y};
}

AVBD_HD inline vec2 operator+(vec2 a, vec2 b)
{
    return {a.x + b.x, a.y + b.y};
}

AVBD_HD inline vec2 operator-(vec2 a, vec2 b)
{
    return {a.x - b.x, a.y - b.y};
}

AVBD_HD inline vec2 operator*(vec2 a, float b)
{
    return {a.x * b, a.y * b};
}

AVBD_HD inline vec2 operator/(vec2 a, float b)
{
    return {a.x / b, a.y / b};
}

AVBD_HD inline vec2 operator*(float2x2 a, vec2 b)
{
    return {dot(a[0], b), dot(a[1], b)};
}

// vec3 operators

AVBD_HD float dot(vec3 a, vec3 b);

AVBD_HD inline vec3 &operator+=(vec3 &a, vec3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

AVBD_HD inline vec3 &operator-=(vec3 &a, vec3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

AVBD_HD inline vec3 operator-(vec3 v)
{
    return {-v.x, -v.y, -v.z};
}

AVBD_HD inline vec3 operator+(vec3 a, vec3 b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

AVBD_HD inline vec3 operator-(vec3 a, vec3 b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

AVBD_HD inline vec3 operator*(vec3 a, float b)
{
    return {a.x * b, a.y * b, a.z * b};
}

AVBD_HD inline vec3 operator/(vec3 a, float b)
{
    return {a.x / b, a.y / b, a.z / b};
}

AVBD_HD inline vec3 operator*(float3x3 a, vec3 b)
{
    return {dot(a[0], b), dot(a[1], b), dot(a[2], b)};
}

// float2x2 operators

AVBD_HD inline float2x2 operator+(float2x2 a, float2x2 b)
{
    return {a[0] + b[0], a[1] + b[1]};
}

AVBD_HD inline float2x2 operator-(float2x2 a, float2x2 b)
{
    return {a[0] - b[0], a[1] - b[1]};
}

AVBD_HD inline float2x2 operator*(float2x2 a, float b)
{
    return {a[0] * b, a[1] * b};
}

AVBD_HD inline float2x2 operator/(float2x2 a, float b)
{
    return {a[0] / b, a[1] / b};
}

AVBD_HD inline float2x2 operator*(float2x2 a, float2x2 b)
{
    return {
        vec2{dot(a.row[0], b.col(0)), dot(a.row[0], b.col(1))},
        vec2{dot(a.row[1], b.col(0)), dot(a.row[1], b.col(1))}};
}

// float3x3 operators

AVBD_HD inline float3x3 &operator+=(float3x3 &a, float3x3 b)
{
    a[0] += b[0];
    a[1] += b[1];
    a[2] += b[2];
    return a;
}

AVBD_HD inline float3x3 operator+(float3x3 a, float3x3 b)
{
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

AVBD_HD inline float3x3 operator-(float3x3 a, float3x3 b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

AVBD_HD inline float3x3 operator*(float3x3 a, float b)
{
    return {a[0] * b, a[1] * b, a[2] * b};
}

AVBD_HD inline float3x3 operator/(float3x3 a, float b)
{
    return {a[0] / b, a[1] / b, a[2] / b};
}

AVBD_HD inline float3x3 operator-(float3x3 a)
{
    return {-a.row[0], -a.row[1], -a.row[2]};
}

AVBD_HD inline float3x3 operator*(float3x3 a, float3x3 b)
{
    return {
        vec3{dot(a.row[0], b.col(0)), dot(a.row[0], b.col(1)), dot(a.row[0], b.col(2))},
        vec3{dot(a.row[1], b.col(0)), dot(a.row[1], b.col(1)), dot(a.row[1], b.col(2))},
        vec3{dot(a.row[2], b.col(0)), dot(a.row[2], b.col(1)), dot(a.row[2], b.col(2))}};
}

// quat operators

AVBD_HD quat normalize(quat q);
AVBD_HD quat inverse(quat q);
AVBD_HD quat conjugate(quat q);

AVBD_HD inline quat operator*(quat a, float b)
{
    return {a.x * b, a.y * b, a.z * b, a.w * b};
}

AVBD_HD inline quat operator/(quat a, float b)
{
    return {a.x / b, a.y / b, a.z / b, a.w / b};
}

AVBD_HD inline quat operator*(quat a, quat b)
{
    return {
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

AVBD_HD inline quat operator+(quat a, quat b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

AVBD_HD inline vec3 operator-(quat a, quat b)
{
    return (a * inverse(b)).vec() * 2.0f;
}

AVBD_HD inline quat operator+(quat a, vec3 b)
{
    return normalize(a + quat{b.x, b.y, b.z, 0} * a * 0.5f);
}

// Math functions

AVBD_HD inline float rad(float deg)
{
    return deg * 0.01745329251994329577f;
}

AVBD_HD inline float sign(float x)
{
    return x < 0 ? -1.0f : x > 0 ? 1.0f
                                 : 0.0f;
}

inline float min(float a, float b)
{
    return a < b ? a : b;
}

inline float max(float a, float b)
{
    return a > b ? a : b;
}

AVBD_HD inline vec3 min(vec3 a, float b)
{
    return {min(a.x, b), min(a.y, b), min(a.z, b)};
}

AVBD_HD inline float clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}

AVBD_HD inline vec3 clamp(vec3 v, float a, float b)
{
    return {clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b)};
}

AVBD_HD inline float dot(vec2 a, vec2 b)
{
    return a.x * b.x + a.y * b.y;
}

AVBD_HD inline float dot(vec3 a, vec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

AVBD_HD inline float lengthSq(vec2 v)
{
    return dot(v, v);
}

AVBD_HD inline float length(vec2 v)
{
    return sqrtf(lengthSq(v));
}

AVBD_HD inline float lengthSq(vec3 v)
{
    return dot(v, v);
}

AVBD_HD inline float length(vec3 v)
{
    return sqrtf(lengthSq(v));
}

AVBD_HD inline float lengthSq(quat q)
{
    return lengthSq(q.vec()) + q.w * q.w;
}

AVBD_HD inline float length(quat q)
{
    return sqrtf(lengthSq(q));
}

AVBD_HD inline vec3 normalize(vec3 v)
{
    return v / length(v);
}

AVBD_HD inline quat normalize(quat q)
{
    return q / length(q);
}

AVBD_HD inline float cross(vec2 a, vec2 b)
{
    return a.x * b.y - a.y * b.x;
}

AVBD_HD inline vec3 cross(vec3 a, vec3 b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x};
}

AVBD_HD inline float3x3 skew(vec3 r)
{
    return float3x3{
        0, -r.z, r.y,
        r.z, 0, -r.x,
        -r.y, r.x, 0};
}

AVBD_HD inline float2x2 outer(vec2 a, vec2 b)
{
    return {b * a.x, b * a.y};
}

AVBD_HD inline float3x3 outer(vec3 a, vec3 b)
{
    return {b * a.x, b * a.y, b * a.z};
}

AVBD_HD inline vec2 abs(vec2 v)
{
    return {fabsf(v.x), fabsf(v.y)};
}

AVBD_HD inline vec3 abs(vec3 v)
{
    return {fabsf(v.x), fabsf(v.y), fabsf(v.z)};
}

AVBD_HD inline float2x2 abs(float2x2 a)
{
    return {abs(a[0]), abs(a[1])};
}

AVBD_HD inline float2x2 transpose(float2x2 a)
{
    return {vec2{a[0][0], a[1][0]}, vec2{a[0][1], a[1][1]}};
}

AVBD_HD inline float3x3 transpose(float3x3 a)
{
    return {vec3{a[0][0], a[1][0], a[2][0]}, vec3{a[0][1], a[1][1], a[2][1]}, vec3{a[0][2], a[1][2], a[2][2]}};
}

// a * diagonal(d), without materializing the diagonal matrix.
//
// The general 3x3 product spends 27 multiplies, 18 of them against structural zeros,
// and keeps nine registers live holding those zeros. This is bit-identical to
// a * diagonal(d.x, d.y, d.z): the dot product it replaces sums two exact zeros onto
// the one surviving term.
AVBD_HD inline float3x3 mulDiag(const float3x3 &a, vec3 d)
{
    return {vec3{a[0][0] * d.x, a[0][1] * d.y, a[0][2] * d.z},
            vec3{a[1][0] * d.x, a[1][1] * d.y, a[1][2] * d.z},
            vec3{a[2][0] * d.x, a[2][1] * d.y, a[2][2] * d.z}};
}

AVBD_HD inline float3x3 diagonal(float m00, float m11, float m22)
{
    return float3x3{
        m00, 0, 0,
        0, m11, 0,
        0, 0, m22};
}

AVBD_HD inline quat conjugate(quat q)
{
    return {-q.x, -q.y, -q.z, q.w};
}

AVBD_HD inline quat inverse(quat q)
{
    return conjugate(q) / lengthSq(q);
}

AVBD_HD inline vec3 rotate(quat angle, vec3 v)
{
    vec3 u = {angle.x, angle.y, angle.z};
    vec3 t = cross(u, v) * 2.0f;
    return v + t * angle.w + cross(u, t);
}

AVBD_HD inline vec3 transform(vec3 qLin, quat qAng, vec3 v)
{
    return rotate(qAng, v) + qLin;
}

AVBD_HD inline float3x3 rotation(quat q)
{
    float x = q.x, y = q.y, z = q.z, w = q.w;

    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, xz = x * z, yz = y * z;
    const float wx = w * x, wy = w * y, wz = w * z;

    // Row-major 3x3
    return float3x3{
        1.0f - 2.0f * (yy + zz), 2.0f * (xy + wz), 2.0f * (xz - wy),
        2.0f * (xy - wz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz + wx),
        2.0f * (xz + wy), 2.0f * (yz - wx), 1.0f - 2.0f * (xx + yy)};
}

AVBD_HD inline float3x3 orthonormal(vec3 normal)
{
    vec3 t1 = fabsf(normal.x) > fabsf(normal.z) ? vec3{-normal.y, normal.x, 0} : vec3{0, -normal.z, normal.y};
    t1 = normalize(t1);
    vec3 t2 = cross(normal, t1);
    return float3x3{normal, t1, t2};
}

AVBD_HD inline float3x3 diagonalize(float3x3 m)
{
    return diagonal(length(m.col(0)), length(m.col(1)), length(m.col(2)));
}

// Pivot guard for the LDL^T factorization below. The 6x6 system is SPD by construction,
// Pivot guard for the LDL^T factorization below. The 6x6 system is SPD by construction,
// but float32 catastrophic cancellation can cause pivots to collapse or turn negative.
// Regularize pivots relative to matrix scale so H remains strictly positive definite.
AVBD_HD inline float ldlPivot(float d, float ldlEps)
{
    return d > ldlEps ? d : ldlEps;
}

AVBD_HD inline void solve(float3x3 aLin, float3x3 aAng, float3x3 aCross, vec3 bLin, vec3 bAng, vec3 &xLin, vec3 &xAng)
{
    // Extract elements from lower triangle storage
    float A11 = aLin[0][0];
    float A21 = aLin[1][0], A22 = aLin[1][1];
    float A31 = aLin[2][0], A32 = aLin[2][1], A33 = aLin[2][2];
    float A41 = aCross[0][0], A42 = aCross[0][1], A43 = aCross[0][2], A44 = aAng[0][0];
    float A51 = aCross[1][0], A52 = aCross[1][1], A53 = aCross[1][2], A54 = aAng[1][0], A55 = aAng[1][1];
    float A61 = aCross[2][0], A62 = aCross[2][1], A63 = aCross[2][2], A64 = aAng[2][0], A65 = aAng[2][1], A66 = aAng[2][2];

    // Scale-aware relative regularization floor for single-precision float32.
    // Preserves strict positive-definiteness and bounds L entries under ill-conditioning.
    float maxDiag = fmaxf(fmaxf(fmaxf(A11, A22), fmaxf(A33, A44)), fmaxf(A55, A66));
    const float ldlEps = fmaxf(maxDiag * 1.0e-5f, 1.0e-6f);

    // Step 1: LDL^T decomposition with positive-definite regularization
    float D1raw = A11;
    float D1 = ldlPivot(D1raw, ldlEps);

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

    // Step 2: Forward substitution: Solve Ly = b
    float y1 = bLin[0];
    float y2 = bLin[1] - L21 * y1;
    float y3 = bLin[2] - L31 * y1 - L32 * y2;
    float y4 = bAng[0] - L41 * y1 - L42 * y2 - L43 * y3;
    float y5 = bAng[1] - L51 * y1 - L52 * y2 - L53 * y3 - L54 * y4;
    float y6 = bAng[2] - L61 * y1 - L62 * y2 - L63 * y3 - L64 * y4 - L65 * y5;

    // Step 3: Diagonal solve: Solve Dz = y
    float z1 = y1 / D1;
    float z2 = y2 / D2;
    float z3 = y3 / D3;
    float z4 = y4 / D4;
    float z5 = y5 / D5;
    float z6 = y6 / D6;

    // Step 4: Backward substitution: Solve L^T x = z
    xAng[2] = z6;
    xAng[1] = z5 - L65 * xAng[2];
    xAng[0] = z4 - L54 * xAng[1] - L64 * xAng[2];
    xLin[2] = z3 - L43 * xAng[0] - L53 * xAng[1] - L63 * xAng[2];
    xLin[1] = z2 - L32 * xLin[2] - L42 * xAng[0] - L52 * xAng[1] - L62 * xAng[2];
    xLin[0] = z1 - L21 * xLin[1] - L31 * xLin[2] - L41 * xAng[0] - L51 * xAng[1] - L61 * xAng[2];
}
