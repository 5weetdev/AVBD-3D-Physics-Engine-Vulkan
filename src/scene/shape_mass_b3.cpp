#include "shape_mass_b3.h"

#include "box3d/collision.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

namespace
{

struct M3
{
    double m[3][3];
};

M3 zero3()
{
    M3 r;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            r.m[i][j] = 0.0;
    return r;
}

// b3Matrix3 is three columns.
M3 fromB3(const b3Matrix3 &a)
{
    M3 r;
    const b3Vec3 c[3] = {a.cx, a.cy, a.cz};
    for (int j = 0; j < 3; j++)
    {
        r.m[0][j] = c[j].x;
        r.m[1][j] = c[j].y;
        r.m[2][j] = c[j].z;
    }
    return r;
}

// Rotation matrix of q.
M3 rotMat(quat q)
{
    const double x = q.x, y = q.y, z = q.z, w = q.w;
    M3 r;
    r.m[0][0] = 1 - 2 * (y * y + z * z);
    r.m[0][1] = 2 * (x * y - z * w);
    r.m[0][2] = 2 * (x * z + y * w);
    r.m[1][0] = 2 * (x * y + z * w);
    r.m[1][1] = 1 - 2 * (x * x + z * z);
    r.m[1][2] = 2 * (y * z - x * w);
    r.m[2][0] = 2 * (x * z - y * w);
    r.m[2][1] = 2 * (y * z + x * w);
    r.m[2][2] = 1 - 2 * (x * x + y * y);
    return r;
}

M3 mul(const M3 &a, const M3 &b)
{
    M3 r = zero3();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                r.m[i][j] += a.m[i][k] * b.m[k][j];
    return r;
}

M3 transpose(const M3 &a)
{
    M3 r;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            r.m[i][j] = a.m[j][i];
    return r;
}

// Jacobi eigen-decomposition of a symmetric matrix: eigenvalues in w, eigenvectors in the
// columns of v.
void jacobi(M3 a, double w[3], M3 &v)
{
    v = zero3();
    for (int i = 0; i < 3; i++)
        v.m[i][i] = 1.0;
    for (int sweep = 0; sweep < 64; sweep++)
    {
        const double off = a.m[0][1] * a.m[0][1] + a.m[0][2] * a.m[0][2] + a.m[1][2] * a.m[1][2];
        if (off == 0.0)
            break;
        for (int p = 0; p < 2; p++)
            for (int q = p + 1; q < 3; q++)
            {
                if (a.m[p][q] == 0.0)
                    continue;
                const double theta = (a.m[q][q] - a.m[p][p]) / (2.0 * a.m[p][q]);
                const double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                // a <- J^T a J, J the rotation in the (p, q) plane.
                for (int k = 0; k < 3; k++)
                {
                    const double akp = a.m[k][p], akq = a.m[k][q];
                    a.m[k][p] = c * akp - s * akq;
                    a.m[k][q] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; k++)
                {
                    const double apk = a.m[p][k], aqk = a.m[q][k];
                    a.m[p][k] = c * apk - s * aqk;
                    a.m[q][k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 3; k++)
                {
                    const double vkp = v.m[k][p], vkq = v.m[k][q];
                    v.m[k][p] = c * vkp - s * vkq;
                    v.m[k][q] = s * vkp + c * vkq;
                }
            }
    }
    for (int i = 0; i < 3; i++)
        w[i] = a.m[i][i];
}

quat quatFromRot(const M3 &r)
{
    double x, y, z, w;
    const double tr = r.m[0][0] + r.m[1][1] + r.m[2][2];
    if (tr > 0.0)
    {
        const double s = std::sqrt(tr + 1.0) * 2.0;
        w = 0.25 * s;
        x = (r.m[2][1] - r.m[1][2]) / s;
        y = (r.m[0][2] - r.m[2][0]) / s;
        z = (r.m[1][0] - r.m[0][1]) / s;
    }
    else if (r.m[0][0] > r.m[1][1] && r.m[0][0] > r.m[2][2])
    {
        const double s = std::sqrt(1.0 + r.m[0][0] - r.m[1][1] - r.m[2][2]) * 2.0;
        w = (r.m[2][1] - r.m[1][2]) / s;
        x = 0.25 * s;
        y = (r.m[0][1] + r.m[1][0]) / s;
        z = (r.m[0][2] + r.m[2][0]) / s;
    }
    else if (r.m[1][1] > r.m[2][2])
    {
        const double s = std::sqrt(1.0 + r.m[1][1] - r.m[0][0] - r.m[2][2]) * 2.0;
        w = (r.m[0][2] - r.m[2][0]) / s;
        x = (r.m[0][1] + r.m[1][0]) / s;
        y = 0.25 * s;
        z = (r.m[1][2] + r.m[2][1]) / s;
    }
    else
    {
        const double s = std::sqrt(1.0 + r.m[2][2] - r.m[0][0] - r.m[1][1]) * 2.0;
        w = (r.m[1][0] - r.m[0][1]) / s;
        x = (r.m[0][2] + r.m[2][0]) / s;
        y = (r.m[1][2] + r.m[2][1]) / s;
        z = 0.25 * s;
    }
    const double n = std::sqrt(x * x + y * y + z * z + w * w);
    return quat{(float)(x / n), (float)(y / n), (float)(z / n), (float)(w / n)};
}

// Closed-mesh mass from signed tetrahedra against the origin. Vertices are welded by exact
// position first (a cooked mesh keeps the caller's duplicated vertices). Closed means every
// undirected edge is used by exactly two triangles in opposite directions; the winding sign
// is taken from the volume, so an inside-out mesh still works.
bool meshClosedVolume(const b3MeshData *mesh, double &volume, double c[3], double cov[3][3])
{
    if (mesh == nullptr)
        return false;
    const b3Vec3 *v = b3GetMeshVertices(mesh);
    const b3MeshTriangle *t = b3GetMeshTriangles(mesh);
    const int vc = mesh->vertexCount, tc = mesh->triangleCount;
    if (v == nullptr || t == nullptr || vc < 4 || tc < 4)
        return false;

    std::map<std::tuple<float, float, float>, int> weld;
    std::vector<int> canon((size_t)vc);
    for (int i = 0; i < vc; i++)
    {
        const auto key = std::make_tuple(v[i].x, v[i].y, v[i].z);
        canon[(size_t)i] = weld.emplace(key, i).first->second;
    }
    std::map<std::pair<int, int>, int> dir; // directed edge -> count
    for (int i = 0; i < tc; i++)
    {
        const int idx[3] = {t[i].index1, t[i].index2, t[i].index3};
        for (int k = 0; k < 3; k++)
            if (idx[k] < 0 || idx[k] >= vc)
                return false;
        const int a = canon[(size_t)idx[0]], b = canon[(size_t)idx[1]], cc = canon[(size_t)idx[2]];
        if (a == b || b == cc || a == cc)
            return false; // degenerate triangle
        dir[{a, b}]++;
        dir[{b, cc}]++;
        dir[{cc, a}]++;
    }
    for (const auto &e : dir)
    {
        if (e.second != 1)
            return false; // an edge used twice the same way
        const auto rev = dir.find({e.first.second, e.first.first});
        if (rev == dir.end() || rev->second != 1)
            return false; // an open edge
    }

    double vol = 0.0, cs[3] = {0, 0, 0}, cc2[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (int i = 0; i < tc; i++)
    {
        const b3Vec3 &p0 = v[t[i].index1], &p1 = v[t[i].index2], &p2 = v[t[i].index3];
        const double a[3] = {p0.x, p0.y, p0.z}, b[3] = {p1.x, p1.y, p1.z}, d[3] = {p2.x, p2.y, p2.z};
        const double det = a[0] * (b[1] * d[2] - b[2] * d[1]) - a[1] * (b[0] * d[2] - b[2] * d[0]) +
                           a[2] * (b[0] * d[1] - b[1] * d[0]);
        const double tv = det / 6.0;
        const double sum[3] = {a[0] + b[0] + d[0], a[1] + b[1] + d[1], a[2] + b[2] + d[2]};
        vol += tv;
        for (int r = 0; r < 3; r++)
        {
            cs[r] += tv * sum[r] * 0.25;
            for (int q = 0; q < 3; q++)
                cc2[r][q] += tv / 20.0 * (a[r] * a[q] + b[r] * b[q] + d[r] * d[q] + sum[r] * sum[q]);
        }
    }
    const double sign = vol < 0.0 ? -1.0 : 1.0;
    vol *= sign;
    if (!(vol > 1e-9))
        return false;
    volume = vol;
    for (int r = 0; r < 3; r++)
    {
        c[r] = sign * cs[r] / vol;
        for (int q = 0; q < 3; q++)
            cov[r][q] = sign * cc2[r][q] - vol * c[r] * c[q]; // about the centroid
    }
    return true;
}

bool meshMassData(const b3MeshData *mesh, float density, b3MassData &out)
{
    double vol, c[3], cov[3][3];
    if (!meshClosedVolume(mesh, vol, c, cov))
        return false;
    const double tr = cov[0][0] + cov[1][1] + cov[2][2];
    out.mass = (float)(density * vol);
    out.center = b3Vec3{(float)c[0], (float)c[1], (float)c[2]};
    const double i[3][3] = {{tr - cov[0][0], -cov[0][1], -cov[0][2]},
                            {-cov[1][0], tr - cov[1][1], -cov[1][2]},
                            {-cov[2][0], -cov[2][1], tr - cov[2][2]}};
    auto f = [&](int r, int q) { return (float)(density * i[r][q]); };
    out.inertia.cx = b3Vec3{f(0, 0), f(1, 0), f(2, 0)};
    out.inertia.cy = b3Vec3{f(0, 1), f(1, 1), f(2, 1)};
    out.inertia.cz = b3Vec3{f(0, 2), f(1, 2), f(2, 2)};
    return true;
}

bool shapeMassData(const ShapeMassInput &s, float density, b3MassData &out)
{
    switch (s.kind)
    {
    case AVBD_SHAPE_BOX:
    {
        const b3BoxHull box = b3MakeBoxHull(s.half.x, s.half.y, s.half.z);
        out = b3ComputeHullMass(&box.base, density);
        return true;
    }
    case AVBD_SHAPE_SPHERE:
    {
        const b3Sphere sphere = {{0.0f, 0.0f, 0.0f}, s.half.x};
        out = b3ComputeSphereMass(&sphere, density);
        return true;
    }
    case AVBD_SHAPE_CAPSULE:
    {
        const float hs = s.half.z - s.half.x > 0.0f ? s.half.z - s.half.x : 0.0f;
        const b3Capsule capsule = {{0.0f, 0.0f, -hs}, {0.0f, 0.0f, hs}, s.half.x};
        out = b3ComputeCapsuleMass(&capsule, density);
        return true;
    }
    case AVBD_SHAPE_HULL:
        if (s.hull == nullptr)
            return false;
        out = b3ComputeHullMass(s.hull, density);
        return true;
    case AVBD_SHAPE_MESH:
        // A closed mesh has volume mass; an open one (or none) has none (a static body ignores it).
        if (s.mesh == nullptr || !meshMassData(s.mesh, density, out))
            out = b3MassData{};
        return true;
    default:
        return false;
    }
}

struct Part
{
    double m;
    double c[3];
    M3 inertia; // about the part's own centre, in body axes
};

// One shape as a part in body axes, at its own density or (if negative) `density`.
bool partOf(const ShapeMassInput &s, float density, Part &pt)
{
    const float rho = s.density >= 0.0f ? s.density : density;
    if (!(rho > 0.0f))
        return false;
    b3MassData md;
    if (!shapeMassData(s, rho, md) || !(md.mass > 0.0f))
        return false;
    const M3 R = rotMat(s.localRot);
    pt.m = md.mass;
    const double lc[3] = {md.center.x, md.center.y, md.center.z};
    for (int r = 0; r < 3; r++)
        pt.c[r] = R.m[r][0] * lc[0] + R.m[r][1] * lc[1] + R.m[r][2] * lc[2];
    pt.c[0] += s.localPos.x;
    pt.c[1] += s.localPos.y;
    pt.c[2] += s.localPos.z;
    pt.inertia = mul(mul(R, fromB3(md.inertia)), transpose(R));
    return true;
}

// Principal moments and rotation of a symmetric tensor (Jacobi), right-handed.
void principal(const M3 &inertia, vec3 &moment, quat &rot)
{
    double w[3];
    M3 v;
    jacobi(inertia, w, v);
    const double det = v.m[0][0] * (v.m[1][1] * v.m[2][2] - v.m[1][2] * v.m[2][1]) -
                       v.m[0][1] * (v.m[1][0] * v.m[2][2] - v.m[1][2] * v.m[2][0]) +
                       v.m[0][2] * (v.m[1][0] * v.m[2][1] - v.m[1][1] * v.m[2][0]);
    if (det < 0.0)
        for (int r = 0; r < 3; r++)
            v.m[r][2] = -v.m[r][2];
    moment = vec3{(float)w[0], (float)w[1], (float)w[2]};
    rot = quatFromRot(v);
}

} // namespace

bool meshIsClosed(const b3MeshData *mesh)
{
    double vol, c[3], cov[3][3];
    return meshClosedVolume(mesh, vol, c, cov);
}

bool computeCompoundMass(const ShapeMassInput *shapes, int count, float density, CompoundMass &out)
{
    out = CompoundMass{};
    if (count <= 0)
        return false;

    std::vector<Part> p((size_t)count);
    double total = 0.0, com[3] = {0, 0, 0};
    int used = 0;
    for (int i = 0; i < count; i++)
    {
        Part &pt = p[(size_t)used];
        if (!partOf(shapes[i], density, pt))
            continue;
        used++;
        total += pt.m;
        for (int r = 0; r < 3; r++)
            com[r] += pt.m * pt.c[r];
    }
    if (used == 0 || !(total > 0.0))
        return false;
    for (int r = 0; r < 3; r++)
        com[r] /= total;

    // Parallel axis: I = sum(I_i + m_i (|d|^2 E - d d^T)) about the combined centre.
    M3 inertia = zero3();
    for (int i = 0; i < used; i++)
    {
        const double d[3] = {p[(size_t)i].c[0] - com[0], p[(size_t)i].c[1] - com[1], p[(size_t)i].c[2] - com[2]};
        const double d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                inertia.m[r][c] += p[(size_t)i].inertia.m[r][c] + p[(size_t)i].m * ((r == c ? d2 : 0.0) - d[r] * d[c]);
    }
    // Symmetrise away rounding before the eigen solve.
    for (int r = 0; r < 3; r++)
        for (int c = r + 1; c < 3; c++)
            inertia.m[r][c] = inertia.m[c][r] = 0.5 * (inertia.m[r][c] + inertia.m[c][r]);

    out.mass = (float)total;
    out.center = vec3{(float)com[0], (float)com[1], (float)com[2]};
    principal(inertia, out.moment, out.principalRot);
    return true;
}

bool computeShapeMass(const ShapeMassInput &shape, float density, float &mass, vec3 &center, float inertia[3][3])
{
    Part pt;
    if (!partOf(shape, density, pt))
    {
        mass = 0.0f;
        center = shape.localPos;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                inertia[r][c] = 0.0f;
        return false;
    }
    mass = (float)pt.m;
    center = vec3{(float)pt.c[0], (float)pt.c[1], (float)pt.c[2]};
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            inertia[r][c] = (float)pt.inertia.m[r][c];
    return true;
}

void principalInertia(const float inertia[3][3], vec3 &moment, quat &rot)
{
    M3 a;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            a.m[r][c] = 0.5 * ((double)inertia[r][c] + (double)inertia[c][r]);
    principal(a, moment, rot);
}
