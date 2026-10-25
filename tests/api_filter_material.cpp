// Gate for the body and material features of the avbd3d layer, through the public API only.
// Y-up, ground top at y = 0, gravity (0, -10, 0).
//   a) motion locks: a body locked in linear Z and angular X / Y keeps its z and its x / y rotation
//      under contact and gravity while an unlocked twin does not; Body_SetMotionLocks works live;
//   b) rolling resistance: a rolling sphere slows down with rollingResistance > 0 and keeps its
//      speed with 0;
//   c) tangent velocity: a box on a static belt box is carried along the belt's shape-frame
//      vector (x, z, a belt rotated about y, and a local-y component that is projected away);
//   d) friction mixing: a sliding box decelerates with the rule's mixed coefficient
//      (geometric default, average, min, max, multiply);
//   e) restitution mixing: a dropped ball bounces by the rule's mixed restitution;
//   f) world-frame inertia (SolverDef.worldFrameInertia): a long thin box spinning about a tilted
//      axis in zero gravity conserves L = R I R^T w; the drift with the flag off and on is printed.



// (merged) // Gate for restitution on the avbd3d layer (restitution.slang). Y-up, ground top at y = 0.
// // A body dropped from rest bounces to e^2 of its drop height (energy goes as speed squared):
// //   a) a sphere with e = 1 comes back to about its drop height;
// //   b) e = 0.5 comes back to about a quarter of it;
// //   c) e = 0 does not bounce (the solver's plastic contact, unchanged);
// //   d) e = 1 below the restitution threshold does not bounce;
// //   e) a flat box (a four-point manifold) with e = 1 bounces without picking up spin;
// //   f) an e = 0.5 sphere comes to rest on the ground;
// //   g) restitution is per shape: a body of two spheres, one above the other, bounces only when the
// //      sphere that touches the ground has e = 1 (the primary, first-created shape is the top one).
// 
// 
// 
#include "test_util.h"
#include <cmath>
#include <cstdio>

namespace materials
{
static int g_fail = 0;
#define CHECK(cond, ...)                                                                                              \
    do                                                                                                                \
    {                                                                                                                 \
        if (!(cond))                                                                                                  \
        {                                                                                                             \
            std::printf("FAIL %s:%d: %s  ", __FILE__, __LINE__, #cond);                                               \
            std::printf(__VA_ARGS__);                                                                                 \
            std::printf("\n");                                                                                        \
            g_fail++;                                                                                                 \
        }                                                                                                             \
    } while (0)

static const float kDt = 1.0f / 60.0f;

static Avbd3dVec3 V(float x, float y, float z) { return Avbd3dVec3{x, y, z}; }

static Avbd3dQuat quatAxisAngle(Avbd3dVec3 axis, float angle)
{
    const float l = std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
    const float s = std::sin(0.5f * angle) / l;
    return Avbd3dQuat{V(axis.x * s, axis.y * s, axis.z * s), std::cos(0.5f * angle)};
}

static Avbd3dWorld *makeWorld(Avbd3dVec3 gravity = V(0, -10.0f, 0))
{
    Avbd3dWorldDef wd = avbd3d_default_world_def();
    wd.gravity = gravity;
    wd.enableSleep = false;
    return t3d_create_world(&wd);
}

// A static box with its top face at y = 0, extents 40 x 1 x 40.
static Avbd3dBodyId makeGround(Avbd3dWorld *w, const Avbd3dSurfaceMaterial *mat = nullptr, Avbd3dQuat rot = {})
{
    Avbd3dBodyDef bd = avbd3d_default_body_def();
    bd.type = AVBD3D_STATIC_BODY;
    bd.position = V(0, -0.5f, 0);
    if (rot.s != 0.0f)
        bd.rotation = rot;
    Avbd3dBodyId g = t3d_create_body(w, &bd);
    Avbd3dShapeDef sd = avbd3d_default_shape_def();
    if (mat)
        sd.baseMaterial = *mat;
    Avbd3dBoxHull hull = avbd3d_make_box_hull(20.0f, 0.5f, 20.0f);
    t3d_create_hull_shape(g, &sd, avbd3d_box_hull_data(&hull));
    return g;
}

static Avbd3dBodyId makeBox(Avbd3dWorld *w, Avbd3dVec3 pos, float half, const Avbd3dSurfaceMaterial *mat,
                             Avbd3dBodyDef *outDef = nullptr)
{
    Avbd3dBodyDef bd = avbd3d_default_body_def();
    bd.type = AVBD3D_DYNAMIC_BODY;
    bd.position = pos;
    if (outDef)
        bd = *outDef, bd.position = pos;
    Avbd3dBodyId b = t3d_create_body(w, &bd);
    Avbd3dShapeDef sd = avbd3d_default_shape_def();
    if (mat)
        sd.baseMaterial = *mat;
    Avbd3dBoxHull hull = avbd3d_make_box_hull(half, half, half);
    t3d_create_hull_shape(b, &sd, avbd3d_box_hull_data(&hull));
    return b;
}

static Avbd3dBodyId makeSphere(Avbd3dWorld *w, Avbd3dVec3 pos, float r, const Avbd3dSurfaceMaterial *mat)
{
    Avbd3dBodyDef bd = avbd3d_default_body_def();
    bd.type = AVBD3D_DYNAMIC_BODY;
    bd.position = pos;
    Avbd3dBodyId b = t3d_create_body(w, &bd);
    Avbd3dShapeDef sd = avbd3d_default_shape_def();
    if (mat)
        sd.baseMaterial = *mat;
    Avbd3dSphere s{V(0, 0, 0), r};
    t3d_create_sphere_shape(b, &sd, &s);
    return b;
}

static Avbd3dSurfaceMaterial material(float friction, float restitution)
{
    Avbd3dSurfaceMaterial m = avbd3d_default_shape_def().baseMaterial;
    m.friction = friction;
    m.restitution = restitution;
    return m;
}

static void stepN(Avbd3dWorld *w, int n)
{
    for (int i = 0; i < n; i++)
        t3d_step(w, kDt);
}

static float len(Avbd3dVec3 v) { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }

// ---- a) motion locks -------------------------------------------------------------------------

static void testLocks()
{
    Avbd3dWorld *w = makeWorld();
    makeGround(w);
    Avbd3dBodyDef bd = avbd3d_default_body_def();
    bd.type = AVBD3D_DYNAMIC_BODY;
    bd.linearVelocity = V(2, 0, 2);
    bd.angularVelocity = V(3, 3, 3);
    Avbd3dBodyDef locked = bd;
    locked.motionLocks.linearZ = true;
    locked.motionLocks.angularX = true;
    locked.motionLocks.angularY = true;
    Avbd3dBodyId a = makeBox(w, V(-5, 2, 0), 0.5f, nullptr, &locked);
    Avbd3dBodyId c = makeBox(w, V(5, 2, 0), 0.5f, nullptr, &bd);
    float maxQxy = 0.0f, maxZ = 0.0f;
    float cQxy = 0.0f, cZ = 0.0f;
    for (int i = 0; i < 180; i++)
    {
        stepN(w, 1);
        const Avbd3dQuat q = t3d_body_get_rotation(a);
        maxQxy = std::fmax(maxQxy, std::fmax(std::fabs(q.v.x), std::fabs(q.v.y)));
        maxZ = std::fmax(maxZ, (float)std::fabs(t3d_body_get_position(a).z));
        const Avbd3dQuat qc = t3d_body_get_rotation(c);
        cQxy = std::fmax(cQxy, std::fmax(std::fabs(qc.v.x), std::fabs(qc.v.y)));
        cZ = std::fmax(cZ, (float)std::fabs(t3d_body_get_position(c).z));
    }
    const Avbd3dVec3 pa = t3d_body_get_position(a) , qa = t3d_body_get_angular_velocity(a);
    (void)pa;
    CHECK(maxZ < 1e-3f, "linear Z locked box moved %g in z", maxZ);
    CHECK(maxQxy < 1e-3f, "angular X/Y locked box rotated (quat xy %g)", maxQxy);
    CHECK(std::fabs(qa.x) < 1e-2f && std::fabs(qa.y) < 1e-2f, "locked angular velocity xy (%g, %g)", qa.x, qa.y);
    CHECK(cZ > 0.5f, "control box z only moved %g", cZ);
    CHECK(cQxy > 0.05f, "control box rotation about x/y only %g", cQxy);
    CHECK(t3d_body_get_position(a).x > -4.5f || t3d_body_get_position(a).x < -5.5f,
          "the free x axis of the locked box did not move");

    // Live: lock a free body's linear Y (gravity) and it hovers.
    Avbd3dBodyId f = makeBox(w, V(0, 8, 8), 0.5f, nullptr);
    stepN(w, 5);
    Avbd3dMotionLocks l{};
    l.linearY = true;
    t3d_body_set_motion_locks(f, l);
    const float y0 = (float)t3d_body_get_position(f).y;
    stepN(w, 30);
    const float y1 = (float)t3d_body_get_position(f).y;
    CHECK(std::fabs(y1 - y0) < 1e-3f, "Body_SetMotionLocks(linearY): box fell from %g to %g", y0, y1);
    t3d_body_set_motion_locks(f, Avbd3dMotionLocks{});
    stepN(w, 30);
    CHECK(t3d_body_get_position(f).y < y1 - 0.5f, "unlocking did not release the body");
    std::printf("locks: locked z %.5f quatXY %.5f | control z %.3f quatXY %.3f\n", maxZ, maxQxy, cZ, cQxy);
    t3d_destroy_world(w);
}

// ---- b) rolling resistance -------------------------------------------------------------------

static float rolledSpeed(float rr)
{
    Avbd3dWorld *w = makeWorld();
    makeGround(w);
    Avbd3dSurfaceMaterial m = material(0.6f, 0.0f);
    m.rollingResistance = rr;
    Avbd3dBodyId s = makeSphere(w, V(-8, 0.5f, 0), 0.5f, &m);
    stepN(w, 20); // settle on the ground first
    t3d_body_set_linear_velocity(s, V(4, 0, 0));
    t3d_body_set_angular_velocity(s, V(0, 0, -8.0f)); // rolling along +x without slip
    stepN(w, 30);
    const float v = t3d_body_get_linear_velocity(s).x;
    t3d_destroy_world(w);
    return v;
}

static void testRolling()
{
    // Same setup as the 2D gate: a ball rolling at speed on flat ground; without resistance it
    // keeps its speed, with resistance it stops. Two bugs used to bleed the rr=0 ball to 3.05 m/s:
    // a sticking contact kept the flat side's anchor while the ball rolled off it (narrowphase),
    // and the velocity pass lost (w dt)^2 / 8 of the spin each step (integrate.slang).
    const float v0 = rolledSpeed(0.0f), v1 = rolledSpeed(0.3f);
    std::printf("rolling resistance: vx after 0.5 s  rr=0 %.3f  rr=0.3 %.3f\n", v0, v1);
    CHECK(v0 > 3.9f, "sphere with rr=0 slowed from 4 to %g", v0);
    CHECK(v1 < 0.7f * v0, "rr=0.3 did not slow the sphere enough (%g vs %g)", v1, v0);

    // A sphere spinning in zero gravity keeps its spin (the velocity pass is the exact inverse of
    // the rotation update).
    Avbd3dWorld *w = makeWorld(V(0, 0, 0));
    Avbd3dBodyId s = makeSphere(w, V(0, 5, 0), 0.5f, nullptr);
    t3d_body_set_angular_velocity(s, V(0, 0, -8.0f));
    stepN(w, 60);
    const float wz = -t3d_body_get_angular_velocity(s).z;
    std::printf("free spin: w after 1 s %.4f (set 8)\n", wz);
    CHECK(std::fabs(wz - 8.0f) < 0.01f, "free spinning sphere went from 8 to %g rad/s", wz);
    t3d_destroy_world(w);
}

// ---- c) tangent velocity ---------------------------------------------------------------------

// A box on a belt box with shape-frame tangent velocity `t`, the belt rotated by `rot`; returns the
// box's displacement over 2 s.
static Avbd3dVec3 beltDisplacement(Avbd3dVec3 t, Avbd3dQuat rot, float *speedOut)
{
    Avbd3dWorld *w = makeWorld();
    Avbd3dSurfaceMaterial belt = material(0.6f, 0.0f);
    belt.tangentVelocity = t;
    makeGround(w, &belt, rot);
    Avbd3dSurfaceMaterial m = material(0.6f, 0.0f);
    Avbd3dBodyId b = makeBox(w, V(0, 0.5f, 0), 0.5f, &m);
    stepN(w, 30); // settle
    const Avbd3dVec3 p0 = t3d_body_get_position(b);
    stepN(w, 120);
    const Avbd3dVec3 p1 = t3d_body_get_position(b);
    const Avbd3dVec3 v = t3d_body_get_linear_velocity(b);
    if (speedOut)
        *speedOut = len(V(v.x, 0, v.z));
    t3d_destroy_world(w);
    return V((float)(p1.x - p0.x), (float)(p1.y - p0.y), (float)(p1.z - p0.z));
}

static void testTangent()
{
    float s;
    const Avbd3dQuat none{V(0, 0, 0), 0.0f};
    // The belt's own scalar speed 1 m/s along x: 2 s moves the box about 2 m.
    const Avbd3dVec3 dx = beltDisplacement(V(1, 0, 0), none, &s);
    std::printf("tangent (1,0,0): d = (%.3f, %.3f, %.3f) speed %.3f\n", dx.x, dx.y, dx.z, s);
    CHECK(std::fabs(std::fabs(dx.x) - 2.0f) < 0.3f && std::fabs(dx.z) < 0.1f, "x belt carried the box by (%g, %g)", dx.x, dx.z);
    CHECK(std::fabs(s - 1.0f) < 0.15f, "x belt speed %g", s);
    const float signX = dx.x > 0 ? 1.0f : -1.0f;

    const Avbd3dVec3 dz = beltDisplacement(V(0, 0, 1), none, &s);
    std::printf("tangent (0,0,1): d = (%.3f, %.3f, %.3f) speed %.3f\n", dz.x, dz.y, dz.z, s);
    CHECK(std::fabs(std::fabs(dz.z) - 2.0f) < 0.3f && std::fabs(dz.x) < 0.1f, "z belt carried the box by (%g, %g)", dz.x, dz.z);
    CHECK(dz.z * signX > 0.0f, "x and z belts disagree in sign convention (%g vs %g)", dx.x, dz.z);

    // The surface normal component of the shape-frame vector is projected away.
    const Avbd3dVec3 dn = beltDisplacement(V(0, 1, 0), none, &s);
    std::printf("tangent (0,1,0) on a flat belt: d = (%.3f, %.3f, %.3f) speed %.3f\n", dn.x, dn.y, dn.z, s);
    CHECK(std::fabs(dn.x) < 0.1f && std::fabs(dn.z) < 0.1f, "normal component moved the box by (%g, %g)", dn.x, dn.z);

    // The vector is in the shape frame: a belt rotated 90 degrees about y turns (1,0,0) into +-z, and
    // one rotated 90 degrees about x turns (0,1,0) into the other horizontal axis (+-z) while
    // (0,0,1) becomes the vertical normal.
    const Avbd3dQuat qy = quatAxisAngle(V(0, 1, 0), 1.5707963f);
    const Avbd3dVec3 dr = beltDisplacement(V(1, 0, 0), qy, &s);
    std::printf("tangent (1,0,0), belt turned about y: d = (%.3f, %.3f, %.3f) speed %.3f\n", dr.x, dr.y, dr.z, s);
    CHECK(std::fabs(std::fabs(dr.z) - 2.0f) < 0.3f && std::fabs(dr.x) < 0.1f, "rotated belt carried the box by (%g, %g)", dr.x, dr.z);
    const Avbd3dVec3 dq = beltDisplacement(V(0, 0, 1), qy, &s);
    std::printf("tangent (0,0,1), belt turned about y: d = (%.3f, %.3f, %.3f) speed %.3f\n", dq.x, dq.y, dq.z, s);
    CHECK(std::fabs(std::fabs(dq.x) - 2.0f) < 0.3f && std::fabs(dq.z) < 0.1f, "belt (0,0,1) turned about y carried the box by (%g, %g)", dq.x, dq.z);
    CHECK(dq.x * dr.z < 0.0f, "rotating the belt about y by +90 degrees: x->-z and z->+x expected");
}

// ---- d) friction mixing ----------------------------------------------------------------------

static float slidSpeed(Avbd3dMixingRule rule, bool set)
{
    Avbd3dWorld *w = makeWorld();
    if (set)
        t3d_world_set_friction_mixing(w, rule);
    Avbd3dSurfaceMaterial gm = material(0.8f, 0.0f);
    makeGround(w, &gm);
    Avbd3dSurfaceMaterial bm = material(0.2f, 0.0f);
    Avbd3dBodyId b = makeBox(w, V(-10, 0.5f, 0), 0.5f, &bm);
    stepN(w, 10);
    t3d_body_set_linear_velocity(b, V(6, 0, 0));
    stepN(w, 30);
    const float v = t3d_body_get_linear_velocity(b).x;
    t3d_destroy_world(w);
    return v;
}

static void testFrictionMixing()
{
    // After 0.5 s of sliding, v = 6 - 5 mu. Coefficients 0.8 (ground) and 0.2 (box).
    struct Case
    {
        const char *name;
        Avbd3dMixingRule rule;
        float mu;
    } cases[] = {{"geometric", AVBD3D_MIX_GEOMETRIC, 0.4f}, {"average", AVBD3D_MIX_AVERAGE, 0.5f},
                 {"min", AVBD3D_MIX_MIN, 0.2f},             {"max", AVBD3D_MIX_MAX, 0.8f},
                 {"multiply", AVBD3D_MIX_MULTIPLY, 0.16f}};
    const float vDefault = slidSpeed(AVBD3D_MIX_GEOMETRIC, false);
    float v[5];
    for (int i = 0; i < 5; i++)
    {
        v[i] = slidSpeed(cases[i].rule, true);
        const float want = 6.0f - 5.0f * cases[i].mu;
        std::printf("friction mixing %-9s mu %.2f: v %.3f (ideal %.3f)\n", cases[i].name, cases[i].mu, v[i], want);
        CHECK(std::fabs(v[i] - want) < 0.35f, "%s: v %g, want about %g", cases[i].name, v[i], want);
    }
    CHECK(std::fabs(vDefault - v[0]) < 1e-5f, "default friction mixing differs from geometric: %g vs %g", vDefault, v[0]);
    // Orderings: more friction, slower.
    CHECK(v[3] < v[1] && v[1] < v[0] && v[0] < v[2] && v[2] < v[4] + 0.01f, "friction mixing order max<avg<geo<min<mul broken");
    Avbd3dWorld *w = makeWorld();
    t3d_world_set_friction_mixing(w, AVBD3D_MIX_MIN);
    t3d_destroy_world(w);
}

// ---- e) restitution mixing -------------------------------------------------------------------

static float bounceRatioMixed(Avbd3dMixingRule rule, bool set)
{
    Avbd3dWorld *w = makeWorld();
    if (set)
        t3d_world_set_restitution_mixing(w, rule);
    Avbd3dSurfaceMaterial gm = material(0.6f, 0.2f);
    makeGround(w, &gm);
    Avbd3dSurfaceMaterial bm = material(0.6f, 1.0f);
    Avbd3dBodyId b = makeSphere(w, V(0, 5.5f, 0), 0.5f, &bm);
    bool falling = true, bounced = false;
    float apex = 0.0f;
    for (int i = 0; i < 600; i++)
    {
        stepN(w, 1);
        const float y = (float)t3d_body_get_position(b).y;
        const float vy = t3d_body_get_linear_velocity(b).y;
        if (falling)
        {
            if (vy > 0.05f)
            {
                falling = false;
                bounced = true;
                apex = y;
            }
            continue;
        }
        apex = std::fmax(apex, y);
        if (vy < 0.0f)
            break;
    }
    t3d_destroy_world(w);
    return bounced ? (apex - 0.5f) / 5.0f : 0.0f;
}

static void testRestitutionMixing()
{
    // Ball e = 1, ground e = 0.2; the bounce height goes as the mixed e squared.
    struct Case
    {
        const char *name;
        Avbd3dMixingRule rule;
        float e;
    } cases[] = {{"max", AVBD3D_MIX_MAX, 1.0f},         {"average", AVBD3D_MIX_AVERAGE, 0.6f},
                 {"geometric", AVBD3D_MIX_GEOMETRIC, 0.4472f}, {"min", AVBD3D_MIX_MIN, 0.2f},
                 {"multiply", AVBD3D_MIX_MULTIPLY, 0.2f}};
    const float rDefault = bounceRatioMixed(AVBD3D_MIX_MAX, false);
    for (const Case &c : cases)
    {
        const float r = bounceRatioMixed(c.rule, true);
        std::printf("restitution mixing %-9s e %.3f: bounce/drop %.3f (ideal %.3f)\n", c.name, c.e, r, c.e * c.e);
        CHECK(std::fabs(r - c.e * c.e) < 0.09f, "%s: bounce/drop %g, want about %g", c.name, r, c.e * c.e);
        if (c.rule == AVBD3D_MIX_MAX)
            CHECK(std::fabs(r - rDefault) < 1e-5f, "default restitution mixing differs from max: %g vs %g", rDefault, r);
    }
}

// ---- f) world-frame inertia ------------------------------------------------------------------

struct Mat3
{
    float m[3][3];
};

static Mat3 matFromQuat(Avbd3dQuat q)
{
    const float x = q.v.x, y = q.v.y, z = q.v.z, s = q.s;
    Mat3 r;
    r.m[0][0] = 1 - 2 * (y * y + z * z); r.m[0][1] = 2 * (x * y - z * s);     r.m[0][2] = 2 * (x * z + y * s);
    r.m[1][0] = 2 * (x * y + z * s);     r.m[1][1] = 1 - 2 * (x * x + z * z); r.m[1][2] = 2 * (y * z - x * s);
    r.m[2][0] = 2 * (x * z - y * s);     r.m[2][1] = 2 * (y * z + x * s);     r.m[2][2] = 1 - 2 * (x * x + y * y);
    return r;
}

// L = R diag(I) R^T w
static Avbd3dVec3 angularMomentum(Avbd3dQuat q, Avbd3dVec3 w, Avbd3dVec3 I)
{
    const Mat3 R = matFromQuat(q);
    const float wl[3] = {R.m[0][0] * w.x + R.m[1][0] * w.y + R.m[2][0] * w.z,
                         R.m[0][1] * w.x + R.m[1][1] * w.y + R.m[2][1] * w.z,
                         R.m[0][2] * w.x + R.m[1][2] * w.y + R.m[2][2] * w.z};
    const float l[3] = {I.x * wl[0], I.y * wl[1], I.z * wl[2]};
    return V(R.m[0][0] * l[0] + R.m[0][1] * l[1] + R.m[0][2] * l[2], R.m[1][0] * l[0] + R.m[1][1] * l[1] + R.m[1][2] * l[2],
             R.m[2][0] * l[0] + R.m[2][1] * l[1] + R.m[2][2] * l[2]);
}

// Worst relative change of L over `steps` steps of a free thin box. Principal inertia of a box with
// half extents (a, b, c) per unit mass is ((b^2+c^2), (a^2+c^2), (a^2+b^2)) / 3.
static float angularMomentumDrift(bool worldFrame, int steps, float *energyDrift)
{
    Avbd3dWorld *w = makeWorld(V(0, 0, 0));
    Avbd3dSolverDef sd = avbd3d_default_solver_def();
    sd.worldFrameInertia = worldFrame;
    t3d_world_set_solver_def(w, &sd);
    const float a = 2.0f, b = 0.1f, c = 0.1f;
    Avbd3dBodyDef bd = avbd3d_default_body_def();
    bd.type = AVBD3D_DYNAMIC_BODY;
    bd.position = V(0, 0, 0);
    bd.rotation = quatAxisAngle(V(0.3f, 1.0f, 0.5f), 0.6f);
    bd.angularVelocity = V(1.5f, 1.0f, 0.5f);
    Avbd3dBodyId body = t3d_create_body(w, &bd);
    Avbd3dShapeDef shd = avbd3d_default_shape_def();
    Avbd3dBoxHull hull = avbd3d_make_box_hull(a, b, c);
    t3d_create_hull_shape(body, &shd, avbd3d_box_hull_data(&hull));
    const Avbd3dVec3 I = V((b * b + c * c) / 3, (a * a + c * c) / 3, (a * a + b * b) / 3);
    const Avbd3dVec3 L0 = angularMomentum(bd.rotation, bd.angularVelocity, I);
    // Kinetic energy 0.5 w . L
    auto energy = [&](Avbd3dQuat q, Avbd3dVec3 wv) {
        const Avbd3dVec3 L = angularMomentum(q, wv, I);
        return 0.5f * (wv.x * L.x + wv.y * L.y + wv.z * L.z);
    };
    const float e0 = energy(bd.rotation, bd.angularVelocity);
    float worst = 0.0f, worstE = 0.0f;
    for (int i = 0; i < steps; i++)
    {
        stepN(w, 1);
        const Avbd3dQuat q = t3d_body_get_rotation(body);
        const Avbd3dVec3 av = t3d_body_get_angular_velocity(body);
        const Avbd3dVec3 L = angularMomentum(q, av, I);
        worst = std::fmax(worst, len(V(L.x - L0.x, L.y - L0.y, L.z - L0.z)) / len(L0));
        worstE = std::fmax(worstE, std::fabs(energy(q, av) - e0) / e0);
    }
    t3d_destroy_world(w);
    if (energyDrift)
        *energyDrift = worstE;
    return worst;
}

static void testWorldFrameInertia()
{
    float eOff, eOn;
    const float off = angularMomentumDrift(false, 120, &eOff);
    const float on = angularMomentumDrift(true, 120, &eOn);
    std::printf("world-frame inertia, 120 steps: |dL|/|L| max  off %.4f  on %.4f   energy drift  off %.4f  on %.4f\n", off, on,
                eOff, eOn);
    CHECK(on < 0.05f, "with worldFrameInertia on, L drifted %g", on);
    CHECK(on < off, "worldFrameInertia did not reduce the drift (off %g, on %g)", off, on);
    // The default is on (since 2026-10-14).
    Avbd3dWorld *w = makeWorld();
    CHECK(avbd3d_default_solver_def().worldFrameInertia, "worldFrameInertia is not on by default");
    t3d_destroy_world(w);
}

int run()
{
    testLocks();
    testRolling();
    testTangent();
    testFrictionMixing();
    testRestitutionMixing();
    testWorldFrameInertia();
    std::printf("avbd_test_vk_api3d_materials: %s\n", g_fail ? "FAILED" : "ok");
    return g_fail ? 1 : 0;
}
#undef CHECK
} // namespace materials

namespace restitution
{
static int g_fail = 0;
#define CHECK(cond, ...)                                                                                              \
    do                                                                                                                \
    {                                                                                                                 \
        if (!(cond))                                                                                                  \
        {                                                                                                             \
            std::printf("FAIL %s:%d: %s  ", __FILE__, __LINE__, #cond);                                               \
            std::printf(__VA_ARGS__);                                                                                 \
            std::printf("\n");                                                                                        \
            g_fail++;                                                                                                 \
        }                                                                                                             \
    } while (0)

static Avbd3dWorld *makeWorld(float threshold)
{
    Avbd3dWorldDef wd = avbd3d_default_world_def();
    wd.restitutionThreshold = threshold;
    Avbd3dWorld *w = t3d_create_world(&wd);
    Avbd3dBodyDef bd = avbd3d_default_body_def();
    bd.type = AVBD3D_STATIC_BODY;
    bd.position = Avbd3dVec3{0, -0.5f, 0};
    Avbd3dBodyId g = t3d_create_body(w, &bd);
    Avbd3dShapeDef sd = avbd3d_default_shape_def();
    Avbd3dBoxHull hull = avbd3d_make_box_hull(20.0f, 0.5f, 20.0f);
    t3d_create_hull_shape(g, &sd, avbd3d_box_hull_data(&hull));
    return w;
}

static Avbd3dBodyId dropBody(Avbd3dWorld *w, float y, float e, bool box)
{
    Avbd3dBodyDef bd = avbd3d_default_body_def();
    bd.type = AVBD3D_DYNAMIC_BODY;
    bd.position = Avbd3dVec3{0, y, 0};
    Avbd3dBodyId b = t3d_create_body(w, &bd);
    Avbd3dShapeDef sd = avbd3d_default_shape_def();
    sd.baseMaterial.restitution = e;
    if (box)
    {
        Avbd3dBoxHull hull = avbd3d_make_box_hull(0.5f, 0.5f, 0.5f);
        t3d_create_hull_shape(b, &sd, avbd3d_box_hull_data(&hull));
    }
    else
    {
        Avbd3dSphere s{Avbd3dVec3{0, 0, 0}, 0.5f};
        t3d_create_sphere_shape(b, &sd, &s);
    }
    return b;
}

struct Bounce
{
    float apex = 0;     // highest centre height after the first bounce
    float maxSpin = 0;  // largest angular speed seen
    bool bounced = false;
};

// Steps until the body has gone up after first touching and come back down (or `steps` run out).
static Bounce firstBounce(Avbd3dWorld *w, Avbd3dBodyId b, int steps)
{
    Bounce r;
    bool falling = true;
    for (int i = 0; i < steps; i++)
    {
        t3d_step(w, 1.0f / 60.0f);
        const Avbd3dVec3 v = t3d_body_get_linear_velocity(b);
        const Avbd3dVec3 av = t3d_body_get_angular_velocity(b);
        const float y = (float)t3d_body_get_position(b).y;
        r.maxSpin = std::fmax(r.maxSpin, std::sqrt(av.x * av.x + av.y * av.y + av.z * av.z));
        if (falling)
        {
            if (v.y > 0.05f)
            {
                falling = false;
                r.bounced = true;
                r.apex = y;
            }
            continue;
        }
        r.apex = std::fmax(r.apex, y);
        if (v.y < 0.0f)
            break;
    }
    return r;
}

// Drop height of the centre above its resting height (0.5): 5 m.
static const float kDropY = 5.5f, kRestY = 0.5f, kH0 = kDropY - kRestY;

static float bounceRatio(float e, float threshold, bool box, Bounce *out = nullptr)
{
    Avbd3dWorld *w = makeWorld(threshold);
    Avbd3dBodyId b = dropBody(w, kDropY, e, box);
    const Bounce r = firstBounce(w, b, 600);
    t3d_destroy_world(w);
    if (out)
        *out = r;
    return r.bounced ? (r.apex - kRestY) / kH0 : 0.0f;
}

// g) Two spheres on one body, top created first (the primary shape) and bottom second. Only the
// bottom sphere touches the ground, so only its restitution counts.
static float twoShapeBounce(float eTop, float eBottom)
{
    Avbd3dWorld *w = makeWorld(1.0f);
    Avbd3dBodyDef bd = avbd3d_default_body_def();
    bd.type = AVBD3D_DYNAMIC_BODY;
    bd.position = Avbd3dVec3{0, 6.0f, 0};
    Avbd3dBodyId b = t3d_create_body(w, &bd);
    const float ys[2] = {0.5f, -0.5f};
    const float es[2] = {eTop, eBottom};
    for (int i = 0; i < 2; i++)
    {
        Avbd3dShapeDef sd = avbd3d_default_shape_def();
        sd.baseMaterial.restitution = es[i];
        Avbd3dSphere sp{Avbd3dVec3{0, ys[i], 0}, 0.5f};
        t3d_create_sphere_shape(b, &sd, &sp);
    }
    const Bounce r = firstBounce(w, b, 600);
    t3d_destroy_world(w);
    return r.bounced ? (r.apex - 1.0f) / 5.0f : 0.0f; // rest height of the body origin is 1.0
}

int run()
{
    const float twoUp = twoShapeBounce(0.0f, 1.0f), twoDown = twoShapeBounce(1.0f, 0.0f);
    CHECK(twoUp > 0.7f && twoUp < 1.1f, "bottom sphere e=1, top e=0: bounce/drop %g (want ~1)", twoUp);
    CHECK(twoDown < 0.05f, "bottom sphere e=0, top e=1: bounce/drop %g (want 0)", twoDown);

    const float r1 = bounceRatio(1.0f, 1.0f, false);
    CHECK(r1 > 0.8f && r1 < 1.1f, "e=1 sphere: bounce/drop %g (want ~1)", r1);

    const float rHalf = bounceRatio(0.5f, 1.0f, false);
    CHECK(rHalf > 0.17f && rHalf < 0.33f, "e=0.5 sphere: bounce/drop %g (want ~0.25)", rHalf);

    const float r0 = bounceRatio(0.0f, 1.0f, false);
    CHECK(r0 < 0.05f, "e=0 sphere: bounce/drop %g (want 0)", r0);

    // Impact speed from 5 m is ~10 m/s; a 20 m/s threshold suppresses the bounce.
    const float rThresh = bounceRatio(1.0f, 20.0f, false);
    CHECK(rThresh < 0.05f, "e=1 under the threshold: bounce/drop %g (want 0)", rThresh);

    Bounce boxB;
    const float rBox = bounceRatio(1.0f, 1.0f, true, &boxB);
    CHECK(rBox > 0.75f && rBox < 1.1f, "e=1 box: bounce/drop %g (want ~1)", rBox);
    CHECK(boxB.maxSpin < 0.5f, "e=1 flat box spun up to %g rad/s", boxB.maxSpin);

    {
        Avbd3dWorld *w = makeWorld(1.0f);
        Avbd3dBodyId b = dropBody(w, kDropY, 0.5f, false);
        for (int i = 0; i < 600; i++)
            t3d_step(w, 1.0f / 60.0f);
        const float y = (float)t3d_body_get_position(b).y;
        const Avbd3dVec3 v = t3d_body_get_linear_velocity(b);
        CHECK(std::fabs(y - kRestY) < 0.05f && std::fabs(v.y) < 0.1f, "e=0.5 sphere after 10 s: y %g vy %g", y, v.y);
        t3d_destroy_world(w);
    }

    std::printf("bounce/drop: e=1 %.3f  e=0.5 %.3f  e=0 %.3f  threshold %.3f  box %.3f\n", r1, rHalf, r0, rThresh,
                rBox);
    std::printf("avbd_test_vk_api3d_restitution: %s\n", g_fail ? "FAILED" : "ok");
    return g_fail ? 1 : 0;
}
#undef CHECK
} // namespace restitution

int main()
{
    int failed = 0;
    failed |= materials::run() != 0;
    failed |= restitution::run() != 0;
    return failed ? 1 : 0;
}
