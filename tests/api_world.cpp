// Gate for the avbd3d world layer (src/api): generational ids, read-your-writes, a box
// dropped onto a static ground under y-up gravity, StepBegin/StepEnd overlap, and one queue
// submission per step across live create/destroy.

#include "test_util.h"

#include <chrono>
#include <cmath>
#include <cstdio>

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

static Avbd3dBodyId makeBox(Avbd3dWorld *w, Avbd3dBodyType type, Avbd3dVec3 p, float hx, float hy, float hz)
{
    Avbd3dBodyDef bd = avbd3d_default_body_def();
    bd.type = type;
    bd.position = p;
    Avbd3dBodyId b = t3d_create_body(w, &bd);
    Avbd3dShapeDef sd = avbd3d_default_shape_def();
    Avbd3dBoxHull hull = avbd3d_make_box_hull(hx, hy, hz);
    t3d_create_hull_shape(b, &sd, avbd3d_box_hull_data(&hull));
    return b;
}

static Avbd3dBodyDef bodyDef(Avbd3dBodyType type, Avbd3dVec3 p)
{
    Avbd3dBodyDef bd = avbd3d_default_body_def();
    bd.type = type;
    bd.position = p;
    return bd;
}

static Avbd3dBodyId makeBodyBox(Avbd3dWorld *w, const Avbd3dBodyDef &bd, float hx, float hy, float hz)
{
    Avbd3dBodyId b = t3d_create_body(w, &bd);
    Avbd3dShapeDef sd = avbd3d_default_shape_def();
    Avbd3dBoxHull hull = avbd3d_make_box_hull(hx, hy, hz);
    t3d_create_hull_shape(b, &sd, avbd3d_box_hull_data(&hull));
    return b;
}

static void step(Avbd3dWorld *w, int n = 1)
{
    for (int i = 0; i < n; i++)
        t3d_step(w, 1.0f / 60.0f);
}

int main()
{
    const auto t0 = std::chrono::steady_clock::now();
    Avbd3dWorldDef wd = avbd3d_default_world_def();
    Avbd3dWorld *w = t3d_create_world(&wd);
    CHECK(w != NULL, "world");
    if (!w)
        return 1;
    Avbd3dCapabilities caps = t3d_get_capabilities(w);
    CHECK(caps.device_name[0] != 0, "capabilities: device name is empty");
    CHECK(caps.subgroup_size == 32, "capabilities: subgroup size %u (want 32)", caps.subgroup_size);
    Avbd3dVec3 g = t3d_world_get_gravity(w);
    CHECK(g.x == 0 && g.y == -10.0f && g.z == 0, "default gravity (%g %g %g)", g.x, g.y, g.z);

    // ---- a box dropped on a static ground, y-up -----------------------------------------
    Avbd3dBodyId ground = makeBox(w, AVBD3D_STATIC_BODY, {0, -0.5f, 0}, 10, 0.5f, 10);
    Avbd3dBodyId box = makeBox(w, AVBD3D_DYNAMIC_BODY, {0, 3.0f, 0}, 0.5f, 0.5f, 0.5f);
    CHECK(avbd3d_body_is_valid(ground) && avbd3d_body_is_valid(box), "ids valid");
    CHECK(t3d_body_get_mass(box) > 0.9f && t3d_body_get_mass(box) < 1.1f, "mass %g", t3d_body_get_mass(box));
    CHECK(t3d_body_get_mass(ground) == 0.0f, "static mass");

    // Deferred visibility: right after create the body reads back its authored pose (before any
    // step has run); only a step moves it.
    Avbd3dVec3 p0 = t3d_body_get_position(box);
    CHECK(p0.y == 3.0f, "authored y %g", p0.y);
    step(w);
    Avbd3dVec3 p1 = t3d_body_get_position(box);
    CHECK(p1.y < 3.0f && p1.y > 2.9f, "after one step y %g", p1.y);
    Avbd3dBodyEvents ev = t3d_world_get_body_events(w);
    bool found = false;
    for (int i = 0; i < ev.moveCount; i++)
        found |= ev.moveEvents[i].bodyId.index1 == box.index1;
    CHECK(found, "move event for the falling box (%d events)", ev.moveCount);
    step(w, 239);
    Avbd3dVec3 p = t3d_body_get_position(box);
    Avbd3dVec3 v = t3d_body_get_linear_velocity(box);
    CHECK(std::fabs(p.y - 0.5f) < 0.05f, "rest height y=%g (want 0.5)", p.y);
    CHECK(std::fabs(p.x) < 0.05f && std::fabs(p.z) < 0.05f, "drift (%g %g)", p.x, p.z);
    CHECK(std::fabs(v.y) < 0.2f, "rest speed %g", v.y);

    // ---- read-your-writes ----------------------------------------------------------------
    t3d_world_set_gravity(w, {0, 0, 0});
    t3d_body_set_transform(box, {2, 5, 1}, {{0, 0, 0}, 1.0f});
    Avbd3dVec3 q = t3d_body_get_position(box);
    CHECK(q.x == 2 && q.y == 5 && q.z == 1, "SetTransform read back (%g %g %g)", q.x, q.y, q.z);
    t3d_body_set_linear_velocity(box, {1, 2, 3});
    Avbd3dVec3 lv = t3d_body_get_linear_velocity(box);
    CHECK(lv.x == 1 && lv.y == 2 && lv.z == 3, "SetLinearVelocity read back");
    t3d_body_set_angular_velocity(box, {0, 0.5f, 0});
    Avbd3dVec3 av = t3d_body_get_angular_velocity(box);
    CHECK(av.y == 0.5f, "SetAngularVelocity read back");
    Avbd3dBodyId ids[1] = {box};
    Avbd3dVec3 batch[1];
    CHECK(t3d_world_get_body_data(w, ids, 1, AVBD3D_BODY_DATA_POSITION, batch) == 1, "batched read");
    CHECK(batch[0].x == 2 && batch[0].y == 5, "batched position");
    step(w, 6);
    q = t3d_body_get_position(box);
    CHECK(std::fabs(q.x - (2 + 0.1f)) < 0.02f && std::fabs(q.y - (5 + 0.2f)) < 0.02f && std::fabs(q.z - (1 + 0.3f)) < 0.02f,
          "moved by velocity: (%g %g %g)", q.x, q.y, q.z);
    t3d_body_set_awake(box, true);
    CHECK(t3d_body_is_awake(box), "awake after SetAwake");
    t3d_body_disable(box);
    CHECK(!t3d_body_is_enabled(box), "disabled");
    t3d_body_enable(box);
    CHECK(t3d_body_is_enabled(box), "enabled");
    t3d_world_set_gravity(w, {0, -10.0f, 0});

    // ---- id reuse ------------------------------------------------------------------------
    Avbd3dBodyId b1 = makeBox(w, AVBD3D_DYNAMIC_BODY, {20, 1, 0}, 0.5f, 0.5f, 0.5f);
    Avbd3dShapeId sh[4];
    CHECK(t3d_body_get_shapes(b1, sh, 4) == 1, "one shape");
    Avbd3dShapeId s1 = sh[0];
    CHECK(avbd3d_shape_is_valid(s1), "shape valid");
    step(w);
    t3d_destroy_body(b1);
    CHECK(!avbd3d_body_is_valid(b1), "destroyed body is invalid");
    CHECK(!avbd3d_shape_is_valid(s1), "destroyed body's shape is invalid");
    step(w, 2);
    Avbd3dBodyId b2 = makeBox(w, AVBD3D_DYNAMIC_BODY, {20, 1, 0}, 0.5f, 0.5f, 0.5f);
    CHECK(b2.index1 == b1.index1, "slot reused: %d vs %d", b2.index1, b1.index1);
    CHECK(b2.generation != b1.generation, "generation bumped: %d vs %d", b2.generation, b1.generation);
    CHECK(!avbd3d_body_is_valid(b1) && avbd3d_body_is_valid(b2), "old id stays invalid, new id valid");
    Avbd3dVec3 stale = {0, 0, 0};
    CHECK(avbd3d_body_get_position(b1, &stale) == AVBD3D_ERR_INVALID_ARG, "stale id returns AVBD3D_ERR_INVALID_ARG");
    CHECK(stale.x == 0 && stale.y == 0, "stale id leaves the out value untouched");
    t3d_body_get_shapes(b2, sh, 4);
    CHECK(sh[0].index1 == s1.index1 && sh[0].generation != s1.generation, "shape slot reused with a new generation");

    // ---- joints: create, step, destroy ---------------------------------------------------
    Avbd3dBodyId ja = makeBox(w, AVBD3D_DYNAMIC_BODY, {-20, 5, 0}, 0.5f, 0.5f, 0.5f);
    Avbd3dBodyId jb = makeBox(w, AVBD3D_DYNAMIC_BODY, {-20, 4, 0}, 0.5f, 0.5f, 0.5f);
    Avbd3dRevoluteJointDef rd = avbd3d_default_revolute_joint_def();
    rd.base.bodyIdA = ja;
    rd.base.bodyIdB = jb;
    rd.base.localFrameA.p = {0, -0.5f, 0};
    rd.base.localFrameB.p = {0, 0.5f, 0};
    Avbd3dJointId j = t3d_create_revolute_joint(w, &rd);
    CHECK(avbd3d_joint_is_valid(j), "joint valid");
    step(w, 10);
    t3d_destroy_joint(j, true);
    CHECK(!avbd3d_joint_is_valid(j), "joint destroyed");
    step(w, 2);

    // ---- one submit per step across create/destroy --------------------------------------
    int bad = 0;
    Avbd3dBodyId last = t3d_null_body;
    for (int i = 0; i < 30; i++)
    {
        Avbd3dBodyId nb = makeBox(w, AVBD3D_DYNAMIC_BODY, {30.0f + i, 1, 0}, 0.5f, 0.5f, 0.5f);
        if (i & 1 && avbd3d_body_is_valid(last))
            t3d_destroy_body(last);
        last = nb;
        t3d_body_set_linear_velocity(nb, {0, 1, 0});
        step(w);
        const Avbd3dStats r = t3d_get_stats(w);
        if (r.submits != 1 || !r.ok)
            bad++;
    }
    CHECK(bad == 0, "%d steps took more than one submit", bad);

    // ---- async overlap -------------------------------------------------------------------
    t3d_step_begin(w, 1.0f / 60.0f);
    CHECK(t3d_is_step_in_flight(w), "in flight after StepBegin");
    volatile double acc = 0; // host work while the GPU runs
    for (int i = 0; i < 1000000; i++)
        acc += std::sqrt((double)i);
    t3d_step_end(w);
    CHECK(!t3d_is_step_in_flight(w), "idle after StepEnd");
    // The read model: calls during a step neither wait for it nor see it. Getters, queries and
    // events read the last finished step; a setter is queued for the next one.
    Avbd3dBodyId fall = makeBox(w, AVBD3D_DYNAMIC_BODY, {-200, 50, 0}, 0.5f, 0.5f, 0.5f);
    step(w, 3);
    const Avbd3dVec3 before = t3d_body_get_position(fall);
    const Avbd3dVec3 velBefore = t3d_body_get_linear_velocity(fall);
    t3d_step_begin(w, 1.0f / 60.0f);
    const Avbd3dVec3 mid = t3d_body_get_position(fall);
    const Avbd3dVec3 midVel = t3d_body_get_linear_velocity(fall);
    t3d_world_cast_ray_closest(w, {-200, 100, 0}, {0, -100, 0}, avbd3d_default_query_filter());
    (void)t3d_world_get_body_events(w);
    CHECK(t3d_is_step_in_flight(w), "getters, queries and events do not settle the step");
    CHECK(mid.y == before.y && midVel.y == velBefore.y, "a getter mid-step reads the last finished step (%g vs %g)",
          mid.y, before.y);
    t3d_body_set_linear_velocity(fall, {0, 5, 0}); // queued for the next step
    CHECK(t3d_body_get_linear_velocity(fall).y == 5.0f, "a write reads back at once");
    t3d_step_end(w);
    const Avbd3dVec3 after = t3d_body_get_position(fall);
    CHECK(t3d_body_get_linear_velocity(fall).y == 5.0f, "the written velocity holds until its step ran");
    CHECK(after.y == before.y, "the write holds the pose too until its step ran (%g vs %g)", after.y, before.y);
    step(w);
    // Applied by the device: gravity took its step's share off the written 5 m/s, and the body
    // rose above where it stood when the write was made.
    const float vyApplied = t3d_body_get_linear_velocity(fall).y;
    const float yApplied = t3d_body_get_position(fall).y;
    CHECK(vyApplied > 4.0f && vyApplied < 5.0f && yApplied > before.y,
          "the step after the write applies it (vy %g, y %g)", vyApplied, yApplied);
    step(w);
    CHECK(t3d_body_get_position(fall).y > yApplied && t3d_body_get_linear_velocity(fall).y < vyApplied,
          "the mirror follows the device again (vy %g)", t3d_body_get_linear_velocity(fall).y);

    // ---- Phase 2 body properties through the C API ---------------------------------------
    // Every body sits in its own column (x far apart) so nothing interacts except the push
    // pair, which is set up to meet only after the checks that need it apart.
    Avbd3dWorldDef pwd = avbd3d_default_world_def();
    Avbd3dWorld *pw = t3d_create_world(&pwd);
    CHECK(pw != NULL, "props world");

    // Gravity scale: zero at creation and zero set at runtime both float; a control falls.
    Avbd3dBodyDef gdef = bodyDef(AVBD3D_DYNAMIC_BODY, {-50, 5, 0});
    gdef.gravityScale = 0.0f;
    Avbd3dBodyId gFloatDef = makeBodyBox(pw, gdef, 0.5f, 0.5f, 0.5f);
    Avbd3dBodyId gFloatRt = makeBodyBox(pw, bodyDef(AVBD3D_DYNAMIC_BODY, {-40, 5, 0}), 0.5f, 0.5f, 0.5f);
    t3d_body_set_gravity_scale(gFloatRt, 0.0f);
    Avbd3dBodyId gCtrl = makeBodyBox(pw, bodyDef(AVBD3D_DYNAMIC_BODY, {-60, 5, 0}), 0.5f, 0.5f, 0.5f);
    step(pw, 30);
    CHECK(t3d_body_get_position(gCtrl).y < 4.5f, "control falls (y %g)", t3d_body_get_position(gCtrl).y);
    CHECK(std::fabs(t3d_body_get_position(gFloatDef).y - 5.0f) < 1e-3f,
          "gravityScale 0 at creation floats (y %g)", t3d_body_get_position(gFloatDef).y);
    CHECK(std::fabs(t3d_body_get_position(gFloatRt).y - 5.0f) < 1e-3f,
          "gravityScale 0 at runtime floats (y %g)", t3d_body_get_position(gFloatRt).y);

    // Damping: a flying body with linear and angular damping slows; the control does not.
    Avbd3dBodyDef ddef = bodyDef(AVBD3D_DYNAMIC_BODY, {-20, 5, 0});
    ddef.gravityScale = 0.0f;
    ddef.linearDamping = 2.0f;
    ddef.angularDamping = 2.0f;
    Avbd3dBodyId dDamped = makeBodyBox(pw, ddef, 0.5f, 0.5f, 0.5f);
    Avbd3dBodyDef cdef = bodyDef(AVBD3D_DYNAMIC_BODY, {-30, 5, 0});
    cdef.gravityScale = 0.0f;
    Avbd3dBodyId dCtrl = makeBodyBox(pw, cdef, 0.5f, 0.5f, 0.5f);
    Avbd3dBodyDef rdef = cdef;
    rdef.position = {-35, 5, 0};
    Avbd3dBodyId dRt = makeBodyBox(pw, rdef, 0.5f, 0.5f, 0.5f);
    t3d_body_set_linear_damping(dRt, 2.0f);
    t3d_body_set_angular_damping(dRt, 2.0f);
    const Avbd3dBodyId flyers[3] = {dDamped, dCtrl, dRt};
    for (int i = 0; i < 3; i++)
    {
        t3d_body_set_linear_velocity(flyers[i], {5, 0, 0});
        t3d_body_set_angular_velocity(flyers[i], {0, 5, 0});
    }
    step(pw, 60);
    const Avbd3dVec3 vc = t3d_body_get_linear_velocity(dCtrl);
    const Avbd3dVec3 vd = t3d_body_get_linear_velocity(dDamped);
    const Avbd3dVec3 vr = t3d_body_get_linear_velocity(dRt);
    const Avbd3dVec3 wcC = t3d_body_get_angular_velocity(dCtrl);
    const Avbd3dVec3 wdD = t3d_body_get_angular_velocity(dDamped);
    // The undamped spin reads about 4.8 after one second in this scene (solver numerics, not the
    // damping under test), so the control only has to stay well above the damped body's speed.
    CHECK(std::fabs(vc.x - 5.0f) < 0.05f && wcC.y > 4.0f, "undamped control keeps speed (%g %g)", vc.x, wcC.y);
    CHECK(vd.x < 2.5f && wdD.y < 2.5f, "damped body slows (v %g w %g)", vd.x, wdD.y);
    CHECK(vr.x < 2.5f, "runtime-damped body slows (v %g)", vr.x);

    // Kinematic: created kinematic and switched to kinematic at runtime both move with their
    // set velocity, unaffected by gravity, and report the type and zero mass.
    Avbd3dBodyDef kdef = bodyDef(AVBD3D_KINEMATIC_BODY, {-80, 5, 0});
    kdef.linearVelocity = {1, 0, 0};
    Avbd3dBodyId kCreate = makeBodyBox(pw, kdef, 0.5f, 0.5f, 0.5f);
    Avbd3dBodyId kRt = makeBodyBox(pw, bodyDef(AVBD3D_DYNAMIC_BODY, {-90, 5, 0}), 0.5f, 0.5f, 0.5f);
    t3d_body_set_type(kRt, AVBD3D_KINEMATIC_BODY);
    CHECK(t3d_body_get_type(kRt) == AVBD3D_KINEMATIC_BODY && t3d_body_get_mass(kRt) == 0.0f,
          "runtime kinematic type/mass");
    // The velocity write lands in the same flush as SetType's 1-byte Awake write: this covers
    // the mutation queue aligning each batch's data (a vec3 after it read garbage).
    t3d_body_set_linear_velocity(kRt, {1, 0, 0});
    step(pw, 60);
    Avbd3dVec3 pc = t3d_body_get_position(kCreate), pr = t3d_body_get_position(kRt);
    CHECK(std::fabs(pc.x - (-79.0f)) < 0.05f && std::fabs(pc.y - 5.0f) < 1e-3f,
          "created kinematic moves v*t, no gravity (%g %g)", pc.x, pc.y);
    CHECK(std::fabs(pr.x - (-89.0f)) < 0.05f && std::fabs(pr.y - 5.0f) < 1e-3f,
          "runtime kinematic moves v*t, no gravity (%g %g)", pr.x, pr.y);

    // Kinematic pusher drives into a dynamic box (gravity off for the target) and pushes it.
    Avbd3dBodyId pusher = makeBodyBox(pw, bodyDef(AVBD3D_KINEMATIC_BODY, {100, 5, 0}), 0.5f, 0.5f, 0.5f);
    Avbd3dBodyDef tdef = bodyDef(AVBD3D_DYNAMIC_BODY, {103, 5, 0});
    tdef.gravityScale = 0.0f;
    Avbd3dBodyId target = makeBodyBox(pw, tdef, 0.5f, 0.5f, 0.5f);
    t3d_body_set_linear_velocity(pusher, {2, 0, 0});
    step(pw, 90);
    const Avbd3dVec3 pp = t3d_body_get_position(pusher), tp = t3d_body_get_position(target);
    CHECK(std::fabs(pp.x - 103.0f) < 0.1f, "pusher kinematic at v*t (x %g)", pp.x);
    CHECK(tp.x > 103.5f, "kinematic pusher moves the dynamic box (x %g)", tp.x);

    // Motion locks: a locked linear axis holds the body; a locked angular axis stops its spin.
    Avbd3dBodyDef ldef = bodyDef(AVBD3D_DYNAMIC_BODY, {-70, 5, 0});
    Avbd3dMotionLocks ml{};
    ml.linearY = true;
    ldef.motionLocks = ml;
    Avbd3dBodyId locked = makeBodyBox(pw, ldef, 0.5f, 0.5f, 0.5f);
    Avbd3dBodyDef adef = bodyDef(AVBD3D_DYNAMIC_BODY, {-75, 5, 0});
    adef.gravityScale = 0.0f;
    Avbd3dBodyId spinLocked = makeBodyBox(pw, adef, 0.5f, 0.5f, 0.5f);
    Avbd3dMotionLocks aml{};
    aml.angularY = true;
    t3d_body_set_motion_locks(spinLocked, aml);
    t3d_body_set_angular_velocity(spinLocked, {0, 5, 0});
    step(pw, 60);
    CHECK(std::fabs(t3d_body_get_position(locked).y - 5.0f) < 1e-3f,
          "linearY lock holds the body (y %g)", t3d_body_get_position(locked).y);
    CHECK(std::fabs(t3d_body_get_angular_velocity(spinLocked).y) < 0.05f,
          "angularY lock stops the spin (w %g)", t3d_body_get_angular_velocity(spinLocked).y);

    // Shape restitution and surface material: stored and read back. Restitution is not
    // simulated by the solver (documented gap); it reaches BodyProps for the primary shape.
    Avbd3dShapeId tsh[4];
    t3d_body_get_shapes(target, tsh, 4);
    t3d_shape_set_restitution(tsh[0], 0.4f);
    Avbd3dSurfaceMaterial sm = avbd3d_default_surface_material();
    sm.friction = 0.3f;
    sm.rollingResistance = 0.05f;
    t3d_shape_set_surface_material(tsh[0], sm);

    t3d_destroy_world(pw);

    t3d_destroy_world(w);

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("avbd_test_vk_api3d: %s (%.1f s)\n", g_fail ? "FAILED" : "ok", secs);
    return g_fail ? 1 : 0;
}
