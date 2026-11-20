// Gate for sleeping and waking through the public avbd3d API. Y-up, default world (sleeping on).
//   a) a box at rest on the ground falls asleep, and a body that is not at rest does not.
//   b) wake by contact: a box dropped onto a sleeping box wakes it.
//   c) wake by joint: a sleeping body tied to another by a spring wakes when the partner is hit.
//   d) wake by teleport: set_transform on a sleeping box wakes it and it falls to the ground.
//   e) wake by kinematic target: a kinematic platform driven into a sleeping box wakes and moves it.

#include "test_util.h"

#include <cmath>
#include <cstdio>

static int g_fail = 0;
#define CHECK(cond, ...)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            std::printf("FAIL %s:%d: %s  ", __FILE__, __LINE__, #cond);                                                \
            std::printf(__VA_ARGS__);                                                                                  \
            std::printf("\n");                                                                                         \
            g_fail++;                                                                                                  \
        }                                                                                                              \
    } while (0)

static Avbd3dBodyId makeBox(Avbd3dWorld *w, Avbd3dBodyType type, Avbd3dVec3 p, float h = 0.5f)
{
    Avbd3dBodyDef bd = avbd3d_default_body_def();
    bd.type = type;
    bd.position = p;
    Avbd3dBodyId b = t3d_create_body(w, &bd);
    Avbd3dShapeDef sd = avbd3d_default_shape_def();
    Avbd3dBoxHull hull = avbd3d_make_box_hull(h, h, h);
    t3d_create_hull_shape(b, &sd, avbd3d_box_hull_data(&hull));
    return b;
}

static void step(Avbd3dWorld *w, int n = 1)
{
    for (int i = 0; i < n; i++)
        t3d_step(w, 1.0f / 60.0f);
}

static Avbd3dWorld *newWorld()
{
    Avbd3dWorldDef wd = avbd3d_default_world_def();
    wd.enableSleep = 1;
    Avbd3dWorld *w = t3d_create_world(&wd);
    return w;
}

static Avbd3dBodyId makeSlab(Avbd3dWorld *w)
{
    Avbd3dBodyDef bd = avbd3d_default_body_def();
    bd.type = AVBD3D_STATIC_BODY;
    bd.position = {0, -0.5f, 0};
    Avbd3dBodyId b = t3d_create_body(w, &bd);
    Avbd3dShapeDef sd = avbd3d_default_shape_def();
    Avbd3dBoxHull hull = avbd3d_make_box_hull(30, 0.5f, 30);
    t3d_create_hull_shape(b, &sd, avbd3d_box_hull_data(&hull));
    return b;
}

static void checkRestSleeps()
{
    Avbd3dWorld *w = newWorld();
    makeSlab(w);
    Avbd3dBodyId box = makeBox(w, AVBD3D_DYNAMIC_BODY, {0, 0.52f, 0});
    Avbd3dBodyId flying = makeBox(w, AVBD3D_DYNAMIC_BODY, {10, 500, 0});
    step(w, 5);
    CHECK(t3d_body_is_awake(box), "a fresh body is awake");
    step(w, 240);
    CHECK(!t3d_body_is_awake(box), "a box at rest falls asleep");
    CHECK(t3d_body_is_awake(flying), "a body in free fall stays awake");
    t3d_destroy_world(w);
}

static void checkWakeByContact()
{
    Avbd3dWorld *w = newWorld();
    makeSlab(w);
    Avbd3dBodyId low = makeBox(w, AVBD3D_DYNAMIC_BODY, {0, 0.52f, 0});
    step(w, 240);
    CHECK(!t3d_body_is_awake(low), "precondition: the lower box sleeps");
    Avbd3dBodyId drop = makeBox(w, AVBD3D_DYNAMIC_BODY, {0, 4.0f, 0});
    bool woke = false;
    for (int i = 0; i < 120 && !woke; i++)
    {
        step(w);
        woke = t3d_body_is_awake(low);
    }
    CHECK(woke, "a box dropped on a sleeping box wakes it");
    (void)drop;
    t3d_destroy_world(w);
}

static void checkWakeByJoint()
{
    Avbd3dWorld *w = newWorld();
    makeSlab(w);
    Avbd3dBodyId a = makeBox(w, AVBD3D_DYNAMIC_BODY, {-2, 0.52f, 0});
    Avbd3dBodyId b = makeBox(w, AVBD3D_DYNAMIC_BODY, {2, 0.52f, 0});
    Avbd3dSpringDef sd = avbd3d_default_spring_def();
    sd.base.bodyIdA = a;
    sd.base.bodyIdB = b;
    sd.stiffness = 50.0f;
    sd.restLength = 4.0f;
    Avbd3dJointId s = t3d_create_spring(w, &sd);
    CHECK(avbd3d_joint_is_valid(s), "spring created");
    step(w, 300);
    CHECK(!t3d_body_is_awake(a) && !t3d_body_is_awake(b), "precondition: both ends of the spring sleep");
    // Hit one end; its partner is in the same joint island and must wake with it.
    makeBox(w, AVBD3D_DYNAMIC_BODY, {-2, 4.0f, 0});
    bool bothAwake = false;
    for (int i = 0; i < 120 && !bothAwake; i++)
    {
        step(w);
        bothAwake = t3d_body_is_awake(a) && t3d_body_is_awake(b);
    }
    CHECK(bothAwake, "hitting one end of a joint wakes the other (a %d, b %d)", (int)t3d_body_is_awake(a),
          (int)t3d_body_is_awake(b));
    t3d_destroy_world(w);
}

static void checkWakeByTeleport()
{
    Avbd3dWorld *w = newWorld();
    makeSlab(w);
    Avbd3dBodyId box = makeBox(w, AVBD3D_DYNAMIC_BODY, {0, 0.52f, 0});
    step(w, 240);
    CHECK(!t3d_body_is_awake(box), "precondition: the box sleeps");
    t3d_body_set_transform(box, {3, 3.0f, 0}, {{0, 0, 0}, 1.0f});
    step(w, 40);
    const float y = t3d_body_get_position(box).y;
    CHECK(y < 2.5f, "a teleported sleeping box falls (y %.3f 40 steps after it was put at y 3)", y);
    t3d_destroy_world(w);
}

static void checkWakeByKinematic()
{
    Avbd3dWorld *w = newWorld();
    makeSlab(w);
    Avbd3dBodyId box = makeBox(w, AVBD3D_DYNAMIC_BODY, {0, 0.52f, 0});
    Avbd3dBodyDef kd = avbd3d_default_body_def();
    kd.type = AVBD3D_KINEMATIC_BODY;
    kd.position = {-4, 0.6f, 0};
    Avbd3dBodyId plat = t3d_create_body(w, &kd);
    Avbd3dShapeDef sd = avbd3d_default_shape_def();
    Avbd3dBoxHull hull = avbd3d_make_box_hull(0.5f, 0.5f, 0.5f);
    t3d_create_hull_shape(plat, &sd, avbd3d_box_hull_data(&hull));
    step(w, 240);
    CHECK(!t3d_body_is_awake(box), "precondition: the box sleeps");
    const float dt = 1.0f / 60.0f;
    for (int i = 1; i <= 120; i++)
    {
        Avbd3dTransform target{{-4.0f + 0.05f * i, 0.6f, 0}, {{0, 0, 0}, 1.0f}};
        t3d_body_set_target_transform(plat, target, dt, true);
        step(w);
    }
    CHECK(t3d_body_get_position(box).x > 0.5f, "a kinematic platform pushes the sleeping box (x %.3f)",
          t3d_body_get_position(box).x);
    t3d_destroy_world(w);
}

int main()
{
    checkRestSleeps();
    checkWakeByContact();
    checkWakeByJoint();
    checkWakeByTeleport();
    checkWakeByKinematic();
    std::printf("api_sleep_wake: %s\n", g_fail ? "FAILED" : "ok");
    return g_fail ? 1 : 0;
}
