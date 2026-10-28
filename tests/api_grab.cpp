// Gate for the mouse grab and the read model around it, through the public avbd3d API. Y-up, default world.
//   a) the top box of a stack that has fallen asleep follows the grab handle up.
//   b) a body woken every frame while it is dragged (what a grab helper does) reads back its new
//      pose: a wake must not hold the mirror the way a pose write does.
//   c) a sleeping body reads awake right after set_awake, before any step has applied the wake.

#include "test_util.h"

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

static Avbd3dBodyId makeBox(Avbd3dWorld *w, Avbd3dBodyType type, Avbd3dVec3 p, Avbd3dVec3 h)
{
    Avbd3dBodyDef bd = avbd3d_default_body_def();
    bd.type = type;
    bd.position = p;
    Avbd3dBodyId b = t3d_create_body(w, &bd);
    Avbd3dShapeDef sd = avbd3d_default_shape_def();
    Avbd3dBoxHull hull = avbd3d_make_box_hull(h.x, h.y, h.z);
    t3d_create_hull_shape(b, &sd, avbd3d_box_hull_data(&hull));
    return b;
}

static void step(Avbd3dWorld *w, int n = 1)
{
    for (int i = 0; i < n; i++)
        t3d_step(w, 1.0f / 60.0f);
}

// A slab and a tower of ten 0.5 m boxes, stepped until the tower sleeps. Returns the top box.
static Avbd3dBodyId sleepingTower(Avbd3dWorld *w)
{
    makeBox(w, AVBD3D_STATIC_BODY, {0, -0.5f, 0}, {30, 0.5f, 30});
    Avbd3dBodyId top{};
    for (int i = 0; i < 10; i++)
        top = makeBox(w, AVBD3D_DYNAMIC_BODY, {0, 0.25f + 0.501f * i, 0}, {0.25f, 0.25f, 0.25f});
    step(w, 300);
    return top;
}

static Avbd3dWorld *newWorld()
{
    Avbd3dWorldDef wd = avbd3d_default_world_def();
    wd.enableSleep = 1;
    return t3d_create_world(&wd);
}

static void dragUp(Avbd3dWorld *w, Avbd3dBodyId body, Avbd3dVec3 target)
{
    Avbd3dGrabHandle h{};
    h.body = body;
    h.localAnchor = {0, 0, 0};
    h.target = target;
    h.weight = 1.0f;
    t3d_grab_move(w, &h, 1);
}

static void checkGrabSleepingStack()
{
    Avbd3dWorld *w = newWorld();
    Avbd3dBodyId top = sleepingTower(w);
    CHECK(!t3d_body_is_awake(top), "the tower has fallen asleep");
    Avbd3dVec3 p0 = t3d_body_get_position(top);
    Avbd3dBodyId group[8];
    CHECK(t3d_grab_begin(w, top, p0, 0.1f, 1, group, 8) == 1, "the grab holds the top box alone");
    for (int i = 0; i < 90; i++)
    {
        dragUp(w, top, {p0.x, p0.y + 3.0f, p0.z});
        step(w);
    }
    float rise = t3d_body_get_position(top).y - p0.y;
    CHECK(rise > 2.5f, "the grabbed top box follows the handle (rose %.3f m of 3)", rise);
    t3d_grab_end(w);
    t3d_destroy_world(w);
}

static void checkWokenEveryFrameReadsBack()
{
    Avbd3dWorld *w = newWorld();
    Avbd3dBodyId top = sleepingTower(w);
    Avbd3dVec3 p0 = t3d_body_get_position(top);
    Avbd3dBodyId group[8];
    t3d_grab_begin(w, top, p0, 0.1f, 1, group, 8);
    float rise = 0.0f;
    for (int frame = 0; frame < 180; frame++)
    {
        // Two frames per step, as a 120 Hz frame rate on a 60 Hz step does. Every frame wakes the held body
        // and drags it while a step is in flight (the async loop), then reads its pose back (a transform sync).
        bool inFlight = frame % 2 == 0;
        if (inFlight)
            t3d_step_begin(w, 1.0f / 60.0f);
        t3d_check(avbd3d_body_set_awake(top, 1), "avbd3d_body_set_awake");
        dragUp(w, top, {p0.x, p0.y + 3.0f, p0.z});
        if (inFlight)
            t3d_step_end(w);
        rise = t3d_body_get_position(top).y - p0.y;
    }
    CHECK(rise > 2.5f, "the pose read while the body is woken every frame follows it (rose %.3f m of 3)", rise);
    t3d_grab_end(w);
    t3d_destroy_world(w);
}

static void checkWakeReadsAwake()
{
    Avbd3dWorld *w = newWorld();
    Avbd3dBodyId top = sleepingTower(w);
    CHECK(!t3d_body_is_awake(top), "the tower has fallen asleep");
    t3d_check(avbd3d_body_set_awake(top, 1), "avbd3d_body_set_awake");
    CHECK(t3d_body_is_awake(top), "set_awake reads awake before a step");
    uint8_t flag = 0;
    int valid = 0;
    t3d_check(avbd3d_world_get_body_data(w, &top, 1, AVBD3D_BODY_DATA_AWAKE, &flag, &valid), "avbd3d_world_get_body_data");
    CHECK(valid == 1 && flag == 1, "the batched read agrees (%d)", flag);
    step(w);
    CHECK(t3d_body_is_awake(top), "and after the step that applies it");
    t3d_destroy_world(w);
}

int main()
{
    checkGrabSleepingStack();
    checkWokenEveryFrameReadsBack();
    checkWakeReadsAwake();
    std::printf(g_fail ? "api_grab: %d failure(s)\n" : "api_grab: ok\n", g_fail);
    return g_fail ? 1 : 0;
}
