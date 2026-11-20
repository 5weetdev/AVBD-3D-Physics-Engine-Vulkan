/*
 * Phase 0.4 gate: a Vulkan failure is an error, never an abort.
 *
 * The failure is injected without a GPU fault through the test hook Device::setDebugFailSubmit
 * (always compiled in, test-only): the Nth step submit reports VK_ERROR_DEVICE_LOST. Checks,
 * for VkWorld and for the C ABI:
 *   - the failing step reports ok=false, failPhase "device", deviceLost, vkError = DEVICE_LOST;
 *   - earlier steps were fine and the error is sticky on the world (deviceError());
 *   - further steps (including stepBegin/stepEnd) are no-ops returning the same error, with
 *     no submit and no new allocation;
 *   - destroying the dead world does not crash;
 *   - the avbd3d C ABI turns it into AVBD3D_ERR_DEVICE_LOST; later steps are no-ops, setters,
 *     getters, queries and creates on the dead world return it, and avbd3d_destroy_world is safe. The process is still alive at the end (that is
 *     the test).
 */

#include "test_util.h"
#include "scene_builder.h"
#include "vk_world.h"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace avbdvk;

static int failures = 0;
static void check(bool ok, const char *what)
{
    printf("%-58s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok)
        failures++;
}

static void testWorld()
{
    SceneBuilder author;
    author.params.gravity = vec3{0.0f, 0.0f, -10.0f};
    author.addBody({40, 40, 1}, 0.0f, 0.5f, {0, 0, -0.5f});
    for (int i = 0; i < 4; i++)
        author.addBody({1, 1, 1}, 1.0f, 0.5f, {0, 0, 1.0f + 1.2f * (float)i});
    const SceneDesc desc = author.finalize();

    {
        VkWorld w;
        w.device().setDebugFailSubmit(3); // steps 1 and 2 succeed, step 3 "loses" the device
        w.init();
        w.build(desc, desc.params);

        const auto r1 = w.step(desc.params);
        const auto r2 = w.step(desc.params);
        check(r1.ok && r2.ok && w.deviceError() == VK_SUCCESS, "first two steps ok, no device error");

        const auto r3 = w.step(desc.params);
        check(!r3.ok, "failing step reports ok = false");
        check(r3.failPhase && std::string(r3.failPhase) == "device", "failPhase is \"device\"");
        check(r3.deviceLost && r3.vkError == (int)VK_ERROR_DEVICE_LOST, "deviceLost / vkError = VK_ERROR_DEVICE_LOST");
        check(w.deviceError() == VK_ERROR_DEVICE_LOST, "error is sticky on the world");
        check(!w.stepInFlight(), "no step left in flight");

        const uint64_t submits = Device::submitCount();
        const uint64_t allocs = Device::allocCount();
        bool allDead = true;
        for (int i = 0; i < 3; i++)
        {
            const auto r = w.step(desc.params);
            allDead = allDead && !r.ok && r.deviceLost && r.vkError == (int)VK_ERROR_DEVICE_LOST;
        }
        w.stepBegin(desc.params);
        const auto rs = w.stepEnd();
        allDead = allDead && !rs.ok && rs.deviceLost;
        check(allDead, "later steps (step and stepBegin/End) return the error");
        check(Device::submitCount() == submits && Device::allocCount() == allocs, "later steps touch no Vulkan");
        check(!w.lastStepResult().ok, "lastStepResult() holds the failure");
    } // ~VkWorld on the dead world
    check(true, "destroying the dead world does not crash");
}

static Avbd3dBodyId addBox(Avbd3dWorld *w, Avbd3dBodyType type, Avbd3dVec3 p, float hx, float hy, float hz)
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

static void testApi()
{
#ifdef _WIN32
    _putenv_s("AVBD_DEBUG_FAIL_SUBMIT", "4");
#else
    setenv("AVBD_DEBUG_FAIL_SUBMIT", "4", 1);
#endif
    Avbd3dWorldDef def = avbd3d_default_world_def();
    Avbd3dWorld *world = NULL;
    const Avbd3dResult created = avbd3d_create_world(&def, &world);
#ifdef _WIN32
    _putenv_s("AVBD_DEBUG_FAIL_SUBMIT", "");
#else
    unsetenv("AVBD_DEBUG_FAIL_SUBMIT");
#endif
    check(created == AVBD3D_OK && world != NULL, "avbd3d: world created");
    if (!world)
        return;
    addBox(world, AVBD3D_STATIC_BODY, Avbd3dVec3{0, -0.5f, 0}, 20, 0.5f, 20);
    const Avbd3dBodyId box = addBox(world, AVBD3D_DYNAMIC_BODY, Avbd3dVec3{0, 3.0f, 0}, 0.5f, 0.5f, 0.5f);

    check(avbd3d_step(world, 1.0f / 60.0f) == AVBD3D_OK, "avbd3d: first step ok");
    // Step until the injected failure; the Result of the failing call names the cause, there is
    // no sticky error to poll.
    Avbd3dResult r = AVBD3D_OK;
    float lastGoodY = 0;
    for (int i = 0; i < 6 && r == AVBD3D_OK; i++)
    {
        lastGoodY = t3d_body_get_position(box).y;
        r = avbd3d_step(world, 1.0f / 60.0f);
    }
    check(r == AVBD3D_ERR_DEVICE_LOST, "avbd3d: a later step returns AVBD3D_ERR_DEVICE_LOST");

    // A dead world answers every call with the error and changes nothing.
    Avbd3dVec3 pos;
    check(avbd3d_body_get_position(box, &pos) == AVBD3D_ERR_DEVICE_LOST, "avbd3d: getters report the lost device");
    check(avbd3d_body_set_linear_velocity(box, Avbd3dVec3{1, 0, 0}) == AVBD3D_ERR_DEVICE_LOST,
          "avbd3d: setters report the lost device");
    Avbd3dBodyDef bd = avbd3d_default_body_def();
    bd.type = AVBD3D_DYNAMIC_BODY;
    Avbd3dBodyId late;
    check(avbd3d_create_body(world, &bd, &late) == AVBD3D_ERR_DEVICE_LOST, "avbd3d: creates report the lost device");
    check(avbd3d_step(world, 1.0f / 60.0f) == AVBD3D_ERR_DEVICE_LOST, "avbd3d: steps report the lost device");
    check(avbd3d_step_begin(world, 1.0f / 60.0f) == AVBD3D_ERR_DEVICE_LOST,
          "avbd3d: step_begin reports the lost device");
    Avbd3dStats st;
    check(avbd3d_get_stats(world, &st) == AVBD3D_ERR_DEVICE_LOST, "avbd3d: get_stats reports the lost device");
    (void)lastGoodY;

    // Only destroy works, and it is safe.
    check(avbd3d_destroy_world(world) == AVBD3D_OK, "avbd3d: destroy of the dead world does not crash");

    // No error outlives the world: a fresh world on the same device works again, and a
    // thread that saw the failure carries nothing over.
    Avbd3dWorld *w2 = t3d_create_world(&def);
    addBox(w2, AVBD3D_DYNAMIC_BODY, Avbd3dVec3{0, 3.0f, 0}, 0.5f, 0.5f, 0.5f);
    bool ok = true;
    for (int i = 0; i < 10; i++)
        ok = ok && avbd3d_step(w2, 1.0f / 60.0f) == AVBD3D_OK;
    check(ok, "avbd3d: a new world after the failure works");
    t3d_destroy_world(w2);
}

int main()
{
    testWorld();
    testApi();
    printf("%s\n", failures ? "FAILED" : "PASSED");
    return failures ? 1 : 0;
}
