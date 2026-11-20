/* A box dropped on a ground comes to rest. C99, public header only. */
#include "avbd3d/avbd3d.h"
#include "avbd3d/avbd3d_geometry.h"

#include <math.h>
#include <stdio.h>

#define CHECK(call)                                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        Avbd3dResult r_ = (call);                                                                                      \
        if (r_ != AVBD3D_OK)                                                                                           \
        {                                                                                                              \
            printf("FAIL: %s -> %s\n", #call, avbd3d_result_string(r_));                                               \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

static int addBox(Avbd3dWorld *world, int dynamic, Avbd3dVec3 p, float half, Avbd3dBodyId *out)
{
    Avbd3dBodyDef bd = avbd3d_default_body_def();
    bd.type = dynamic ? AVBD3D_DYNAMIC_BODY : AVBD3D_STATIC_BODY;
    bd.position = p;
    CHECK(avbd3d_create_body(world, &bd, out));
    Avbd3dShapeDef sd = avbd3d_default_shape_def();
    Avbd3dBoxHull hull = avbd3d_make_box_hull(half, half, half);
    Avbd3dShapeId shape;
    CHECK(avbd3d_create_hull_shape(*out, &sd, avbd3d_box_hull_data(&hull), &shape));
    return 0;
}

int main(void)
{
    if ((avbd3d_abi_version() >> 16) != AVBD3D_ABI_VERSION_MAJOR)
    {
        printf("FAIL: ABI major mismatch\n");
        return 1;
    }
    Avbd3dWorldDef wd = avbd3d_default_world_def();
    Avbd3dWorld *world = NULL;
    CHECK(avbd3d_create_world(&wd, &world));

    Avbd3dBodyId ground, box;
    Avbd3dVec3 gp = {0.0f, -5.0f, 0.0f};
    Avbd3dVec3 bp = {0.0f, 3.0f, 0.0f};
    if (addBox(world, 0, gp, 5.0f, &ground) || addBox(world, 1, bp, 0.5f, &box))
        return 1;

    for (int i = 0; i < 120; i++)
        CHECK(avbd3d_step(world, 1.0f / 60.0f));

    Avbd3dVec3 p;
    CHECK(avbd3d_body_get_position(box, &p));
    CHECK(avbd3d_destroy_world(world));
    printf("box y after 120 steps: %g\n", (double)p.y);
    /* Resting height is 0.5 (ground top at y = 0). */
    if (!(p.y > 0.3f && p.y < 0.7f) || fabsf(p.x) > 0.1f || fabsf(p.z) > 0.1f)
    {
        printf("FAIL: unexpected pose\n");
        return 1;
    }
    printf("OK\n");
    return 0;
}
