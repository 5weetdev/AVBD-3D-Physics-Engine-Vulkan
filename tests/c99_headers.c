/* The public headers are plain C99: include all three in one translation unit and use a type from
   each. Nothing is linked. */
#include "avbd3d/avbd3d.h"
#include "avbd3d/avbd3d_geometry.h"
#include "avbd3d/avbd3d_vulkan.h"

int main(void)
{
    Avbd3dBoxHull box;
    Avbd3dGpuDef gpu;
    Avbd3dWorldDef def;
    (void)box;
    (void)gpu;
    (void)def;
    return AVBD3D_ABI_VERSION_MAJOR == 1 ? 0 : 1;
}
