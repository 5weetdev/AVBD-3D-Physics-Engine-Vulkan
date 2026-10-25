/*
 * M4 gate: one vkQueueSubmit per step, and no device allocation per step after warmup
 * (docs/VULKAN_PROGRESS.md, M4 exit gates).
 *
 * For a handful of scenes (contacts, a big pile, joints + springs), each with sleep off
 * and on: warm up, then measure every step:
 *
 *   - a step whose StepResult reports no rerun of any kind must have used exactly one
 *     submit;
 *   - a step without a capacity rerun must not have called vkCreateBuffer or
 *     vkAllocateMemory (Device::allocCount());
 *   - reruns must be rare (<= 10% of measured steps), or "one submission" is a
 *     best case rather than the normal case.
 *
 * Usage: avbd_test_vk_onesubmit [warmup=60] [steps=120]
 */

#include "scene_builder.h"
#include "scenes.h"
#include "vk_world.h"
#include "vk_grab.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

using namespace avbdvk;

static int findScene(const char *name)
{
    for (int s = 0; s < gpuSceneCount; s++)
        if (strcmp(gpuSceneNames[s], name) == 0)
            return s;
    return -1;
}

// The demo's multi-grab frame: dragBodies and the survivor test before and after the step,
// as DemoApp::update does while a jointed group is held. Holding a group must not cost
// submissions or allocations of its own (each dragBodies used to be four Buffer::upload
// one-shots plus a dispatch, and the survivor test a full jBroken readback: the demo's
// frame rate halved while a bridge was held). And the held group must still follow the
// cursor, which proves the queued update lands in the step.
static int multiGrabFrames(VkWorld &vk, int warmup, int steps)
{
    SceneBuilder author; author.params.gravity = vec3{0.0f, 0.0f, -10.0f};
    gpuScenes[findScene("Bridge")](&author);
    author.params.sleepEnabled = false;
    const SceneDesc desc = author.finalize();
    vk.build(desc, desc.params);
    for (int i = 0; i < warmup; i++)
        vk.step(desc.params);

    int seed = -1;
    for (int i = 0; i < desc.jointCount && seed < 0; i++)
        if (desc.mass[desc.jointB[i]] > 0.0f)
            seed = desc.jointB[i];
    std::vector<vec3> pos;
    vk.downloadPositions(pos);
    std::vector<int> group;
    grabGroup(vk, desc, seed, pos[seed], 1000.0f, true, group);
    GrabTopology topo;
    buildGrabTopology(desc, group, topo);

    const vec3 lift{0, 0, 1.0f};
    std::vector<vec3> targets(group.size()), locals(group.size(), vec3{0, 0, 0});
    for (size_t k = 0; k < group.size(); k++)
        targets[k] = pos[group[k]] + lift;

    std::vector<unsigned char> keep;
    int badSubmit = 0, badAlloc = 0, dropped = 0;
    auto hold = [&] {
        dropped += (int)group.size() - grabSurvivorsCached(vk, topo, keep);
        vk.dragBodies(group.data(), targets.data(), locals.data(), (int)group.size(), nullptr, 1.0f);
    };
    for (int i = 0; i < steps; i++)
    {
        const uint64_t s0 = Device::submitCount(), a0 = Device::allocCount();
        hold();
        const VkWorld::StepResult r = vk.step(desc.params);
        hold();
        const uint64_t submits = Device::submitCount() - s0, allocs = Device::allocCount() - a0;
        if (r.totalReruns() == 0 && submits != 1 && badSubmit++ < 3)
            printf("   grab frame %d: %llu submits (step alone %d)\n", i, (unsigned long long)submits, r.submits);
        if (r.capacityReruns == 0 && allocs != 0 && badAlloc++ < 3)
            printf("   grab frame %d: %llu allocations\n", i, (unsigned long long)allocs);
    }
    std::vector<vec3> after;
    vk.downloadPositions(after);
    // The seed is the plank next to an anchor and barely moves; the bridge's middle must.
    float rise = -1e30f;
    for (int b : group)
        rise = std::max(rise, after[b].z - pos[b].z);
    vk.releaseDrag();

    // Unbreakable bridge joints: nothing may come off. The held span must have risen most of the lift.
    const bool follows = rise > 0.5f * lift.z;
    const bool ok = badSubmit == 0 && badAlloc == 0 && dropped == 0 && follows;
    printf("%-12s grab  %s  %d bodies held, max rise %.3f m of %.1f, extra submits/allocs: %d/%d frames, "
           "dropped %d\n",
           "Bridge", ok ? "ok  " : "FAIL", (int)group.size(), rise, lift.z, badSubmit, badAlloc, dropped);
    fflush(stdout);
    return ok ? 0 : 1;
}

int main(int argc, char **argv)
{
    const int warmup = argc > 1 ? atoi(argv[1]) : 60;
    const int steps = argc > 2 ? atoi(argv[2]) : 120;
    const char *names[] = {"Pyramid", "Pile Settle", "Massive 10k", "Bridge", "Soft Body"};

    VkWorld vk;
    vk.init();

    int failures = 0;
    for (const char *name : names)
    {
        const int s = findScene(name);
        if (s < 0)
        {
            printf("no scene named %s\n", name);
            return 2;
        }
        for (int sleep = 0; sleep < 2; sleep++)
        {
            SceneBuilder author; author.params.gravity = vec3{0.0f, 0.0f, -10.0f};
            gpuScenes[s](&author);
            author.params.sleepEnabled = sleep != 0;
            const SceneDesc desc = author.finalize();
            vk.build(desc, desc.params);

            for (int i = 0; i < warmup; i++)
                vk.step(desc.params);

            int rerunSteps = 0, badSubmit = 0, badAlloc = 0, capReruns = 0, flood = 0, island = 0, jp = 0,
                color = 0, allocAfterWarm = 0;
            int hist[8] = {};
            for (int i = 0; i < steps; i++)
            {
                const uint64_t a0 = Device::allocCount();
                const VkWorld::StepResult r = vk.step(desc.params);
                const uint64_t allocs = Device::allocCount() - a0;
                const int total = r.totalReruns();
                hist[r.submits < 7 ? r.submits : 7]++;
                if (total > 0)
                    rerunSteps++;
                capReruns += r.capacityReruns;
                flood += r.floodReruns;
                island += r.islandReruns;
                jp += r.jpReruns;
                color += r.colorReruns;
                if (total == 0 && r.submits != 1)
                {
                    if (badSubmit++ < 3)
                        printf("   step %d: %d submits with no reruns\n", i, r.submits);
                }
                if (r.capacityReruns == 0)
                    allocAfterWarm += (int)allocs;
                if (r.capacityReruns == 0 && allocs != 0)
                {
                    if (badAlloc++ < 3)
                        printf("   step %d: %llu allocations without a capacity rerun\n", i,
                               (unsigned long long)allocs);
                }
                if (!r.ok)
                {
                    printf("   step %d: truncated (%s/%s %lld > %lld)\n", i, r.failPhase, r.failResource,
                           (long long)r.required, (long long)r.capacity);
                    failures++;
                }
            }
            const bool rare = rerunSteps * 10 <= steps;
            const bool ok = badSubmit == 0 && badAlloc == 0 && rare;
            printf("%-12s sleep %d  %s  rerun steps %3d/%d (cap %d flood %d island %d jp %d color %d)  "
                   "submits{1:%d 2:%d 3+:%d}  allocs after warmup %d\n",
                   name, sleep, ok ? "ok  " : "FAIL", rerunSteps, steps, capReruns, flood, island, jp, color,
                   hist[1], hist[2], hist[3] + hist[4] + hist[5] + hist[6] + hist[7], allocAfterWarm);
            fflush(stdout);
            if (!ok)
                failures++;
        }
    }

    failures += multiGrabFrames(vk, warmup, steps);

    printf("\n%s (%d failure%s)\n", failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
