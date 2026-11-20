/*
 * M4 gate: every capacity / bound the one-submit step guesses must recover exactly.
 *
 * VkWorld::step() records a whole step blind to data-dependent sizes (pair and contact
 * capacity, wake-flood rounds, island-flood rounds, Jones-Plassmann rounds, colour
 * count) and reruns itself when the device flags a guess as too small. A rerun must be
 * invisible: the same scene stepped with a deliberately starved guess has to produce
 * BIT-IDENTICAL body poses after every step to the unconstrained run.
 *
 * For each forced case (tiny initial pair cap, tiny initial contact cap, flood bound 1,
 * island bound 1, JP bound 1, colour bound 1) and each scenario below, the trajectory is
 * hashed per step and compared with the default-configured reference; and each case must
 * have triggered its own rerun counter at least once somewhere (a forced case that never
 * fires tested nothing). Finally a hard ceiling is forced so low the step has to commit
 * clamped: StepResult must say so, naming phase and resource with required > capacity,
 * and nothing may crash or go non-finite.
 *
 * Scenarios: Pyramid (sleep off), Square Towers 20k (sleep on), a settled 24-box tower that is
 * nudged (explicit wake flood), and a settled jointed column hit by a slug (implicit wake,
 * island flood).
 */

#include "scene_builder.h"
#include "scenes.h"
#include "vk_world.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>
#include "test_zup.h"

using namespace avbdvk;
using Overrides = VkWorld::DebugOverrides;

namespace
{

uint64_t fnv(uint64_t h, const void *data, size_t bytes)
{
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < bytes; i++)
    {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

struct Run
{
    std::vector<uint64_t> hashes; // one per step
    int cap = 0, flood = 0, island = 0, jp = 0, color = 0;
    int truncatedSteps = 0;
    bool lastTruncated = false;
    bool finite = true;
    const char *failPhase = nullptr, *failResource = nullptr;
    int64_t required = 0, capacity = 0;
};

int findScene(const char *name)
{
    for (int s = 0; s < gpuSceneCount; s++)
        if (strcmp(gpuSceneNames[s], name) == 0)
            return s;
    return -1;
}

// Steps the world once and appends its state hash and StepResult to the run.
void stepAndRecord(VkWorld &w, const SolverParams &p, Run &run)
{
    const VkWorld::StepResult r = w.step(p);
    run.cap += r.capacityReruns;
    run.flood += r.floodReruns;
    run.island += r.islandReruns;
    run.jp += r.jpReruns;
    run.color += r.colorReruns;
    run.lastTruncated = r.truncated;
    if (r.truncated)
    {
        run.truncatedSteps++;
        run.failPhase = r.failPhase;
        run.failResource = r.failResource;
        run.required = r.required;
        run.capacity = r.capacity;
    }
    std::vector<vec3> pos;
    std::vector<quat> rot;
    w.downloadPositions(pos);
    w.downloadRotations(rot);
    for (const vec3 &v : pos)
        if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z))
            run.finite = false;
    uint64_t h = 1469598103934665603ull;
    h = fnv(h, pos.data(), pos.size() * sizeof(vec3));
    h = fnv(h, rot.data(), rot.size() * sizeof(quat));
    run.hashes.push_back(h);
}

Run runSceneByName(VkWorld &w, const Overrides &o, const char *name, bool sleep, int steps)
{
    Run run;
    w.setDebugOverrides(o);
    SceneBuilder author;
    author.params.gravity = vec3{0.0f, 0.0f, -10.0f};
    gpuScenes[findScene(name)](&author);
    author.params.sleepEnabled = sleep;
    const SceneDesc desc = author.finalize();
    w.build(desc, desc.params);
    for (int i = 0; i < steps; i++)
        stepAndRecord(w, desc.params, run);
    return run;
}

// Settled tower, then a 0.2 m/s nudge on the bottom box: the explicit wake flood has to
// climb the whole tower.
Run runTower(VkWorld &w, const Overrides &o)
{
    Run run;
    w.setDebugOverrides(o);
    SolverParams p = zUpParams();
    p.dt = 1.0f / 60.0f;
    p.sleepEnabled = true;
    const int layers = 24;
    SceneBuilder s;
    s.params.gravity = vec3{0.0f, 0.0f, -10.0f};
    s.clear();
    s.addBody(vec3{40, 40, 1}, 0.0f, 0.5f, vec3{0, 0, -0.5f});
    for (int i = 0; i < layers; i++)
        s.addBody(vec3{1, 1, 1}, 1.0f, 0.5f, vec3{0, 0, 0.5f + i * 1.0f});
    w.build(s.finalize(), p);
    for (int i = 0; i < 2000; i++)
    {
        stepAndRecord(w, p, run);
        if (w.activeBodyCount() == 0)
            break;
    }
    w.setLinearVelocity(1, vec3{0.2f, 0, 0});
    for (int i = 0; i < 40; i++)
        stepAndRecord(w, p, run);
    return run;
}

// Settled jointed column on a pedestal, a slug spawned into the live world and fired at
// the pedestal: the implicit wake has to reach the column through its joints (the island
// flood), and the spawn exercises append() under starved capacities.
Run runIsland(VkWorld &w, const Overrides &o)
{
    Run run;
    w.setDebugOverrides(o);
    SolverParams p = zUpParams();
    p.dt = 1.0f / 60.0f;
    p.sleepEnabled = true;
    SceneBuilder s;
    s.params.gravity = vec3{0.0f, 0.0f, -10.0f};
    s.clear();
    s.addBody(vec3{60, 60, 1}, 0.0f, 0.5f, vec3{0, 0, -0.5f});
    s.addBody(vec3{1, 1, 1}, 1.0f, 0.5f, vec3{0, 0, 0.5f});
    int prev = -1;
    for (int i = 0; i < 8; i++)
    {
        const float z = 1.5f + i;
        const int body = s.addBody(vec3{1, 1, 1}, 1.0f, 0.5f, vec3{0, 0, z});
        if (prev >= 0)
            s.addJointWorld(prev, body, vec3{0, 0, z - 0.5f}, INFINITY, 1.0e6f, INFINITY);
        prev = body;
    }
    w.build(s.finalize(), p);
    for (int i = 0; i < 3000; i++)
    {
        stepAndRecord(w, p, run);
        if (w.activeBodyCount() == 0)
            break;
    }
    const int slug = s.spawnBody(vec3{1, 1, 1}, 40.0f, 0.5f, vec3{-14, 0, 0.5f});
    w.append(s.finalize());
    w.setLinearVelocity(slug, vec3{90, 0, 0});
    for (int i = 0; i < 160; i++)
        stepAndRecord(w, p, run);
    return run;
}

} // namespace

int main()
{
    if (findScene("Pyramid") < 0 || findScene("Square Towers 20k") < 0)
    {
        printf("missing scene\n");
        return 2;
    }

    struct Case
    {
        const char *name;
        Overrides o;
        int Run::*counter;
    };
    Overrides pairs{}, contacts{}, flood{}, island{}, jp{}, color{};
    pairs.initialPairCapPerBody = 1;
    contacts.initialContactCap = 4;
    flood.floodBound = 1;
    island.islandBound = 1;
    jp.jpBound = 1;
    color.colorBound = 1;
    const Case cases[] = {
        {"tiny pair cap", pairs, &Run::cap},   {"tiny contact cap", contacts, &Run::cap},
        {"flood bound 1", flood, &Run::flood}, {"island bound 1", island, &Run::island},
        {"jp bound 1", jp, &Run::jp},          {"colour bound 1", color, &Run::color},
    };

    // Each scenario is a function of the overrides; the world is reused (build() resets
    // the scene, and the overrides are re-applied before it).
    using Fn = std::function<Run(VkWorld &, const Overrides &)>;
    struct Sc
    {
        const char *name;
        Fn fn;
    };
    const Sc scs[] = {
        {"Pyramid/sleep off",
         [](VkWorld &w, const Overrides &o) { return runSceneByName(w, o, "Pyramid", false, 100); }},
        {"Square Towers 20k/sleep on",
         [](VkWorld &w, const Overrides &o) { return runSceneByName(w, o, "Square Towers 20k", true, 150); }},
        {"tower nudge", runTower},
        {"joint island wake", runIsland},
    };

    VkWorld vk;
    vk.init();
    int failures = 0;

    std::vector<Run> refs;
    for (const Sc &sc : scs)
    {
        refs.push_back(sc.fn(vk, Overrides{}));
        const Run &r = refs.back();
        printf("reference %-22s %3zu steps  reruns cap %d flood %d island %d jp %d colour %d\n", sc.name,
               r.hashes.size(), r.cap, r.flood, r.island, r.jp, r.color);
        if (r.truncatedSteps || !r.finite)
        {
            printf("   reference run truncated or non-finite\n");
            failures++;
        }
    }
    printf("\n");

    for (const Case &c : cases)
    {
        int fired = 0;
        bool identical = true;
        for (size_t k = 0; k < sizeof(scs) / sizeof(scs[0]); k++)
        {
            const Run r = scs[k].fn(vk, c.o);
            const int n = r.*(c.counter);
            fired += n;
            size_t diverge = (size_t)-1;
            const size_t m = std::min(r.hashes.size(), refs[k].hashes.size());
            for (size_t i = 0; i < m; i++)
                if (r.hashes[i] != refs[k].hashes[i])
                {
                    diverge = i;
                    break;
                }
            const bool same = diverge == (size_t)-1 && r.hashes.size() == refs[k].hashes.size();
            if (!same)
            {
                identical = false;
                if (diverge == (size_t)-1)
                    printf("   %-18s %-22s length %zu vs reference %zu\n", c.name, scs[k].name, r.hashes.size(),
                           refs[k].hashes.size());
                else
                    printf("   %-18s %-22s DIVERGES at step %zu\n", c.name, scs[k].name, diverge);
            }
            printf("   %-18s %-22s %s  reruns cap %d flood %d island %d jp %d colour %d\n", c.name, scs[k].name,
                   same ? "identical" : "DIFFERENT", r.cap, r.flood, r.island, r.jp, r.color);
            if (r.truncatedSteps)
            {
                printf("   truncated unexpectedly\n");
                identical = false;
            }
        }
        const bool ok = identical && fired > 0;
        printf("%-18s %s  (counter fired %d times)\n\n", c.name, ok ? "ok" : "FAIL", fired);
        if (!ok)
            failures++;
    }

    // Hard ceilings: commit clamped, say so, do not crash.
    struct Hard
    {
        const char *name;
        Overrides o;
        const char *phase, *resource;
    };
    Overrides hp{}, hc{};
    hp.hardPairCap = 8;
    hc.hardContactCap = 8;
    const Hard hards[] = {{"hard pair cap 8", hp, "broadphase", "pairs"},
                          {"hard contact cap 8", hc, "narrowphase", "contacts"}};
    for (const Hard &h : hards)
    {
        const Run r = runSceneByName(vk, h.o, "Square Towers 20k", false, 40);
        const bool named = r.truncatedSteps > 0 && r.failPhase && r.failResource && strcmp(r.failPhase, h.phase) == 0 &&
                           strcmp(r.failResource, h.resource) == 0;
        const bool over = r.required > r.capacity;
        // The first steps may have no contact yet (the pile is still falling): what must
        // hold is that once the pile is in contact the ceiling is reported, every step.
        const bool ok = named && over && r.finite && r.lastTruncated && r.truncatedSteps * 2 >= (int)r.hashes.size();
        printf("%-18s %s  truncated %d/%zu steps, %s/%s required %lld > capacity %lld, finite %d, cap reruns %d\n",
               h.name, ok ? "ok" : "FAIL", r.truncatedSteps, r.hashes.size(), r.failPhase ? r.failPhase : "-",
               r.failResource ? r.failResource : "-", (long long)r.required, (long long)r.capacity, (int)r.finite,
               r.cap);
        if (!ok)
            failures++;
    }

    // The world must still work after the ceilings are lifted.
    {
        const Run r = runSceneByName(vk, Overrides{}, "Massive 10k", false, 20);
        const bool ok = r.truncatedSteps == 0 && r.finite;
        printf("recovers after ceiling  %s\n", ok ? "ok" : "FAIL");
        if (!ok)
            failures++;
    }

    printf("\n%s (%d failure%s)\n", failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
