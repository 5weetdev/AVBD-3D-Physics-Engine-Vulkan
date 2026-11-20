/*
 * Copyright (c) 2026 Chris Giles
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies.
 * Chris Giles makes no representations about the suitability
 * of this software for any purpose.
 * It is provided "as is" without express or implied warranty.
 */

#pragma once

// The grab's host-side helpers, shared by the demo and the C ABI: the two breadth-first searches that decide which bodies a grab holds
// (ported from the drag path). The topology comes from the authored scene
// (joints and springs), the live poses and fracture flags from the world. A grab is a
// once-per-click event, so the readbacks cost nothing that matters.

#include "vk_world.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace avbdvk
{

// One grab holds at most this many bodies: VkWorld reserves this many drag joint slots.
static constexpr int kMaxGrabBodies = 256;

// A joint that holds its bodies together: not broken.
inline bool grabJointIntact(const std::vector<unsigned char> &broken, int j)
{
    return j < 0 || j >= (int)broken.size() || !broken[j];
}

// The seed plus, with followConnections, everything reachable through joints and springs
// whose live position lies within `radius` of `center`. Statics are never included.
inline void grabGroup(VkWorld &world, const SceneDesc &scene, int seed, vec3 center, float radius,
                      bool followConnections, std::vector<int> &out)
{
    out.clear();
    if (seed < 0 || seed >= scene.count)
        return;

    out.push_back(seed);

    // Measured against live poses, so the radius selects what is near the cursor now.
    std::vector<vec3> live;
    if (followConnections && radius > 0.0f)
        world.downloadPositions(live);

    const int n = scene.count;
    std::vector<unsigned char> seen(n, 0);
    seen[seed] = 1;

    if (!followConnections || radius <= 0.0f)
        return;

    std::vector<std::vector<int>> adj(n);
    auto link = [&](int a, int b)
    {
        if (a < 0 || b < 0 || a >= n || b >= n)
            return;
        adj[a].push_back(b);
        adj[b].push_back(a);
    };
    for (int i = 0; i < scene.jointCount; i++)
        link(scene.jointA[i], scene.jointB[i]);
    for (int i = 0; i < scene.springCount; i++)
        link(scene.springA[i], scene.springB[i]);

    // The walk starts from the seed.
    const float r2 = radius * radius;
    std::vector<int> queue(out.begin(), out.end());
    for (size_t head = 0; head < queue.size() && (int)out.size() < kMaxGrabBodies; head++)
    {
        const int cur = queue[head];
        for (int nb : adj[cur])
        {
            if (seen[nb])
                continue;
            seen[nb] = 1;
            // A pinned anchor would fight the handle and cannot move anyway.
            if (!scene.mass.empty() && scene.mass[nb] <= 0.0f)
                continue;
            if (lengthSq(live[nb] - center) > r2)
                continue;
            out.push_back(nb);
            queue.push_back(nb);
            if ((int)out.size() >= kMaxGrabBodies)
                break;
        }
    }
}

// Which entries of `group` are still attached to `seed` through unbroken joints and
// springs. Writes keep[i] per entry and returns how many survived.
inline int grabSurvivors(VkWorld &world, const SceneDesc &scene, int seed, const std::vector<int> &group,
                         std::vector<unsigned char> &keep)
{
    keep.assign(group.size(), 0);
    if (group.empty())
        return 0;

    std::unordered_map<int, int> slotOf;
    slotOf.reserve(group.size() * 2);
    for (size_t i = 0; i < group.size(); i++)
        slotOf[group[i]] = (int)i;

    auto it = slotOf.find(seed);
    if (it == slotOf.end())
        return 0;

    // Only the authored joints matter; the drag slots are the hand.
    std::vector<unsigned char> broken;
    world.downloadJointBroken(broken);

    std::vector<std::vector<int>> adj(group.size());
    auto link = [&](int a, int b)
    {
        auto ia = slotOf.find(a), ib = slotOf.find(b);
        if (ia == slotOf.end() || ib == slotOf.end())
            return;
        adj[ia->second].push_back(ib->second);
        adj[ib->second].push_back(ia->second);
    };
    for (int i = 0; i < scene.jointCount; i++)
    {
        if (!grabJointIntact(broken, i))
            continue;
        link(scene.jointA[i], scene.jointB[i]);
    }
    // Springs stretch but never break, so they always still connect.
    for (int i = 0; i < scene.springCount; i++)
        link(scene.springA[i], scene.springB[i]);

    std::vector<int> queue;
    keep[it->second] = 1;
    queue.push_back(it->second);
    int found = 1;
    for (size_t head = 0; head < queue.size(); head++)
        for (int nb : adj[queue[head]])
            if (!keep[nb])
            {
                keep[nb] = 1;
                found++;
                queue.push_back(nb);
            }
    return found;
}

// The part of the scene's topology a held group can lose: edges between two members,
// indexed by slot in `group`. Built once per grab (it walks every authored joint and
// spring) so the per-frame survivor test touches only the group's own edges.
struct GrabTopology
{
    std::vector<int> a, b;      // slots in the group
    std::vector<int> joint;     // authored joint index, or -1 for a spring
    std::vector<int> breakable; // indices into the edge arrays whose joint can fracture
    int slotCount = 0;
};

inline void buildGrabTopology(const SceneDesc &scene, const std::vector<int> &group, GrabTopology &t)
{
    t = GrabTopology{};
    t.slotCount = (int)group.size();
    if (group.size() < 2)
        return;
    std::unordered_map<int, int> slotOf;
    slotOf.reserve(group.size() * 2);
    for (size_t i = 0; i < group.size(); i++)
        slotOf[group[i]] = (int)i;
    auto edge = [&](int a, int b, int joint)
    {
        auto ia = slotOf.find(a), ib = slotOf.find(b);
        if (ia == slotOf.end() || ib == slotOf.end())
            return;
        if (joint >= 0 && joint < (int)scene.jointFrac.size() && std::isfinite(scene.jointFrac[joint]))
            t.breakable.push_back((int)t.a.size());
        t.a.push_back(ia->second);
        t.b.push_back(ib->second);
        t.joint.push_back(joint);
    };
    for (int i = 0; i < scene.jointCount; i++)
        edge(scene.jointA[i], scene.jointB[i], i);
    for (int i = 0; i < scene.springCount; i++)
        edge(scene.springA[i], scene.springB[i], -1);
}

// grabSurvivors over a prebuilt topology; slot 0 is the seed. A group whose joints cannot
// fracture needs no readback at all, which is what keeps holding a rope or bridge free.
inline int grabSurvivorsCached(VkWorld &world, const GrabTopology &t, std::vector<unsigned char> &keep)
{
    keep.assign(t.slotCount, 0);
    if (t.slotCount == 0)
        return 0;
    std::vector<unsigned char> broken;
    if (!t.breakable.empty())
        world.downloadJointBroken(broken);

    std::vector<std::vector<int>> adj(t.slotCount);
    for (size_t e = 0; e < t.a.size(); e++)
    {
        if (!grabJointIntact(broken, t.joint[e]))
            continue;
        adj[t.a[e]].push_back(t.b[e]);
        adj[t.b[e]].push_back(t.a[e]);
    }
    std::vector<int> queue;
    keep[0] = 1;
    queue.push_back(0);
    int found = 1;
    for (size_t head = 0; head < queue.size(); head++)
        for (int nb : adj[queue[head]])
            if (!keep[nb])
            {
                keep[nb] = 1;
                found++;
                queue.push_back(nb);
            }
    return found;
}

} // namespace avbdvk
