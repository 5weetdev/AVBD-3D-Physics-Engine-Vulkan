// Grab on the avbd3d world layer: the topology walk runs on the host mirror (live joints, springs,
// mirror poses); the handles are VkWorld's drag joint pool, staged on the host and landed by the
// next step.
#include "avbd3d_world.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

using namespace avbd3d_impl;

namespace
{

vec3 posToV(avbd3d_Pos p) { return vec3{(float)p.x, (float)p.y, (float)p.z}; }

} // namespace

namespace avbd3d_impl
{

// ---- grab ---------------------------------------------------------------------------------

int World::grabGroup(int seed, vec3 center, float radius, bool follow, avbd3d_BodyId *out, int maxCount)
{
    syncBodies(); // the flood tests every reached body's position
    const int cap = std::min(maxCount, AVBD3D_MAX_GRAB_BODIES);
    if (!out || cap <= 0)
        return 0;
    const SceneDesc &d = desc;
    auto holdable = [&](int b) {
        const BodyRec &r = bodies[b];
        return r.alive && r.enabled && r.type == avbd3d_dynamicBody && d.mass[b] > 0.0f;
    };
    if (!holdable(seed))
        return 0;
    int n = 0;
    out[n++] = bodyId(seed);
    if (!follow || radius <= 0.0f || n >= cap)
        return n;

    // Adjacency over live joints and springs.
    const int count = d.count;
    std::vector<std::vector<int>> adj(count);
    auto link = [&](int a, int b) {
        if (a <= 0 || b <= 0 || a >= count || b >= count || !bodies[a].alive || !bodies[b].alive)
            return;
        adj[a].push_back(b);
        adj[b].push_back(a);
    };
    for (const JointRec &j : joints)
        if (j.alive && j.rowCount > 0) // a filter joint connects nothing; springs are linked below
            link(j.a, j.b);
    for (int k = 0; k < d.springCount; k++)
        if (d.springK[k] > 0.0f) // a destroyed spring
            link(d.springA[k], d.springB[k]);

    const float r2 = radius * radius;
    std::vector<unsigned char> seen(count, 0);
    seen[seed] = 1;
    std::vector<int> queue{seed};
    for (size_t head = 0; head < queue.size() && n < cap; head++)
        for (int nb : adj[queue[head]])
        {
            if (seen[nb])
                continue;
            seen[nb] = 1;
            // A pinned anchor would fight the handle and cannot move anyway.
            if (!holdable(nb) || lengthSq(originPos(nb) - center) > r2)
                continue;
            out[n++] = bodyId(nb);
            queue.push_back(nb);
            if (n >= cap)
                break;
        }
    return n;
}

int World::grabSurvivors(int seed, const avbd3d_BodyId *group, int count, uint8_t *keep)
{
    if (!group || !keep || count <= 0)
        return 0;
    std::fill(keep, keep + count, (uint8_t)0);
    std::unordered_map<int, int> slotOf;
    slotOf.reserve((size_t)count * 2);
    for (int i = 0; i < count; i++)
        if (bodyValid(group[i]))
            slotOf.emplace(group[i].index1, i);
    auto it = slotOf.find(seed);
    if (it == slotOf.end())
        return 0;

    // A joint holds while none of its rows has broken (the device's flags; one readback).
    std::vector<unsigned char> broken;
    if (m_error == avbd3d_errorNone && m_vkJoints > 0)
    {
        try
        {
            settle();
            m_vk.downloadJointBroken(broken);
        }
        catch (...)
        {
            noteException();
        }
    }
    std::vector<std::vector<int>> adj((size_t)count);
    auto link = [&](int a, int b) {
        auto ia = slotOf.find(a), ib = slotOf.find(b);
        if (ia == slotOf.end() || ib == slotOf.end())
            return;
        adj[ia->second].push_back(ib->second);
        adj[ib->second].push_back(ia->second);
    };
    for (const JointRec &j : joints)
    {
        if (!j.alive || j.rowCount == 0)
            continue;
        bool intact = true;
        for (int row = j.firstRow; row < j.firstRow + j.rowCount; row++)
            if (row < (int)broken.size() && broken[row])
                intact = false;
        if (intact)
            link(j.a, j.b);
    }
    for (int k = 0; k < desc.springCount; k++)
        if (desc.springK[k] > 0.0f)
            link(desc.springA[k], desc.springB[k]);

    int n = 0;
    std::vector<int> queue{it->second};
    keep[it->second] = 1;
    n++;
    for (size_t head = 0; head < queue.size(); head++)
        for (int nb : adj[queue[head]])
            if (!keep[nb])
            {
                keep[nb] = 1;
                n++;
                queue.push_back(nb);
            }
    return n;
}

void World::grabDrag(const avbd3d_GrabHandle *h, int count)
{
    if (!h || count <= 0 || m_error != avbd3d_errorNone)
        return;
    count = std::min(count, AVBD3D_MAX_GRAB_BODIES);
    int ids[AVBD3D_MAX_GRAB_BODIES];
    vec3 targets[AVBD3D_MAX_GRAB_BODIES], locals[AVBD3D_MAX_GRAB_BODIES];
    float weights[AVBD3D_MAX_GRAB_BODIES];
    int n = 0;
    for (int k = 0; k < count; k++)
    {
        if (!bodyValid(h[k].body))
            continue;
        const int b = h[k].body.index1;
        const BodyRec &r = bodies[b];
        // A body not yet on the device is held from the next grab call on.
        if (!r.enabled || b >= m_vkBodies)
            continue;
        ids[n] = b;
        targets[n] = posToV(h[k].target);
        // Body origin frame -> the device frame (centre of mass, principal axes).
        locals[n] = rotate(conjugate(desc.bodyPrincipalRot[b]), toV(h[k].localAnchor) - desc.bodyLocalCenter[b]);
        weights[n] = std::min(std::max(h[k].weight, 0.0f), 1.0f);
        n++;
    }
    if (n == 0)
        return;
    try
    {
        settle();
        m_vk.dragBodies(ids, targets, locals, n, weights, params.grabStrength);
    }
    catch (...)
    {
        noteException();
    }
}

void World::grabRelease()
{
    if (m_error != avbd3d_errorNone)
        return;
    try
    {
        settle();
        m_vk.releaseDrag();
    }
    catch (...)
    {
        noteException();
    }
}

} // namespace avbd3d_impl

// ---- C API --------------------------------------------------------------------------------

extern "C"
{

int avbd3d_World_GrabGroup(avbd3d_WorldId id, avbd3d_BodyId seed, avbd3d_Pos center, float radius,
                           bool followConnections, avbd3d_BodyId *outBodies, int maxCount)
{
    World *w = settled(id);
    if (!w || !w->bodyValid(seed))
        return 0;
    return w->grabGroup(seed.index1, posToV(center), radius, followConnections, outBodies, maxCount);
}

int avbd3d_World_GrabSurvivors(avbd3d_WorldId id, avbd3d_BodyId seed, const avbd3d_BodyId *group, int groupCount,
                               uint8_t *outKeep)
{
    World *w = settled(id);
    if (!w || !w->bodyValid(seed))
    {
        if (outKeep && groupCount > 0)
            std::memset(outKeep, 0, (size_t)groupCount);
        return 0;
    }
    return w->grabSurvivors(seed.index1, group, groupCount, outKeep);
}

void avbd3d_World_GrabDrag(avbd3d_WorldId id, const avbd3d_GrabHandle *handles, int count)
{
    if (World *w = settled(id))
        w->grabDrag(handles, count);
}

void avbd3d_World_GrabRelease(avbd3d_WorldId id)
{
    if (World *w = settled(id))
        w->grabRelease();
}

} // extern "C"
