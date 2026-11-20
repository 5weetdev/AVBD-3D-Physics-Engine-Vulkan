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

#include "scene_types.h"

#include <cmath>
#include <vector>

// Multi-body objects thrown into a running scene by the spawn tool.
//
// The pieces are authored in their own frame and only placed at the end. That is what
// lets a scene builder be reused as a spawnable object: the passenger plane is written
// around a runway at the origin, and SpawnEmitter collects the same calls, recentres
// them on the assembly's own bounding box, scales, rotates and launches the result.
//
// The emitter deliberately mirrors the three SceneBuilder methods a scene uses -- clear(),
// addBody() and addJointWorld() -- so a scene function templated on its solver type can
// be handed one of these instead, with no second copy of the geometry to keep in step.
struct SpawnEmitter
{
    struct Part
    {
        vec3 size;
        float density;
        float friction;
        vec3 position; // in the emitter's own authoring frame
        quat rotation;
        bool dynamic;
    };

    struct Link
    {
        int a, b; // indices into parts, or -1 for a world anchor
        vec3 anchor;
        float stiffnessLin, stiffnessAng, fracture;
    };

    std::vector<Part> parts;
    std::vector<Link> links;

    // Scene builders open with clear(); for a spawned copy that just means starting the
    // collection over, not touching the scene it is about to be thrown into.
    void clear()
    {
        parts.clear();
        links.clear();
    }

    int addBody(vec3 size, float density, float friction, vec3 position, vec3 velocity = vec3{0, 0, 0},
                vec3 angularVelocity = vec3{0, 0, 0}, quat rotation = quat{0, 0, 0, 1})
    {
        (void)velocity;
        (void)angularVelocity;
        parts.push_back(Part{size, density, friction, position, rotation, density > 0.0f});
        return (int)parts.size() - 1;
    }

    int addJointWorld(int bodyA, int bodyB, vec3 worldAnchor, float stiffnessLin = INFINITY, float stiffnessAng = 0.0f,
                      float fracture = INFINITY)
    {
        links.push_back(Link{bodyA, bodyB, worldAnchor, stiffnessLin, stiffnessAng, fracture});
        return (int)links.size() - 1;
    }

    // Centre of the dynamic parts' bounding box, which is what the assembly is placed by.
    // The static scenery a scene builder puts in (the plane's runway) is left out of it
    // and dropped at commit time -- a thrown airliner should not bring a runway with it.
    vec3 centre() const
    {
        vec3 lo{0, 0, 0}, hi{0, 0, 0};
        bool any = false;
        for (const Part &p : parts)
        {
            if (!p.dynamic)
                continue;
            const vec3 h = p.size * 0.5f;
            const vec3 a = p.position - h, b = p.position + h;
            if (!any)
            {
                lo = a;
                hi = b;
                any = true;
            }
            else
            {
                lo = vec3{min(lo.x, a.x), min(lo.y, a.y), min(lo.z, a.z)};
                hi = vec3{max(hi.x, b.x), max(hi.y, b.y), max(hi.z, b.z)};
            }
        }
        return any ? (lo + hi) * 0.5f : vec3{0, 0, 0};
    }

    // Half the bounding diagonal about that centre, so the caller can keep the spawn
    // point far enough out that the object does not appear inside the camera.
    float radius() const
    {
        const vec3 c = centre();
        float r = 0.0f;
        for (const Part &p : parts)
        {
            if (!p.dynamic)
                continue;
            r = max(r, length(p.position - c) + length(p.size * 0.5f));
        }
        return r;
    }

    // Place the assembly and hand it to the solver.
    //
    // massMultiplier scales the density of all parts. Since size is also scaled,
    // the total mass of the object will scale with size^3 * massMultiplier.
    template <typename Solver>
    int commit(Solver &solver, vec3 origin, quat rotation, float scale, float massMultiplier, vec3 velocity,
               vec3 angularVelocity = vec3{0, 0, 0}) const
    {
        const vec3 c = centre();
        const float densityScale = massMultiplier;

        std::vector<SpawnBody> bodies;
        std::vector<int> remap(parts.size(), -1);
        bodies.reserve(parts.size());

        for (size_t i = 0; i < parts.size(); i++)
        {
            const Part &p = parts[i];
            if (!p.dynamic)
                continue; // scenery belongs to the scene, not to the object
            SpawnBody b;
            b.size = p.size * scale;
            b.density = p.density * densityScale;
            b.friction = p.friction;
            b.position = origin + rotate(rotation, (p.position - c) * scale);
            b.rotation = rotation * p.rotation;
            // Rigid-body velocity of the whole assembly, so it flies as one piece rather
            // than shearing itself apart on the first step.
            b.velocity = velocity + cross(angularVelocity, b.position - origin);
            b.angularVelocity = angularVelocity;
            remap[i] = (int)bodies.size();
            bodies.push_back(b);
        }

        if (bodies.empty())
            return -1;

        const int base = solver.bodyCount();

        std::vector<SpawnJoint> joints;
        joints.reserve(links.size());
        for (const Link &l : links)
        {
            // A joint onto a part that was dropped would otherwise become a world
            // anchor, which would nail the object to the sky.
            if (l.b < 0 || l.b >= (int)parts.size() || remap[l.b] < 0)
                continue;
            if (l.a >= 0 && (l.a >= (int)parts.size() || remap[l.a] < 0))
                continue;

            SpawnJoint j;
            j.bodyA = l.a >= 0 ? base + remap[l.a] : -1;
            j.bodyB = base + remap[l.b];
            j.stiffnessLin = l.stiffnessLin;
            // A fracture load is a force and scales with the object's mass; an angular
            // stiffness is a torque per radian, so it takes one more factor of length.
            const float massScale = densityScale * scale * scale * scale;
            j.stiffnessAng = l.stiffnessAng * massScale * scale;
            j.fracture = l.fracture * massScale;

            // Anchors are body-local, so they are only scaled, never rotated: the
            // placement is a rigid transform of both body and anchor alike.
            const vec3 anchorLocalA =
                l.a >= 0 ? rotate(conjugate(parts[l.a].rotation), l.anchor - parts[l.a].position) : l.anchor;
            const vec3 anchorLocalB = rotate(conjugate(parts[l.b].rotation), l.anchor - parts[l.b].position);
            j.rA = l.a >= 0 ? anchorLocalA * scale : origin + rotate(rotation, (l.anchor - c) * scale);
            j.rB = anchorLocalB * scale;
            joints.push_back(j);
        }

        return solver.spawnGroup(bodies.data(), (int)bodies.size(), joints.empty() ? nullptr : joints.data(),
                                 (int)joints.size());
    }
};

// A simple ragdoll in an A pose: one box for the head with no neck, one for the torso,
// and two per limb. Built about 1.7m tall at scale 1, standing on the origin plane, and
// recentred by the emitter when it is placed.
//
// The joints are ball joints -- zero angular stiffness, as the plane's propeller
// bearings are -- so the limbs swing freely and the thing tumbles like a body rather
// than like a piece of furniture.
inline void buildRagdoll(SpawnEmitter &e)
{
    e.clear();

    const float kFriction = 0.6f;
    const float kDensity = 1.0f;
    const float kDensityHead = 1.1f;

    // Torso, X forward, Y across, Z up. Everything else hangs off its measurements.
    const float torsoH = 0.60f;
    const float torsoZ = 1.10f;
    const float torsoTop = torsoZ + torsoH * 0.5f;
    const float torsoBot = torsoZ - torsoH * 0.5f;
    const float halfWidth = 0.18f;

    const int torso = e.addBody({0.22f, halfWidth * 2.0f, torsoH}, kDensity, kFriction, {0.0f, 0.0f, torsoZ});

    // Head sits straight on the shoulders: no neck, so the joint is the contact line.
    const float headH = 0.24f;
    const int head = e.addBody({0.22f, 0.22f, headH}, kDensityHead, kFriction, {0.0f, 0.0f, torsoTop + headH * 0.5f});
    e.addJointWorld(torso, head, {0.0f, 0.0f, torsoTop}, INFINITY, 0.0f);

    // A limb segment: a box whose local +Z runs from the joint down the limb, so the
    // rotation is the one that maps +Z onto dir. dir stays in the YZ plane here, which
    // makes that a single rotation about X.
    auto limb = [&](int parent, vec3 from, vec3 dir, float len, float thick)
    {
        const float theta = atan2f(-dir.y, dir.z);
        const quat rot = quat{sinf(theta * 0.5f), 0.0f, 0.0f, cosf(theta * 0.5f)};
        const int seg =
            e.addBody({thick, thick, len}, kDensity, kFriction, from + dir * (len * 0.5f), {0, 0, 0}, {0, 0, 0}, rot);
        e.addJointWorld(parent, seg, from, INFINITY, 0.0f);
        return seg;
    };

    for (int s = -1; s <= 1; s += 2)
    {
        const float sy = (float)s;

        // Arms out at 45 degrees, which is the A of the pose.
        const float armCos = 0.7071f;
        const vec3 armDir = {0.0f, sy * armCos, -armCos};
        const vec3 shoulder = {0.0f, sy * halfWidth, torsoTop - 0.06f};
        const float upperArm = 0.30f, lowerArm = 0.28f, armThick = 0.10f;
        const int upper = limb(torso, shoulder, armDir, upperArm, armThick);
        limb(upper, shoulder + armDir * upperArm, armDir, lowerArm, armThick * 0.9f);

        // Legs splayed by ten degrees, so the ragdoll stands on a base rather than a line.
        const vec3 legDir = {0.0f, sy * 0.1736f, -0.9848f};
        const vec3 hip = {0.0f, sy * 0.10f, torsoBot};
        const float upperLeg = 0.44f, lowerLeg = 0.42f, legThick = 0.14f;
        const int thigh = limb(torso, hip, legDir, upperLeg, legThick);
        limb(thigh, hip + legDir * upperLeg, legDir, lowerLeg, legThick * 0.85f);
    }
}
