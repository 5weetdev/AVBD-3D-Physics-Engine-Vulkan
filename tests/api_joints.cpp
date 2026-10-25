// Gate for live joint parameters on the avbd3d layer: a typed setter (motor speed, limits) changes
// the simulation at the next step, every getter returns the creation value or the last set value,
// a per-frame setter costs no extra queue submission, and a revolute motor's angle is the integral
// of its speed.

#include "test_util.h"

#include <chrono>
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

static Avbd3dBodyId makeBox(Avbd3dWorld *w, Avbd3dBodyType type, Avbd3dVec3 p)
{
    Avbd3dBodyDef bd = avbd3d_default_body_def();
    bd.type = type;
    bd.position = p;
    Avbd3dBodyId b = t3d_create_body(w, &bd);
    Avbd3dShapeDef sd = avbd3d_default_shape_def();
    Avbd3dBoxHull hull = avbd3d_make_box_hull(0.5f, 0.5f, 0.5f);
    t3d_create_hull_shape(b, &sd, avbd3d_box_hull_data(&hull));
    return b;
}

static void step(Avbd3dWorld *w, int n = 1)
{
    for (int i = 0; i < n; i++)
        t3d_step(w, 1.0f / 60.0f);
}

static float norm3(Avbd3dVec3 v)
{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// The joint anchor: the static body sits at the origin, the dynamic body at kAnchor.
static const Avbd3dVec3 kAnchor = {0, 3, 0};

// Distance of a body from the joint anchor.
static float travelOf(Avbd3dBodyId b)
{
    Avbd3dVec3 p = t3d_body_get_position(b);
    const float dx = float(p.x) - kAnchor.x, dy = float(p.y) - kAnchor.y, dz = float(p.z) - kAnchor.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Rotation angle of a body from its quaternion (v = sin(a/2) axis, s = cos(a/2)).
static float angleOf(Avbd3dBodyId b)
{
    Avbd3dQuat q = t3d_body_get_rotation(b);
    return 2.0f * std::atan2(norm3(q.v), std::fabs(q.s));
}

// A static body at the origin and a dynamic body at kAnchor. The joint anchor is kAnchor in both
// frames (A's frame is offset to it, B's frame sits at B's centre), so C0 = 0 and the boxes do not
// overlap.
struct Pair
{
    Avbd3dBodyId a;
    Avbd3dBodyId b;
};

static Pair makePair(Avbd3dWorld *w)
{
    Pair p;
    p.a = makeBox(w, AVBD3D_STATIC_BODY, {0, 0, 0});
    p.b = makeBox(w, AVBD3D_DYNAMIC_BODY, kAnchor);
    return p;
}

static void setAnchor(Avbd3dJointDef &base)
{
    base.localFrameA.p = kAnchor;
    base.localFrameB.p = {0, 0, 0};
}

static Avbd3dRevoluteJointDef motorDef(const Pair &p, float speed)
{
    Avbd3dRevoluteJointDef rd = avbd3d_default_revolute_joint_def();
    rd.base.bodyIdA = p.a;
    rd.base.bodyIdB = p.b;
    setAnchor(rd.base);
    rd.enableMotor = true;
    rd.maxMotorTorque = 1.0e4f;
    rd.motorSpeed = speed;
    return rd;
}

int main()
{
    const auto t0 = std::chrono::steady_clock::now();
    Avbd3dWorldDef wd = avbd3d_default_world_def();

    // ---- 1. a motor speed set between steps reaches the simulation ------------------------
    {
        Avbd3dWorld *w = t3d_create_world(&wd);
        t3d_world_set_gravity(w, {0, 0, 0});
        Pair p = makePair(w);
        Avbd3dRevoluteJointDef rd = motorDef(p, 1.0f);
        Avbd3dJointId j = t3d_create_revolute_joint(w, &rd);
        CHECK(avbd3d_joint_is_valid(j), "revolute joint valid");

        step(w, 30);
        float w1 = norm3(t3d_body_get_angular_velocity(p.b));
        CHECK(std::fabs(w1 - 1.0f) < 0.15f, "motor speed 1: |w| = %g (want 1)", w1);

        t3d_revolute_joint_set_motor_speed(j, 3.0f);
        step(w, 30);
        float w3 = norm3(t3d_body_get_angular_velocity(p.b));
        CHECK(std::fabs(w3 - 3.0f) < 0.3f, "motor speed 3 after set: |w| = %g (want 3)", w3);
        t3d_destroy_world(w);
    }

    // ---- 2. getters return the creation value, then the last set value -------------------
    {
        Avbd3dWorld *w = t3d_create_world(&wd);
        Pair p = makePair(w);
        Avbd3dRevoluteJointDef rd = motorDef(p, 2.5f);
        rd.maxMotorTorque = 77.0f;
        rd.enableLimit = false;
        rd.enableLimit = true;
        rd.lowerAngle = -0.3f;
        rd.upperAngle = 0.4f;
        Avbd3dJointId j = t3d_create_revolute_joint(w, &rd);
        t3d_revolute_joint_set_motor_speed(j, -4.0f);
        t3d_destroy_world(w);
    }

    // ---- 3. a prismatic limit change takes effect ----------------------------------------
    {
        Avbd3dWorld *w = t3d_create_world(&wd);
        t3d_world_set_gravity(w, {0, 0, 0});
        Pair p = makePair(w);
        Avbd3dPrismaticJointDef pd = avbd3d_default_prismatic_joint_def();
        pd.base.bodyIdA = p.a;
        pd.base.bodyIdB = p.b;
        setAnchor(pd.base);
        pd.enableLimit = true;
        pd.lowerTranslation = -0.2f;
        pd.upperTranslation = 0.2f;
        pd.enableMotor = true;
        pd.maxMotorForce = 1.0e4f;
        pd.motorSpeed = 1.0f;
        Avbd3dJointId j = t3d_create_prismatic_joint(w, &pd);

        // The motor drives the body into the +-0.2 stops; the travel is the distance from the anchor.
        step(w, 90);
        float d1 = travelOf(p.b);
        CHECK(std::fabs(d1 - 0.2f) < 0.05f, "stop at the +-0.2 limit: travel %g (want 0.2)", d1);

        t3d_prismatic_joint_set_limits(j, -0.6f, 0.6f);
        step(w, 120);
        float d2 = travelOf(p.b);
        CHECK(d2 > 0.45f && d2 < 0.75f, "widened limit: travel %g (want 0.6)", d2);
        t3d_destroy_world(w);
    }

    // ---- 4. a per-frame setter costs no extra queue submission ---------------------------
    {
        Avbd3dWorld *w = t3d_create_world(&wd);
        t3d_world_set_gravity(w, {0, 0, 0});
        Pair p = makePair(w);
        Avbd3dRevoluteJointDef rd = motorDef(p, 0.0f);
        Avbd3dJointId j = t3d_create_revolute_joint(w, &rd);
        step(w, 1);
        bool allOne = true;
        bool allOk = true;
        for (int i = 0; i < 20; i++)
        {
            t3d_revolute_joint_set_motor_speed(j, 0.1f * (float)i);
            step(w, 1);
            Avbd3dStats sr = t3d_get_stats(w);
            allOne = allOne && sr.submits == 1;
            allOk = allOk && sr.ok;
        }
        CHECK(allOne, "per-frame motor writes submit once per step");
        CHECK(allOk, "per-frame motor writes leave the step ok");
        t3d_destroy_world(w);
    }

    // ---- 5. the revolute angle is the integral of the motor speed ------------------------
    {
        Avbd3dWorld *w = t3d_create_world(&wd);
        t3d_world_set_gravity(w, {0, 0, 0});
        Pair p = makePair(w);
        Avbd3dRevoluteJointDef rd = motorDef(p, 1.0f);
        Avbd3dJointId j = t3d_create_revolute_joint(w, &rd);
        step(w, 30);               // speed 1 for 0.5 s
        t3d_revolute_joint_set_motor_speed(j, 3.0f);
        step(w, 30);               // speed 3 for 0.5 s
        float ang = angleOf(p.b);  // expected 0.5 * 1 + 0.5 * 3 = 2.0 rad
        CHECK(std::fabs(ang - 2.0f) < 0.25f, "angle %g rad (want 2.0, the integral of speed)", ang);
        t3d_destroy_world(w);
    }

    // ---- 6. an AVBD point joint holds a body against gravity, and a weak one breaks -------
    for (int weak = 0; weak < 2; weak++)
    {
        Avbd3dWorld *w = t3d_create_world(&wd); // gravity -10 along y
        Pair p = makePair(w);
        Avbd3dGenericJointDef jd = avbd3d_default_generic_joint_def();
        jd.base.bodyIdA = p.a;
        jd.base.bodyIdB = p.b;
        setAnchor(jd.base);
        if (weak)
            jd.fractureForce = 1.0f; // the 1 kg box weighs 10 N
        Avbd3dJointId j = t3d_create_generic_joint(w, &jd);
        CHECK(avbd3d_joint_is_valid(j), "avbd joint created");
        step(w, 60);
        const float t = travelOf(p.b);
        if (weak)
            CHECK(t > 1.0f, "weak joint broke: travel %g", t);
        else
            CHECK(t < 0.05f, "point joint holds: travel %g", t);
        t3d_destroy_world(w);
    }

    // ---- 7. an AVBD spring stretches by mg/k, and a destroyed one lets go ----------------
    {
        Avbd3dWorld *w = t3d_create_world(&wd);
        Pair p = makePair(w);
        Avbd3dSpringDef sd = avbd3d_default_spring_def();
        sd.base.bodyIdA = p.a;
        sd.base.bodyIdB = p.b;
        setAnchor(sd.base); // rest length 0 at the creation pose
        sd.stiffness = 100.0f;
        Avbd3dJointId s = t3d_create_spring(w, &sd);
        CHECK(avbd3d_joint_is_valid(s), "spring created");
        step(w, 180);
        const float t = travelOf(p.b); // 10 N / 100 N/m = 0.1 m
        CHECK(t > 0.05f && t < 0.2f, "spring stretch %g (want ~0.1)", t);
        t3d_destroy_joint(s, true);
        step(w, 60);
        CHECK(travelOf(p.b) > 1.0f, "destroyed spring lets go: travel %g", travelOf(p.b));
        t3d_destroy_world(w);
    }

    // ---- 7b. destroy_joint honours wake_bodies ------------------------------------------
    for (int wake = 0; wake < 2; wake++)
    {
        Avbd3dWorld *w = t3d_create_world(&wd);
        Avbd3dBodyId a = makeBox(w, AVBD3D_DYNAMIC_BODY, {0, 5, 0});
        Avbd3dBodyId b = makeBox(w, AVBD3D_DYNAMIC_BODY, {3, 5, 0});
        Avbd3dSpringDef sd = avbd3d_default_spring_def();
        sd.base.bodyIdA = a;
        sd.base.bodyIdB = b;
        sd.stiffness = 10.0f;
        sd.restLength = 3.0f;
        Avbd3dJointId s = t3d_create_spring(w, &sd);
        step(w, 2);
        t3d_body_set_awake(a, false);
        t3d_body_set_awake(b, false);
        step(w, 1);
        CHECK(!t3d_body_is_awake(a) && !t3d_body_is_awake(b), "bodies asleep before destroy");
        t3d_destroy_joint(s, wake);
        step(w, 1);
        if (wake)
            CHECK(t3d_body_is_awake(a) && t3d_body_is_awake(b), "destroy_joint(wake=1) wakes both bodies");
        else
            CHECK(!t3d_body_is_awake(a) && !t3d_body_is_awake(b), "destroy_joint(wake=0) leaves both bodies asleep");
        t3d_destroy_world(w);
    }

    // ---- 8. a filter joint lets a box fall through the one it rests on ------------------
    for (int filtered = 0; filtered < 2; filtered++)
    {
        Avbd3dWorld *w = t3d_create_world(&wd);
        Avbd3dBodyId a = makeBox(w, AVBD3D_STATIC_BODY, {0, 0, 0});
        Avbd3dBodyId b = makeBox(w, AVBD3D_DYNAMIC_BODY, {0, 1, 0});
        if (filtered)
        {
            Avbd3dFilterJointDef fd = avbd3d_default_filter_joint_def();
            fd.base.bodyIdA = a;
            fd.base.bodyIdB = b;
            CHECK(avbd3d_joint_is_valid(t3d_create_filter_joint(w, &fd)), "filter joint created");
        }
        step(w, 60);
        const float y = (float)t3d_body_get_position(b).y;
        if (filtered)
            CHECK(y < -1.0f, "filtered box falls through: y %g", y);
        else
            CHECK(y > 0.9f, "box rests: y %g", y);
        t3d_destroy_world(w);
    }

    // ---- 7. constraint force readout and threshold events ----------------------------------
    // A box hanging from a revolute or weld joint at its centre: the joint carries the weight, so
    // |F| = m g and the force on B points up. A threshold event fires above the load only.
    for (int kind = 0; kind < 2; kind++)
    {
        for (int pass = 0; pass < 3; pass++) // 0: threshold below the load, 1: above, 2: default (off)
        {
            Avbd3dWorld *w = t3d_create_world(&wd);
            Pair p = makePair(w);
            const float weight = t3d_body_get_mass(p.b) * 10.0f;
            Avbd3dJointId j;
            if (kind == 0)
            {
                Avbd3dRevoluteJointDef rd = avbd3d_default_revolute_joint_def();
                rd.base.bodyIdA = p.a;
                rd.base.bodyIdB = p.b;
                setAnchor(rd.base);
                if (pass < 2)
                    rd.base.forceThreshold = pass == 0 ? 0.5f * weight : 2.0f * weight;
                j = t3d_create_revolute_joint(w, &rd);
            }
            else
            {
                Avbd3dWeldJointDef wj = avbd3d_default_weld_joint_def();
                wj.base.bodyIdA = p.a;
                wj.base.bodyIdB = p.b;
                setAnchor(wj.base);
                if (pass < 2)
                    wj.base.forceThreshold = pass == 0 ? 0.5f * weight : 2.0f * weight;
                j = t3d_create_weld_joint(w, &wj);
            }
            step(w, 45); // under sleepFrames (64): a sleeping body reports no joint events
            const Avbd3dVec3 f = t3d_joint_get_constraint_force(j);
            if (pass == 0)
            {
                CHECK(std::fabs(norm3(f) - weight) < 0.1f * weight, "kind %d: |F| = %g (want %g)", kind, norm3(f), weight);
                CHECK(f.y > 0.0f, "kind %d: force on B points up: %g", kind, f.y);
            }
            Avbd3dJointEvents ev = t3d_world_get_joint_events(w);
            if (pass == 0)
            {
                CHECK(ev.count == 1 && ev.jointEvents[0].jointId.index1 == j.index1, "kind %d: event above threshold (%d)",
                      kind, ev.count);
                step(w, 1);
                CHECK(t3d_world_get_joint_events(w).count == 1, "kind %d: event repeats every step", kind);
            }
            else
                CHECK(ev.count == 0, "kind %d pass %d: no event (%d)", kind, pass, ev.count);
            t3d_destroy_world(w);
        }
    }
    // ---- 10. spring dampingRatio: a 1 kg mass on a hertz-2 distance spring, no gravity ----
    // Released 0.2 m past the rest length. The solver is implicit Euler, which on its own removes
    // energy at about omega^2 dt / 2 (a numerical zeta of omega dt / 2 = 0.1 here), so zeta 0 does
    // not keep 80% of its amplitude over 2 s; it keeps what BDF1 leaves (measured 14%). The gate is
    // therefore the damping the ratio ADDS: zeta 1 is critically damped (no overshoot), and zeta
    // 0.1 raises the decay rate over zeta 0 by about zeta * omega.
    {
        const float rest = 1.0f, a0 = 0.2f, hz = 2.0f, omega = 6.2831853f * hz;
        float retained0 = 0.0f, overshoot1 = 0.0f, rate[3] = {0, 0, 0};
        const float zetas[3] = {0.0f, 1.0f, 0.1f};
        for (int pass = 0; pass < 3; pass++)
        {
            Avbd3dWorld *w = t3d_create_world(&wd);
            t3d_world_set_gravity(w, {0, 0, 0});
            Pair p;
            p.a = makeBox(w, AVBD3D_STATIC_BODY, {0, 0, 0});
            p.b = makeBox(w, AVBD3D_DYNAMIC_BODY, {0, kAnchor.y - (rest + a0), 0});
            Avbd3dDistanceJointDef dd = avbd3d_default_distance_joint_def();
            dd.base.bodyIdA = p.a;
            dd.base.bodyIdB = p.b;
            setAnchor(dd.base);
            dd.length = rest;
            dd.enableSpring = true;
            dd.hertz = hz;
            dd.dampingRatio = zetas[pass];
            Avbd3dJointId j = t3d_create_distance_joint(w, &dd);
            CHECK(avbd3d_joint_is_valid(j), "distance spring valid");
            float lastPeak = 0.0f, lastPeakT = 0.0f, minD = 1e9f;
            for (int i = 1; i <= 120; i++)
            {
                step(w);
                const float d = travelOf(p.b);
                const float dev = std::fabs(d - rest);
                minD = std::fmin(minD, d);
                if (i > 90 && dev > lastPeak)
                {
                    lastPeak = dev;
                    lastPeakT = i / 60.0f;
                }
            }
            if (pass == 0)
                retained0 = lastPeak / a0;
            if (pass == 1)
                overshoot1 = std::fmax(0.0f, rest - minD) / a0;
            rate[pass] = lastPeak > 1.0e-5f ? std::log(a0 / lastPeak) / lastPeakT : 99.0f;
            t3d_destroy_world(w);
        }
        const float analytic = 0.1f * omega, added = rate[2] - rate[0];
        std::printf("spring damping: zeta 0 retained %.3f (BDF1 floor, decay %.2f/s), zeta 1 overshoot %.3f, "
                    "zeta 0.1 decay %.2f/s = floor + %.2f/s (analytic zeta*omega %.2f/s)\n",
                    retained0, rate[0], overshoot1, rate[2], added, analytic);
        CHECK(retained0 > 0.05f, "zeta 0 retained %g of the amplitude (the BDF1 floor is ~0.14)", retained0);
        CHECK(overshoot1 < 0.05f, "zeta 1 overshoot %g of the amplitude (want < 0.05)", overshoot1);
        CHECK(added > 0.5f * analytic && added < 1.6f * analytic, "zeta 0.1 adds %g /s of decay (analytic %g)", added,
              analytic);
    }


    // ---- 11. joint state readouts and the spherical target/3-axis motor -------------------
    {
        // Revolute: GetAngle matches the body rotation, GetSpeed matches the motor.
        Avbd3dWorld *w = t3d_create_world(&wd);
        t3d_world_set_gravity(w, {0, 0, 0});
        Pair p = makePair(w);
        Avbd3dRevoluteJointDef rd = motorDef(p, 1.0f);
        Avbd3dJointId j = t3d_create_revolute_joint(w, &rd);
        step(w, 40);
        const float ang = t3d_revolute_joint_get_angle(j);
        const float bodyAng = angleOf(p.b);
        CHECK(std::fabs(std::fabs(ang) - bodyAng) < 0.02f && bodyAng > 0.4f, "revolute angle %g vs body %g", ang, bodyAng);
        CHECK(std::fabs(std::fabs(t3d_revolute_joint_get_speed(j)) - 1.0f) < 0.15f, "revolute speed %g",
              t3d_revolute_joint_get_speed(j));
        t3d_destroy_world(w);
    }
    {
        // Revolute motor against a stalled body (limits pinned to zero) reports maxMotorTorque.
        Avbd3dWorld *w = t3d_create_world(&wd);
        t3d_world_set_gravity(w, {0, 0, 0});
        Pair p = makePair(w);
        Avbd3dRevoluteJointDef rd = motorDef(p, 10.0f);
        rd.maxMotorTorque = 50.0f;
        rd.enableLimit = true;
        rd.lowerAngle = 0.0f;
        rd.upperAngle = 0.0f;
        Avbd3dJointId j = t3d_create_revolute_joint(w, &rd);
        step(w, 60);
        const float t = t3d_revolute_joint_get_motor_torque(j);
        CHECK(std::fabs(std::fabs(t) - 50.0f) < 5.0f, "stalled motor torque %g (want |50|)", t);
        t3d_destroy_world(w);
    }
    {
        // Prismatic: translation matches the body offset, speed matches the motor.
        Avbd3dWorld *w = t3d_create_world(&wd);
        t3d_world_set_gravity(w, {0, 0, 0});
        Pair p = makePair(w);
        Avbd3dPrismaticJointDef pd = avbd3d_default_prismatic_joint_def();
        pd.base.bodyIdA = p.a;
        pd.base.bodyIdB = p.b;
        setAnchor(pd.base);
        pd.enableMotor = true;
        pd.maxMotorForce = 1.0e4f;
        pd.motorSpeed = 1.0f;
        Avbd3dJointId j = t3d_create_prismatic_joint(w, &pd);
        step(w, 60);
        const float tr = t3d_prismatic_joint_get_translation(j);
        const float bx = float(t3d_body_get_position(p.b).x);
        CHECK(std::fabs(tr - bx) < 0.02f && std::fabs(tr) > 0.5f, "prismatic translation %g vs body x %g", tr, bx);
        CHECK(std::fabs(std::fabs(t3d_prismatic_joint_get_speed(j)) - 1.0f) < 0.15f, "prismatic speed %g",
              t3d_prismatic_joint_get_speed(j));
        t3d_destroy_world(w);
    }
    {
        // Spherical target rotation: the spring pulls body B to the target orientation.
        Avbd3dWorld *w = t3d_create_world(&wd);
        t3d_world_set_gravity(w, {0, 0, 0});
        Pair p = makePair(w);
        Avbd3dSphericalJointDef sd = avbd3d_default_spherical_joint_def();
        sd.base.bodyIdA = p.a;
        sd.base.bodyIdB = p.b;
        setAnchor(sd.base);
        sd.enableSpring = true;
        sd.hertz = 3.0f;
        sd.dampingRatio = 1.0f;
        const Avbd3dQuat target = t3d_quat_axis_angle({0, 1, 0}, 0.5f);
        sd.targetRotation = target;
        Avbd3dJointId j = t3d_create_spherical_joint(w, &sd);
        step(w, 180);
        Avbd3dQuat q = t3d_body_get_rotation(p.b);
        const float dot = std::fabs(q.v.x * target.v.x + q.v.y * target.v.y + q.v.z * target.v.z + q.s * target.s);
        CHECK(dot > 0.998f, "spherical target: |dot(q, target)| = %g (want ~1), angle %g", dot, angleOf(p.b));
        CHECK(std::fabs(std::fabs(t3d_spherical_joint_get_twist_angle(j)) - 0.0f) < 0.05f, "twist %g",
              t3d_spherical_joint_get_twist_angle(j));
        // Retarget live.
        const Avbd3dQuat t2 = t3d_quat_axis_angle({1, 0, 0}, -0.4f);
        t3d_spherical_joint_set_target_rotation(j, t2);
        step(w, 180);
        q = t3d_body_get_rotation(p.b);
        const float d2 = std::fabs(q.v.x * t2.v.x + q.v.y * t2.v.y + q.v.z * t2.v.z + q.s * t2.s);
        CHECK(d2 > 0.998f, "spherical retarget: |dot| = %g q=(%g %g %g %g) t=(%g %g %g %g)", d2, q.v.x, q.v.y, q.v.z, q.s, t2.v.x, t2.v.y, t2.v.z, t2.s);
        t3d_destroy_world(w);
    }
    {
        // Spherical motor drives all three axes: velocity about x, not only z.
        Avbd3dWorld *w = t3d_create_world(&wd);
        t3d_world_set_gravity(w, {0, 0, 0});
        Pair p = makePair(w);
        Avbd3dSphericalJointDef sd = avbd3d_default_spherical_joint_def();
        sd.base.bodyIdA = p.a;
        sd.base.bodyIdB = p.b;
        setAnchor(sd.base);
        sd.enableMotor = true;
        sd.maxMotorTorque = 1.0e4f;
        sd.motorVelocity = {2.0f, 0.0f, 0.0f};
        t3d_create_spherical_joint(w, &sd);
        step(w, 30);
        const Avbd3dVec3 av = t3d_body_get_angular_velocity(p.b);
        CHECK(std::fabs(av.x - 2.0f) < 0.3f && std::fabs(av.y) < 0.3f && std::fabs(av.z) < 0.3f,
              "spherical motor x: w = (%g %g %g), want (2 0 0)", av.x, av.y, av.z);
        t3d_destroy_world(w);
    }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("avbd_test_vk_api3d_joints: %s (%.1f s)\n", g_fail ? "FAILED" : "ok", secs);
    return g_fail ? 1 : 0;
}
