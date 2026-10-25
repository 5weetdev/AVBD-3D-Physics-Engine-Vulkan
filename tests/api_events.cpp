// Gate for the avbd3d contact events and contact data (src/api, Phase 4b-1). Y-up, the same
// convention as test_vk_api3d_query.cpp. Checks:
//   a) begin: a flagged box dropped on the ground gets exactly one begin event, and a flagged box
//      resting on the ground gets one at the first step.
//   b) end: moving a touching box away emits one end event for its pair, and the old contact id
//      goes invalid.
//   c) hit: the falling box's impact is a hit event above the threshold with the contact normal
//      along +-y; a resting box produces none; the threshold round-trips.
//   d) control: a box with no event flags on the same ground produces no events and no contact
//      data.
//   e) contact data: capacity, shape and body getters, the manifold (normal and points) and
//      Contact_IsValid / Contact_GetData agree with each other.
//   f) truncation: 300 flagged boxes on the ground overflow the 256-record buffer. The first step
//      is flagged truncated, drops the excess without ending it, and the next step reports every
//      pair once (total begins 300, no ends).
//   g) destroy: destroying a touching body emits its end event at the next step, and its
//      contact id goes invalid.



// (merged) // Gate for the avbd3d sensor overlap events and the explosion (src/api, Phase 4b-2). Y-up,
// // the same convention as test_vk_api3d_events.cpp. Checks:
// //   a) a static sensor box above static ground, and a dynamic box dropped through it: exactly one
// //      begin event for the pair while inside, one end event after the box lands, and
// //      Shape_GetSensorData lists the visitor only while it is inside.
// //   b) a visitor with enableSensorEvents=false gives no events, and a second sensor overlapping
// //      the first gives no events (sensors do not detect sensors).
// //   c) destroying a visitor while it is inside emits an end event naming it at the next step, and
// //      its shape id goes invalid.
// //   d) explosion with three resting boxes at 1, 2 and 10 m, radius 5: the two inside move
// //      radially outward, the far one is untouched, and the sleeping inside boxes are woken.
// //   e) a world with no sensors produces no sensor events.
// 
// 
// 
#include "test_util.h"
#include <cmath>
#include <cstdio>
#include <vector>

namespace events
{
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

static bool sameShape(Avbd3dShapeId a, Avbd3dShapeId b) { return a.index1 == b.index1 && a.generation == b.generation; }

static Avbd3dQuat identityQuat() { return Avbd3dQuat{Avbd3dVec3{0, 0, 0}, 1.0f}; }

// A box body with one hull shape. `flagged` turns on contact and hit events for the shape.
static Avbd3dBodyId makeBox(Avbd3dWorld *w, Avbd3dBodyType type, Avbd3dVec3 p, float hx, float hy, float hz,
                             bool flagged, Avbd3dShapeId *shapeOut = nullptr)
{
    Avbd3dBodyDef bd = avbd3d_default_body_def();
    bd.type = type;
    bd.position = p;
    Avbd3dBodyId b = t3d_create_body(w, &bd);
    Avbd3dShapeDef sd = avbd3d_default_shape_def();
    sd.enableContactEvents = flagged;
    sd.enableHitEvents = flagged;
    Avbd3dBoxHull hull = avbd3d_make_box_hull(hx, hy, hz);
    Avbd3dShapeId s = t3d_create_hull_shape(b, &sd, avbd3d_box_hull_data(&hull));
    if (shapeOut)
        *shapeOut = s;
    return b;
}

// Totals over the steps run so far, plus the per-event details the checks need.
struct Tally
{
    int begin = 0, end = 0, hit = 0;
    std::vector<Avbd3dShapeId> beginShapes;
    std::vector<Avbd3dShapeId> hitShapes;
    std::vector<Avbd3dShapeId> endShapes; // both shapes of every end event, in pairs
    float hitSpeed = 0, hitNy = 0;
    bool hitNormalAlongY = false;
};

static void stepTally(Avbd3dWorld *w, Tally &t)
{
    t3d_step(w, 1.0f / 60.0f);
    const Avbd3dContactEvents e = t3d_world_get_contact_events(w);
    t.begin += e.beginCount;
    t.end += e.endCount;
    t.hit += e.hitCount;
    for (int i = 0; i < e.beginCount; i++)
        t.beginShapes.push_back(e.beginEvents[i].shapeIdA), t.beginShapes.push_back(e.beginEvents[i].shapeIdB);
    for (int i = 0; i < e.endCount; i++)
        t.endShapes.push_back(e.endEvents[i].shapeIdA), t.endShapes.push_back(e.endEvents[i].shapeIdB);
    for (int i = 0; i < e.hitCount; i++)
    {
        const Avbd3dContactHitEvent &h = e.hitEvents[i];
        t.hitShapes.push_back(h.shapeIdA);
        t.hitShapes.push_back(h.shapeIdB);
        t.hitSpeed = h.approachSpeed;
        t.hitNy = std::fabs(h.normal.y);
        t.hitNormalAlongY = t.hitNy > 0.9f;
    }
}

static int countShape(const std::vector<Avbd3dShapeId> &v, Avbd3dShapeId s)
{
    int n = 0;
    for (size_t i = 0; i < v.size(); i++)
        if (sameShape(v[i], s))
            n++;
    return n;
}

int run()
{
    Avbd3dWorldDef wd = avbd3d_default_world_def();
    Avbd3dWorld *w = t3d_create_world(&wd);
    CHECK((w != NULL), "world create");

    // Ground (unflagged) and the three boxes on it.
    Avbd3dBodyId ground = makeBox(w, AVBD3D_STATIC_BODY, Avbd3dVec3{0, -0.5f, 0}, 10, 0.5f, 10, false);
    (void)ground;
    Avbd3dShapeId s1{}, s2{}, s3{};
    Avbd3dBodyId b1 = makeBox(w, AVBD3D_DYNAMIC_BODY, Avbd3dVec3{0, 2.0f, 0}, 0.5f, 0.5f, 0.5f, true, &s1);
    Avbd3dBodyId b2 = makeBox(w, AVBD3D_DYNAMIC_BODY, Avbd3dVec3{5, 0.5f, 0}, 0.5f, 0.5f, 0.5f, false, &s2);
    Avbd3dBodyId b3 = makeBox(w, AVBD3D_DYNAMIC_BODY, Avbd3dVec3{-5, 0.5f, 0}, 0.5f, 0.5f, 0.5f, true, &s3);

    // ---- c) threshold round-trip -----------------------------------------------------------
    t3d_world_set_hit_event_threshold(w, 1.0f);

    // ---- a, c: fall and settle ---------------------------------------------------------------
    Tally t;
    for (int i = 0; i < 120; i++)
        stepTally(w, t);
    // box1 may bounce on landing: each separation is an end and each re-touch a begin, so only
    // the balance is fixed: begins minus ends is 1 while it rests on the ground.
    const int nBeg1 = countShape(t.beginShapes, s1), nEnd1 = countShape(t.endShapes, s1);
    CHECK(nBeg1 >= 1 && nBeg1 - nEnd1 == 1, "box1 begin/end balance %d/%d (want begin - end == 1)", nBeg1, nEnd1);
    CHECK(countShape(t.beginShapes, s3) == 1, "box3 begin events %d (want 1)", countShape(t.beginShapes, s3));
    CHECK(countShape(t.endShapes, s3) == 0, "resting box3 produced an end event");
    CHECK(countShape(t.beginShapes, s2) == 0, "unflagged box2 produced a begin event");
    CHECK(countShape(t.hitShapes, s1) >= 1, "box1 impact produced no hit event");
    CHECK(countShape(t.hitShapes, s3) == 0, "resting box3 produced a hit event");
    CHECK(countShape(t.hitShapes, s2) == 0, "unflagged box2 produced a hit event");
    CHECK(t.hitSpeed >= 1.0f, "hit approach speed %g below the threshold", t.hitSpeed);
    CHECK(t.hitNormalAlongY, "hit normal not along y (|ny| %g)", t.hitNy);

    // ---- e: contact data getters -------------------------------------------------------------
    CHECK(t3d_shape_get_contact_capacity(s1) == 1, "box1 contact capacity %d (want 1)", t3d_shape_get_contact_capacity(s1));
    Avbd3dContactData sd1[2];
    const int n1 = t3d_shape_get_contact_data(s1, sd1, 2);
    CHECK(n1 == 1, "box1 contact data count %d (want 1)", n1);
    Avbd3dContactId oldContact = sd1[0].contactId;
    if (n1 == 1)
    {
        CHECK(sd1[0].manifoldCount == 1, "box1 manifold count %d", sd1[0].manifoldCount);
        CHECK(sd1[0].manifolds && sd1[0].manifolds->pointCount >= 1, "box1 manifold has no points");
        if (sd1[0].manifolds)
            CHECK(std::fabs(sd1[0].manifolds->normal.y) > 0.9f, "box1 manifold normal y %g",
                  sd1[0].manifolds->normal.y);
    }
    CHECK(t3d_body_get_contact_capacity(b1) == 1, "box1 body contact capacity %d", t3d_body_get_contact_capacity(b1));
    Avbd3dContactData bd1[2];
    CHECK(t3d_body_get_contact_data(b1, bd1, 2) == 1, "box1 body contact data count");
    CHECK(avbd3d_contact_is_valid(oldContact), "box1 contact id not valid while touching");
    const Avbd3dContactData cd = t3d_contact_get_data(oldContact);
    CHECK(cd.manifolds && cd.manifolds->pointCount >= 1, "Contact_GetData has no manifold points");
    CHECK(t3d_shape_get_contact_capacity(s2) == 0, "unflagged box2 has contact data");
    CHECK(t3d_body_get_contact_capacity(b2) == 0, "unflagged box2 body has contact data");

    // ---- d: the control box is quiet -------------------------------------------------------
    CHECK(countShape(t.beginShapes, s2) == 0 && countShape(t.hitShapes, s2) == 0, "control box not quiet");

    // ---- b: moving the box away ends its contact -------------------------------------------
    t3d_body_set_transform(b1, Avbd3dVec3{0, 10.0f, 0}, identityQuat());
    Tally tb;
    stepTally(w, tb);
    CHECK(tb.end == 1, "end events after moving box1 away: %d (want 1)", tb.end);
    CHECK(countShape(tb.endShapes, s1) == 1, "end event does not name box1");
    CHECK(!avbd3d_contact_is_valid(oldContact), "contact id still valid after its end event");
    CHECK(t3d_shape_get_contact_capacity(s1) == 0, "box1 still has contact data after its end event");

    // ---- g: destroying a touching body ends its contact at the next step -------------------
    // box3 has been touching since the first step; find its contact id before destroying it.
    Avbd3dContactData sd3[2];
    CHECK(t3d_shape_get_contact_data(s3, sd3, 2) == 1, "box3 has no contact before destroy");
    Avbd3dContactId c3 = sd3[0].contactId;
    t3d_destroy_body(b3);
    Tally tg;
    stepTally(w, tg);
    CHECK(tg.end == 1, "end events after destroying box3: %d (want 1)", tg.end);
    CHECK(countShape(tg.endShapes, s3) == 1, "destroy end event does not name box3");
    CHECK(!avbd3d_contact_is_valid(c3), "box3 contact id still valid after destroy");

    t3d_destroy_world(w);

    // ---- f: truncation of a full buffer ----------------------------------------------------
    Avbd3dWorld *wt = t3d_create_world(&wd);
    makeBox(wt, AVBD3D_STATIC_BODY, Avbd3dVec3{0, -0.5f, 0}, 40, 0.5f, 40, false);
    const int nBoxes = 300;
    for (int i = 0; i < nBoxes; i++)
    {
        const float x = (float)(i % 20) * 1.5f - 14.25f;
        const float z = (float)(i / 20) * 1.5f - 10.0f;
        makeBox(wt, AVBD3D_DYNAMIC_BODY, Avbd3dVec3{x, 0.5f, z}, 0.5f, 0.5f, 0.5f, true);
    }
    Tally tf1, tf2;
    stepTally(wt, tf1);
    CHECK((t3d_get_stats(wt).contact_events_truncated != 0), "a 300-pair step was not flagged truncated");
    CHECK(tf1.end == 0, "truncated step emitted %d end events", tf1.end);
    stepTally(wt, tf2);
    CHECK(!(t3d_get_stats(wt).contact_events_truncated != 0), "the grown buffer still truncated the next step");
    CHECK(tf1.begin + tf2.begin == nBoxes, "begin events over two steps %d (want %d)", tf1.begin + tf2.begin, nBoxes);
    CHECK(tf2.end == 0, "second step emitted %d end events", tf2.end);
    t3d_destroy_world(wt);

    // ---- h: 64-bit filters and Shape_SetFilter ---------------------------------------------
    // The ground carries only category bit 33 and every mask bit; the box's category is bit 34.
    // Everything that decides the pair is above bit 31. Mask bit 33 admits the ground (collide),
    // mask bit 35 does not (pass through).
    for (int pass = 0; pass < 2; pass++)
    {
        Avbd3dWorld *wh = t3d_create_world(&wd);
        Avbd3dBodyDef gbd = avbd3d_default_body_def();
        gbd.type = AVBD3D_STATIC_BODY;
        gbd.position = Avbd3dVec3{0, -0.5f, 0};
        Avbd3dBodyId gb = t3d_create_body(wh, &gbd);
        Avbd3dShapeDef gsd = avbd3d_default_shape_def();
        gsd.filter.categoryBits = 1ull << 33;
        gsd.filter.maskBits = ~0ull;
        Avbd3dBoxHull gh = avbd3d_make_box_hull(10, 0.5f, 10);
        t3d_create_hull_shape(gb, &gsd, avbd3d_box_hull_data(&gh));

        Avbd3dBodyDef bbd = avbd3d_default_body_def();
        bbd.type = AVBD3D_DYNAMIC_BODY;
        bbd.position = Avbd3dVec3{0, 0.6f, 0};
        Avbd3dBodyId bb = t3d_create_body(wh, &bbd);
        Avbd3dShapeDef bsd = avbd3d_default_shape_def();
        bsd.filter.categoryBits = 1ull << 34;
        bsd.filter.maskBits = pass == 0 ? (1ull << 33) : (1ull << 35);
        Avbd3dBoxHull bh = avbd3d_make_box_hull(0.5f, 0.5f, 0.5f);
        t3d_create_hull_shape(bb, &bsd, avbd3d_box_hull_data(&bh));
        for (int i = 0; i < 90; i++)
            t3d_step(wh, 1.0f / 60.0f);
        const float y = (float)t3d_body_get_position(bb).y;
        if (pass == 0)
            CHECK(y > 0.4f, "high-bit mask admits the ground, box should rest: y %g", y);
        else
            CHECK(y < -1.0f, "high-bit mask excludes the ground, box should fall through: y %g", y);
        t3d_destroy_world(wh);
    }

    // Shape_SetFilter at runtime: a resting box with contact events falls through the ground
    // and its pair ends (one end event); restoring the filter makes it collide again.
    {
        Avbd3dWorld *wh = t3d_create_world(&wd);
        makeBox(wh, AVBD3D_STATIC_BODY, Avbd3dVec3{0, -0.5f, 0}, 10, 0.5f, 10, false);
        Avbd3dShapeId sh{};
        Avbd3dBodyId bh = makeBox(wh, AVBD3D_DYNAMIC_BODY, Avbd3dVec3{0, 0.5f, 0}, 0.5f, 0.5f, 0.5f, true, &sh);
        Tally th;
        for (int i = 0; i < 120; i++)
            stepTally(wh, th);
        CHECK(th.begin == 1 && th.end == 0, "setfilter: resting begin/end %d/%d", th.begin, th.end);
        CHECK((float)t3d_body_get_position(bh).y > 0.4f, "setfilter: box did not rest");
        Avbd3dFilter f = avbd3d_default_filter();
        f.maskBits = 1ull << 40;
        t3d_shape_set_filter(sh, f);
        Tally th2;
        for (int i = 0; i < 90; i++)
            stepTally(wh, th2);
        CHECK(th2.end == 1, "setfilter: end events %d (want 1)", th2.end);
        CHECK((float)t3d_body_get_position(bh).y < -1.0f, "setfilter: box did not fall through: y %g",
              (float)t3d_body_get_position(bh).y);
        t3d_destroy_world(wh);
    }

    if (g_fail == 0)
        std::printf("avbd_test_vk_api3d_events: all checks passed\n");
    else
        std::printf("avbd_test_vk_api3d_events: %d failure(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
#undef CHECK
} // namespace events

namespace sensors
{
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

static bool sameShape(Avbd3dShapeId a, Avbd3dShapeId b) { return a.index1 == b.index1 && a.generation == b.generation; }

static Avbd3dBodyId makeBody(Avbd3dWorld *w, Avbd3dBodyType type, Avbd3dVec3 p)
{
    Avbd3dBodyDef bd = avbd3d_default_body_def();
    bd.type = type;
    bd.position = p;
    return t3d_create_body(w, &bd);
}

// A box shape on `b`. `isSensor` makes a sensor; `sensorEvents` is enableSensorEvents.
static Avbd3dShapeId addBox(Avbd3dBodyId b, float hx, float hy, float hz, bool isSensor, bool sensorEvents)
{
    Avbd3dShapeDef sd = avbd3d_default_shape_def();
    sd.isSensor = isSensor;
    sd.enableSensorEvents = sensorEvents;
    Avbd3dBoxHull hull = avbd3d_make_box_hull(hx, hy, hz);
    return t3d_create_hull_shape(b, &sd, avbd3d_box_hull_data(&hull));
}

// Sensor and visitor events accumulated over the steps of one scenario.
struct Log
{
    std::vector<Avbd3dSensorBeginTouchEvent> begins;
    std::vector<Avbd3dSensorEndTouchEvent> ends;
};

static void step(Avbd3dWorld *w, Log &log)
{
    t3d_step(w, 1.0f / 60.0f);
    const Avbd3dSensorEvents e = t3d_world_get_sensor_events(w);
    for (int i = 0; i < e.beginCount; i++)
        log.begins.push_back(e.beginEvents[i]);
    for (int i = 0; i < e.endCount; i++)
        log.ends.push_back(e.endEvents[i]);
}

static int countBegin(const Log &log, Avbd3dShapeId sensor, Avbd3dShapeId visitor)
{
    int n = 0;
    for (size_t i = 0; i < log.begins.size(); i++)
        if (sameShape(log.begins[i].sensorShapeId, sensor) && sameShape(log.begins[i].visitorShapeId, visitor))
            n++;
    return n;
}

static int countEnd(const Log &log, Avbd3dShapeId sensor, Avbd3dShapeId visitor)
{
    int n = 0;
    for (size_t i = 0; i < log.ends.size(); i++)
        if (sameShape(log.ends[i].sensorShapeId, sensor) && sameShape(log.ends[i].visitorShapeId, visitor))
            n++;
    return n;
}

// Number of visitors the sensor lists now; `contains` is set when `visitor` is one of them.
static int overlapCount(Avbd3dShapeId sensor, Avbd3dShapeId visitor, bool &contains)
{
    const int cap = t3d_shape_get_sensor_capacity(sensor);
    std::vector<Avbd3dShapeId> ids(cap > 0 ? cap : 1);
    const int n = t3d_shape_get_sensor_data(sensor, ids.data(), cap);
    contains = false;
    for (int i = 0; i < n && i < cap; i++)
        if (sameShape(ids[i], visitor))
            contains = true;
    return n;
}

static Avbd3dWorld *newWorld()
{
    Avbd3dWorldDef wd = avbd3d_default_world_def();
    return t3d_create_world(&wd);
}

// A static ground box with its top at y = 0, spanning x and z in [-30, 30].
static void addGround(Avbd3dWorld *w)
{
    Avbd3dBodyId g = makeBody(w, AVBD3D_STATIC_BODY, Avbd3dVec3{0, -0.5f, 0});
    addBox(g, 30, 0.5f, 30, false, false);
}

// ---- a) sensor begin/end, and the overlap list ----------------------------------------------
static void checkSensorBasics()
{
    Avbd3dWorld *w = newWorld();
    CHECK((w != NULL), "world a");
    addGround(w);

    // Static sensor spanning y in [2, 4], above the ground.
    Avbd3dBodyId sb = makeBody(w, AVBD3D_STATIC_BODY, Avbd3dVec3{0, 3, 0});
    const Avbd3dShapeId sensor = addBox(sb, 1, 1, 1, true, true);
    CHECK(t3d_shape_is_sensor(sensor), "sensor flag set");

    // Queries do not see sensors (Box3D): a ray down through the sensor hits the ground at y = 0.
    Avbd3dRayResult ray = t3d_world_cast_ray_closest(w, {0, 10, 0}, {0, -20, 0}, avbd3d_default_query_filter());
    CHECK(ray.hit && !sameShape(ray.shapeId, sensor) && ray.point.y < 0.5f, "ray skips the sensor, hit=%d y=%f",
          (int)ray.hit, (double)ray.point.y);

    // Dynamic visitor dropped from y = 8, through the sensor, onto the ground.
    Avbd3dBodyId vb = makeBody(w, AVBD3D_DYNAMIC_BODY, Avbd3dVec3{0, 8, 0});
    const Avbd3dShapeId visitor = addBox(vb, 0.5f, 0.5f, 0.5f, false, true);

    Log log;
    int insideSteps = 0;
    bool listedWhileInside = false;
    for (int i = 0; i < 240; i++)
    {
        step(w, log);
        bool contains = false;
        const int n = overlapCount(sensor, visitor, contains);
        if (n > 0)
            insideSteps++;
        if (contains)
            listedWhileInside = true;
    }

    const Avbd3dVec3 p = t3d_body_get_position(vb);
    CHECK(p.y < 2.0f, "visitor fell through the sensor band, y=%f", (double)p.y);
    CHECK(countBegin(log, sensor, visitor) == 1, "one begin event, got %d", countBegin(log, sensor, visitor));
    CHECK(countEnd(log, sensor, visitor) == 1, "one end event, got %d", countEnd(log, sensor, visitor));
    CHECK(insideSteps > 0, "the visitor was inside the sensor for some steps");
    CHECK(listedWhileInside, "Shape_GetSensorData listed the visitor while it was inside");
    bool containsAtRest = true;
    overlapCount(sensor, visitor, containsAtRest);
    CHECK(!containsAtRest, "Shape_GetSensorData no longer lists the landed visitor");

    t3d_step(w, 1.0f / 60.0f);
    t3d_destroy_world(w);
}

// ---- b) visitor with events off, and sensor-sensor overlap ----------------------------------
static void checkNoEventsCases()
{
    Avbd3dWorld *w = newWorld();
    addGround(w);

    // Sensor S1 over the ground. Visitor V has enableSensorEvents = false.
    Avbd3dBodyId s1b = makeBody(w, AVBD3D_STATIC_BODY, Avbd3dVec3{0, 3, 0});
    const Avbd3dShapeId s1 = addBox(s1b, 1, 1, 1, true, true);
    Avbd3dBodyId vb = makeBody(w, AVBD3D_DYNAMIC_BODY, Avbd3dVec3{0, 8, 0});
    const Avbd3dShapeId v = addBox(vb, 0.5f, 0.5f, 0.5f, false, false);

    // Sensor S2 and sensor S3 overlap each other at z = 6, away from the falling box.
    Avbd3dBodyId s2b = makeBody(w, AVBD3D_STATIC_BODY, Avbd3dVec3{0, 3, 6});
    const Avbd3dShapeId s2 = addBox(s2b, 1, 1, 1, true, true);
    Avbd3dBodyId s3b = makeBody(w, AVBD3D_STATIC_BODY, Avbd3dVec3{0, 3, 6});
    const Avbd3dShapeId s3 = addBox(s3b, 1, 1, 1, true, true);

    Log log;
    int listed = 0;
    for (int i = 0; i < 240; i++)
    {
        step(w, log);
        bool c = false;
        overlapCount(s1, v, c);
        if (c)
            listed++;
    }
    CHECK(log.begins.empty() && log.ends.empty(), "no events for the events-off visitor or the sensor pair, got %zu/%zu",
          log.begins.size(), log.ends.size());
    CHECK(listed == 0, "the events-off visitor is never listed by the sensor");
    bool sensorSees = false;
    overlapCount(s2, s3, sensorSees);
    CHECK(!sensorSees, "a sensor does not list an overlapping sensor");
    t3d_destroy_world(w);
}

// ---- c) destroying a visitor inside a sensor -------------------------------------------------
static void checkDestroyVisitor()
{
    Avbd3dWorld *w = newWorld();
    addGround(w);
    Avbd3dBodyId sb = makeBody(w, AVBD3D_STATIC_BODY, Avbd3dVec3{0, 3, 0});
    const Avbd3dShapeId sensor = addBox(sb, 1, 1, 1, true, true);
    Avbd3dBodyId vb = makeBody(w, AVBD3D_DYNAMIC_BODY, Avbd3dVec3{0, 3, 0});
    const Avbd3dShapeId visitor = addBox(vb, 0.5f, 0.5f, 0.5f, false, true);

    Log log;
    step(w, log);
    CHECK(countBegin(log, sensor, visitor) == 1, "begin while inside, got %d", countBegin(log, sensor, visitor));

    t3d_destroy_body(vb);
    step(w, log);
    CHECK(countEnd(log, sensor, visitor) == 1, "destroying the visitor emits one end event, got %d",
          countEnd(log, sensor, visitor));
    CHECK(!avbd3d_shape_is_valid(visitor), "the destroyed visitor id is invalid");
    bool contains = true;
    overlapCount(sensor, visitor, contains);
    CHECK(!contains, "the sensor no longer lists the destroyed visitor");
    t3d_destroy_world(w);
}

// ---- d) explosion ----------------------------------------------------------------------------
static void checkExplosion()
{
    Avbd3dWorld *w = newWorld();
    addGround(w);
    // Boxes resting on the ground at 1 m, 2 m and 10 m from the origin (the blast centre).
    Avbd3dBodyId a = makeBody(w, AVBD3D_DYNAMIC_BODY, Avbd3dVec3{1, 0.5f, 0});
    addBox(a, 0.5f, 0.5f, 0.5f, false, false);
    Avbd3dBodyId b = makeBody(w, AVBD3D_DYNAMIC_BODY, Avbd3dVec3{0, 0.5f, 2});
    addBox(b, 0.5f, 0.5f, 0.5f, false, false);
    Avbd3dBodyId c = makeBody(w, AVBD3D_DYNAMIC_BODY, Avbd3dVec3{10, 0.5f, 0});
    addBox(c, 0.5f, 0.5f, 0.5f, false, false);

    Log log;
    for (int i = 0; i < 600; i++)
        step(w, log);
    CHECK(!t3d_body_is_awake(a) && !t3d_body_is_awake(b) && !t3d_body_is_awake(c),
          "precondition: the three boxes are asleep before the blast");

    Avbd3dExplosionDef def = avbd3d_default_explosion_def();
    def.position = Avbd3dVec3{0, 0, 0};
    def.radius = 5;
    def.falloff = 0;
    def.impulsePerArea = 10;
    t3d_world_explode(w, &def);
    step(w, log);

    CHECK(t3d_body_is_awake(a), "inside box A woken by the blast");
    CHECK(t3d_body_is_awake(b), "inside box B woken by the blast");

    const Avbd3dVec3 va = t3d_body_get_linear_velocity(a);
    const Avbd3dVec3 vb = t3d_body_get_linear_velocity(b);
    const Avbd3dVec3 vc = t3d_body_get_linear_velocity(c);
    // Radial: A sits on +x, B on +z, so their horizontal velocity must point away from the origin.
    CHECK(va.x > 0.0f, "box A pushed radially outward, vx=%f", (double)va.x);
    CHECK(vb.z > 0.0f, "box B pushed radially outward, vz=%f", (double)vb.z);
    const float cz = std::fabs(vc.x) + std::fabs(vc.z) + std::fabs(vc.y);
    CHECK(cz < 1e-3f, "far box C unaffected, |v|=%f", (double)cz);
    t3d_destroy_world(w);
}

// ---- e) a world with no sensors --------------------------------------------------------------
static void checkNoSensors()
{
    Avbd3dWorld *w = newWorld();
    addGround(w);
    for (int i = 0; i < 3; i++)
    {
        Avbd3dBodyId b = makeBody(w, AVBD3D_DYNAMIC_BODY, Avbd3dVec3{(float)i * 3.0f, 3.0f, 0});
        addBox(b, 0.5f, 0.5f, 0.5f, false, false);
    }
    Log log;
    for (int i = 0; i < 180; i++)
        step(w, log);
    CHECK(log.begins.empty() && log.ends.empty(), "a world with no sensors reports no sensor events");
    t3d_destroy_world(w);
}

int run()
{
    checkSensorBasics();
    checkNoEventsCases();
    checkDestroyVisitor();
    checkExplosion();
    checkNoSensors();

    if (g_fail)
    {
        std::printf("%d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
#undef CHECK
} // namespace sensors

int main()
{
    int failed = 0;
    failed |= events::run() != 0;
    failed |= sensors::run() != 0;
    return failed ? 1 : 0;
}
