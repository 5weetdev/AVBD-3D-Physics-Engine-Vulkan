#pragma once

#include "scene_common.h"

// Four-engine passenger airliner simulation scene.
//
// Built from ~2,250 distinct physical bodies of diverse shapes:
//   - Fuselage barrel: 16-facet aerodynamic skin, internal ring formers,
//     window mullions, passenger cabin floor, lower cargo deck, overhead bins,
//     dual passenger seat rows, cockpit glare shield, dashboard, and windshield frames.
//   - Multi-spar wings: front & rear spars, upper & lower torque box skins, leading edge
//     slats, chordwise ribs, spanwise stringers, trailing edge flaps/ailerons, and winglets.
//   - 4 engine nacelles: heavy crankcase/turbine cores, cantilever pylon trusses,
//     multi-ring octagonal cowlings, and aerodynamic intake guide vanes.
//   - 4 rotating propellers: 3-DOF spherical bearing joints (stiffnessAng = 0.0f)
//     providing complete rotational freedom, initialized with counter-rotating spin.
//   - Empennage: swept dorsal fin with segmented rudder, horizontal stabilizers with elevators.
//   - Heavy-duty tricycle landing gear: shock struts, bogie beams, and rotating wheel pairs.
//
// Every joint is breakable with physically calibrated fracture loads:
//   - Primary airframe / wing spar joints: 6,000 - 8,000 N
//   - Fuselage formers & pylons: 4,000 N
//   - Skin panels & decks: 2,500 N
//   - Minor fittings, interior, and control surfaces: 1,500 - 2,000 N
//   - Propeller bearings and blades: 1,800 - 3,500 N
//
// Zero initial interpenetration: strict analytical coordinates with clearance offsets.
//
// Templated on the builder so the same aircraft can be emitted somewhere other than
// into a scene: the spawn tool hands this a SpawnEmitter, which collects the very same
// addBody/addJointWorld calls and then places, scales and launches them. Anything this
// function calls has to exist on both, which is why it stays to clear(), addBody() and
// addJointWorld().
struct PassengerPlaneConfig
{
    float breakHeavy = 5000.0f;   // Main wing spars, gear trunnions, center carry-through
    float breakFrame = 2000.0f;   // Fuselage formers, engine pylons, fin spars
    float breakSkin = 2500.0f;    // Outer skin panels, floor slabs, tail skins
    float breakMinor = 1500.0f;   // Interior seats, window mullions, control flaps
    float breakPropHub = 1800.0f; // Propeller shaft / bearing fracture load
    float breakBlade = 1500.0f;   // Propeller blade fracture load
    float breakPylon = 2000.0f;   // Engine pylons (shear off cleanly)
    float breakGear = 4000.0f;    // Landing gear (collapse under hard impact)
};

template <typename Builder>
static SceneView buildPassengerPlane(Builder *s, const PassengerPlaneConfig &cfg = PassengerPlaneConfig{})
{
    s->clear();

    // -------------------------------------------------------------------------
    // 0. Physical Constants & Calibrated Fracture Forces
    // -------------------------------------------------------------------------
    const float kBreakHeavy = cfg.breakHeavy;
    const float kBreakFrame = cfg.breakFrame;
    const float kBreakSkin = cfg.breakSkin;
    const float kBreakMinor = cfg.breakMinor;
    const float kBreakPropHub = cfg.breakPropHub;
    const float kBreakBlade = cfg.breakBlade;
    const float kBreakPylon = cfg.breakPylon;
    const float kBreakGear = cfg.breakGear;

    // A finite angular stiffness is a spring: the penalty is clamped to it and the dual
    // variable is never accumulated, so the joint carries load only in proportion to how
    // far it has already bent. INFINITY selects the hard-constraint path instead -- dual
    // update plus C0 stabilization -- which is what an airframe joint should be. The
    // secondary joints stay finite but an order of magnitude stiffer than before, so
    // skin panels and control surfaces still give a little rather than sagging.
    const float kAngStiffRigid = INFINITY;   // Structural beams: hard angular constraint
    const float kAngStiffPanel = 5000000.0f; // Multi-bay skin, slightly compliant
    const float kAngStiffHinge = 1000000.0f; // Flaps and control surfaces

    const float kDensityAlloy = 1.0f; // Standard structural density
    const float kDensityHeavy = 2.0f; // Engine blocks and landing gear struts
    const float kDensityLight = 0.5f; // Cladding, seats, cowlings
    const float kFrictionAir = 0.5f;

    // Centerline altitude of aircraft fuselage: rests on tricycle landing gear
    const float zPlane = 5.20f;

    // Wing box / carry-through centerline. The passenger floor sits at zPlane - 0.35m
    // and the cargo deck at zPlane - 1.15m, so the torque box must thread between them.
    const float zWingBox = zPlane - 0.78f;

    // -------------------------------------------------------------------------
    // 1. Static Runway & Ground
    // -------------------------------------------------------------------------
    // Main runway tarmac surface is at z = 0.0m (thickness 1.0m, center at z = -0.5m)
    s->addBody({200.0f, 60.0f, 1.0f}, 0.0f, 0.6f, {0.0f, 0.0f, -0.5f});

    // Runway centerline markings (embedded flush, density = 0.0f)
    for (int rm = -8; rm <= 8; rm++)
    {
        s->addBody({6.0f, 0.8f, 0.05f}, 0.0f, 0.6f, {rm * 10.0f, 0.0f, 0.025f});
    }

    // -------------------------------------------------------------------------
    // 2. Fuselage Barrel (Skin, Formers, Floors, Interior, Windows)
    // -------------------------------------------------------------------------
    // Length: 42.0m from X = -21.0m to +21.0m.
    // 35 bays along X (pitch = 1.20m, element length = 1.16m, 0.04m clearance).
    const int nBays = 35;
    const float bayPitch = 1.20f;
    const float bayLen = 1.16f;
    const float xStart = -20.4f;

    const int nFacets = 16; // 16-sided faceted cylinder
    const float dPhi = 2.0f * 3.1415926535f / (float)nFacets;

    // Base fuselage radius
    const float rBase = 1.80f;
    const float skinThick = 0.08f;
    const float formerThick = 0.10f;
    const float formerDepth = 0.14f;

    auto getBayRadius = [&](float x) -> float
    {
        if (x > 13.0f) // Nose taper
        {
            float u = (21.0f - x) / (21.0f - 13.0f);
            if (u < 0.15f)
                u = 0.15f;
            return rBase * sqrtf(u);
        }
        else if (x < -10.0f) // Tail cone taper
        {
            float u = (x - (-21.0f)) / (-10.0f - (-21.0f));
            if (u < 0.12f)
                u = 0.12f;
            return rBase * powf(u, 0.75f);
        }
        return rBase; // Main cabin
    };

    auto getBayCenterZ = [&](float x) -> float
    {
        if (x > 13.0f) // Slight cockpit downward nose droop
        {
            float u = (21.0f - x) / (21.0f - 13.0f);
            return zPlane - 0.25f * (1.0f - u);
        }
        else if (x < -10.0f) // Upswept tail cone
        {
            float u = (x - (-21.0f)) / (-10.0f - (-21.0f));
            return zPlane + 0.85f * (1.0f - u);
        }
        return zPlane;
    };

    int skinBodies[35][16];
    int formerBodies[35][16];
    for (int b = 0; b < nBays; b++)
    {
        for (int f = 0; f < nFacets; f++)
        {
            skinBodies[b][f] = -1;
            formerBodies[b][f] = -1;
        }
    }

    int cabinFloorL[35];
    int cabinFloorR[35];
    int cargoFloor[35];
    for (int b = 0; b < nBays; b++)
    {
        cabinFloorL[b] = cabinFloorR[b] = cargoFloor[b] = -1;
    }

    // Meridian slope of the fuselage surface at a station: how fast the radius grows
    // and the centerline rises along +X. Zero through the constant-radius cabin, and
    // non-zero through the nose and tail cones, where the panels have to tilt to lie on
    // the surface instead of staying parallel to the axis.
    auto getBaySlope = [&](float x, float *drdx, float *dzdx)
    {
        const float h = bayPitch * 0.5f;
        *drdx = (getBayRadius(x + h) - getBayRadius(x - h)) / (2.0f * h);
        *dzdx = (getBayCenterZ(x + h) - getBayCenterZ(x - h)) / (2.0f * h);
    };

    // Generate Fuselage Skin and Internal Formers
    for (int b = 0; b < nBays; b++)
    {
        const float xC = xStart + b * bayPitch;
        const float rCur = getBayRadius(xC);
        const float zC = getBayCenterZ(xC);

        float drdx = 0.0f, dzdx = 0.0f;
        getBaySlope(xC, &drdx, &dzdx);

        const float uBay = fabsf((float)b - 17.5f) / 17.5f;
        const float localBreakFrame = kBreakFrame * (1.0f - 0.6f * uBay);
        const float localBreakSkin = kBreakSkin * (1.0f - 0.6f * uBay);

        for (int f = 0; f < nFacets; f++)
        {
            const float phi = f * dPhi;
            const float cosP = cosf(phi);
            const float sinP = sinf(phi);

            // Bodies sit at radial direction (sin phi, cos phi) in (Y, Z). A quaternion
            // about +X by angle a maps local +Z to (0, -sin a, cos a), so the roll that
            // aligns the panel's thickness with the radius is -phi, not +phi.
            const float halfPhi = -phi * 0.5f;
            const quat rotRoll = {sinf(halfPhi), 0.0f, 0.0f, cosf(halfPhi)};

            // Then pitch about the rolled body's own Y so its long axis follows the
            // surface meridian. After the roll, local +Z is the outward radial
            // direction e_r, and the surface tangent along X is (1, drdx + dzdx*cos phi)
            // in the local XZ plane. A rotation about +Y by `a` sends local +X to
            // (cos a, 0, -sin a), hence the negated arc tangent.
            const float meridianSlope = drdx + dzdx * cosP;
            const float halfPitch = -atan2f(meridianSlope, 1.0f) * 0.5f;
            const quat rotPitch = {0.0f, sinf(halfPitch), 0.0f, cosf(halfPitch)};

            // Intrinsic composition: roll first, then pitch in the rolled frame.
            const quat rotPanel = rotRoll * rotPitch;

            // Facet chord, sized off the smallest radius the body actually reaches --
            // not its mid-surface radius. Two things pull a panel inboard of its own
            // centre: its thickness, and the meridian pitch above, which swings a
            // 1.16m-long panel through half its length times sin(theta) in radius. The
            // ring is a 16-gon, so the circumferential room available shrinks with
            // radius, and a panel sized for its centre has its corners inside its
            // neighbours. Invisible at the cabin radius of 1.80m, where theta is zero,
            // and severe at the cone tips, where the radius is down to 0.37m and the
            // pitch reaches 15 degrees.
            const float pitchAngle = atan2f(meridianSlope, 1.0f);
            const float sinPitch = fabsf(sinf(pitchAngle));
            const float cosPitch = fabsf(cosf(pitchAngle));

            const float rSkinMin = rCur - skinThick * 0.5f - 0.5f * (bayLen * sinPitch + skinThick * cosPitch);
            const float facetW = 2.0f * fmaxf(rSkinMin, 0.05f) * sinf(dPhi * 0.5f) * 0.96f;

            const float rFormerMin =
                rCur - skinThick - formerDepth * 0.5f - 0.5f * (bayLen * 0.40f * sinPitch + formerDepth * cosPitch);
            const float formerW = 2.0f * fmaxf(rFormerMin, 0.04f) * sinf(dPhi * 0.5f) * 0.95f;

            // 1. Outer Skin Panel
            const float rSkinCenter = rCur - skinThick * 0.5f;
            const vec3 pSkin = {xC, rSkinCenter * sinP, zC + rSkinCenter * cosP};
            int sBody = s->addBody({bayLen, facetW, skinThick}, kDensityLight, kFrictionAir, pSkin, {0, 0, 0},
                                   {0, 0, 0}, rotPanel);
            skinBodies[b][f] = sBody;

            // 2. Internal Ring Former Strut
            const float rFormerCenter = rCur - skinThick - formerDepth * 0.5f;
            const vec3 pFormer = {xC, rFormerCenter * sinP, zC + rFormerCenter * cosP};
            int fBody = s->addBody({bayLen * 0.40f, formerW, formerDepth}, kDensityAlloy, kFrictionAir, pFormer,
                                   {0, 0, 0}, {0, 0, 0}, rotPanel);
            formerBodies[b][f] = fBody;

            // Joint: Skin to Former
            const vec3 pSkinFormerFace = {xC, (rCur - skinThick) * sinP, zC + (rCur - skinThick) * cosP};
            s->addJointWorld(sBody, fBody, pSkinFormerFace, INFINITY, kAngStiffRigid, localBreakFrame);

            // Joint: Former to adjacent former in the ring
            if (f > 0)
            {
                const float phiMid = (f - 0.5f) * dPhi;
                const vec3 pRingJoint = {xC, rFormerCenter * sinf(phiMid), zC + rFormerCenter * cosf(phiMid)};
                s->addJointWorld(formerBodies[b][f - 1], fBody, pRingJoint, INFINITY, kAngStiffRigid, localBreakFrame);
            }
            if (f == nFacets - 1)
            {
                const float phiMid = (nFacets - 0.5f) * dPhi;
                const vec3 pRingJoint = {xC, rFormerCenter * sinf(phiMid), zC + rFormerCenter * cosf(phiMid)};
                s->addJointWorld(fBody, formerBodies[b][0], pRingJoint, INFINITY, kAngStiffRigid, localBreakFrame);
            }

            // Joint: Longitudinal connection along fuselage between consecutive bays
            if (b > 0)
            {
                const vec3 pLongSkin = {xC - bayPitch * 0.5f, pSkin.y, pSkin.z};
                s->addJointWorld(skinBodies[b - 1][f], sBody, pLongSkin, INFINITY, kAngStiffPanel, localBreakSkin);

                const vec3 pLongFormer = {xC - bayPitch * 0.5f, pFormer.y, pFormer.z};
                s->addJointWorld(formerBodies[b - 1][f], fBody, pLongFormer, INFINITY, kAngStiffRigid, localBreakFrame);
            }
        }

        // 3. Passenger Cabin Floor, Lower Cargo Deck, Seats & Windows (b in [9, 28])
        if (b >= 9 && b <= 28)
        {
            const float zFloor = zC - 0.35f;
            const float zCargo = zC - 1.15f;

            // Left passenger floor slab
            // Half-width 0.645m, so the outboard edge sits at y = 1.47m. The former ring
            // has an inner radius of 1.58m and the lower corner of the slab is 0.41m below
            // the axis, which leaves 1.50m of room -- the old 1.35m slab reached exactly
            // that limit and broke through as soon as the radius tapered at bay 28.
            cabinFloorL[b] = s->addBody({bayLen, 1.29f, 0.12f}, kDensityAlloy, kFrictionAir, {xC, -0.825f, zFloor});
            // Right passenger floor slab
            cabinFloorR[b] = s->addBody({bayLen, 1.29f, 0.12f}, kDensityAlloy, kFrictionAir, {xC, +0.825f, zFloor});
            // Cargo hold deck slab
            cargoFloor[b] = s->addBody({bayLen, 1.80f, 0.10f}, kDensityAlloy, kFrictionAir, {xC, 0.0f, zCargo});

            // Anchor floors to fuselage ring formers
            s->addJointWorld(formerBodies[b][4], cabinFloorR[b], {xC, +1.45f, zFloor}, INFINITY, kAngStiffRigid,
                             localBreakFrame);
            s->addJointWorld(formerBodies[b][12], cabinFloorL[b], {xC, -1.45f, zFloor}, INFINITY, kAngStiffRigid,
                             localBreakFrame);
            s->addJointWorld(formerBodies[b][8], cargoFloor[b], {xC, 0.0f, zCargo - 0.05f}, INFINITY, kAngStiffRigid,
                             localBreakFrame);

            // Connect floor slabs longitudinally
            if (b > 9)
            {
                s->addJointWorld(cabinFloorL[b - 1], cabinFloorL[b], {xC - bayPitch * 0.5f, -0.825f, zFloor}, INFINITY,
                                 kAngStiffRigid, localBreakFrame);
                s->addJointWorld(cabinFloorR[b - 1], cabinFloorR[b], {xC - bayPitch * 0.5f, +0.825f, zFloor}, INFINITY,
                                 kAngStiffRigid, localBreakFrame);
                s->addJointWorld(cargoFloor[b - 1], cargoFloor[b], {xC - bayPitch * 0.5f, 0.0f, zCargo}, INFINITY,
                                 kAngStiffRigid, localBreakFrame);
            }

            // 4. Passenger Seats (2-seat benches)
            if (b >= 10 && b <= 27)
            {
                int seatL =
                    s->addBody({0.55f, 0.90f, 0.65f}, kDensityLight, kFrictionAir, {xC, -0.90f, zFloor + 0.385f});
                int seatR =
                    s->addBody({0.55f, 0.90f, 0.65f}, kDensityLight, kFrictionAir, {xC, +0.90f, zFloor + 0.385f});

                s->addJointWorld(cabinFloorL[b], seatL, {xC, -0.90f, zFloor + 0.06f}, INFINITY, kAngStiffRigid,
                                 kBreakMinor);
                s->addJointWorld(cabinFloorR[b], seatR, {xC, +0.90f, zFloor + 0.06f}, INFINITY, kAngStiffRigid,
                                 kBreakMinor);

                // 5. Overhead Luggage Bins above passenger seats
                int binL =
                    s->addBody({bayLen * 0.86f, 0.50f, 0.30f}, kDensityLight, kFrictionAir, {xC, -1.00f, zC + 0.75f});
                int binR =
                    s->addBody({bayLen * 0.86f, 0.50f, 0.30f}, kDensityLight, kFrictionAir, {xC, +1.00f, zC + 0.75f});

                s->addJointWorld(formerBodies[b][14], binL, {xC, -1.00f, zC + 0.85f}, INFINITY, kAngStiffRigid,
                                 kBreakMinor);
                s->addJointWorld(formerBodies[b][2], binR, {xC, +1.00f, zC + 0.85f}, INFINITY, kAngStiffRigid,
                                 kBreakMinor);

                // 6. Passenger Window Mullions (Vertical frames along cabin walls)
                for (int side = 0; side < 2; side++)
                {
                    float sgn = (side == 0) ? -1.0f : +1.0f;
                    int fIndex = (side == 0) ? 12 : 4;
                    for (int m = -1; m <= 1; m++)
                    {
                        float xMullion = xC + m * 0.35f;
                        // Inboard of the inner face of the former ring (r = 1.58m). The
                        // mullion is 0.45m tall, so its top corners swing out to a larger
                        // radius than its centre: at y = 1.47m those corners are still
                        // inside its own facet, but not inside the one 22.5 degrees along.
                        const float yMullion = sgn * (rCur - 0.36f);
                        int mullion = s->addBody({0.08f, 0.06f, 0.45f}, kDensityLight, kFrictionAir,
                                                 {xMullion, yMullion, zFloor + 0.60f});
                        s->addJointWorld(formerBodies[b][fIndex], mullion, {xMullion, yMullion, zFloor + 0.60f},
                                         INFINITY, kAngStiffRigid, kBreakMinor);
                    }
                }
            }
        }
    }

    // 7. Partition Bulkheads
    const int bulkheadBays[3] = {9, 20, 28};
    for (int bi = 0; bi < 3; bi++)
    {
        int bb = bulkheadBays[bi];
        float xC = xStart + bb * bayPitch;
        float zC = getBayCenterZ(xC);

        // On the bay boundary, not on the bay centre: a wall at xC lands inside the seat
        // row and the overhead bins of that bay, which are not jointed to it.
        const float xWall = xC - bayPitch * 0.5f;

        // Floor to shoulder height, y from 0.15m to 1.20m. The old 1.40 x 1.80m panel put
        // its top outboard corner at r = 1.79m, well outside the 1.58m former ring, and
        // its base 0.35m below the cabin floor it is supposed to stand on.
        const float zWall = zC + 0.365f;
        int wallL = s->addBody({0.08f, 1.05f, 1.27f}, kDensityAlloy, kFrictionAir, {xWall, -0.675f, zWall});
        int wallR = s->addBody({0.08f, 1.05f, 1.27f}, kDensityAlloy, kFrictionAir, {xWall, +0.675f, zWall});

        if (cabinFloorL[bb] >= 0)
            s->addJointWorld(cabinFloorL[bb], wallL, {xC - bayLen * 0.5f, -0.675f, zC - 0.29f}, INFINITY,
                             kAngStiffRigid, kBreakFrame);
        if (cabinFloorR[bb] >= 0)
            s->addJointWorld(cabinFloorR[bb], wallR, {xC - bayLen * 0.5f, +0.675f, zC - 0.29f}, INFINITY,
                             kAngStiffRigid, kBreakFrame);
    }

    // 8. Cockpit Canopy & Windshield Visor (Matching reference image)
    {
        float xCockpit = xStart + 31 * bayPitch;
        float zCockpit = getBayCenterZ(xCockpit);
        int dash =
            s->addBody({0.70f, 1.50f, 0.40f}, kDensityLight, kFrictionAir, {xCockpit + 0.20f, 0.0f, zCockpit + 0.15f});
        if (cabinFloorL[28] >= 0)
            s->addJointWorld(cabinFloorL[28], dash, {xCockpit, 0.0f, zCockpit - 0.10f}, INFINITY, kAngStiffRigid,
                             kBreakMinor);

        // Pilot and Co-pilot seats
        int seatP1 = s->addBody({0.50f, 0.48f, 0.60f}, kDensityLight, kFrictionAir,
                                {xCockpit - 0.20f, -0.42f, zCockpit + 0.15f});
        int seatP2 = s->addBody({0.50f, 0.48f, 0.60f}, kDensityLight, kFrictionAir,
                                {xCockpit - 0.20f, +0.42f, zCockpit + 0.15f});
        s->addJointWorld(dash, seatP1, {xCockpit - 0.10f, -0.42f, zCockpit}, INFINITY, kAngStiffRigid, kBreakMinor);
        s->addJointWorld(dash, seatP2, {xCockpit - 0.10f, +0.42f, zCockpit}, INFINITY, kAngStiffRigid, kBreakMinor);

        // Kept inside the nose cone radius: the outboard panels used to protrude
        // through the skin bays they are not jointed to.
        for (int w = -1; w <= 1; w++)
        {
            float yWin = w * 0.45f;
            int win = s->addBody({0.60f, 0.34f, 0.08f}, kDensityLight, kFrictionAir,
                                 {xCockpit + 0.35f, yWin, zCockpit + 0.80f});
            s->addJointWorld(formerBodies[31][0], win, {xCockpit + 0.35f, yWin, zCockpit + 0.75f}, INFINITY,
                             kAngStiffRigid, kBreakMinor);
        }
    }

    // -------------------------------------------------------------------------
    // 3. Center Wing Carry-Through Box
    // -------------------------------------------------------------------------
    int carryThroughSpars[3];
    const float xCarry[3] = {+1.60f, 0.0f, -1.80f};
    for (int cs = 0; cs < 3; cs++)
    {
        // Half-span 1.15m. The box is 0.55m deep and its centreline sits 0.78m below the
        // fuselage axis, so its lower corners are 1.055m down, where the former ring only
        // leaves sqrt(1.58^2 - 1.055^2) = 1.18m of half-width. The old 3.60m spar drove
        // its ends straight out through the skin, the single worst overlap in the scene;
        // the wing root joints below reach the rest of the way as a lever.
        carryThroughSpars[cs] =
            s->addBody({0.28f, 2.30f, 0.55f}, kDensityHeavy, kFrictionAir, {xCarry[cs], 0.0f, zWingBox});
        s->addJointWorld(formerBodies[17][8], carryThroughSpars[cs], {xCarry[cs], 0.0f, zPlane - 0.90f}, INFINITY,
                         kAngStiffRigid, kBreakHeavy);
        s->addJointWorld(formerBodies[17][4], carryThroughSpars[cs], {xCarry[cs], +1.15f, zWingBox}, INFINITY,
                         kAngStiffRigid, kBreakHeavy);
        s->addJointWorld(formerBodies[17][12], carryThroughSpars[cs], {xCarry[cs], -1.15f, zWingBox}, INFINITY,
                         kAngStiffRigid, kBreakHeavy);
    }

    // -------------------------------------------------------------------------
    // 4. Main Wings (Multi-Spar Box Structure with Slats, Ribs, Flaps & Ailerons)
    // -------------------------------------------------------------------------
    const int nWingStations = 18;
    const float yWingRoot = 1.80f;
    const float yWingTip = 19.20f;
    const float dSpan = (yWingTip - yWingRoot) / (float)nWingStations; // ~0.967m per station
    const float wingBayW = dSpan * 0.96f;
    const float tanDihedral = 0.0524f; // 3.0 degrees dihedral

    int wingFrontSpar[2][18];
    int wingRearSpar[2][18];
    int wingUpperSkin[2][18];
    int wingLowerSkin[2][18];
    int wingRib[2][18];
    int wingFlapAileron[2][18];

    for (int side = 0; side < 2; side++)
    {
        const float sgn = (side == 0) ? -1.0f : +1.0f;

        for (int st = 0; st < nWingStations; st++)
        {
            const float u = (float)st / (float)(nWingStations - 1);
            const float yC = sgn * (yWingRoot + (st + 0.5f) * dSpan);
            const float zC = zWingBox + tanDihedral * ((st + 0.5f) * dSpan);

            const float rollAngle = sgn * 0.0524f;
            const quat rotDihedral = {sinf(rollAngle * 0.5f), 0.0f, 0.0f, cosf(rollAngle * 0.5f)};

            const float xLE = +2.20f - 1.60f * u;
            const float xTE = -3.00f + 1.80f * u;
            const float chord = xLE - xTE;
            const float thick = (0.55f - 0.28f * u) * 0.95f;

            // Skin mid-plane. The front spar is 0.90 x thick deep, so a skin centred on
            // thick * 0.45 is half buried in it; this puts the inner face of the skin on
            // the spar cap instead, with 5mm of clearance.
            const float zSkinOff = thick * 0.45f + 0.045f;

            const float xFS = xLE - chord * 0.22f;
            const float xRS = xTE + chord * 0.25f;
            const float boxChord = xFS - xRS;

            // 1. Leading Edge Aerodynamic Slat / Nose Block
            const float slatChord = (xLE - xFS) * 0.92f;
            int slat =
                s->addBody({slatChord, wingBayW, thick * 0.55f}, kDensityLight, kFrictionAir,
                           {xFS + slatChord * 0.5f + 0.02f, yC, zC + thick * 0.05f}, {0, 0, 0}, {0, 0, 0}, rotDihedral);

            // 2. Front Spar (Primary bending shear web)
            wingFrontSpar[side][st] = s->addBody({0.20f, wingBayW, thick * 0.90f}, kDensityHeavy, kFrictionAir,
                                                 {xFS, yC, zC}, {0, 0, 0}, {0, 0, 0}, rotDihedral);

            // 3. Rear Spar (Secondary shear web)
            wingRearSpar[side][st] = s->addBody({0.18f, wingBayW, thick * 0.75f}, kDensityAlloy, kFrictionAir,
                                                {xRS, yC, zC}, {0, 0, 0}, {0, 0, 0}, rotDihedral);

            // 4. Upper Skin Panel
            wingUpperSkin[side][st] =
                s->addBody({boxChord * 0.95f, wingBayW, 0.08f}, kDensityLight, kFrictionAir,
                           {(xFS + xRS) * 0.5f, yC, zC + zSkinOff}, {0, 0, 0}, {0, 0, 0}, rotDihedral);

            // 5. Lower Skin Panel
            wingLowerSkin[side][st] =
                s->addBody({boxChord * 0.95f, wingBayW, 0.08f}, kDensityLight, kFrictionAir,
                           {(xFS + xRS) * 0.5f, yC, zC - zSkinOff}, {0, 0, 0}, {0, 0, 0}, rotDihedral);

            // 6. Internal Chordwise Wing Rib
            wingRib[side][st] = s->addBody({boxChord * 0.90f, 0.12f, thick * 0.80f}, kDensityAlloy, kFrictionAir,
                                           {(xFS + xRS) * 0.5f, yC, zC}, {0, 0, 0}, {0, 0, 0}, rotDihedral);

            // 7. Internal Spanwise Stiffener Stringer
            // Tucked immediately under the upper skin, which is where a stiffener goes and
            // also the only place it does not share volume with the skin it stiffens.
            const float zStringer = zC + zSkinOff - 0.085f;
            int stringer = s->addBody({0.08f, wingBayW, 0.08f}, kDensityAlloy, kFrictionAir,
                                      {(xFS + xRS) * 0.5f, yC, zStringer}, {0, 0, 0}, {0, 0, 0}, rotDihedral);
            s->addJointWorld(wingRib[side][st], stringer, {(xFS + xRS) * 0.5f, yC, zStringer}, INFINITY, kAngStiffRigid,
                             kBreakHeavy);

            // 8. Trailing Edge Control Surface (Flaps inboard st <= 10, Ailerons outboard st > 10)
            const float flapChord = (xRS - xTE) * 0.90f;
            wingFlapAileron[side][st] =
                s->addBody({flapChord, wingBayW, thick * 0.35f}, kDensityLight, kFrictionAir,
                           {xRS - flapChord * 0.5f - 0.05f, yC, zC - thick * 0.05f}, {0, 0, 0}, {0, 0, 0}, rotDihedral);

            // Structural chordwise joints
            s->addJointWorld(slat, wingFrontSpar[side][st], {xFS, yC, zC}, INFINITY, kAngStiffRigid, kBreakSkin);
            s->addJointWorld(wingFrontSpar[side][st], wingRib[side][st], {xFS - 0.10f, yC, zC}, INFINITY,
                             kAngStiffRigid, kBreakHeavy);
            s->addJointWorld(wingRearSpar[side][st], wingRib[side][st], {xRS + 0.10f, yC, zC}, INFINITY, kAngStiffRigid,
                             kBreakHeavy);
            s->addJointWorld(wingUpperSkin[side][st], wingRib[side][st], {(xFS + xRS) * 0.5f, yC, zC + thick * 0.40f},
                             INFINITY, kAngStiffPanel, kBreakSkin);
            s->addJointWorld(wingLowerSkin[side][st], wingRib[side][st], {(xFS + xRS) * 0.5f, yC, zC - thick * 0.40f},
                             INFINITY, kAngStiffPanel, kBreakSkin);
            s->addJointWorld(wingRearSpar[side][st], wingFlapAileron[side][st], {xRS - 0.05f, yC, zC}, INFINITY,
                             kAngStiffHinge, kBreakMinor);

            // Spanwise continuity
            if (st > 0)
            {
                const float yInterface = sgn * (yWingRoot + st * dSpan);
                const float zInterface = zWingBox + tanDihedral * (st * dSpan);
                const float localBreakHeavy = kBreakHeavy * (1.0f - 0.7f * u); // Decreases to 30% strength at tips
                const float localBreakSkin = kBreakSkin * (1.0f - 0.7f * u);

                s->addJointWorld(wingFrontSpar[side][st - 1], wingFrontSpar[side][st], {xFS, yInterface, zInterface},
                                 INFINITY, kAngStiffRigid, localBreakHeavy);
                s->addJointWorld(wingRearSpar[side][st - 1], wingRearSpar[side][st], {xRS, yInterface, zInterface},
                                 INFINITY, kAngStiffRigid, localBreakHeavy);
                s->addJointWorld(wingUpperSkin[side][st - 1], wingUpperSkin[side][st],
                                 {(xFS + xRS) * 0.5f, yInterface, zInterface + zSkinOff}, INFINITY, kAngStiffPanel,
                                 localBreakSkin);
                s->addJointWorld(wingLowerSkin[side][st - 1], wingLowerSkin[side][st],
                                 {(xFS + xRS) * 0.5f, yInterface, zInterface - zSkinOff}, INFINITY, kAngStiffPanel,
                                 localBreakSkin);
            }
            else
            {
                s->addJointWorld(carryThroughSpars[0], wingFrontSpar[side][0], {xCarry[0], sgn * yWingRoot, zWingBox},
                                 INFINITY, kAngStiffRigid, kBreakHeavy);
                s->addJointWorld(carryThroughSpars[2], wingRearSpar[side][0], {xCarry[2], sgn * yWingRoot, zWingBox},
                                 INFINITY, kAngStiffRigid, kBreakHeavy);
            }
        }

        // Wingtip Fence / Winglet
        const float yTipEdge = sgn * (yWingTip + 0.10f);
        const float zTipEdge = zWingBox + tanDihedral * (yWingTip - yWingRoot);
        int winglet =
            s->addBody({1.20f, 0.08f, 0.85f}, kDensityLight, kFrictionAir, {0.0f, yTipEdge, zTipEdge + 0.50f});
        s->addJointWorld(wingFrontSpar[side][nWingStations - 1], winglet, {0.0f, yTipEdge, zTipEdge}, INFINITY,
                         kAngStiffRigid, kBreakSkin);
    }

    // -------------------------------------------------------------------------
    // 5. Four Engine Nacelles & Rotating Propellers
    // -------------------------------------------------------------------------
    const int engWingStation[4] = {10, 4, 4, 10};
    const int engSide[4] = {0, 0, 1, 1};

    const float nacLen = 3.80f;
    const float nacRadius = 0.65f;
    const float propSpinRate = 25.0f; // ~240 RPM

    for (int e = 0; e < 4; e++)
    {
        const int side = engSide[e];
        const int st = engWingStation[e];
        const float sgnEng = (side == 0) ? -1.0f : +1.0f;

        // Sit exactly under the centre of the wing station the pylons bolt to.
        const float yEng = sgnEng * (yWingRoot + (st + 0.5f) * dSpan);
        const float zWingAtEng = zWingBox + tanDihedral * ((st + 0.5f) * dSpan);

        // Underside of the wing box at this station (skin panel bottom face).
        const float uEng = (float)st / (float)(nWingStations - 1);
        const float thickEng = (0.55f - 0.28f * uEng) * 0.95f;
        const float zWingLower = zWingAtEng - thickEng * 0.45f - 0.085f;

        // Nacelle axis low enough that the cowling ring clears the wing underside.
        const float zEng = zWingLower - 0.95f - nacRadius;
        const float xEngFront = +3.80f;
        const float xEngCenter = xEngFront - nacLen * 0.5f;

        // 1. Heavy Central Engine Core / Crankcase Block
        int engCore = s->addBody({nacLen * 0.70f, 0.70f, 0.70f}, kDensityHeavy, kFrictionAir, {xEngCenter, yEng, zEng});

        // 2. Engine Pylon Struts
        // Run from the core top face up to the wing underside, with a 0.04m gap so the
        // pylons never bury themselves in skin panels they are not jointed to.
        const float pylonH = (zWingLower - 0.04f) - (zEng + 0.35f);
        const float zPylon = (zWingLower - 0.04f + zEng + 0.35f) * 0.5f;

        int pylonFront =
            s->addBody({0.30f, 0.18f, pylonH}, kDensityHeavy, kFrictionAir, {xEngCenter + 0.60f, yEng, zPylon});
        int pylonRear =
            s->addBody({0.30f, 0.18f, pylonH}, kDensityHeavy, kFrictionAir, {xEngCenter - 0.60f, yEng, zPylon});

        s->addJointWorld(engCore, pylonFront, {xEngCenter + 0.60f, yEng, zEng + 0.35f}, INFINITY, kAngStiffRigid,
                         kBreakPylon);
        s->addJointWorld(engCore, pylonRear, {xEngCenter - 0.60f, yEng, zEng + 0.35f}, INFINITY, kAngStiffRigid,
                         kBreakPylon);

        s->addJointWorld(wingFrontSpar[side][st], pylonFront, {xEngCenter + 0.60f, yEng, zWingLower}, INFINITY,
                         kAngStiffRigid, kBreakPylon);
        s->addJointWorld(wingLowerSkin[side][st], pylonRear, {xEngCenter - 0.60f, yEng, zWingLower}, INFINITY,
                         kAngStiffRigid, kBreakPylon);

        // 3. Octagonal Engine Cowling Shell (3 rings of 8 panels = 24 cowl panels per nacelle)
        const int nCowlFacets = 8;
        const float dPhiCowl = 2.0f * 3.1415926535f / (float)nCowlFacets;
        const float cowlRingLen = nacLen / 3.0f;
        const float cowlPanelW = 2.0f * nacRadius * sinf(dPhiCowl * 0.5f) * 0.94f;

        for (int cr = 0; cr < 3; cr++)
        {
            float xRingC = (xEngFront - nacLen) + (cr + 0.5f) * cowlRingLen;

            for (int cf = 0; cf < nCowlFacets; cf++)
            {
                if (cf == 0)
                    continue; // Top facet omitted: the pylons pass through this gap.

                float phi = cf * dPhiCowl;
                float cosP = cosf(phi);
                float sinP = sinf(phi);

                float halfP = -phi * 0.5f; // see fuselage rotRoll: radial roll is -phi
                quat rotCowl = {sinf(halfP), 0.0f, 0.0f, cosf(halfP)};

                vec3 pCowl = {xRingC, yEng + nacRadius * sinP, zEng + nacRadius * cosP};
                int cowl = s->addBody({cowlRingLen * 0.96f, cowlPanelW, 0.06f}, kDensityLight, kFrictionAir, pCowl,
                                      {0, 0, 0}, {0, 0, 0}, rotCowl);

                vec3 pAnchorCore = {xRingC, yEng + 0.35f * sinP, zEng + 0.35f * cosP};
                s->addJointWorld(engCore, cowl, pAnchorCore, INFINITY, kAngStiffRigid, kBreakSkin);
            }
        }

        // 4. Aerodynamic Intake Guide Vanes (Front lip airflow guides)
        for (int v = 0; v < 4; v++)
        {
            float vPhi = v * 3.1415926535f * 0.5f;
            float halfV = -vPhi * 0.5f; // see fuselage rotRoll: radial roll is -phi
            quat rotVane = {sinf(halfV), 0.0f, 0.0f, cosf(halfV)};
            vec3 pVane = {xEngFront - 0.15f, yEng + 0.48f * sinf(vPhi), zEng + 0.48f * cosf(vPhi)};
            int vane =
                s->addBody({0.20f, 0.08f, 0.22f}, kDensityLight, kFrictionAir, pVane, {0, 0, 0}, {0, 0, 0}, rotVane);
            s->addJointWorld(engCore, vane, pVane, INFINITY, kAngStiffRigid, kBreakMinor);
        }

        // 5. Propeller Assembly (Freely Rotating Bearing Joint)
        const float xPropFront = xEngFront + 0.35f;
        const vec3 pPropHub = {xPropFront, yEng, zEng};

        const float spinDir = (e < 2) ? +1.0f : -1.0f;
        const vec3 propAngVel = {spinDir * propSpinRate, 0.0f, 0.0f};

        int propHub = s->addBody({0.55f, 0.36f, 0.36f}, kDensityHeavy, kFrictionAir, pPropHub, {0, 0, 0}, propAngVel);

        // Spherical bearing joint: stiffnessLin = INFINITY, stiffnessAng = 0.0f
        const vec3 pBearingAnchor = {xEngFront, yEng, zEng};
        s->addJointWorld(engCore, propHub, pBearingAnchor, INFINITY, 0.0f, kBreakPropHub);

        // 4 Propeller Blades
        const float bladeLen = 1.15f;
        const float bladeW = 0.22f;
        const float bladeT = 0.05f;

        const vec3 bladeOffsets[4] = {{0.0f, 0.0f, +(0.18f + bladeLen * 0.5f)},
                                      {0.0f, +(0.18f + bladeLen * 0.5f), 0.0f},
                                      {0.0f, 0.0f, -(0.18f + bladeLen * 0.5f)},
                                      {0.0f, -(0.18f + bladeLen * 0.5f), 0.0f}};

        const vec3 bladeSizes[4] = {{bladeT, bladeW, bladeLen},
                                    {bladeT, bladeLen, bladeW},
                                    {bladeT, bladeW, bladeLen},
                                    {bladeT, bladeLen, bladeW}};

        for (int bl = 0; bl < 4; bl++)
        {
            vec3 pBlade = pPropHub + bladeOffsets[bl];
            int blade = s->addBody(bladeSizes[bl], kDensityLight, kFrictionAir, pBlade, {0, 0, 0}, propAngVel);

            vec3 pBladeRoot = pPropHub + bladeOffsets[bl] * 0.40f;
            s->addJointWorld(propHub, blade, pBladeRoot, INFINITY, kAngStiffRigid, kBreakBlade);
        }
    }

    // -------------------------------------------------------------------------
    // 6. Empennage (Vertical Fin & Rudder, Horizontal Tailplanes & Elevators)
    // -------------------------------------------------------------------------
    // 1. Vertical Stabilizer (Dorsal Fin & Rudder)
    const int nFinTiers = 10;
    // Seat the fin on the tail cone crown under its leading edge (it used to float).
    const float xFinBase = xStart + 5 * bayPitch; // -14.4m
    const float zFinRoot = getBayCenterZ(xFinBase) + getBayRadius(xFinBase);
    const float zFinTip = zFinRoot + 6.00f;
    const float dFinZ = (zFinTip - zFinRoot) / (float)nFinTiers;

    int prevFinSpar = -1;
    for (int ft = 0; ft < nFinTiers; ft++)
    {
        float u = (float)ft / (float)(nFinTiers - 1);
        float zTier = zFinRoot + (ft + 0.5f) * dFinZ;

        float xFinLE = -14.50f - 4.50f * u;
        float xFinTE = -20.20f - 0.40f * u;
        float finChord = xFinLE - xFinTE;
        float finThick = (0.36f - 0.18f * u) * 0.95f;

        float xFinSpar = xFinLE - finChord * 0.40f;
        float xRudder = xFinTE + finChord * 0.25f;

        int finSpar =
            s->addBody({0.24f, finThick, dFinZ * 0.95f}, kDensityAlloy, kFrictionAir, {xFinSpar, 0.0f, zTier});

        int finSkinL = s->addBody({finChord * 0.50f, 0.05f, dFinZ * 0.95f}, kDensityLight, kFrictionAir,
                                  {(xFinLE + xFinSpar) * 0.5f, -finThick * 0.45f, zTier});
        int finSkinR = s->addBody({finChord * 0.50f, 0.05f, dFinZ * 0.95f}, kDensityLight, kFrictionAir,
                                  {(xFinLE + xFinSpar) * 0.5f, +finThick * 0.45f, zTier});

        int rudder = s->addBody({(xFinSpar - xFinTE) * 0.85f, finThick * 0.60f, dFinZ * 0.95f}, kDensityLight,
                                kFrictionAir, {xRudder, 0.0f, zTier});

        s->addJointWorld(finSpar, finSkinL, {xFinSpar, -finThick * 0.45f, zTier}, INFINITY, kAngStiffPanel, kBreakSkin);
        s->addJointWorld(finSpar, finSkinR, {xFinSpar, +finThick * 0.45f, zTier}, INFINITY, kAngStiffPanel, kBreakSkin);
        s->addJointWorld(finSpar, rudder, {xFinSpar - 0.15f, 0.0f, zTier}, INFINITY, kAngStiffHinge, kBreakMinor);

        if (prevFinSpar >= 0)
        {
            float localBreakFrame = kBreakFrame * (1.0f - 0.7f * u);
            s->addJointWorld(prevFinSpar, finSpar, {xFinSpar, 0.0f, zTier - dFinZ * 0.5f}, INFINITY, kAngStiffRigid,
                             localBreakFrame);
        }
        else
        {
            s->addJointWorld(formerBodies[5][0], finSpar, {xFinSpar, 0.0f, zFinRoot}, INFINITY, kAngStiffRigid,
                             kBreakHeavy);
        }
        prevFinSpar = finSpar;
    }

    // 2. Horizontal Stabilizers (Tailplanes & Elevators)
    const int nTailStations = 9;
    const float yTailRoot = 0.80f;
    const float yTailTip = 7.20f;
    const float dTailSpan = (yTailTip - yTailRoot) / (float)nTailStations;

    for (int side = 0; side < 2; side++)
    {
        const float sgn = (side == 0) ? -1.0f : +1.0f;
        int prevTailSpar = -1;

        for (int tst = 0; tst < nTailStations; tst++)
        {
            float u = (float)tst / (float)(nTailStations - 1);
            float yT = sgn * (yTailRoot + (tst + 0.5f) * dTailSpan);
            float zT = zPlane + 0.60f + 0.035f * (yTailRoot + tst * dTailSpan);

            float xTLE = -16.50f - 1.20f * u;
            float xTTE = -19.50f + 0.60f * u;
            float tChord = xTLE - xTTE;
            float tThick = (0.30f - 0.14f * u) * 0.95f;

            float xTSpar = xTLE - tChord * 0.35f;
            float xElev = xTTE + tChord * 0.28f;

            int tailSpar =
                s->addBody({0.20f, dTailSpan * 0.95f, tThick}, kDensityAlloy, kFrictionAir, {xTSpar, yT, zT});
            int elevator = s->addBody({(xTSpar - xTTE) * 0.85f, dTailSpan * 0.95f, tThick * 0.40f}, kDensityLight,
                                      kFrictionAir, {xElev, yT, zT});

            s->addJointWorld(tailSpar, elevator, {xTSpar - 0.10f, yT, zT}, INFINITY, kAngStiffHinge, kBreakMinor);

            if (prevTailSpar >= 0)
            {
                float localBreakFrame = kBreakFrame * (1.0f - 0.7f * u);
                s->addJointWorld(prevTailSpar, tailSpar, {xTSpar, sgn * (yTailRoot + tst * dTailSpan), zT}, INFINITY,
                                 kAngStiffRigid, localBreakFrame);
            }
            else
            {
                s->addJointWorld(formerBodies[4][(side == 0) ? 12 : 4], tailSpar, {xTSpar, sgn * yTailRoot, zT},
                                 INFINITY, kAngStiffRigid, kBreakHeavy);
            }
            prevTailSpar = tailSpar;
        }
    }

    // -------------------------------------------------------------------------
    // 7. Tricycle Landing Gear (Nose Gear & Main Bogies)
    // -------------------------------------------------------------------------
    // 1. Nose Gear: located under forward fuselage at X = +14.0m, Y = 0.0m
    {
        const float xNose = +14.00f;
        const float zGround = 0.0f;
        const float wheelR = 0.42f;

        // Reach the fuselage underside at this station rather than floating below it.
        // The underside across the whole footprint of the strut, not just at its
        // centreline: the radius grows aft here, so the skin bay behind the strut hangs
        // lower than the station the strut is measured at.
        const float zNoseAttach = fminf(getBayCenterZ(xNose - 0.6f) - getBayRadius(xNose - 0.6f),
                                        getBayCenterZ(xNose + 0.6f) - getBayRadius(xNose + 0.6f)) -
                                  0.05f;
        const float noseStrutH = zNoseAttach - (zGround + wheelR);

        int noseStrut = s->addBody({0.24f, 0.24f, noseStrutH}, kDensityHeavy, kFrictionAir,
                                   {xNose, 0.0f, (zNoseAttach + zGround + wheelR) * 0.5f});
        s->addJointWorld(formerBodies[29][8], noseStrut, {xNose, 0.0f, zNoseAttach}, INFINITY, kAngStiffRigid,
                         kBreakGear);

        // Twin nose wheels
        for (int nw = 0; nw < 2; nw++)
        {
            float yNW = (nw == 0) ? -0.32f : +0.32f;
            int wheel = s->addBody({wheelR * 1.90f, 0.22f, wheelR * 1.90f}, kDensityHeavy, 0.7f,
                                   {xNose, yNW, zGround + wheelR});
            s->addJointWorld(noseStrut, wheel, {xNose, yNW, zGround + wheelR}, INFINITY, 0.0f, kBreakGear);
        }
    }

    // 2. Main Landing Gear Bogies: located under inboard wings at X = -0.60m, Y = ±5.20m
    for (int mg = 0; mg < 2; mg++)
    {
        const float sgn = (mg == 0) ? -1.0f : +1.0f;
        const float yMG = sgn * 5.20f;
        const float xMG = -0.60f;
        const float zGround = 0.0f;
        const float wheelR = 0.52f;

        // Stop clear of the wing underside at this spanwise position. The lower skin
        // hangs 0.30m below the box centreline here and the dihedral roll drops its
        // outboard edge another 25mm, so 0.42m leaves about 0.1m of daylight.
        const float zMGTop = zWingBox + tanDihedral * (5.20f - yWingRoot) - 0.42f;
        const float mainStrutH = zMGTop - (zGround + wheelR);

        int mainStrut = s->addBody({0.36f, 0.36f, mainStrutH}, kDensityHeavy, kFrictionAir,
                                   {xMG, yMG, (zMGTop + zGround + wheelR) * 0.5f});
        s->addJointWorld(wingFrontSpar[mg][4], mainStrut, {xMG, yMG, zMGTop}, INFINITY, kAngStiffRigid, kBreakGear);

        // 4-Wheel Main Gear Bogie Truck
        int bogieBeam = s->addBody({1.60f, 0.28f, 0.25f}, kDensityHeavy, kFrictionAir, {xMG, yMG, zGround + wheelR});
        s->addJointWorld(mainStrut, bogieBeam, {xMG, yMG, zGround + wheelR + 0.15f}, INFINITY, kAngStiffRigid,
                         kBreakGear);

        for (int fx = -1; fx <= 1; fx += 2)
        {
            for (int fy = -1; fy <= 1; fy += 2)
            {
                float xWheel = xMG + fx * 0.65f;
                float yWheel = yMG + fy * 0.35f;
                int wheel = s->addBody({wheelR * 1.92f, 0.26f, wheelR * 1.92f}, kDensityHeavy, 0.8f,
                                       {xWheel, yWheel, zGround + wheelR});
                s->addJointWorld(bogieBeam, wheel, {xWheel, yWheel, zGround + wheelR}, INFINITY, 0.0f, kBreakGear);
            }
        }
    }

    // Camera view: distance 65m, target height 4.5m along Z
    return {65.0f, 4.5f};
}

static SceneView scenePassengerPlaneGpu(SceneBuilder *s)
{
    return buildPassengerPlane(s);
}
