#include "scene_import.h"

#include "collision.h"
#include "shape_mass.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace
{

Avbd3dVec3 v3(const vec3 &v)
{
    return Avbd3dVec3{v.x, v.y, v.z};
}
Avbd3dQuat q4(const quat &q)
{
    return Avbd3dQuat{Avbd3dVec3{q.x, q.y, q.z}, q.w};
}

// A failed create leaves the id null; the demo reports it and carries on.
void must(Avbd3dResult r, const char *what)
{
    if (r != AVBD3D_OK)
        std::fprintf(stderr, "scene import: %s failed: %s\n", what, avbd3d_result_string(r));
}

unsigned long long pairKey(int a, int b)
{
    const int lo = std::min(a, b), hi = std::max(a, b);
    return ((unsigned long long)(unsigned)lo << 32) | (unsigned)hi;
}

} // namespace

void SceneImport::reset()
{
    m_bodies.clear();
    m_joints = m_springs = 0;
    m_pairs.clear();
}

void SceneImport::importTail(Avbd3dWorld *w, const SceneDesc &d)
{
    // Bodies. One centred primary shape each (the catalogue never calls addShape); its density
    // is chosen so the API's closed-form mass reproduces the desc's mass and moment exactly.
    Avbd3dBoxHull hull{};
    vec3 hullHalf{-1, -1, -1};
    for (int i = (int)m_bodies.size(); i < d.count; i++)
    {
        const bool dynamic = d.mass[i] > 0.0f;
        Avbd3dBodyDef bd = avbd3d_default_body_def();
        bd.type = dynamic ? AVBD3D_DYNAMIC_BODY : AVBD3D_STATIC_BODY;
        bd.position = v3(d.pos[i]);
        bd.rotation = q4(d.rot[i]);
        bd.linearVelocity = v3(d.velLin[i]);
        bd.angularVelocity = v3(d.velAng[i]);
        bd.isAwake = !(i < (int)d.asleep.size() && d.asleep[i]);
        Avbd3dBodyId b{};
        must(avbd3d_create_body(w, &bd, &b), "create_body");
        m_bodies.push_back(b);

        const int kind = d.shapeType[i];
        const vec3 h = d.halfExtents[i];
        Avbd3dShapeDef sd = avbd3d_default_shape_def();
        Avbd3dShapeId sid{};
        sd.baseMaterial.friction = d.friction[i];
        sd.density = 0.0f;
        if (dynamic)
        {
            float unitMass;
            vec3 unitMoment;
            shapeMassProperties(kind, h, 1.0f, unitMass, unitMoment);
            sd.density = unitMass > 0.0f ? d.mass[i] / unitMass : 1.0f;
        }
        if (kind == AVBD_SHAPE_SPHERE)
        {
            const Avbd3dSphere s{Avbd3dVec3{0, 0, 0}, h.x};
            must(avbd3d_create_sphere_shape(b, &sd, &s, &sid), "create_sphere_shape");
        }
        else if (kind == AVBD_SHAPE_CAPSULE)
        {
            const float seg = h.z - h.x;
            const Avbd3dCapsule c{Avbd3dVec3{0, 0, -seg}, Avbd3dVec3{0, 0, seg}, h.x};
            must(avbd3d_create_capsule_shape(b, &sd, &c, &sid), "create_capsule_shape");
        }
        else
        {
            if (!(h.x == hullHalf.x && h.y == hullHalf.y && h.z == hullHalf.z))
            {
                hull = avbd3d_make_box_hull(h.x, h.y, h.z);
                hullHalf = h;
            }
            must(avbd3d_create_hull_shape(b, &sd, avbd3d_box_hull_data(&hull), &sid), "create_hull_shape");
        }
    }

    // Is a pair in the desc's ignore list (jointed neighbours and ignoreCollision)?
    const auto ignored = [&](int a, int b)
    {
        if (a < 0 || b < 0)
            return false;
        return std::binary_search(d.ignorePairs.begin(), d.ignorePairs.end(), pairKey(a, b));
    };
    // A joint or spring between two bodies collides unless the desc ignores the pair; the API
    // then adds the same ignore, so the pair needs no filter joint below.
    const auto baseOf = [&](Avbd3dJointDef &base, int a, int b)
    {
        base.bodyIdA = body(a);
        base.bodyIdB = body(b);
        base.collideConnected = a >= 0 && b >= 0 && !ignored(a, b);
        if (a >= 0 && b >= 0 && !base.collideConnected)
            m_pairs.insert(pairKey(a, b));
    };

    for (; m_joints < d.jointCount; m_joints++)
    {
        const int j = m_joints;
        Avbd3dGenericJointDef jd = avbd3d_default_generic_joint_def();
        baseOf(jd.base, d.jointA[j], d.jointB[j]);
        jd.base.localFrameA.p = v3(d.jointRA[j]);
        jd.base.localFrameB.p = v3(d.jointRB[j]);
        jd.kind = d.jointGeneric[j];
        jd.linearStiffness = d.jointKLin[j];
        jd.angularStiffness = d.jointKAng[j];
        jd.fractureForce = d.jointFrac[j];
        jd.fractureTorque = d.jointFracAng[j];
        if (jd.kind != AVBD3D_GENERIC_POINT)
        {
            jd.base.localFrameA.q = q4(d.jointFrameA[j]);
            jd.base.localFrameB.q = q4(d.jointFrameB[j]);
            for (int k = 0; k < 6; k++)
                jd.axes[k] =
                    Avbd3dGenericAxis{d.jointAxisMode[j * 6 + k], d.jointAxisLo[j * 6 + k], d.jointAxisHi[j * 6 + k]};
        }
        Avbd3dJointId jid{};
        must(avbd3d_create_generic_joint(w, &jd, &jid), "create_generic_joint");
    }

    // Springs.
    for (; m_springs < d.springCount; m_springs++)
    {
        const int s = m_springs;
        Avbd3dSpringDef sd = avbd3d_default_spring_def();
        baseOf(sd.base, d.springA[s], d.springB[s]);
        sd.base.localFrameA.p = v3(d.springRA[s]);
        sd.base.localFrameB.p = v3(d.springRB[s]);
        sd.stiffness = d.springK[s];
        sd.restLength = d.springRest[s];
        Avbd3dJointId sid{};
        must(avbd3d_create_spring(w, &sd, &sid), "create_spring");
    }

    // The ignoreCollision pairs no joint or spring covered become filter joints.
    for (unsigned long long k : d.ignorePairs)
    {
        const int a = (int)(k >> 32), b = (int)(k & 0xFFFFFFFFu);
        if (a >= (int)m_bodies.size() || b >= (int)m_bodies.size())
            continue;
        if (!m_pairs.insert(k).second)
            continue;
        Avbd3dFilterJointDef fd = avbd3d_default_filter_joint_def();
        fd.base.bodyIdA = body(a);
        fd.base.bodyIdB = body(b);
        Avbd3dJointId fid{};
        must(avbd3d_create_filter_joint(w, &fd, &fid), "create_filter_joint");
    }
}

Avbd3dSolverDef solverDefOf(const SolverParams &p)
{
    Avbd3dSolverDef d = avbd3d_default_solver_def();
    d.iterations = p.iterations;
    d.primalLanes = p.primalLanes;
    d.alpha = p.alpha;
    d.betaLin = p.betaLin;
    d.betaAng = p.betaAng;
    d.gamma = p.gamma;
    d.autoKStart = p.autoKStart;
    d.dualDamping = p.dualDamping;
    d.dualDampVel = p.dualDampVel;
    d.sleepFrames = p.sleepFrames;
    d.sleepLinearVelocity = p.sleepLinVel;
    d.sleepAngularVelocity = p.sleepAngVel;
    d.sleepDisplacement = p.sleepDisp;
    d.sleepFracture = p.sleepFracture;
    d.sleepLeak = p.sleepLeak;
    d.enableKillFallen = p.killFallenEnabled;
    d.killHeight = p.killHeight;
    d.grabStrength = p.grabStrength;
    d.contactCap = p.contactCap;
    d.coarseIterations = p.coarseIterations;
    d.coarseCell = p.coarseCell;
    d.coarseDamping = p.coarseDamping;
    return d;
}
