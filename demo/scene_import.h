#pragma once

// SceneImport: replays a SceneBuilder scene into an avbd3d world through the public API.
//
// The demo keeps SceneBuilder as its authoring library (the scene catalogue and the spawn
// helpers are written against it) and runs the solver only through src/api. A SceneDesc
// is append-only, so a spawn re-finalizes the builder and imports just the new tail: bodies,
// joints and springs past the counts already imported, and the collision-ignore pairs not yet
// seen. Desc body i maps to one Avbd3dBodyId.

#include "scene_desc.h"
#include "solver_params.h"

#include "avbd3d/avbd3d.h"
#include "avbd3d/avbd3d_geometry.h"

#include <unordered_set>
#include <vector>

class SceneImport
{
  public:
    // Forget everything imported (a new world).
    void reset();

    // Create everything in `d` past what is already imported.
    void importTail(Avbd3dWorld *w, const SceneDesc &d);

    Avbd3dBodyId body(int descIndex) const
    {
        return descIndex >= 0 && descIndex < (int)m_bodies.size() ? m_bodies[descIndex] : Avbd3dBodyId{};
    }

  private:
    std::vector<Avbd3dBodyId> m_bodies;
    int m_joints = 0, m_springs = 0;
    std::unordered_set<unsigned long long> m_pairs; // ignore pairs already applied
};

// SolverParams -> the API's defs. Gravity and sleep are separate world calls.
Avbd3dSolverDef solverDefOf(const SolverParams &p);
