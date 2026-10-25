// Most gates were written against the z-up frame (ground at z = 0, gravity -z). The engine default is now
// y-up, so those gates ask for z-up gravity explicitly instead of rewriting their geometry.
#pragma once
#include "solver_params.h"

inline SolverParams zUpParams()
{
    SolverParams p = defaultSolverParams();
    p.gravity = vec3{0.0f, 0.0f, -10.0f};
    return p;
}
