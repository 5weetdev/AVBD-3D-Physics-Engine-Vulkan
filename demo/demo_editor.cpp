// The scene editor: while the simulation is paused, bodies can be selected (click, shift /
// ctrl click, a marquee box, the usual hotkeys) and moved along the three world axes with a
// gizmo drawn at the middle of the selection. Everything here is part of DemoApp; the state
// lives in demo_app.h.
//
// Selection is a set of desc body indices (SceneImport order). Nothing in here touches the
// world from drawUI(), which runs while the frame's command buffer is recording: the panel
// and the hotkeys queue an EditAction and update() carries it out.

#include "demo_app.h"

#include "shape_mass.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <float.h>
#include <stdio.h>
#include <unordered_map>

static Avbd3dVec3 toApi(vec3 v)
{
    return Avbd3dVec3{v.x, v.y, v.z};
}
static vec3 fromApi(Avbd3dVec3 v)
{
    return vec3{v.x, v.y, v.z};
}
static void must(Avbd3dResult r, const char *what)
{
    if (r != AVBD3D_OK)
        fprintf(stderr, "avbd3d_demo: %s failed: %s\n", what, avbd3d_result_string(r));
}

static const int kMaxOutlines = 1500; // selected bodies that get an outline; the rest still move
static const float kGizmoPixels = 110.0f;
static const float kPickPixels = 9.0f;
static const float kClickSlopPixels = 4.0f;

// ---------------------------------------------------------------------------
// Selection set
// ---------------------------------------------------------------------------

void DemoApp::editorReset()
{
    m_edSel.clear();
    m_edMask.clear();
    m_edDeleted.clear();
    m_edPending = EditNone;
    m_edDown = m_edBox = false;
    m_edAxis = m_edAxisHover = -1;
}

bool DemoApp::editorStatic(int i) const
{
    Avbd3dBodyType type = AVBD3D_STATIC_BODY;
    return avbd3d_body_get_type(m_import.body(i), &type) != AVBD3D_OK || type == AVBD3D_STATIC_BODY;
}

void DemoApp::editorSelect(int i, bool on)
{
    if (!editorAlive(i))
        return;
    if ((int)m_edMask.size() <= i)
        m_edMask.resize(editorBodyCount(), 0);
    if ((bool)m_edMask[i] == on)
        return;
    m_edMask[i] = on;
    if (on)
        m_edSel.push_back(i);
    else
        m_edSel.erase(std::find(m_edSel.begin(), m_edSel.end(), i));
}

void DemoApp::editorClear()
{
    for (int i : m_edSel)
        m_edMask[i] = 0;
    m_edSel.clear();
}

// ---------------------------------------------------------------------------
// Projection and picking
// ---------------------------------------------------------------------------

// World point -> drawable pixels, the inverse of rayFromPixel(). False behind the camera.
bool DemoApp::project(vec3 p, float &sx, float &sy) const
{
    int w = 0, h = 0;
    m_host.drawableSize(w, h);
    if (w <= 0 || h <= 0)
        return false;
    const vec3 eye = camEye();
    const vec3 f = normalize(m_dispTarget - eye);
    vec3 r = cross(f, vec3{0, 1, 0});
    if (lengthSq(r) < 1e-6f)
        r = cross(f, vec3{0, 0, 1});
    r = normalize(r);
    const vec3 u = cross(r, f);
    const vec3 v = p - eye;
    const float z = dot(v, f);
    if (z < kNear)
        return false;
    const float tanHalf = tanf(rad(kFovY) * 0.5f);
    const float aspect = (float)w / (float)h;
    const float x = dot(v, r) / (z * tanHalf * aspect);
    const float y = dot(v, u) / (z * tanHalf);
    sx = (x * 0.5f + 0.5f) * (float)w;
    sy = (0.5f - y * 0.5f) * (float)h;
    return true;
}

// Closest body under the pixel, statics included (a click can select a wall; the bulk
// operations never do). -1 when the ray hits nothing.
int DemoApp::editorPick(int px, int py) const
{
    vec3 o, d;
    if (!rayFromPixel(px, py, o, d))
        return -1;
    struct Pick
    {
        Avbd3dShapeId shape;
        float fraction;
    } pick{{}, 1.0f};
    auto onHit = [](Avbd3dShapeId shape, Avbd3dVec3, Avbd3dVec3, float fraction, uint64_t, int, int, void *ctx) -> float
    {
        Pick *p = (Pick *)ctx;
        p->shape = shape;
        p->fraction = fraction;
        return fraction;
    };
    must(avbd3d_world_cast_ray(m_world, toApi(o), toApi(d * kFar), avbd3d_default_query_filter(), onHit, &pick),
         "world_cast_ray");
    if (pick.shape.index1 == 0)
        return -1;
    Avbd3dBodyId hit{};
    if (avbd3d_shape_get_body(pick.shape, &hit) != AVBD3D_OK)
        return -1;
    for (int i = 0; i < editorBodyCount(); i++)
        if (editorAlive(i) && m_import.body(i).index1 == hit.index1)
            return i;
    return -1;
}

// ---------------------------------------------------------------------------
// Gizmo
// ---------------------------------------------------------------------------

static const vec3 kAxes[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

// The world length that makes the gizmo kGizmoPixels long on screen.
static float gizmoWorldLength(const DemoApp &app, vec3 c, float pixelsPerUnitAtOne)
{
    (void)pixelsPerUnitAtOne;
    return length(c - app.camEye()) * 0.18f;
}

// Screen segments of the three axes: out[a] = {x0, y0, x1, y1} in drawable pixels.
bool DemoApp::gizmoAxes(float out[3][4], float &len) const
{
    if (m_edSel.empty())
        return false;
    len = gizmoWorldLength(*this, m_edCenter, 1.0f);
    float cx, cy;
    if (!project(m_edCenter, cx, cy))
        return false;
    for (int a = 0; a < 3; a++)
    {
        float ex, ey;
        if (!project(m_edCenter + kAxes[a] * len, ex, ey))
        {
            // The tip is behind the camera: fall back to a short stub along the axis.
            if (!project(m_edCenter + kAxes[a] * (len * 0.2f), ex, ey))
                return false;
        }
        out[a][0] = cx;
        out[a][1] = cy;
        out[a][2] = ex;
        out[a][3] = ey;
    }
    return true;
}

static float distToSegment(float px, float py, float x0, float y0, float x1, float y1)
{
    const float dx = x1 - x0, dy = y1 - y0;
    const float l2 = dx * dx + dy * dy;
    float t = l2 > 0.0f ? ((px - x0) * dx + (py - y0) * dy) / l2 : 0.0f;
    t = std::min(std::max(t, 0.0f), 1.0f);
    const float qx = x0 + t * dx - px, qy = y0 + t * dy - py;
    return sqrtf(qx * qx + qy * qy);
}

int DemoApp::gizmoHit(int px, int py) const
{
    float seg[3][4], len;
    if (!gizmoAxes(seg, len))
        return -1;
    int best = -1;
    float bestD = kPickPixels;
    for (int a = 0; a < 3; a++)
    {
        const float d = distToSegment((float)px, (float)py, seg[a][0], seg[a][1], seg[a][2], seg[a][3]);
        if (d < bestD)
        {
            bestD = d;
            best = a;
        }
    }
    return best;
}

// Where along the line (origin + t * axis) the mouse ray passes closest. False when the
// ray is parallel to the axis, i.e. the axis points at the camera and cannot be dragged.
bool DemoApp::axisParam(int px, int py, vec3 origin, vec3 axis, float &t) const
{
    vec3 o, d;
    if (!rayFromPixel(px, py, o, d))
        return false;
    const vec3 w0 = o - origin;
    const float b = dot(d, axis);
    const float dd = dot(d, w0), e = dot(axis, w0);
    const float denom = 1.0f - b * b;
    if (denom < 1e-3f)
        return false;
    t = (e - b * dd) / denom;
    return true;
}

void DemoApp::editorApplyDelta(vec3 delta)
{
    for (size_t k = 0; k < m_edSel.size(); k++)
    {
        const Avbd3dBodyId id = m_import.body(m_edSel[k]);
        must(avbd3d_body_set_transform(id, toApi(m_edStartPos[k] + delta),
                                       Avbd3dQuat{Avbd3dVec3{m_edStartRot[k].x, m_edStartRot[k].y, m_edStartRot[k].z},
                                                  m_edStartRot[k].w}),
             "body_set_transform");
    }
}

// ---------------------------------------------------------------------------
// Mouse
// ---------------------------------------------------------------------------

void DemoApp::editorMouseDown(int px, int py)
{
    const SDL_Keymod mod = SDL_GetModState();
    m_edMode = (mod & KMOD_CTRL) ? 2 : (mod & KMOD_SHIFT) ? 1 : 0;
    m_edDown = true;
    m_edBox = false;
    m_edX0 = m_edX1 = px;
    m_edY0 = m_edY1 = py;

    const int axis = gizmoHit(px, py);
    if (axis < 0)
        return;
    // Start a gizmo drag: remember where every selected body is, and where on the axis
    // the cursor grabbed it.
    float t0;
    if (!axisParam(px, py, m_edCenter, kAxes[axis], t0))
        return;
    m_edAxis = axis;
    m_edAxisDir = kAxes[axis];
    m_edAnchor = m_edCenter;
    m_edT0 = t0;
    m_edStartPos.resize(m_edSel.size());
    m_edStartRot.resize(m_edSel.size());
    for (size_t k = 0; k < m_edSel.size(); k++)
    {
        Avbd3dTransform xf{};
        must(avbd3d_body_get_transform(m_import.body(m_edSel[k]), &xf), "body_get_transform");
        m_edStartPos[k] = fromApi(xf.p);
        m_edStartRot[k] = quat{xf.q.v.x, xf.q.v.y, xf.q.v.z, xf.q.s};
    }
}

void DemoApp::editorMouseUp(int px, int py)
{
    m_edDown = false;
    m_edX1 = px;
    m_edY1 = py;

    if (m_edAxis >= 0)
    {
        // Teleported bodies should not carry the velocity they had, and a sleeper that was
        // moved has to wake or it stays frozen at its old contacts.
        for (int i : m_edSel)
        {
            const Avbd3dBodyId id = m_import.body(i);
            if (!editorStatic(i))
            {
                must(avbd3d_body_set_linear_velocity(id, Avbd3dVec3{0, 0, 0}), "set_linear_velocity");
                must(avbd3d_body_set_angular_velocity(id, Avbd3dVec3{0, 0, 0}), "set_angular_velocity");
                must(avbd3d_body_set_awake(id, 1), "body_set_awake");
            }
        }
        m_edAxis = -1;
        m_edBox = false;
        return;
    }

    const bool dragged = m_edBox || std::abs(px - m_edX0) > kClickSlopPixels || std::abs(py - m_edY0) > kClickSlopPixels;
    m_edBox = false;

    if (!dragged)
    {
        const int hit = editorPick(px, py);
        if (m_edMode == 0)
        {
            editorClear();
            if (hit >= 0)
                editorSelect(hit, true);
        }
        else if (hit >= 0)
            editorSelect(hit, m_edMode == 1 ? true : !editorSelected(hit));
        return;
    }

    // Marquee: every non-static body whose centre lands inside the rectangle.
    const int x0 = std::min(m_edX0, px), x1 = std::max(m_edX0, px);
    const int y0 = std::min(m_edY0, py), y1 = std::max(m_edY0, py);
    if (m_edMode == 0)
        editorClear();
    for (int i = 0; i < editorBodyCount(); i++)
    {
        if (!editorAlive(i) || editorStatic(i))
            continue;
        Avbd3dVec3 p{};
        if (avbd3d_body_get_position(m_import.body(i), &p) != AVBD3D_OK)
            continue;
        float sx, sy;
        if (!project(fromApi(p), sx, sy) || sx < x0 || sx > x1 || sy < y0 || sy > y1)
            continue;
        editorSelect(i, m_edMode == 2 ? !editorSelected(i) : true);
    }
}

// ---------------------------------------------------------------------------
// Keyboard
// ---------------------------------------------------------------------------

// Editor hotkeys, active while paused:
//   Ctrl+A select all     Esc / Ctrl+Shift+A deselect     Ctrl+I invert
//   Delete / X delete     Ctrl+D duplicate                F focus the camera on the selection
bool DemoApp::editorKey(const SDL_KeyboardEvent &k)
{
    const bool ctrl = (k.keysym.mod & KMOD_CTRL) != 0;
    const bool shift = (k.keysym.mod & KMOD_SHIFT) != 0;
    switch (k.keysym.sym)
    {
    case SDLK_a:
        if (!ctrl)
            return false;
        m_edPending = shift ? EditSelectNone : EditSelectAll;
        return true;
    case SDLK_i:
        if (!ctrl)
            return false;
        m_edPending = EditInvert;
        return true;
    case SDLK_ESCAPE:
        if (m_edSel.empty())
            return false; // nothing selected: Esc keeps its usual meaning
        m_edPending = EditSelectNone;
        return true;
    case SDLK_DELETE:
    case SDLK_x:
        if (m_edSel.empty() || (k.keysym.sym == SDLK_x && ctrl))
            return false;
        m_edPending = EditDelete;
        return true;
    case SDLK_d:
        if (!ctrl)
            return false;
        m_edPending = EditDuplicate;
        return true;
    case SDLK_f:
        if (ctrl)
            return false;
        m_edPending = EditFocus;
        return true;
    default:
        return false;
    }
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

void DemoApp::editorRun(EditAction a)
{
    switch (a)
    {
    case EditSelectAll:
        for (int i = 0; i < editorBodyCount(); i++)
            if (editorAlive(i) && !editorStatic(i))
                editorSelect(i, true);
        break;

    case EditSelectNone:
        editorClear();
        break;

    case EditInvert:
        for (int i = 0; i < editorBodyCount(); i++)
            if (editorAlive(i) && !editorStatic(i))
                editorSelect(i, !editorSelected(i));
        // Statics are never part of an inversion; drop any that were clicked in.
        for (int k = (int)m_edSel.size() - 1; k >= 0; k--)
            if (editorStatic(m_edSel[k]))
                editorSelect(m_edSel[k], false);
        break;

    case EditDelete:
    {
        if (m_edSel.empty())
            break;
        m_edDeleted.resize(editorBodyCount(), 0);
        for (int i : m_edSel)
        {
            must(avbd3d_destroy_body(m_import.body(i)), "destroy_body");
            m_edDeleted[i] = 1;
            m_edMask[i] = 0;
        }
        m_edSel.clear();
        break;
    }

    case EditDuplicate:
    {
        if (m_edSel.empty())
            break;
        // The copies go beside the selection, one selection-width along +X, so they do not
        // start inside the originals.
        float lo = FLT_MAX, hi = -FLT_MAX;
        for (int i : m_edSel)
        {
            Avbd3dAABB box{};
            if (avbd3d_body_compute_aabb(m_import.body(i), &box) == AVBD3D_OK)
            {
                lo = std::min(lo, box.lowerBound.x);
                hi = std::max(hi, box.upperBound.x);
            }
        }
        const float shift = hi > lo ? hi - lo + 0.1f : 1.0f;

        const SceneDesc desc = m_author.finalize();
        std::vector<int> made;
        for (int i : m_edSel)
        {
            Avbd3dTransform xf{};
            if (i >= desc.count || avbd3d_body_get_transform(m_import.body(i), &xf) != AVBD3D_OK)
                continue;
            const vec3 pos = fromApi(xf.p) + vec3{shift, 0, 0};
            const quat rot{xf.q.v.x, xf.q.v.y, xf.q.v.z, xf.q.s};
            const int kind = desc.shapeType[i];
            const vec3 h = desc.halfExtents[i];
            float unitMass = 0.0f;
            vec3 unitMoment;
            shapeMassProperties(kind, h, 1.0f, unitMass, unitMoment);
            const float density = (desc.mass[i] > 0.0f && unitMass > 0.0f) ? desc.mass[i] / unitMass : 0.0f;
            const float friction = desc.friction[i];
            int n;
            if (kind == AVBD_SHAPE_SPHERE)
                n = m_author.addSphere(h.x, density, friction, pos, vec3{0, 0, 0}, vec3{0, 0, 0}, rot);
            else if (kind == AVBD_SHAPE_CAPSULE)
                n = m_author.addCapsule(h.x, h.z - h.x, density, friction, pos, vec3{0, 0, 0}, vec3{0, 0, 0}, rot);
            else
                n = m_author.addBody(h * 2.0f, density, friction, pos, vec3{0, 0, 0}, vec3{0, 0, 0}, rot);
            made.push_back(n);
        }
        if (made.empty())
            break;
        m_import.importTail(m_world, m_author.finalize());
        editorClear();
        for (int n : made)
            editorSelect(n, true);
        break;
    }

    case EditFocus:
    {
        if (m_edSel.empty())
            break;
        vec3 lo{FLT_MAX, FLT_MAX, FLT_MAX}, hi{-FLT_MAX, -FLT_MAX, -FLT_MAX};
        for (int i : m_edSel)
        {
            Avbd3dAABB box{};
            if (avbd3d_body_compute_aabb(m_import.body(i), &box) != AVBD3D_OK)
                continue;
            lo = vec3{std::min(lo.x, box.lowerBound.x), std::min(lo.y, box.lowerBound.y),
                      std::min(lo.z, box.lowerBound.z)};
            hi = vec3{std::max(hi.x, box.upperBound.x), std::max(hi.y, box.upperBound.y),
                      std::max(hi.z, box.upperBound.z)};
        }
        if (lo.x > hi.x)
            break;
        const vec3 c = (lo + hi) * 0.5f;
        const float radius = std::max(length(hi - lo) * 0.5f, 0.5f);
        // Keep the viewing direction, move the look point onto the selection and back off
        // far enough to see all of it.
        const vec3 eye = rawEye();
        m_camTarget = c;
        m_camDistance = clamp(radius / tanf(rad(kFovY) * 0.5f) * 1.3f, 2.0f, 3000.0f);
        (void)eye;
        break;
    }

    default:
        break;
    }
}

// Per frame, before the draw: queued actions, the gizmo drag, the selection centre.
void DemoApp::editorUpdate()
{
    if (!m_paused)
        return;

    if (m_edPending != EditNone)
    {
        const EditAction a = m_edPending;
        m_edPending = EditNone;
        editorRun(a);
    }

    // The selection's centre: mean of the body positions.
    vec3 sum{0, 0, 0};
    int n = 0;
    for (int i : m_edSel)
    {
        Avbd3dVec3 p{};
        if (avbd3d_body_get_position(m_import.body(i), &p) == AVBD3D_OK)
        {
            sum = sum + fromApi(p);
            n++;
        }
    }
    if (n > 0 && m_edAxis < 0)
        m_edCenter = sum * (1.0f / (float)n);

    ImGuiIO &io = ImGui::GetIO();
    int mx = 0, my = 0;
    SDL_GetMouseState(&mx, &my);
    int px, py;
    toPixels(mx, my, px, py);

    if (m_edAxis >= 0)
    {
        float t;
        if (axisParam(px, py, m_edAnchor, m_edAxisDir, t))
        {
            float d = t - m_edT0;
            if (m_edSnap)
            {
                // Snap the moved distance rather than the world position, so a body that
                // started off-grid stays a whole number of steps from where it was.
                const float step = std::max(m_edSnapStep, 0.001f);
                d = roundf(d / step) * step;
            }
            editorApplyDelta(m_edAxisDir * d);
            m_edCenter = m_edAnchor + m_edAxisDir * d;
        }
        m_edAxisHover = m_edAxis;
        return;
    }

    m_edAxisHover = (io.WantCaptureMouse || m_edDown) ? -1 : gizmoHit(px, py);

    if (m_edDown)
    {
        m_edX1 = px;
        m_edY1 = py;
        if (std::abs(px - m_edX0) > kClickSlopPixels || std::abs(py - m_edY0) > kClickSlopPixels)
            m_edBox = true;
    }
}

// ---------------------------------------------------------------------------
// Drawing (ImGui draw lists: no renderer changes)
// ---------------------------------------------------------------------------

void DemoApp::editorDrawUI()
{
    if (!m_paused)
        return;
    ImDrawList *dl = ImGui::GetBackgroundDrawList();

    int dw = 1, dh = 1, ww = 1, wh = 1;
    m_host.drawableSize(dw, dh);
    m_host.windowSize(ww, wh);
    const float sx = dw > 0 ? (float)ww / (float)dw : 1.0f;
    const float sy = dh > 0 ? (float)wh / (float)dh : 1.0f;
    auto pt = [&](float x, float y) { return ImVec2(x * sx, y * sy); };

    // Outlines of the selected bodies' world AABBs.
    const ImU32 outline = IM_COL32(255, 190, 40, 255);
    int drawn = 0;
    for (int i : m_edSel)
    {
        if (drawn++ >= kMaxOutlines)
            break;
        Avbd3dAABB b{};
        if (avbd3d_body_compute_aabb(m_import.body(i), &b) != AVBD3D_OK)
            continue;
        ImVec2 c[8];
        bool ok = true;
        for (int k = 0; k < 8 && ok; k++)
        {
            float x, y;
            ok = project(vec3{(k & 1) ? b.upperBound.x : b.lowerBound.x, (k & 2) ? b.upperBound.y : b.lowerBound.y,
                              (k & 4) ? b.upperBound.z : b.lowerBound.z},
                         x, y);
            c[k] = pt(x, y);
        }
        if (!ok)
            continue;
        static const int edges[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3},
                                         {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
        for (const auto &e : edges)
            dl->AddLine(c[e[0]], c[e[1]], outline, 1.5f);
    }

    // Marquee.
    if (m_edDown && m_edBox && m_edAxis < 0)
    {
        const ImVec2 a = pt((float)std::min(m_edX0, m_edX1), (float)std::min(m_edY0, m_edY1));
        const ImVec2 b = pt((float)std::max(m_edX0, m_edX1), (float)std::max(m_edY0, m_edY1));
        dl->AddRectFilled(a, b, IM_COL32(90, 160, 255, 50));
        dl->AddRect(a, b, IM_COL32(120, 180, 255, 230), 0.0f, 0, 1.0f);
    }

    // Gizmo: three axis arrows from the selection centre, drawn above everything else.
    float seg[3][4], len;
    if (!gizmoAxes(seg, len))
        return;
    ImDrawList *top = ImGui::GetForegroundDrawList();
    const ImU32 cols[3] = {IM_COL32(235, 70, 70, 255), IM_COL32(90, 220, 90, 255), IM_COL32(80, 130, 245, 255)};
    for (int a = 0; a < 3; a++)
    {
        const bool hot = m_edAxisHover == a;
        const ImU32 col = hot ? IM_COL32(255, 235, 80, 255) : cols[a];
        const ImVec2 p0 = pt(seg[a][0], seg[a][1]), p1 = pt(seg[a][2], seg[a][3]);
        top->AddLine(p0, p1, IM_COL32(0, 0, 0, 160), hot ? 6.0f : 5.0f);
        top->AddLine(p0, p1, col, hot ? 3.5f : 2.5f);
        // Arrow head: a filled triangle at the tip, pointing away from the centre.
        ImVec2 dir(p1.x - p0.x, p1.y - p0.y);
        const float l = sqrtf(dir.x * dir.x + dir.y * dir.y);
        if (l > 1.0f)
        {
            dir = ImVec2(dir.x / l, dir.y / l);
            const ImVec2 nrm(-dir.y, dir.x);
            const float hl = 14.0f, hw = 6.0f;
            const ImVec2 tip(p1.x + dir.x * hl, p1.y + dir.y * hl);
            top->AddTriangleFilled(tip, ImVec2(p1.x + nrm.x * hw, p1.y + nrm.y * hw),
                                   ImVec2(p1.x - nrm.x * hw, p1.y - nrm.y * hw), col);
        }
    }
    const ImVec2 c0 = pt(seg[0][0], seg[0][1]);
    top->AddCircleFilled(c0, 4.0f, IM_COL32(240, 240, 240, 255));
    (void)kGizmoPixels;
}

// ---------------------------------------------------------------------------
// Panel
// ---------------------------------------------------------------------------

void DemoApp::editorPanel()
{
    if (!m_paused)
    {
        ImGui::TextUnformatted("Pause (Space) to select and move bodies.");
        return;
    }
    ImGui::Text("%d selected", (int)m_edSel.size());
    auto button = [&](const char *label, EditAction a, const char *tip)
    {
        if (ImGui::Button(label))
            m_edPending = a;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", tip);
    };
    button("All", EditSelectAll, "Select every dynamic body.  Ctrl+A");
    ImGui::SameLine();
    button("None", EditSelectNone, "Clear the selection.  Esc or Ctrl+Shift+A");
    ImGui::SameLine();
    button("Invert", EditInvert, "Select what is not selected (dynamic bodies only).  Ctrl+I");
    button("Delete", EditDelete, "Remove the selected bodies from the world.  Delete or X");
    ImGui::SameLine();
    button("Duplicate", EditDuplicate,
           "Copy the selected bodies (shape, size and mass; joints are not copied) beside the "
           "selection, and select the copies.  Ctrl+D");
    ImGui::SameLine();
    button("Focus", EditFocus, "Move the camera to frame the selection.  F");

    ImGui::Checkbox("Snap moves", &m_edSnap);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Round the distance moved with the gizmo to whole steps.");
    if (m_edSnap)
    {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(110.0f);
        ImGui::SliderFloat("##snapstep", &m_edSnapStep, 0.05f, 5.0f, "%.2f m", ImGuiSliderFlags_Logarithmic);
    }
    ImGui::TextWrapped("Click selects, Shift+click adds, Ctrl+click toggles, drag draws a box "
                       "(Shift adds, Ctrl toggles). Drag a gizmo arrow to move the selection. "
                       "Boxes and Select all skip static bodies; a click can pick them.");
}

void DemoApp::smokeEdit(int step, int w, int h)
{
    if (step == 0)
    {
        setPaused(true);
        editorMouseDown(2, 2);
        editorMouseUp(w - 2, h - 2); // a box over everything
    }
    else if (step == 1)
    {
        editorRun(EditDuplicate);
        m_edStartPos.resize(m_edSel.size());
        m_edStartRot.resize(m_edSel.size());
        for (size_t k = 0; k < m_edSel.size(); k++)
        {
            Avbd3dTransform xf{};
            must(avbd3d_body_get_transform(m_import.body(m_edSel[k]), &xf), "body_get_transform");
            m_edStartPos[k] = fromApi(xf.p);
            m_edStartRot[k] = quat{xf.q.v.x, xf.q.v.y, xf.q.v.z, xf.q.s};
        }
        editorApplyDelta(vec3{1, 0, 0});
        m_stepOnce = true;
    }
    else if (step == 2)
    {
        editorRun(EditDelete);
        m_stepOnce = true;
    }
}
