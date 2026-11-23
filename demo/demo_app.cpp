// DemoApp on the public avbd3d_* API: the window calls go through DemoHost, the solver calls
// to the world m_world names.

#include "demo_app.h"

#include "spawn_objects.h"

#include <vulkan/vulkan.h>

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static const float kSpawnMinDepth = 0.5f;
static const char *spawnKindNames[] = {"Box", "Passenger plane", "Ragdoll"};

// Wrapped tooltip for the item just submitted. The overlay tooltips say what the
// thing being shown actually is, which takes a sentence or two -- unwrapped, that is a
// tooltip wider than the window.
static void itemTooltip(const char *tip)
{
    if (!tip || !ImGui::IsItemHovered())
        return;
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 26.0f);
    ImGui::TextUnformatted(tip);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
}

static Avbd3dVec3 toApi(vec3 v)
{
    return Avbd3dVec3{v.x, v.y, v.z};
}
static vec3 fromApi(Avbd3dVec3 v)
{
    return vec3{v.x, v.y, v.z};
}

// An id is null when it was never filled in.
template <class Id> static bool isNull(Id id)
{
    return id.index1 == 0;
}

// The demo reports a failed call and carries on; a lost device shows up as a failed step.
static void must(Avbd3dResult r, const char *what)
{
    if (r != AVBD3D_OK)
        fprintf(stderr, "avbd3d_demo: %s failed: %s\n", what, avbd3d_result_string(r));
}

static VkDevice gpuDevice(const Avbd3dGpu *gpu)
{
    Avbd3dVulkanContext ctx{};
    must(avbd3d_gpu_get_vulkan_context(gpu, &ctx), "gpu_get_vulkan_context");
    return (VkDevice)ctx.device;
}

void DemoApp::init(const DemoHost &host)
{
    m_host = host;
}

void DemoApp::setGpu(Avbd3dGpu *gpu)
{
    m_gpu = gpu;
    Avbd3dVulkanContext ctx{};
    must(avbd3d_gpu_get_vulkan_context(gpu, &ctx), "gpu_get_vulkan_context");
    snprintf(m_deviceName, sizeof(m_deviceName), "%s", ctx.deviceName ? ctx.deviceName : "");
}

void DemoApp::shutdown()
{
    if (m_world)
        must(avbd3d_destroy_world(m_world), "destroy_world");
    m_world = nullptr;
}

void DemoApp::applyParams()
{
    const SolverParams &p = m_author.params;
    const Avbd3dSolverDef sd = solverDefOf(p);
    must(avbd3d_world_set_solver_def(m_world, &sd), "world_set_solver_def");
    must(avbd3d_world_set_gravity(m_world, toApi(p.gravity)), "world_set_gravity");
    must(avbd3d_world_enable_sleeping(m_world, p.sleepEnabled), "world_enable_sleeping"); // off wakes every sleeper
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

vec3 DemoApp::rawEye() const
{
    vec3 dir = {cosf(m_camElevation) * cosf(m_camAzimuth), sinf(m_camElevation),
                -cosf(m_camElevation) * sinf(m_camAzimuth)};
    return m_camTarget + dir * m_camDistance;
}

// What is on screen: the input-driven camera above, with its view direction low-pass filtered when smoothing is on.
vec3 DemoApp::camEye() const
{
    vec3 dir = {cosf(m_dispElevation) * cosf(m_dispAzimuth), sinf(m_dispElevation),
                -cosf(m_dispElevation) * sinf(m_dispAzimuth)};
    return m_dispTarget + dir * m_dispDistance;
}

// Frame-rate-independent exponential follow. Smoothness 0..1 maps to a time constant of
// 0.03 s .. ~1.2 s (quadratic, so the top of the slider is the slow cinematic glide).
void DemoApp::updateSmoothCamera(float dtSec)
{
    if (!m_smoothCamera || m_camSnap)
    {
        m_dispTarget = m_camTarget;
        m_dispDistance = m_camDistance;
        m_dispAzimuth = m_camAzimuth;
        m_dispElevation = m_camElevation;
        m_camSnap = false;
        return;
    }
    const float s = clamp(m_smoothCameraAmount, 0.0f, 1.0f);
    const float tau = 0.03f + 1.17f * s * s;
    const float a = 1.0f - expf(-dtSec / tau);
    // Only the view direction is filtered; the eye stays exactly where the input put it.
    m_dispAzimuth += (m_camAzimuth - m_dispAzimuth) * a;
    m_dispElevation += (m_camElevation - m_dispElevation) * a;
    m_dispDistance = m_camDistance;
    const vec3 fwd = normalize(vec3{-cosf(m_dispElevation) * cosf(m_dispAzimuth), -sinf(m_dispElevation),
                                    cosf(m_dispElevation) * sinf(m_dispAzimuth)});
    m_dispTarget = rawEye() + fwd * m_camDistance;
}

// Unit view direction, i.e. where the camera is pointing. The eye sits at
// camTarget + dir * camDistance, so the camera looks back along -dir.
vec3 DemoApp::camForward() const
{
    return normalize(vec3{-cosf(m_camElevation) * cosf(m_camAzimuth), -sinf(m_camElevation),
                          cosf(m_camElevation) * sinf(m_camAzimuth)});
}

// Camera right, in the horizontal plane. Handedness matters here: with the view
// direction f and world up y, screen-right is cross(f, y), not cross(y, f) -- the latter
// is what made A and D swap.
vec3 DemoApp::camRight() const
{
    vec3 flat = {-cosf(m_camAzimuth), 0.0f, sinf(m_camAzimuth)};
    return normalize(cross(flat, vec3{0, 1, 0}));
}

// The same two vectors for the view on screen (they lag the raw ones while smoothing
// catches up). Movement and zoom follow these, so you travel along what you actually see.
vec3 DemoApp::dispForward() const
{
    return normalize(vec3{-cosf(m_dispElevation) * cosf(m_dispAzimuth), -sinf(m_dispElevation),
                          cosf(m_dispElevation) * sinf(m_dispAzimuth)});
}

vec3 DemoApp::dispRight() const
{
    vec3 flat = {-cosf(m_dispAzimuth), 0.0f, sinf(m_dispAzimuth)};
    return normalize(cross(flat, vec3{0, 1, 0}));
}

void DemoApp::viewMatrix(float out[16]) const
{
    const vec3 eye = camEye();
    vec3 f = normalize(m_dispTarget - eye);
    vec3 s = normalize(cross(f, vec3{0, 1, 0}));
    vec3 u = cross(s, f);

    out[0] = s.x;
    out[4] = s.y;
    out[8] = s.z;
    out[12] = -dot(s, eye);
    out[1] = u.x;
    out[5] = u.y;
    out[9] = u.z;
    out[13] = -dot(u, eye);
    out[2] = -f.x;
    out[6] = -f.y;
    out[10] = -f.z;
    out[14] = dot(f, eye);
    out[3] = 0;
    out[7] = 0;
    out[11] = 0;
    out[15] = 1;
}

void DemoApp::perspective(float out[16], float aspect, bool vulkanClip)
{
    const float f = 1.0f / tanf(rad(kFovY) * 0.5f);
    const float zn = kNear, zf = kFar;
    for (int i = 0; i < 16; i++)
        out[i] = 0.0f;
    out[0] = f / aspect;
    out[11] = -1.0f;
    if (vulkanClip)
    {
        // Vulkan clip space: Y points down and depth runs 0..1.
        out[5] = -f;
        out[10] = zf / (zn - zf);
        out[14] = (zf * zn) / (zn - zf);
    }
    else
    {
        out[5] = f;
        out[10] = (zf + zn) / (zn - zf);
        out[14] = (2.0f * zf * zn) / (zn - zf);
    }
}

void DemoApp::matMul(float out[16], const float a[16], const float b[16])
{
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++)
        {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++)
                sum += a[k * 4 + r] * b[c * 4 + k];
            out[c * 4 + r] = sum;
        }
}

void DemoApp::toPixels(int x, int y, int &px, int &py) const
{
    int w = 0, h = 0, winW = 0, winH = 0;
    m_host.drawableSize(w, h);
    m_host.windowSize(winW, winH);
    float scaleX = (winW > 0) ? (float)w / (float)winW : 1.0f;
    float scaleY = (winH > 0) ? (float)h / (float)winH : 1.0f;
    px = (int)(x * scaleX);
    py = (int)(y * scaleY);
}

bool DemoApp::rayFromPixel(int mouseX, int mouseY, vec3 &rayOrigin, vec3 &rayDir) const
{
    int screenW = 0, screenH = 0;
    m_host.drawableSize(screenW, screenH);
    if (screenW <= 0 || screenH <= 0)
        return false;

    const vec3 eye = camEye();
    float aspect = (float)screenW / (float)screenH;
    float ndcX = (float)mouseX / (float)screenW * 2.0f - 1.0f;
    float ndcY = 1.0f - (float)mouseY / (float)screenH * 2.0f;

    vec3 forward = normalize(m_dispTarget - eye);
    vec3 right = cross(forward, vec3{0, 1, 0});
    if (lengthSq(right) < 1e-6f)
        right = cross(forward, vec3{0, 0, 1});
    right = normalize(right);
    vec3 up = cross(right, forward);

    float tanHalfFovY = tanf(rad(kFovY) * 0.5f);
    float px = ndcX * aspect * tanHalfFovY;
    float py = ndcY * tanHalfFovY;

    rayOrigin = eye;
    rayDir = normalize(forward + right * px + up * py);
    return true;
}

// Free-fly movement: W/S along the full view direction (pitch included, so you fly where
// you are looking), A/D strafe horizontally, Q/E drop and rise along world up. Read from
// the keyboard state once a frame rather than from key events, so holding a key glides
// instead of repeating.
void DemoApp::updateCameraKeys(float dtSec)
{
    if (ImGui::GetIO().WantCaptureKeyboard)
        return;

    if (m_paused && (SDL_GetModState() & KMOD_CTRL))
        return; // Ctrl+A / Ctrl+D ... are editor hotkeys, not fly keys
    const Uint8 *keys = SDL_GetKeyboardState(nullptr);
    float fwd = (keys[SDL_SCANCODE_W] ? 1.0f : 0.0f) - (keys[SDL_SCANCODE_S] ? 1.0f : 0.0f);
    float side = (keys[SDL_SCANCODE_D] ? 1.0f : 0.0f) - (keys[SDL_SCANCODE_A] ? 1.0f : 0.0f);
    float lift = (keys[SDL_SCANCODE_E] ? 1.0f : 0.0f) - (keys[SDL_SCANCODE_Q] ? 1.0f : 0.0f);
    if (fwd == 0.0f && side == 0.0f && lift == 0.0f)
        return;

    float step = m_camMoveSpeed * dtSec;
    if (keys[SDL_SCANCODE_LSHIFT] || keys[SDL_SCANCODE_RSHIFT])
        step *= 4.0f;

    // Moving the look point drags the eye along with it, since the eye is defined
    // relative to it -- that is the whole camera flying, not a pan around a fixed pivot.
    m_camTarget += dispForward() * (fwd * step) + dispRight() * (side * step) + vec3{0, lift * step, 0};
}

// ---------------------------------------------------------------------------
// Scenes
// ---------------------------------------------------------------------------

void DemoApp::loadScene(int index, bool resetCamera)
{
    m_stepFailMsg[0] = '\0';
    m_accum = 0.0f;
    editorReset();
    m_dragging = false;
    m_dragBody = Avbd3dBodyId{};
    m_dragGroup.clear();
    m_dragSeed = Avbd3dBodyId{};
    if (m_world)
        must(avbd3d_grab_end(m_world), "grab_end");

    // The world's buffers may still be referenced by a submitted command buffer from the
    // frame in flight, and destroying the world frees them. Scene switches are rare; drain first.
    vkDeviceWaitIdle(gpuDevice(m_gpu));

    SceneView view;
    if (index < gpuSceneCount)
        view = gpuScenes[index](&m_author);
    else
        view = buildSpheresCapsulesScene(); // demo extra scene, uploads internally
    const SceneDesc desc = m_author.finalize();
    shutdown();
    Avbd3dWorldDef wd = avbd3d_default_world_def();
    wd.gravity = toApi(m_author.params.gravity);
    wd.enableSleep = m_author.params.sleepEnabled;
    must(avbd3d_create_world_on_gpu(m_gpu, &wd, &m_world), "create_world_on_gpu");
    m_import.reset();
    m_import.importTail(m_world, desc);
    m_cellSize = desc.cellSize;
    applyParams();
    m_stepCount = 0;
    if (resetCamera)
    {
        m_camDistance = view.camDistance;
        m_camTarget = vec3{0, view.camTargetHeight, 0};
        m_camSnap = true;
    }
}

// The one extra scene no entry in gpuScenes[] covers: every scene there is box-only, and
// this one is the sphere/capsule rendering test bed. Authored directly against
// addSphere()/addCapsule(), index gpuSceneCount in the scene combo
// (DemoHost::extraSceneNames).
SceneView DemoApp::buildSpheresCapsulesScene()
{
    m_author.clear();
    m_author.addBody(vec3{40, 1, 40}, 0.0f, 0.6f, vec3{0, -1, 0});

    const int n = 5;
    const float gap = 2.2f;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            vec3 pos{(i - n * 0.5f + 0.5f) * gap, 3.0f + (float)((i + j) % 3), (j - n * 0.5f + 0.5f) * gap};
            if ((i + j) % 2 == 0)
                m_author.addSphere(0.75f, 1.0f, 0.6f, pos);
            else
                m_author.addCapsule(0.5f, 0.6f, 1.0f, 0.6f, pos, vec3{0, 0, 0}, vec3{0, 0, 0},
                                    kZUpToYUp); // axis +z, stood up
        }
    return SceneView{30.0f, 3.0f};
}

// Counts and timings of the last step, from the world's status block (no readback). The
// cell size is the authoring grid, which is what the broadphase overlay wants.
void DemoApp::stats(DemoStats &s)
{
    Avbd3dStats t{};
    must(avbd3d_get_stats(m_world, &t), "get_stats");
    const Avbd3dStats &r = t;
    s.bodies = t.bodies;
    s.awake = t.awake_bodies;
    s.joints = t.joint_rows;
    s.springs = t.springs;
    s.pairs = t.pairs;
    s.manifolds = t.manifolds;
    s.contacts = t.contacts;
    s.colors = t.colors;
    s.jpRounds = t.coloring_rounds;
    s.pairCapacity = t.pair_capacity;
    s.contactCapacity = t.contact_capacity;
    s.manifoldCapacity = t.manifold_capacity;
    s.reruns = r.capacity_reruns + r.flood_reruns + r.island_reruns + r.jp_reruns + r.color_reruns;
    uint64_t allocated = 0;
    must(avbd3d_gpu_get_allocated_bytes(m_gpu, &allocated), "gpu_get_allocated_bytes");
    s.deviceMB = (float)((double)allocated / (1024.0 * 1024.0));
    s.deviceName = m_deviceName;
    s.cellSize = m_cellSize;
    int32_t requested = 0;
    must(avbd3d_world_get_contact_markers_requested(m_world, &requested), "world_get_contact_markers_requested");
    s.debugRequested = requested;
    s.stepMs = t.step_ms;
    s.broadMs = t.broadphase_ms;
    s.narrowMs = t.narrowphase_ms;
    s.solveMs = t.solve_ms;
    s.buildMs = t.build_ms;
    s.iterMs = t.iterations_ms;
    s.primalMs = t.primal_ms;
    s.dualMs = t.dual_ms;
    s.frontMs = t.front_ms;
    s.gpuBuildMs = t.gpu_build_ms;
    s.tailMs = t.tail_ms;
    s.gpuMs = t.gpu_ms;
    s.hostWaitMs = t.host_wait_ms;
}

bool DemoApp::loadInitialScene(const DemoLaunch &launch)
{
    // Benchmark mode (--frames N) runs unattended with vsync off, flipping culling on and
    // off in alternating blocks and printing the timings to stdout. With vsync on the
    // frame time is pinned to the refresh and says nothing about cost.
    m_benchFrames = launch.benchFrames > 0 ? launch.benchFrames : 0;
    m_unattended = launch.unattended || m_benchFrames > 0;
    m_vsync = !m_unattended;
    if (m_host.setVsync)
        m_host.setVsync(m_vsync);

    const int sceneTotal = sceneCount();
    if (launch.scene >= 0 && launch.scene < sceneTotal)
        m_scene = launch.scene;
    if (launch.sceneName)
    {
        char *end = nullptr;
        const long idx = strtol(launch.sceneName, &end, 10);
        int found = -1;
        if (end != launch.sceneName && *end == '\0')
            found = (idx >= 0 && idx < sceneTotal) ? (int)idx : -1;
        else
        {
            for (int i = 0; i < sceneTotal; i++)
            {
                const char *name = i < gpuSceneCount ? gpuSceneNames[i] : m_host.extraSceneNames[i - gpuSceneCount];
                if (SDL_strcasecmp(name, launch.sceneName) == 0)
                {
                    found = i;
                    break;
                }
            }
        }
        if (found < 0)
        {
            fprintf(stderr, "avbd3d_demo: unknown scene '%s' (0..%d or a scene name)\n", launch.sceneName,
                    sceneTotal - 1);
            return false;
        }
        m_scene = found;
    }
    loadScene(m_scene);
    if (launch.noSleep)
    {
        // The panel's "Never sleep": params are pushed every step, so it holds across
        // scene switches; EnableSleeping(false) also wakes what the scene authored asleep.
        m_author.params.sleepEnabled = false;
        applyParams();
    }

    if (launch.camDist > 0.0f)
    {
        m_camDistance = launch.camDist;
        m_camSnap = true;
    }
    return true;
}

void DemoApp::loadSceneIndex(int index)
{
    if (index < 0 || index >= sceneCount())
        return;
    m_scene = index;
    m_paused = false;
    loadScene(index);
}

void DemoApp::setPaused(bool paused)
{
    m_paused = paused;
    if (!paused)
    {
        m_stepFailMsg[0] = '\0'; // resuming acknowledges the failure
        editorReset();
    }
}

// A capacity ceiling made the step commit clamped results. Stop the sim so the state
// that exposes it stays on screen, and say what ran out.
void DemoApp::noteStepResult(Avbd3dResult res, const Avbd3dStats &r)
{
    if (res == AVBD3D_OK && r.ok && !r.truncated)
        return;
    m_stepFailures++;
    snprintf(m_stepFailMsg, sizeof(m_stepFailMsg), "step failed: ok %d truncated %d (%s)", r.ok ? 1 : 0,
             r.truncated ? 1 : 0, avbd3d_result_string(res));
    m_paused = true;
    if (m_unattended)
        fprintf(stderr, "avbd3d_demo: %s\n", m_stepFailMsg);
}

// ---------------------------------------------------------------------------
// Grab
//
// A grab holds a whole group, not one box: the picked body plus, within a radius the UI
// sets, everything jointed or sprung to it. Grabbing a single component of a bridge or
// a house pulls that component out of the structure, which is rarely what you meant --
// spreading the handle over the neighbourhood carries the construction instead.
//
// The group moves rigidly with the cursor: every held body keeps the offset it had from
// the hit point when the grab started, so the assembly is translated rather than
// funnelled towards one point.
// ---------------------------------------------------------------------------

// Handle strength for a body whose centre started `dist` from the grab point.
//
// Smoothstep rather than a straight (1 - t): its zero derivative at both ends is what
// keeps the grip from stepping as a body crosses the radius, and from feeling like a
// hard cone near the cursor. The exponent then shapes how quickly that falls away.
float DemoApp::grabWeight(float dist, float radius) const
{
    if (radius <= 0.0f)
        return 1.0f;
    float t = clamp(dist / radius, 0.0f, 1.0f);
    float s = 1.0f - t * t * (3.0f - 2.0f * t); // smoothstep, 1 at the cursor, 0 at the rim
    float w = powf(s, m_grabFalloff > 0.0f ? m_grabFalloff : 0.0f);
    return clamp(m_grabFalloffMin + (1.0f - m_grabFalloffMin) * w, 0.0f, 1.0f);
}

// Works out what one click takes hold of, and pins down how each held body sits
// relative to the cursor. The picked body is held at the exact point the ray hit it;
// the rest of the group is held at its centre, which is enough to carry it and avoids
// raycasting every neighbour.
void DemoApp::beginGrab(Avbd3dBodyId hit, vec3 worldHit, vec3 localHit)
{
    m_dragGroup.clear();
    m_dragLocals.clear();
    m_dragOffsets.clear();
    m_dragWeights.clear();
    m_dragSeed = hit;

    // The seed comes back in slot 0.
    m_dragGroup.resize(AVBD3D_MAX_GRAB_BODIES);
    int32_t n = 0;
    must(avbd3d_grab_begin(m_world, hit, toApi(worldHit), m_grabConnected ? m_grabRadius : 0.0f, m_grabConnected,
                           m_dragGroup.data(), AVBD3D_MAX_GRAB_BODIES, &n),
         "grab_begin");
    m_dragGroup.resize(n);
    if (m_dragGroup.empty())
        return;

    // Poses for the offsets. Read here rather than kept current every frame: a grab is a
    // once-per-click event.
    std::vector<Avbd3dVec3> pos(m_dragGroup.size());
    int32_t got = 0;
    must(avbd3d_world_get_body_data(m_world, m_dragGroup.data(), n, AVBD3D_BODY_DATA_POSITION, pos.data(), &got),
         "world_get_body_data");

    m_dragLocals.resize(m_dragGroup.size());
    m_dragOffsets.resize(m_dragGroup.size());
    m_dragWeights.resize(m_dragGroup.size());
    for (size_t i = 0; i < m_dragGroup.size(); i++)
    {
        if (m_dragGroup[i].index1 == hit.index1)
        {
            m_dragLocals[i] = localHit;
            m_dragOffsets[i] = vec3{0, 0, 0};
        }
        else
        {
            m_dragLocals[i] = vec3{0, 0, 0};
            m_dragOffsets[i] = fromApi(pos[i]) - worldHit;
        }
        // Frozen at grab time along with the offsets. Recomputing it as the group moves
        // would mean the grip changed strength while you were carrying something, which
        // reads as the construction sagging for no reason.
        m_dragWeights[i] = grabWeight(length(m_dragOffsets[i]), m_grabRadius);
    }
    m_dragCheckedStep = -1;
}

// Drops from the grab anything that has come off the piece being held.
//
// Grabbing a structure and pulling until it fractures should leave the broken-off pieces
// falling, not still flying with the cursor: what the hand holds is what is still
// attached to the body under it. The connectivity is re-walked through the joints that
// are still intact, so a piece is released the moment its last link to the seed breaks.
void DemoApp::dropDetached()
{
    if (m_dragGroup.size() < 2 || isNull(m_dragSeed))
        return;
    // Fractures only happen inside a step, and updateGrab runs twice a frame.
    if (m_dragCheckedStep == m_stepCount)
        return;
    m_dragCheckedStep = m_stepCount;

    m_dragKeep.resize(m_dragGroup.size());
    int32_t survivors = 0;
    must(avbd3d_grab_survivors(m_world, m_dragSeed, m_dragGroup.data(), (int)m_dragGroup.size(), m_dragKeep.data(),
                               &survivors),
         "grab_survivors");
    if (survivors == (int)m_dragGroup.size())
        return;

    size_t w = 0;
    for (size_t i = 0; i < m_dragGroup.size(); i++)
    {
        if (!m_dragKeep[i])
            continue;
        m_dragGroup[w] = m_dragGroup[i];
        m_dragLocals[w] = m_dragLocals[i];
        m_dragOffsets[w] = m_dragOffsets[i];
        m_dragWeights[w] = m_dragWeights[i];
        w++;
    }
    m_dragGroup.resize(w);
    m_dragLocals.resize(w);
    m_dragOffsets.resize(w);
    m_dragWeights.resize(w);
}

void DemoApp::updateGrab(vec3 cursorPoint)
{
    if (m_dragGroup.empty())
        return;
    dropDetached();
    if (m_dragGroup.empty())
        return;
    m_dragHandles.resize(m_dragGroup.size());
    for (size_t i = 0; i < m_dragGroup.size(); i++)
        m_dragHandles[i] = Avbd3dGrabHandle{m_dragGroup[i], toApi(m_dragLocals[i]),
                                            toApi(cursorPoint + m_dragOffsets[i]), m_dragWeights[i]};
    must(avbd3d_grab_move(m_world, m_dragHandles.data(), (int)m_dragHandles.size()), "grab_move");
}

void DemoApp::updateGrabFromCursor()
{
    if (!m_dragging || m_dragGroup.empty())
        return;
    vec3 rayOrigin, rayDir;
    if (rayFromPixel(m_dragMouseX, m_dragMouseY, rayOrigin, rayDir))
        updateGrab(rayOrigin + rayDir * m_dragDist);
}

void DemoApp::endGrab()
{
    m_dragging = false;
    m_dragBody = Avbd3dBodyId{};
    m_dragGroup.clear();
    m_dragSeed = Avbd3dBodyId{};
    must(avbd3d_grab_end(m_world), "grab_end");
}

// ---------------------------------------------------------------------------
// Spawn
// ---------------------------------------------------------------------------

// The author stages the new bodies, the desc is re-exported in full, and importTail()
// creates only what is new in the live world: existing bodies keep their state, the contact
// cache survives, and authored joints keep their duals and fracture flags.
void DemoApp::spawnBody(vec3 size, float density, float friction, vec3 position, vec3 velocity)
{
    m_author.spawnBody(size, density, friction, position, velocity);
    const SceneDesc desc = m_author.finalize();
    m_import.importTail(m_world, desc);
}

void DemoApp::spawnAssembly(const SpawnEmitter &emitter, vec3 origin, quat rotation, float scale, float massMultiplier,
                            vec3 velocity)
{
    emitter.commit(m_author, origin, rotation, scale, massMultiplier, velocity);
    const SceneDesc desc = m_author.finalize();
    m_import.importTail(m_world, desc);
}

// Throws the currently selected object along the cursor ray.
//
// The distance is from the camera, not from the look point, and it is only pushed out
// far enough that the object cannot appear straddling the near plane -- for a box that
// is its half width, and for an assembly the radius of its own bounding sphere, so an
// airliner does not spawn wrapped around the viewer.
void DemoApp::spawnAlongRay(vec3 rayOrigin, vec3 rayDir)
{
    if (m_spawnKind == SpawnKindBox)
    {
        float depth = m_spawnDist;
        if (depth < kSpawnMinDepth + m_spawnSize)
            depth = kSpawnMinDepth + m_spawnSize;
        const vec3 at = rayOrigin + rayDir * depth;
        spawnBody(vec3{m_spawnSize, m_spawnSize, m_spawnSize}, m_spawnMass, 0.5f, at, rayDir * m_spawnSpeed);
        return;
    }

    SpawnEmitter emitter;
    float scale = 1.0f;
    quat rotation = kZUpToYUp; // assembly sources are z-up

    if (m_spawnKind == SpawnKindPlane)
    {
        buildPassengerPlane(&emitter);

        // Nose first. The aircraft is built along +X with +Z up (the assembly sources are z-up),
        // so pitching about Y and then yawing about Z puts its nose on the ray while keeping its
        // wings level. That is worked out in the source frame (the ray mapped back to z-up), then
        // the whole assembly is rotated into the y-up world.
        const vec3 rayZ = vec3{rayDir.x, -rayDir.z, rayDir.y};
        const float yaw = atan2f(rayZ.y, rayZ.x);
        const float pitch = -asinf(clamp(rayZ.z, -1.0f, 1.0f));
        const quat qYaw = quat{0, 0, sinf(yaw * 0.5f), cosf(yaw * 0.5f)};
        const quat qPitch = quat{0, sinf(pitch * 0.5f), 0, cosf(pitch * 0.5f)};
        rotation = kZUpToYUp * (qYaw * qPitch);
    }
    else
    {
        buildRagdoll(emitter);
        scale = m_spawnSize;
        // Upright, whichever way it is thrown: a ragdoll launched nose-down along a
        // steep ray reads as a glitch rather than as a throw.
    }

    const float radius = emitter.radius() * scale;
    float depth = m_spawnDist;
    if (depth < kSpawnMinDepth + radius)
        depth = kSpawnMinDepth + radius;
    const vec3 at = rayOrigin + rayDir * depth;

    spawnAssembly(emitter, at, rotation, scale, m_spawnMass, rayDir * m_spawnSpeed);
}

// ---------------------------------------------------------------------------
// Input and frame
// ---------------------------------------------------------------------------

// Click-to-grab, from a drawable-pixel position: what a left click does, and what the
// smoke run calls.
bool DemoApp::grabAtPixel(int px, int py)
{
    vec3 rayOrigin, rayDir;
    if (!rayFromPixel(px, py, rayOrigin, rayDir))
        return false;
    // Closest dynamic body: statics are clicked through, as the grab never holds one.
    struct Pick
    {
        Avbd3dShapeId shape;
        float fraction;
    } pick{{}, 1.0f};
    auto onHit = [](Avbd3dShapeId shape, Avbd3dVec3, Avbd3dVec3, float fraction, uint64_t, int, int, void *ctx) -> float
    {
        Avbd3dBodyId owner{};
        Avbd3dBodyType type = AVBD3D_STATIC_BODY;
        if (avbd3d_shape_get_body(shape, &owner) != AVBD3D_OK || avbd3d_body_get_type(owner, &type) != AVBD3D_OK ||
            type == AVBD3D_STATIC_BODY)
            return -1.0f;
        Pick *p = (Pick *)ctx;
        p->shape = shape;
        p->fraction = fraction;
        return fraction;
    };
    must(avbd3d_world_cast_ray(m_world, toApi(rayOrigin), toApi(rayDir * kFar), avbd3d_default_query_filter(), onHit,
                               &pick),
         "world_cast_ray");
    if (isNull(pick.shape))
        return false;
    Avbd3dBodyId hit{};
    must(avbd3d_shape_get_body(pick.shape, &hit), "shape_get_body");
    const float hitDist = pick.fraction * kFar;
    Avbd3dVec3 localApi{};
    must(avbd3d_body_get_local_point(hit, toApi(rayOrigin + rayDir * hitDist), &localApi), "body_get_local_point");
    const vec3 localHit = fromApi(localApi);
    m_dragging = true;
    m_dragBody = hit;
    m_dragDist = hitDist;
    m_dragMouseX = px;
    m_dragMouseY = py;
    beginGrab(hit, rayOrigin + rayDir * hitDist, localHit);
    return true;
}

void DemoApp::moveGrab(int px, int py)
{
    m_dragMouseX = px;
    m_dragMouseY = py;
}

// Middle click: throw the selected object along the ray through the pixel.
bool DemoApp::spawnAtPixel(int px, int py)
{
    vec3 rayOrigin, rayDir;
    if (!rayFromPixel(px, py, rayOrigin, rayDir))
        return false;
    spawnAlongRay(rayOrigin, rayDir);
    return true;
}

void DemoApp::handleEvent(const SDL_Event &e)
{
    ImGuiIO &io = ImGui::GetIO();

    switch (e.type)
    {
    case SDL_QUIT:
        m_running = false;
        break;

    case SDL_KEYDOWN:
        if (io.WantCaptureKeyboard)
            break;
        if (e.key.keysym.sym == SDLK_F11 && m_host.setFullscreen && m_host.isFullscreen)
            m_host.setFullscreen(!m_host.isFullscreen());
        else if (m_paused && editorKey(e.key))
            break;
        else if (e.key.keysym.sym == SDLK_ESCAPE)
            m_running = false;
        else if (e.key.keysym.sym == SDLK_SPACE)
            setPaused(!m_paused);
        else if (e.key.keysym.sym == SDLK_r)
            loadScene(m_scene, false);
        break;

    case SDL_MOUSEBUTTONDOWN:
    {
        if (io.WantCaptureMouse)
            break;
        int px, py;
        toPixels(e.button.x, e.button.y, px, py);
        if (e.button.button == SDL_BUTTON_LEFT)
        {
            if (m_paused)
                editorMouseDown(px, py);
            else
                grabAtPixel(px, py);
        }
        else if (e.button.button == SDL_BUTTON_MIDDLE)
            spawnAtPixel(px, py);
        break;
    }

    case SDL_MOUSEBUTTONUP:
        if (e.button.button == SDL_BUTTON_LEFT && m_edDown)
        {
            int px, py;
            toPixels(e.button.x, e.button.y, px, py);
            editorMouseUp(px, py);
        }
        else if (e.button.button == SDL_BUTTON_LEFT && m_dragging)
            endGrab(); // the drag is a joint, so letting go has to switch it off
        break;

    case SDL_MOUSEMOTION:
        if (io.WantCaptureMouse)
            break;
        // Right button is FPS mouse look: the camera turns in place rather than orbiting
        // a pivot. The eye is stored indirectly, as camTarget + dir * camDistance, so
        // holding the eye still means recomputing camTarget once the angles have moved.
        if (e.motion.state & SDL_BUTTON_RMASK)
        {
            const vec3 eye = rawEye();
            m_camAzimuth -= e.motion.xrel * m_lookSensitivity;
            m_camElevation += e.motion.yrel * m_lookSensitivity;
            m_camElevation = clamp(m_camElevation, -1.5f, 1.5f);
            m_camTarget = eye + camForward() * m_camDistance;
        }
        if (m_edDown)
            toPixels(e.motion.x, e.motion.y, m_edX1, m_edY1);
        if (m_dragging && !isNull(m_dragBody))
        {
            int px, py;
            toPixels(e.motion.x, e.motion.y, px, py);
            moveGrab(px, py);
        }
        break;

    case SDL_MOUSEWHEEL:
        if (io.WantCaptureMouse)
            break;
    {
        // Dolly: slide the eye along the view you see, keeping the look point fixed.
        const float newDist = clamp(m_camDistance * ((e.wheel.y > 0) ? 0.9f : 1.1f), 1.0f, 3000.0f);
        const vec3 eye = rawEye() + dispForward() * (m_camDistance - newDist);
        m_camDistance = newDist;
        m_camTarget = eye + camForward() * newDist;
    }
        break;
    }
}

void DemoApp::update()
{
    if (m_pendingScene >= 0)
    {
        const int index = m_pendingScene;
        m_pendingScene = -1;
        loadScene(index, m_pendingResetCamera);
    }

    updateCameraKeys(m_lastFrameSec);
    updateSmoothCamera(m_lastFrameSec);

    updateGrabFromCursor();

    runPhysics();
    editorUpdate();

    updateGrabFromCursor();
}

// Physics runs at a fixed rate (SolverParams::dt), whatever the frame rate is. Each frame
// adds its wall-clock time to an accumulator and takes whole ticks out of it, so a 144 Hz
// display steps at the physics rate rather than 144 times a second, and a frame that took
// longer than a tick runs several. When the ticks cost more than they are worth -- a cap on
// ticks per frame and a wall-clock budget -- the backlog is dropped: the simulation runs in
// slow motion and the frame rate stays up, instead of the usual spiral of ever more ticks.
// (The solver shares the render queue, so ticks cannot run on another thread.)
void DemoApp::runPhysics()
{
    float dt = m_author.params.dt;
    // Top of the rate slider: no fixed rate, one tick per frame covering that frame's time.
    const bool matchFrame = !m_unattended && !m_stepOnce && !m_paused && dt <= 1.0f / kMaxTickHz + 1e-6f;
    if (matchFrame)
        dt = clamp(m_lastFrameSec * m_timeScale, 0.001f, 0.05f);
    int steps = 0;
    if (matchFrame)
        steps = 1;
    else if (m_unattended)
        steps = (!m_paused || m_stepOnce) ? 1 : 0; // benchmarks and --smoke: one tick per frame
    else if (m_stepOnce)
        steps = 1;
    else if (!m_paused)
        m_accum += m_lastFrameSec * m_timeScale;
    if (m_paused && !m_stepOnce)
        m_accum = 0.0f;
    if (m_stepOnce && !m_unattended)
        m_accum = 0.0f;

    const int kMaxTicks = 8;
    const float kBudgetMs = 24.0f;
    const bool fixedRate = !m_unattended && !m_stepOnce && !m_paused && !matchFrame;
    const Uint64 t0 = SDL_GetPerformanceCounter();
    int done = 0;
    for (;;)
    {
        if (fixedRate)
        {
            if (m_accum < dt || done >= kMaxTicks)
                break;
            if (done > 0 && (float)(SDL_GetPerformanceCounter() - t0) * 1000.0f / (float)SDL_GetPerformanceFrequency() > kBudgetMs)
                break;
            m_accum -= dt;
        }
        else if (done >= steps)
            break;

        applyParams();
        const Avbd3dResult res = avbd3d_step(m_world, dt);
        Avbd3dStats r{};
        if (res == AVBD3D_OK)
            must(avbd3d_get_stats(m_world, &r), "get_stats");
        m_stepCount++;
        done++;
        noteStepResult(res, r);
        if (m_paused && fixedRate)
            break; // a failed step pauses
    }
    if (fixedRate && m_accum >= dt)
        m_accum = fmodf(m_accum, dt); // behind: drop the backlog, keep the phase
    m_stepOnce = false;

    const float inst = m_lastFrameSec > 0.0f ? (float)done / m_lastFrameSec : 0.0f;
    m_simHz += (inst - m_simHz) * 0.05f;
    if (done == 0)
        return;
    DemoStats s;
    stats(s);
    if (s.stepMs >= 0.0f)
        m_avgStepMs += (s.stepMs - m_avgStepMs) * 0.1f;
    m_histHostMs[m_histPos] = std::max(s.stepMs, 0.0f);
    m_histGpuMs[m_histPos] = std::max(s.gpuMs, 0.0f);
    m_histPos = (m_histPos + 1) % kTimeHistory;
}

void DemoApp::endFrame(float frameMs)
{
    m_lastFrameSec = clamp(frameMs * 0.001f, 0.0f, 0.1f);
    m_avgFrameMs += (frameMs - m_avgFrameMs) * 0.1f;

    if (m_benchFrames <= 0)
        return;
    m_frameNo++;
    // Flip every 120 frames, and only report the last frame of each block, so every
    // number printed has had 120 frames of rolling average behind it and none of them
    // straddle a settings change.
    const int block = 120;
    if (m_frameNo % block == 0)
    {
        DemoStats s;
        stats(s);
        printf(
            "frame %5d cull %d bodies %7d drawn %7d | physics %7.3f render %7.3f ui %6.3f | frame %7.3f (%5.1f fps)\n",
            m_frameNo, m_cull ? 1 : 0, s.bodies, m_lastDrawn, m_avgStepMs, m_avgDrawMs, m_avgUiMs, m_avgFrameMs,
            m_avgFrameMs > 0 ? 1000.0f / m_avgFrameMs : 0.0f);
        fflush(stdout);
        m_cull = !m_cull;
    }
    if (m_frameNo >= m_benchFrames)
        m_running = false;
}

// ---------------------------------------------------------------------------
// Panel
// ---------------------------------------------------------------------------

// One overlay toggle. These are checkboxes rather than a combo because that is what
// they read as -- a row of things you switch on to look at -- but only one can be on at
// a time, since they all write to the same channel: the body's colour. Ticking one
// therefore unticks the rest, and ticking the one already on turns overlays off.
void DemoApp::overlayToggle(const char *label, OverlayMode mode, const char *tip)
{
    bool on = m_overlay.mode == mode;
    if (ImGui::Checkbox(label, &on))
        m_overlay.mode = on ? mode : OVERLAY_NONE;
    itemTooltip(tip);
}

void DemoApp::drawUI()
{
    SolverParams &params = m_author.params;
    DemoStats st;
    stats(st);

    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(340, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin(m_host.panelTitle);

    if (m_stepFailMsg[0])
        ImGui::TextColored(ImVec4(1.0f, 0.25f, 0.25f, 1.0f), "%s", m_stepFailMsg);

    // gpuScenes[] first, then any demo-provided extras, under the single m_scene index
    // loadScene() receives.
    std::vector<const char *> sceneNames(gpuSceneNames, gpuSceneNames + gpuSceneCount);
    sceneNames.insert(sceneNames.end(), m_host.extraSceneNames.begin(), m_host.extraSceneNames.end());
    if (ImGui::Combo("Scene", &m_scene, sceneNames.data(), (int)sceneNames.size()))
        requestScene(m_scene);

    if (ImGui::Button(m_paused ? "Play" : "Pause"))
        setPaused(!m_paused);
    ImGui::SameLine();
    if (ImGui::Button("Step"))
    {
        m_paused = true;
        m_stepOnce = true;
        m_accum = 0.0f;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset"))
        requestScene(m_scene, false);

    ImGui::Separator();

    // Everything below is folded away and shut by default. The panel had grown to a
    // single scrolling column of sliders, and most of them are set once per session --
    // what you actually reach for repeatedly is the scene, the transport, and whichever
    // one group you are currently tuning.
    if (ImGui::CollapsingHeader("Solver"))
    {
        ImGui::SliderFloat("Time rate", &m_timeScale, 0.05f, 4.0f, "%.2fx", ImGuiSliderFlags_Logarithmic);
        itemTooltip("How fast simulated time runs compared to the clock: 1x is real time, 0.25x is slow motion, "
                    "2x makes blocks fall twice as fast. The physics rate below is unchanged, so slow motion "
                    "runs fewer ticks per second and fast forward runs more.");
        int hz = (int)lroundf(1.0f / std::max(params.dt, 1e-4f));
        if (ImGui::SliderInt("Physics rate", &hz, 10, (int)kMaxTickHz,
                             hz >= (int)kMaxTickHz ? "Max (match frame rate)" : "%d Hz"))
            params.dt = 1.0f / (float)std::max(hz, 1);
        itemTooltip("Physics ticks per second. Each tick advances the world by 1/rate seconds, and the "
                    "simulation is stepped on its own clock: rendering keeps its own frame rate, a fast display "
                    "does not step the physics faster, and a slow frame runs several ticks to catch up. Dragging to "
                    "the maximum removes the cap: one tick per rendered frame, as long as that frame. Higher "
                    "rates are more accurate and stiffer per second of simulated time but cost proportionally "
                    "more GPU time. If the ticks cannot keep up (more than 8 per frame, or about 24 ms of "
                    "work) the backlog is dropped and the simulation runs in slow motion while the frame "
                    "rate stays up. A tick is one step of simulated time; Iterations below are solver passes "
                    "inside each tick. Raise the rate for fast motion that tunnels or looks choppy, raise "
                    "iterations for stacks that sink or sag.");
        if (!m_paused && m_simHz > 0.0f)
            ImGui::TextDisabled("running at %.0f ticks/s%s", m_simHz,
                                m_simHz < 0.9f / std::max(params.dt, 1e-4f) ? "  (behind: slow motion)" : "");
        ImGui::SliderInt("Iterations", &params.iterations, 1, 64);
        itemTooltip("AVBD iterations per step. Each one is a primal pass (every body's "
                    "position is updated in parallel, by colour) followed by the dual "
                    "pass that updates the Lagrange multipliers and stiffnesses. Cost is "
                    "linear in this; the paper's large scenes run at 3-5. Unlike a dual "
                    "method, a low count here does not make the sim explode -- it just "
                    "leaves more constraint error, which the multipliers carry into the "
                    "next step rather than turning into momentum. These passes repeat inside one "
                    "physics tick and do not advance time (see Physics rate): more iterations make each step "
                    "stiffer and more accurate, not smoother in time.");

        ImGui::SliderFloat("Alpha", &params.alpha, 0.0f, 1.0f);
        itemTooltip("Stabilisation, paper Eq. 18: the solver targets "
                    "C(x) - alpha*C(x_prev) instead of C(x), so only (1 - alpha) of the "
                    "error that already existed at the start of the step gets corrected "
                    "now. alpha = 1 ignores all pre-existing error (nothing pushes "
                    "overlaps apart); alpha = 0 corrects it all in one step, which is "
                    "what makes a deep pile pop or jitter -- a hard constraint that was "
                    "violated last frame can inject an arbitrarily large force. The "
                    "paper uses 0.95, which bleeds the error out over several frames.");

        ImGui::SliderFloat("Gamma", &params.gamma, 0.0f, 1.0f);
        itemTooltip("Warm-start decay, paper Eq. 19. The stiffness and multiplier from "
                    "the end of the last step are reused as this step's starting guess, "
                    "scaled down first: k0 = max(gamma*k, k_start) and lambda0 = "
                    "alpha*gamma*lambda. The scaling matters because the ramp only ever "
                    "raises k, so without decay a stiffness needed for one hard frame "
                    "would be stuck there forever. gamma = 0 throws the history away "
                    "(every step restarts cold, far worse convergence); the paper uses "
                    "0.99. Note lambda is also scaled by alpha, so warm starting cannot "
                    "feed energy back in to correct old error.");

        ImGui::SliderFloat("Beta linear", &params.betaLin, 1.0f, 100000.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
        itemTooltip("Penalty ramp rate for linear (contact/attachment) constraints, "
                    "paper Eq. 12: k <- k + beta*|C(x)| after each iteration, so a "
                    "constraint that is still violated gets stiffer as the step "
                    "proceeds. This is what lets AVBD start soft -- keeping the Hessian "
                    "well conditioned and letting weak forces propagate globally -- and "
                    "end stiff enough to actually hold. The paper reports results are "
                    "insensitive across beta in [1, 1000]; even beta = 0 works, leaving "
                    "the multipliers to do all the work, but needs more iterations.");

        ImGui::SliderFloat("Gravity", &params.gravity.y, -30.0f, 0.0f);
        itemTooltip("Gravitational acceleration along -z, folded into the inertial "
                    "target y = x + dt*v + dt^2*a_ext that every primal step pulls "
                    "towards. Not a solver parameter -- it is the scene load, and "
                    "raising it makes the same stack harder to hold.");

        // "Never sleep" is the A/B lever: unticking it solves every dynamic body every
        // step, exactly as the solver did before sleeping existed, so the two can be
        // compared live on the same scene without restarting or swapping binaries.
        bool neverSleep = !params.sleepEnabled;
        if (ImGui::Checkbox("Never sleep", &neverSleep))
        {
            params.sleepEnabled = !neverSleep;
            // The next step's applyParams() turns sleeping off, which wakes everything:
            // with deactivation off nothing runs the wake kernels at all, so anything
            // already frozen would otherwise stay frozen forever.
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Solve every dynamic body every step. Use this to A/B the\n"
                              "simulation against itself and judge whether sleeping\n"
                              "changes anything you can see.");

        ImGui::BeginDisabled(neverSleep);
        ImGui::SliderInt("Sleep frames", &params.sleepFrames, 0, 120);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Settled steps before a body may freeze. 0 disables sleeping.");
        ImGui::SliderFloat("Sleep travel", &params.sleepDisp, 0.0001f, 0.05f, "%.4f m", ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Net movement allowed across the whole window. This is the\n"
                              "rest test; the velocity thresholds are only a fast reject.");
        ImGui::SliderFloat("Sleep leak", &params.sleepLeak, 0.0f, 4.0f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("0: any velocity spike restarts the rest window. Above 0: a spike\n"
                              "only drains progress in proportion to its size (leaky streak).");
        ImGui::SliderFloat("Sleep speed", &params.sleepLinVel, 0.01f, 2.0f, "%.2f m/s", ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Fast reject, applied at 2x, and the wake hysteresis.");
        ImGui::EndDisabled();

        ImGui::Checkbox("Kill fallen bodies", &params.killFallenEnabled);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Freezes and stops simulating anything that falls below\n"
                              "the Z threshold below -- for bodies that missed the floor.");
        ImGui::BeginDisabled(!params.killFallenEnabled);
        ImGui::SliderFloat("Kill height", &params.killHeight, -500.0f, 0.0f, "%.0f m");
        ImGui::EndDisabled();
    }

    if (ImGui::CollapsingHeader("Grab"))
    {
        ImGui::SliderFloat("Grab strength", &params.grabStrength, 0.05f, 2.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Multiplier on the mouse-drag spring stiffness.\n"
                              "Lower it if a heavy structure feels glued to the cursor.");
        ImGui::Checkbox("Grab connected bodies", &m_grabConnected);
        if (m_grabConnected)
        {
            ImGui::SliderFloat("Grab radius", &m_grabRadius, 0.0f, 100.0f, "%.1f m", ImGuiSliderFlags_Logarithmic);
            ImGui::TextUnformatted("Follows joints and springs out from the");
            ImGui::TextUnformatted("body you clicked, within the radius.");
            ImGui::SliderFloat("Falloff", &m_grabFalloff, 0.0f, 8.0f, "%.2f");
            ImGui::SliderFloat("Min strength", &m_grabFalloffMin, 0.0f, 1.0f, "%.2f");
            ImGui::TextUnformatted("0 falloff holds the whole group equally;");
            ImGui::TextUnformatted("higher concentrates the pull on the click.");

            // The curve itself, sampled from the cursor out to the rim. A slider pair
            // shaping a falloff is hard to read as numbers; the plot makes the knee of
            // the curve, which is the thing being tuned, visible at a glance.
            float curve[48];
            for (int i = 0; i < 48; i++)
                curve[i] = grabWeight(m_grabRadius * (float)i / 47.0f, m_grabRadius);
            ImGui::PlotLines("##grabfalloff", curve, 48, 0, "strength vs distance", 0.0f, 1.0f, ImVec2(0, 45));
        }
        else
        {
            ImGui::TextUnformatted("Single body.");
        }
        if (m_dragging)
            ImGui::Text("holding %d bodies", (int)m_dragGroup.size());
    }

    if (ImGui::CollapsingHeader("Spawn"))
    {
        ImGui::Combo("Object", &m_spawnKind, spawnKindNames, SpawnKindCount);

        // The plane is a fixed piece of geometry with its own proportions, so the size
        // slider has nothing to do there; it still takes mass, speed and distance.
        const bool sizeApplies = m_spawnKind != SpawnKindPlane;
        if (!sizeApplies)
            ImGui::BeginDisabled();
        ImGui::SliderFloat("Spawn size", &m_spawnSize, 0.25f, 8.0f);
        if (!sizeApplies)
            ImGui::EndDisabled();

        ImGui::SliderFloat("Mass multiplier", &m_spawnMass, 0.05f, 200000.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::TextUnformatted("Multiplier for the spawned object's density;");
        ImGui::TextUnformatted("mass scales with the object's volume.");
        ImGui::SliderFloat("Spawn speed", &m_spawnSpeed, 0.0f, 60.0f);
        ImGui::SliderFloat("Spawn dist", &m_spawnDist, 0.5f, 200.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
    }

    if (ImGui::CollapsingHeader("Camera & controls"))
    {
        ImGui::SliderFloat("Look sensitivity", &m_lookSensitivity, 0.001f, 0.02f, "%.4f",
                           ImGuiSliderFlags_Logarithmic);
        ImGui::Checkbox("Smooth camera", &m_smoothCamera);
        if (m_smoothCamera)
        {
            ImGui::SliderFloat("Smoothness", &m_smoothCameraAmount, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
            if (m_smoothCameraAmount >= 0.99f)
            {
                ImGui::SameLine();
                ImGui::TextUnformatted("(cinematic)");
            }
        }
        ImGui::SliderFloat("Fly speed", &m_camMoveSpeed, 1.0f, 300.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
        ImGui::TextUnformatted("LMB drag  RMB look  MMB spawn  wheel dolly");
        ImGui::TextUnformatted("WASD fly  Q/E down/up  shift fast");
    }

    if (ImGui::CollapsingHeader("Display"))
    {
        ImGui::Checkbox("Frustum culling", &m_cull);
        ImGui::SameLine();
        if (ImGui::Checkbox("VSync", &m_vsync) && m_host.setVsync)
            m_host.setVsync(m_vsync);
        if (m_host.setFullscreen && m_host.isFullscreen)
        {
            bool fs = m_host.isFullscreen();
            if (ImGui::Checkbox("Fullscreen (F11)", &fs))
                m_host.setFullscreen(fs);
        }
        ImGui::Checkbox("FPS & scene overlay", &m_showOverlay);
        itemTooltip("Frame rate and the current scene's name, in the top right corner.");
        ImGui::BeginDisabled(!m_showOverlay);
        ImGui::SliderFloat("Overlay font size", &m_overlayFontSize, 8.0f, 64.0f, "%.0f px");
        itemTooltip("Size of the top right overlay text, in pixels.");
        ImGui::EndDisabled();
    }

    if (ImGui::CollapsingHeader("Editor"))
        editorPanel();

    if (m_host.graphicsUI && ImGui::CollapsingHeader("Graphics"))
        m_host.graphicsUI();

    if (ImGui::CollapsingHeader("Debug overlays"))
    {
        // Every overlay recolours the instances the solver already writes for the
        // renderer, inside the same kernel, so all of them survive 100k bodies and none
        // of them changes a number the solver computes.
        overlayToggle("Sleeping bodies", OVERLAY_SLEEP,
                      "Blue: frozen, and dropped from the constraint graph entirely. Amber to pale: awake, cooling as "
                      "its rest timer fills.");
        overlayToggle("Broadphase grid", OVERLAY_GRID,
                      "One hue per occupied hash-grid cell. Bodies are binned by centre and tested against the 27-cell "
                      "neighbourhood, so this is the partition that decides which pairs are considered at all.");
        overlayToggle("Graph colors", OVERLAY_GRAPH_COLOR,
                      "Jones-Plassmann coloring of the constraint graph. Bodies sharing a hue share no constraint and "
                      "are solved in the same dispatch -- one dispatch per hue, per iteration.");
        overlayToggle("Contact load", OVERLAY_LOAD,
                      "Peak contact normal force on the body. This is where the force chains through an arch, a tower "
                      "or a pile become visible.");
        overlayToggle("Speed", OVERLAY_SPEED, "Linear speed. Shows the creep a settled-looking pile is hiding.");
        overlayToggle("Contact degree", OVERLAY_DEGREE,
                      "How many bodies this one is touching. Needs sleeping enabled: the count is maintained as the "
                      "sleep logic's contact fingerprint.");

        ImGui::SeparatorText("Structure");
        overlayToggle("Joint count", OVERLAY_JOINT_DEGREE,
                      "How many unbroken joints and springs are attached to this body. Grey means none at all, which "
                      "is a different thing from zero load. A structure's articulation points are its high counts.");
        overlayToggle("Joint load", OVERLAY_JOINT_LOAD,
                      "Axial force in the connected joints and springs, signed: blue is compression (being squeezed), "
                      "red is tension (being pulled apart), pale grey is unloaded. Only the axial component counts -- "
                      "a strut is judged by what it carries along its length.");
        overlayToggle("Structural load", OVERLAY_AXIAL_LOAD,
                      "The same diverging scale, but contacts count as well as joints. Every contact can only push, so "
                      "this reads a masonry pile as solidly blue and shows tension only where something is actually "
                      "holding it together.");
        overlayToggle(
            "Joint stress", OVERLAY_JOINT_STRESS,
            "Worst connected joint as a fraction of its own fracture limit, so 1.0 is the frame it snaps on. This is "
            "the weak-point view: whatever is white is what breaks next. Grey means no connected joint is breakable.");
        overlayToggle("Friction cone", OVERLAY_FRICTION,
                      "How much of the available friction each contact has spent. White is a contact that has run out "
                      "and is sliding -- a structure shears along whatever is white here before it visibly moves.");
        overlayToggle(
            "Penetration", OVERLAY_PENETRATION,
            "Deepest overlap the last step left behind, measured where the narrowphase picked it up. Bright means the "
            "solve is not holding that contact, which is where a stack starts sinking into itself.");

        switch (m_overlay.mode)
        {
        case OVERLAY_SPEED:
            ImGui::SliderFloat("Speed range", &m_overlay.speedMax, 0.05f, 100.0f, "%.2f m/s",
                               ImGuiSliderFlags_Logarithmic);
            break;
        case OVERLAY_LOAD:
            ImGui::SliderFloat("Load range", &m_overlay.loadMax, 1.0f, 100000.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
            break;
        case OVERLAY_DEGREE:
            ImGui::SliderInt("Degree range", &m_overlay.degreeMax, 1, 32);
            if (params.sleepFrames <= 0)
                ImGui::TextUnformatted("(sleeping is off, so there is no count to show)");
            break;
        case OVERLAY_JOINT_DEGREE:
            ImGui::SliderInt("Count range", &m_overlay.jointDegreeMax, 1, 32);
            if (st.joints == 0 && st.springs == 0)
                ImGui::TextUnformatted("(this scene has no joints or springs)");
            break;
        case OVERLAY_JOINT_LOAD:
        case OVERLAY_AXIAL_LOAD:
            ImGui::SliderFloat("Load range", &m_overlay.jointLoadMax, 1.0f, 100000.0f, "%.0f N",
                               ImGuiSliderFlags_Logarithmic);
            ImGui::TextUnformatted("blue compression | grey unloaded | red tension");
            break;
        case OVERLAY_JOINT_STRESS:
            ImGui::TextUnformatted("0 = slack, 1 = fracturing this frame");
            break;
        case OVERLAY_FRICTION:
            ImGui::TextUnformatted("0 = held with friction to spare, 1 = sliding");
            break;
        case OVERLAY_PENETRATION:
            ImGui::SliderFloat("Depth range", &m_overlay.penetrationMax, 0.0005f, 0.5f, "%.4f m",
                               ImGuiSliderFlags_Logarithmic);
            break;
        case OVERLAY_GRID:
            ImGui::Text("cell %.2f m  (largest dynamic body + margin)", st.cellSize);
            break;
        case OVERLAY_GRAPH_COLOR:
            ImGui::Text("%d colors over %d awake bodies", st.colors, st.awake);
            break;
        case OVERLAY_SLEEP:
            ImGui::Text("%d of %d bodies awake", st.awake, st.bodies);
            break;
        default:
            break;
        }

        if (m_overlay.mode != OVERLAY_NONE)
        {
            ImGui::Checkbox("Include statics", &m_overlay.colorStatics);
            itemTooltip(
                "Statics are in none of these data sets -- not binned in the grid, not in the constraint graph, never "
                "asleep -- so by default they stay neutral grey rather than say something untrue about themselves.");
        }
    }

    if (ImGui::CollapsingHeader("Contact markers"))
    {
        // Unlike the overlays above, these draw extra geometry rather than recolouring
        // what is already there, so they are the one debug view with a cost worth
        // mentioning in the UI -- hence the budget and the truncation notice.
        ImGui::Checkbox("Contact points", &m_debug.contactPoints);
        itemTooltip("A small plate at each contact point, lying in the contact plane. A box-box face contact is a "
                    "four-point polygon clip, so a flat landing shows four plates, not one.");
        ImGui::Checkbox("Contact normals", &m_debug.contactNormals);
        itemTooltip("A needle along each contact normal, its length scaled by the normal force. Together these are a "
                    "force diagram: the direction each contact is pushing, and how hard.");
        ImGui::Checkbox("Manifold links", &m_debug.manifoldLinks);
        itemTooltip(
            "One rod per touching pair, centre to centre, coloured by the total force the pair carries. This is the "
            "force network -- the load path down through an arch or a tower is the bright chain running through it.");
        ImGui::Checkbox("Joint load on rods", &m_debug.jointLoadRods);
        itemTooltip("Colour the joint rods by signed axial load -- blue compression, red tension -- instead of by how "
                    "close they are to fracturing.");

        if (m_debug.jointLoadRods && !m_debug.any())
            ImGui::SliderFloat("Force range", &m_debug.forceMax, 1.0f, 100000.0f, "%.0f N",
                               ImGuiSliderFlags_Logarithmic);

        if (m_debug.any())
        {
            const char *tints[] = {"Normal force", "Friction cone", "Stick / slip", "Flat"};
            int tintIdx = (int)m_debug.contactTint;
            if (ImGui::Combo("Marker color", &tintIdx, tints, 4))
                m_debug.contactTint = (ContactTint)tintIdx;
            itemTooltip("Normal force: how hard the contact is pressing. Friction cone: how much of its friction it "
                        "has spent, white where it is sliding. Stick / slip: green where static friction is holding "
                        "the anchor, orange where it has let go.");

            ImGui::SliderFloat("Marker size", &m_debug.contactPointSize, 0.005f, 0.5f, "%.3f m",
                               ImGuiSliderFlags_Logarithmic);
            if (m_debug.contactNormals)
                ImGui::SliderFloat("Normal length", &m_debug.normalLength, 0.02f, 2.0f, "%.2f m",
                                   ImGuiSliderFlags_Logarithmic);
            if (m_debug.contactTint == CONTACT_TINT_FORCE || m_debug.manifoldLinks || m_debug.contactNormals)
                ImGui::SliderFloat("Force range", &m_debug.forceMax, 1.0f, 100000.0f, "%.0f N",
                                   ImGuiSliderFlags_Logarithmic);

            ImGui::SliderInt("Marker budget", &m_debug.maxMarkers, 1000, 1000000);
            itemTooltip("Hard ceiling on marker instances. Past it the extra markers are dropped rather than the set "
                        "being thinned, so what you are looking at stays put between frames instead of reshuffling.");

            if (st.debugRequested > m_debug.maxMarkers)
                ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "showing %d of %d (raise the budget)",
                                   m_debug.maxMarkers, st.debugRequested);
            else
                ImGui::Text("%d markers", st.debugRequested);
        }
    }

    if (ImGui::CollapsingHeader("Scene stats"))
    {
        ImGui::Text("device    %s", st.deviceName);
        ImGui::Text("bodies    %d (%d awake)", st.bodies, st.awake);
        ImGui::Text("joints    %d   springs %d", st.joints, st.springs);
        ImGui::Text("drawn     %d (%.0f%% culled)", m_lastDrawn,
                    st.bodies > 0 ? 100.0f * (1.0f - (float)m_lastDrawn / st.bodies) : 0.0f);
        ImGui::Text("pairs     %d / %d", st.pairs, st.pairCapacity);
        ImGui::Text("manifolds %d / %d", st.manifolds, st.manifoldCapacity);
        ImGui::Text("contacts  %d / %d", st.contacts, st.contactCapacity);
        ImGui::Text("colors    %d (JP rounds %d)", st.colors, st.jpRounds);
        ImGui::Text("device mem %.1f MB", st.deviceMB);
    }

    if (ImGui::CollapsingHeader("Timings"))
    {
        // Where the GPU time goes, each phase as a share of the whole pass.
        if (st.gpuMs > 0.0f)
        {
            ImGui::Text("GPU total %.3f ms", st.gpuMs);
            auto row = [&](const char *name, float ms)
            {
                char label[64];
                snprintf(label, sizeof(label), "%.3f ms", ms);
                ImGui::ProgressBar(std::min(std::max(ms / st.gpuMs, 0.0f), 1.0f), ImVec2(120, 0), label);
                ImGui::SameLine();
                ImGui::TextUnformatted(name);
            };
            row("integrate + front", st.frontMs);
            row("broadphase", st.broadMs);
            row("narrowphase + wake", st.narrowMs);
            row("graph + colouring", st.gpuBuildMs);
            row("iterations (primal + dual)", st.iterMs);
            row("velocity, guard, sleep", st.tailMs);
        }
        else
            ImGui::TextUnformatted("GPU timestamps unavailable");
        float maxMs = 1.0f;
        for (int i = 0; i < kTimeHistory; i++)
            maxMs = std::max(maxMs, m_histHostMs[i]);
        ImGui::PlotLines("##hosthist", m_histHostMs, kTimeHistory, m_histPos, "host step ms", 0.0f, maxMs,
                         ImVec2(0, 50));
        ImGui::PlotLines("##gpuhist", m_histGpuMs, kTimeHistory, m_histPos, "GPU ms", 0.0f, maxMs, ImVec2(0, 50));
        ImGui::Separator();

        ImGui::Text("step      %6.3f ms", m_avgStepMs);
        if (st.broadMs >= 0.0f)
        {
            ImGui::Text("  broad   %6.3f ms", st.broadMs);
            ImGui::Text("  narrow  %6.3f ms", st.narrowMs);
            ImGui::Text("  solve   %6.3f ms", st.solveMs);
            // The solve is two unrelated costs: rebuilding and coloring the constraint
            // graph, then the iterations themselves. Primal and dual are per iteration.
            ImGui::Text("    graph %6.3f ms", st.buildMs);
            ImGui::Text("    iter  %6.3f ms (primal %.3f, dual %.3f per it)", st.iterMs, st.primalMs, st.dualMs);
        }
        if (st.gpuMs >= 0.0f)
        {
            ImGui::Text("  front   %6.3f ms", st.frontMs);
            ImGui::Text("  gpu     %6.3f ms (build %.3f, tail %.3f)", st.gpuMs, st.gpuBuildMs, st.tailMs);
        }
        if (st.hostWaitMs >= 0.0f)
            ImGui::Text("  host wait %5.3f ms", st.hostWaitMs);
        if (st.reruns > 0)
            ImGui::Text("  reruns  %d (capacity grown, step redone)", st.reruns);
        ImGui::Text("render    %6.3f ms (boxes)", m_avgDrawMs);
        ImGui::Text("  ui      %6.3f ms", m_avgUiMs);
        ImGui::Text("frame     %6.3f ms (%.0f fps)", m_avgFrameMs, m_avgFrameMs > 0 ? 1000.0f / m_avgFrameMs : 0.0f);

        // Physics and render are disjoint; what is left over in the frame is vsync, the
        // swap and the CPU -- so this is a share of the two, not of the frame.
        const float busy = m_avgStepMs + m_avgDrawMs + m_avgUiMs;
        if (busy > 0.0f)
            ImGui::Text("physics %.0f%% / render %.0f%% / ui %.0f%%", 100.0f * m_avgStepMs / busy,
                        100.0f * m_avgDrawMs / busy, 100.0f * m_avgUiMs / busy);
    }

    ImGui::End();

    editorDrawUI();

    if (m_showOverlay)
    {
        const char *name = m_scene < gpuSceneCount ? gpuSceneNames[m_scene]
                                                   : m_host.extraSceneNames[m_scene - gpuSceneCount];
        char fps[32];
        snprintf(fps, sizeof(fps), "%.0f FPS", m_avgFrameMs > 0.0f ? 1000.0f / m_avgFrameMs : 0.0f);
        ImDrawList *dl = ImGui::GetForegroundDrawList();
        ImFont *font = ImGui::GetFont();
        const float fs = m_overlayFontSize;
        const float right = ImGui::GetIO().DisplaySize.x - 12.0f;
        float y = 8.0f;
        char tps[32];
        snprintf(tps, sizeof(tps), "%.0f ticks/s", m_paused ? 0.0f : m_simHz);
        for (const char *line : {(const char *)fps, (const char *)tps, name})
        {
            const ImVec2 sz = font->CalcTextSizeA(fs, FLT_MAX, 0.0f, line);
            const ImVec2 p(right - sz.x, y);
            dl->AddText(font, fs, ImVec2(p.x + 1.0f, p.y + 1.0f), IM_COL32(0, 0, 0, 200), line);
            dl->AddText(font, fs, p, IM_COL32(235, 240, 250, 255), line);
            y += sz.y + 2.0f;
        }
    }
}
