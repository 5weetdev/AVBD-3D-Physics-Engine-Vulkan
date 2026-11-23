#pragma once

// The demo, minus the window and the renderer.
//
// Scene selection, the ImGui panel, the fly camera, click-to-grab, middle-click spawning,
// the benchmark mode and the hooks the --smoke driver scripts the same paths through.
// DemoApp owns the simulation: a SceneBuilder that authors scenes and holds the solver
// parameters, and the avbd3d world that runs them, reached only through the public C API
// (src/api). SceneImport replays each authored scene into the world. demo owns
// the window, the swapchain and the renderer, and reads what it draws from world(), overlay()
// and debugDraw().

#include "scene_types.h"
#include "solver_params.h"
#include "maths.h"
#include "scenes.h"
#include "scene_builder.h"
#include "scene_import.h"

#include "avbd3d/avbd3d.h"
#include "avbd3d/avbd3d_vulkan.h"

#include <SDL2/SDL.h>

#include <functional>
#include <vector>

struct DemoHost
{
    const char *panelTitle = "Physics Engine";
    // Drawable size in pixels (differs from the window size on high-DPI displays).
    std::function<void(int &w, int &h)> drawableSize;
    // Window size in points, for scaling mouse coordinates into drawable pixels.
    std::function<void(int &w, int &h)> windowSize;
    // The renderer's own section of the panel ("Graphics"). Optional.
    std::function<void()> graphicsUI;
    std::function<void(bool)> setVsync;
    // Borderless desktop fullscreen. Optional.
    std::function<void(bool)> setFullscreen;
    std::function<bool()> isFullscreen;
    // Demo-provided scenes appended after gpuScenes[] in the scene combo. Selecting
    // the i'th name loads index gpuSceneCount + i; DemoApp::loadScene owns what those
    // indices mean (currently the one "Spheres & Capsules" scene).
    std::vector<const char *> extraSceneNames;
};

struct SpawnEmitter;

// What the command line asked for; see loadInitialScene().
struct DemoLaunch
{
    int scene = -1;                  // index into the scene combo; -1 = the default
    const char *sceneName = nullptr; // "3" or a case-insensitive scene name; wins over scene
    int benchFrames = 0;             // > 0: unattended run, vsync off, quit after N frames
    float camDist = -1.0f;           // > 0 overrides the scene's camera distance
    bool unattended = false;         // bench/smoke: step failures go to stderr
    bool noSleep = false;            // start with "Never sleep" ticked (benchmarks)
};

struct DemoStats
{
    int bodies = 0, awake = 0, joints = 0, springs = 0;
    int pairs = 0, manifolds = 0, contacts = 0, colors = 0;
    float cellSize = 0.0f;
    int debugRequested = 0;

    // Per-phase timings in ms; negative means not measured and the panel hides the line.
    float stepMs = -1, broadMs = -1, narrowMs = -1, solveMs = -1;
    float buildMs = -1, iterMs = -1, primalMs = -1, dualMs = -1;
    float frontMs = -1, gpuBuildMs = -1, tailMs = -1, gpuMs = -1, hostWaitMs = -1;

    int jpRounds = 0;
    int pairCapacity = 0, contactCapacity = 0, manifoldCapacity = 0;
    int reruns = 0;
    float deviceMB = 0.0f;
    const char *deviceName = "";
};

class DemoApp
{
public:
    static constexpr float kFovY = 45.0f;
    static constexpr float kNear = 0.5f;
    static constexpr float kFar = 4000.0f;

    // The demo shares the solver's GPU for its swapchain rather than creating a second
    // device (Vulkan objects are device-scoped). The window creates the GPU (presentable,
    // with its surface) and hands it over before init(); DemoApp creates its worlds on it.
    void setGpu(Avbd3dGpu *gpu);
    Avbd3dWorld *world() const { return m_world; }
    // Destroys the world (the GPU stays with whoever created it).
    void shutdown();

    // Live solver parameters. The scene builders set per-scene values on these.
    SolverParams &params() { return m_author.params; }
    bool findShadowPlane(vec3 &point, vec3 &normal) { return m_author.findShadowPlane(point, normal); }

    void init(const DemoHost &host);

    // Applies the launch options and loads the first scene. Returns false (after saying
    // why on stderr) when the requested scene does not exist.
    bool loadInitialScene(const DemoLaunch &launch);

    void handleEvent(const SDL_Event &e);

    // Camera keys, grab, and the solver step. Call once a frame before drawing.
    void update();

    // The panel. Call between ImGui::NewFrame() and ImGui::Render().
    void drawUI();

    // Frame bookkeeping and benchmark mode. frameMs is the wall-clock frame time.
    void endFrame(float frameMs);

    bool running() const { return m_running; }
    void quit() { m_running = false; }

    // --- Scripted driving (the --smoke run) --------------------------------------------
    // The same code the mouse and the panel reach, without faking SDL events. Pixel
    // arguments are drawable pixels.
    int sceneCount() const { return gpuSceneCount + (int)m_host.extraSceneNames.size(); }
    void loadSceneIndex(int index); // also un-pauses, so every scene gets stepped
    bool spawnAtPixel(int px, int py);
    bool grabAtPixel(int px, int py);
    void moveGrab(int px, int py);
    void releaseGrab()
    {
        if (m_dragging)
            endGrab();
    }
    bool grabbing() const { return m_dragging; }
    // The editor's paths, scripted: 0 pause + marquee over the window, 1 duplicate and move
    // the selection along X, 2 delete it. Pixels are drawable pixels.
    void smokeEdit(int step, int w, int h);
    // Steps that came back not ok / truncated since launch.
    int stepFailures() const { return m_stepFailures; }

    // --- Camera -----------------------------------------------------------------------
    vec3 camEye() const;
    vec3 camTarget() const { return m_dispTarget; }
    float camDistance() const { return m_dispDistance; }
    // Column-major view matrix, looking from camEye() at camTarget(), +Y up.
    void viewMatrix(float out[16]) const;
    // Column-major perspective. vulkanClip: depth 0..1 and Y flipped; otherwise -1..1 depth (OpenGL-style).
    static void perspective(float out[16], float aspect, bool vulkanClip);
    static void matMul(float out[16], const float a[16], const float b[16]);

    // --- State the renderer consumes ---------------------------------------------------
    const OverlayOptions &overlay() const { return m_overlay; }
    const DebugDrawOptions &debugDraw() const { return m_debug; }
    bool cullEnabled() const { return m_cull; }
    bool vsyncEnabled() const { return m_vsync; }

    // --- Measurements the renderer feeds back, shown in the panel ---------------------
    void setRenderTimings(float drawMs, float uiMs)
    {
        m_avgDrawMs += (drawMs - m_avgDrawMs) * 0.1f;
        m_avgUiMs += (uiMs - m_avgUiMs) * 0.1f;
    }
    void setDrawnCount(int n) { m_lastDrawn = n; }

private:
    void loadScene(int index, bool resetCamera = true);
    // drawUI() runs while the frame's command buffer is being recorded, with draws already
    // bound to the world's instance buffer. loadScene() destroys that buffer, so the panel only
    // asks and update() loads at the top of the next frame.
    void requestScene(int index, bool resetCamera = true)
    {
        m_pendingScene = index;
        m_pendingResetCamera = resetCamera;
    }
    int m_pendingScene = -1;
    bool m_pendingResetCamera = true;
    void setPaused(bool paused);
    // Runs the fixed-rate physics ticks the elapsed frame time asks for (see update()).
    void runPhysics();
    void noteStepResult(Avbd3dResult res, const Avbd3dStats &r);
    bool rayFromPixel(int px, int py, vec3 &origin, vec3 &dir) const;
    void toPixels(int x, int y, int &px, int &py) const;
    float grabWeight(float dist, float radius) const;
    void beginGrab(Avbd3dBodyId hit, vec3 worldHit, vec3 localHit);
    void dropDetached();
    void updateGrab(vec3 cursorPoint);
    void endGrab();
    void updateGrabFromCursor();
    void spawnAlongRay(vec3 rayOrigin, vec3 rayDir);
    void updateCameraKeys(float dtSec);
    void updateSmoothCamera(float dtSec);
    vec3 rawEye() const;
    vec3 dispForward() const;
    vec3 dispRight() const;
    vec3 camForward() const;
    vec3 camRight() const;
    void overlayToggle(const char *label, OverlayMode mode, const char *tip);
    // Pushes the live SolverParams into the world (plain field copies; done every step).
    void applyParams();

    // Scene construction flows one way -- gpuScenes[i](&m_author), m_author.finalize(),
    // m_import.importTail -- and interaction reaches the live world directly.
    SceneView buildSpheresCapsulesScene();
    void stats(DemoStats &out);
    void spawnBody(vec3 size, float density, float friction, vec3 position, vec3 velocity);
    void spawnAssembly(const SpawnEmitter &emitter, vec3 origin, quat rotation, float scale, float massMultiplier,
                       vec3 velocity);

    // --- Scene editor (demo_editor.cpp; paused only) -----------------------------------
    // Selection is a set of desc body indices (the SceneImport order). Everything that
    // touches the world runs from update() / handleEvent(), never from drawUI(), which
    // executes while the frame's command buffer is being recorded.
    enum EditAction
    {
        EditNone = 0,
        EditSelectAll,
        EditSelectNone,
        EditInvert,
        EditDelete,
        EditDuplicate,
        EditFocus
    };
    void editorReset();
    void editorUpdate();       // per frame: pending actions, gizmo drag, selection centre
    void editorMouseDown(int px, int py);
    void editorMouseUp(int px, int py);
    bool editorKey(const SDL_KeyboardEvent &k); // true when the key was an editor hotkey
    void editorDrawUI();       // selection outlines, box, gizmo (ImGui draw lists)
    void editorPanel();        // the "Editor" section of the panel
    void editorRun(EditAction a);
    int  editorBodyCount() const { return m_import.bodyCount(); }
    bool editorAlive(int i) const { return i >= 0 && i < m_import.bodyCount() && !(i < (int)m_edDeleted.size() && m_edDeleted[i]); }
    bool editorStatic(int i) const;
    bool editorSelected(int i) const { return i >= 0 && i < (int)m_edMask.size() && m_edMask[i]; }
    void editorSelect(int i, bool on);
    void editorClear();
    int  editorPick(int px, int py) const; // desc index under the pixel, or -1
    bool project(vec3 p, float &sx, float &sy) const; // world -> drawable pixels
    bool gizmoAxes(float out[3][4], float &len) const; // screen segments of the three axes
    int  gizmoHit(int px, int py) const;
    bool axisParam(int px, int py, vec3 origin, vec3 axis, float &t) const;
    void editorApplyDelta(vec3 delta);

    std::vector<int> m_edSel;
    std::vector<uint8_t> m_edMask, m_edDeleted;
    EditAction m_edPending = EditNone;
    bool m_edDown = false;      // left button held in editor mode
    bool m_edBox = false;       // ... and it has become a box selection
    int m_edX0 = 0, m_edY0 = 0, m_edX1 = 0, m_edY1 = 0; // drawable pixels
    int m_edMode = 0;           // 0 replace, 1 add (shift), 2 toggle (ctrl)
    vec3 m_edCenter = {0, 0, 0};
    int m_edAxisHover = -1, m_edAxis = -1; // gizmo axis under the cursor / being dragged
    vec3 m_edAxisDir = {1, 0, 0}, m_edAnchor = {0, 0, 0};
    float m_edT0 = 0.0f;
    std::vector<vec3> m_edStartPos;
    std::vector<quat> m_edStartRot;
    bool m_edSnap = false;
    float m_edSnapStep = 0.5f;

    // --- Fixed-rate physics ----------------------------------------------------------------
    static constexpr float kMaxTickHz = 240.0f; // top of the rate slider = uncapped
    float m_timeScale = 1.0f; // simulated seconds per wall-clock second
    float m_accum = 0.0f;   // wall-clock seconds not yet turned into physics ticks
    float m_simHz = 0.0f;   // smoothed ticks per second actually executed

    // --- Overlay (top right) ---------------------------------------------------------------
    bool m_showOverlay = true;
    float m_overlayFontSize = 18.0f;

    SceneBuilder m_author; // scene authoring + params only
    Avbd3dGpu *m_gpu = nullptr;
    char m_deviceName[256] = {};
    Avbd3dWorld *m_world = nullptr;
    SceneImport m_import;
    float m_cellSize = 0.0f; // the authoring grid, for the broadphase overlay

    DemoHost m_host;
    bool m_running = true;

    int m_scene = 4;
    bool m_paused = false;
    bool m_stepOnce = false;
    bool m_unattended = false;
    int m_stepFailures = 0;
    char m_stepFailMsg[256] = {}; // non-empty: shown red at the top of the panel

    float m_camDistance = 40.0f;
    float m_camAzimuth = 1.5707963f;
    float m_camElevation = 0.35f;
    vec3 m_camTarget = {0, 0, 5.0f};
    float m_camMoveSpeed = 25.0f;
    float m_lookSensitivity = 0.005f; // radians per pixel of mouse look
    // Displayed (smoothed) camera; the m_cam* values above are the raw input-driven one.
    float m_dispDistance = 40.0f;
    float m_dispAzimuth = 1.5707963f;
    float m_dispElevation = 0.35f;
    vec3 m_dispTarget = {0, 0, 5.0f};
    bool m_smoothCamera = true;
    float m_smoothCameraAmount = 0.25f;
    bool m_camSnap = true;

    enum SpawnKind
    {
        SpawnKindBox = 0,
        SpawnKindPlane,
        SpawnKindRagdoll,
        SpawnKindCount
    };
    int m_spawnKind = SpawnKindBox;
    float m_spawnSize = 1.0f;
    float m_spawnSpeed = 20.0f;
    float m_spawnMass = 1.0f;
    float m_spawnDist = 10.0f;

    bool m_dragging = false;
    Avbd3dBodyId m_dragBody = {};
    float m_dragDist = 0.0f;
    int m_dragMouseX = 0, m_dragMouseY = 0;
    Avbd3dBodyId m_dragSeed = {};
    std::vector<Avbd3dBodyId> m_dragGroup;
    std::vector<vec3> m_dragLocals, m_dragOffsets;
    std::vector<float> m_dragWeights;
    std::vector<uint8_t> m_dragKeep;
    std::vector<Avbd3dGrabHandle> m_dragHandles;
    int m_dragCheckedStep = -1; // step count of the last survivor test: once per step

    bool m_grabConnected = true;
    float m_grabRadius = 8.0f;
    float m_grabFalloff = 2.0f;
    float m_grabFalloffMin = 0.05f;

    OverlayOptions m_overlay;
    DebugDrawOptions m_debug;
    bool m_cull = true;
    bool m_vsync = true;

    float m_avgStepMs = 0, m_avgFrameMs = 0, m_avgDrawMs = 0, m_avgUiMs = 0;
    // Step-time history for the Timings plots: host step ms and GPU ms, newest at histPos-1.
    static constexpr int kTimeHistory = 180;
    float m_histHostMs[kTimeHistory] = {};
    float m_histGpuMs[kTimeHistory] = {};
    int m_histPos = 0;
    float m_lastFrameSec = 1.0f / 60.0f;
    int m_lastDrawn = 0;

    int m_benchFrames = 0;
    int m_frameNo = 0;
    int m_stepCount = 0; // ticks since the scene was loaded
};
