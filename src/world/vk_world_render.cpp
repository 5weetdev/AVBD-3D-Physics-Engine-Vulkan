#include "vk_world.h"
#include "vk_world_internal.h"
#include "vk_capacity.h"
#include "vk_small_world.h"
#include "collision.h" // MAX_MANIFOLD_CONTACTS
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace avbdvk
{


void VkWorld::computeRenderGroups(const SceneDesc &scene)
{
    // Render groups: groups 0..2 are the shape classes and
    // always exist; a body's authored group (>= 0) overrides its shape. Every group's
    // region is sized for every shape that could land in it, so the kernel's atomic can
    // never run past its region and culling only ever compacts within one. Instances are
    // per shape, so the group is per shape: the owning body's authored group, else the
    // shape's own class.
    const int n = scene.shapeCount;
    std::vector<int> eff(n);
    int groups = 3;
    for (int i = 0; i < n; i++)
    {
        const int body = scene.shapeBody[i];
        const int authored = body < (int)scene.renderGroup.size() ? scene.renderGroup[body] : -1;
        const unsigned char s = scene.shapeKind[i];
        eff[i] = authored >= 0 ? authored : (s < 3 ? (int)s : 0);
        groups = std::max(groups, eff[i] + 1);
    }
    std::vector<int> count(groups, 0);
    for (int i = 0; i < n; i++)
        count[eff[i]]++;
    m_groupBase.assign(groups, 0);
    for (int k = 1; k < groups; k++)
        m_groupBase[k] = m_groupBase[k - 1] + count[k - 1];
    m_groupVisible.assign(groups, 0);

    if (groupIdx.size() < (VkDeviceSize)std::max(n, 1) * sizeof(int))
        groupIdx.resize((VkDeviceSize)std::max(n, 1) * sizeof(int));
    if (n > 0)
        groupIdx.upload(eff.data(), n * sizeof(int));
    if (shapeBaseBuf.size() < (VkDeviceSize)groups * sizeof(int32_t))
    {
        shapeBaseBuf.resize(groups * sizeof(int32_t));
        shapeVisBuf.resize(groups * sizeof(int32_t));
    }
    shapeBaseBuf.upload(m_groupBase.data(), groups * sizeof(int32_t));
}


void VkWorld::writeInstances(const RenderInstancesParams &params)
{
    assertIdle("VkWorld::writeInstances");
    VkDeviceSize need = (VkDeviceSize)std::max(m_shapeCount, 1) * sizeof(BodyInstanceGpu);
    if (instances.size() < need)
        instances.resize(need, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    std::fill(m_groupVisible.begin(), m_groupVisible.end(), 0);
    if (m_bodyCount == 0)
    {
        m_instanceCount = 0;
        return;
    }

    const FullBodySoAGpu b = bodyView();
    WriteInstancesArgsGpu args{};
    args.b = b;
    args.out = instances.address();
    args.maxInstances = m_shapeCount;
    args.s = shapeView();
    args.ov = params.overlay;

    // Overlay inputs that live in this VkWorld's own buffers -- the demo cannot pass
    // them, the same way the instance writer gathers them solver-internally on the reference: graphColor
    // is the constraint graph's per-body Jones-Plassmann color, and the contact degree is
    // the sleep logic's touch fingerprint, only maintained while the rest test is running.
    // render.slang null-guards every one, so a mode whose input is absent falls through
    // to plain instance colour.
    const int mode = params.overlay.mode;
    args.ov.colorCount = constraints.colorCount();
    args.ov.graphColor = mode == OVERLAY_GRAPH_COLOR ? constraints.bodyColor().address() : 0;
    args.ov.degree = mode == OVERLAY_DEGREE && params.overlay.sleepFrames > 0.0f ? touchCount.address() : 0;
    args.ov.load = 0;
    args.ov.push = args.ov.pull = args.ov.stress = args.ov.cone = args.ov.penetration = args.ov.jointDegree = 0;

    const VkDeviceSize perBody = (VkDeviceSize)m_bodyCount * sizeof(float);
    auto ensure = [perBody](Buffer &buf)
    {
        if (buf.size() < perBody)
            buf.resize(perBody);
        return buf.address();
    };

    CommandList cmd;
    HostRingBuffer ring;
    cmd.init(dev);
    ring.init(dev, 1 << 16);
    cmd.begin();

    const int slots = narrow.manifoldSlots();
    if (mode == OVERLAY_LOAD)
    {
        OverlayLoadClearArgsGpu lc{};
        lc.load = ensure(overlayLoad);
        lc.n = m_bodyCount;
        launchOverlayLoadClear(cmd, ring, renderPl, lc, m_bodyCount);
        cmd.barrier();
        if (slots > 0)
        {
            OverlayLoadAccumArgsGpu la{};
            la.b = b;
            la.m = manifoldView();
            la.slots = slots;
            la.load = lc.load;
            launchOverlayLoadAccum(cmd, ring, renderPl, la, slots);
            cmd.barrier();
        }
        args.ov.load = lc.load;
    }

    // Structural overlays: a reduction from the constraint arrays onto the bodies, run
    // only for the overlay selected and only over the buffers it reads (the instance writer).
    const bool wantPush = mode == OVERLAY_JOINT_LOAD || mode == OVERLAY_AXIAL_LOAD;
    const bool wantStress = mode == OVERLAY_JOINT_STRESS;
    const bool wantCone = mode == OVERLAY_FRICTION;
    const bool wantPen = mode == OVERLAY_PENETRATION;
    const bool wantJointDeg = mode == OVERLAY_JOINT_DEGREE;
    if (wantPush || wantStress || wantCone || wantPen || wantJointDeg)
    {
        OverlayStatsGpu st{};
        if (wantPush)
        {
            st.push = ensure(overlayPush);
            st.pull = ensure(overlayPull);
        }
        if (wantStress)
            st.stress = ensure(overlayStress);
        if (wantCone)
            st.cone = ensure(overlayCone);
        if (wantPen)
            st.penetration = ensure(overlayPen);
        if (wantJointDeg)
            st.jointDegree = ensure(overlayJointDeg);

        OverlayStatsClearArgsGpu sc{};
        sc.s = st;
        sc.n = m_bodyCount;
        launchOverlayStatsClear(cmd, ring, renderPl, sc, m_bodyCount);
        cmd.barrier();

        // Contacts feed push, cone and penetration; joints feed push, pull, stress and
        // degree. Each launch is skipped when nothing it produces was asked for.
        if (slots > 0 && (st.push || st.cone || st.penetration))
        {
            OverlayStatsContactsArgsGpu ca{};
            ca.m = manifoldView();
            ca.slots = slots;
            ca.s = st;
            launchOverlayStatsContacts(cmd, ring, renderPl, ca, slots);
            cmd.barrier();
        }
        // Only authored joints: the drag slots are a UI affordance, and counting them
        // would put a permanent +1 on whatever the mouse is holding.
        const int forceCount = m_jointCount + m_springCount;
        if (forceCount > 0 && (st.push || st.pull || st.stress || st.jointDegree))
        {
            OverlayStatsForcesArgsGpu fa{};
            fa.b = b;
            fa.j = jointView();
            fa.sp = springView();
            fa.jointsAuthored = m_jointCount;
            fa.s = st;
            launchOverlayStatsForces(cmd, ring, renderPl, fa, forceCount);
            cmd.barrier();
        }
        args.ov.push = st.push;
        args.ov.pull = st.pull;
        args.ov.stress = st.stress;
        args.ov.cone = st.cone;
        args.ov.penetration = st.penetration;
        args.ov.jointDegree = st.jointDegree;
    }

    if (params.frustum)
    {
        frustumBuf.upload(params.frustum, 24 * sizeof(float));
        args.frustum = frustumBuf.address();
        visibleCounterBuf.zero(sizeof(int32_t));
        args.visible = visibleCounterBuf.address();
    }

    const int groups = (int)m_groupBase.size();
    if (params.segmentByShape)
    {
        shapeVisBuf.zero(groups * sizeof(int32_t));
        args.groupOf = groupIdx.address();
        args.shapeBase = shapeBaseBuf.address();
        args.shapeVisible = shapeVisBuf.address();
    }

    launchWriteInstances(cmd, ring, renderPl, args, m_shapeCount);
    cmd.submit();
    cmd.wait();

    if (params.segmentByShape)
    {
        shapeVisBuf.readback(m_groupVisible.data(), groups * sizeof(int32_t));
        int total = 0;
        for (int k = 0; k < groups; k++)
            total += m_groupVisible[k];
        m_instanceCount = std::min(total, m_shapeCount);
    }
    else if (params.frustum)
    {
        int32_t vis = 0;
        visibleCounterBuf.readback(&vis, sizeof(int32_t));
        m_instanceCount = std::min((int)vis, m_shapeCount);
    }
    else
    {
        m_instanceCount = m_shapeCount;
    }

    cmd.destroy();
    ring.destroy();
}


void VkWorld::writeDebugInstances(const DebugDrawViewGpu &view, int maxMarkers)
{
    assertIdle("VkWorld::writeDebugInstances");
    m_debugInstanceCount = 0;
    m_debugRequested = 0;
    const bool any = view.contactPoints || view.contactNormals || view.manifoldLinks;
    const int slots = narrow.manifoldSlots();
    if (!any || slots <= 0 || maxMarkers <= 0)
        return;

    // Debug instance budget: what the enabled markers could want -- a point
    // and a normal per contact, a link per manifold -- capped by maxMarkers. Sizing to the
    // cap alone would grow the buffer to 120k instances for a five-block tower.
    long long want = 0;
    const long long contacts = narrow.contactTotal();
    if (view.contactPoints)
        want += contacts;
    if (view.contactNormals)
        want += contacts;
    if (view.manifoldLinks)
        want += slots;
    const int limit = (int)std::min<long long>(want, maxMarkers);
    if (limit <= 0)
        return;
    VkDeviceSize need = (VkDeviceSize)limit * sizeof(BodyInstanceGpu);
    if (debugInstances.size() < need)
        debugInstances.resize(need, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    WriteDebugInstancesArgsGpu args{};
    args.b = bodyView();
    args.m = manifoldView();
    args.slots = slots;
    args.out = debugInstances.address();
    args.maxInstances = limit;
    debugCounterBuf.zero(sizeof(int32_t));
    args.counter = debugCounterBuf.address();
    args.d = view;

    CommandList cmd;
    HostRingBuffer ring;
    cmd.init(dev);
    ring.init(dev, 1 << 16);
    cmd.begin();
    launchWriteDebugInstances(cmd, ring, renderPl, args, slots);
    cmd.submit();
    cmd.wait();

    // The counter overshoots `limit` by however many markers did not fit, which is
    // exactly what the UI needs in order to say the view is incomplete.
    int32_t requested = 0;
    debugCounterBuf.readback(&requested, sizeof(int32_t));
    m_debugRequested = requested;
    m_debugInstanceCount = std::min((int)requested, limit);
    cmd.destroy();
    ring.destroy();
}



} // namespace avbdvk
