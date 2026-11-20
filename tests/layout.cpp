/*
 * End-to-end layout safety gate: src/shaders/layout_probe.slang, dispatched on the
 * GPU, reading a sentinel-filled instance of each host struct the Vulkan backend really
 * uploads (src/world/vk_args.h: FullBodySoAGpu, FullManifoldSoAGpu,
 * FullJointSoAGpu, FullSpringSoAGpu and every *ArgsGpu, plus SolverParamsGpu packed from
 * the real SolverParams) through src/shaders/types.slang's struct declarations, and
 * writing every field back in declaration order.
 *
 * Each struct is filled with a distinct sentinel per field, its RAW BYTES are uploaded
 * unchanged (no repacking on this side), and a Slang kernel compiled from types.slang
 * reads it back field by field. If a declaration has a missing pad, a reordered field or
 * a wrong pointee width on either side, the compiler-computed offset is wrong and a
 * sentinel lands in the wrong slot -- caught here by name. A hand-written offsetof check
 * could not do this: only the Slang compiler knows the offsets it computed.
 *
 * Plain C++, no reference solver. Until M2 this file probed the reference SoA structs (gpu_types.cuh /
 * gpu_forces.cuh) and so had to compile as device code; those structs are not what the Vulkan
 * backend uploads, and they go away at M7, so the four SoA probes now fill the vk_args.h
 * mirrors instead. The reference<->Vulkan field-order tie is no longer asserted here; once
 * the reference is gone the vk_args.h <-> Slang contract is the only one that matters.
 */

#include "collision.h"
#include "solver_params_pack.h"

#include "vk_args.h"
#include "vk_buffer.h"
#include "vk_cmd.h"
#include "vk_device.h"
#include "vk_pipeline.h"

#include "layout_probe_spirv.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace avbdvk;

namespace
{

int g_fail = 0;

// Pointer sentinel for field index i: a recognizable, never-null, 8-byte-aligned
// address pattern that is also distinct per field so a swap between two pointer fields
// is caught (not just "some pointer" vs "some other pointer").
uint64_t ptrSentinel(int i)
{
    return 0xA000000000000000ull | ((uint64_t)i << 8);
}

struct FieldSpec
{
    const char *name;
    bool isPointer; // true: compare all 64 bits; false: compare low 32 bits only
    uint64_t expected;
};

// Dispatches one probe kernel against `srcBytes` (a raw byte image of the real struct,
// sentinel-filled) and compares the readback against `fields` in order. Returns the
// number of mismatches (0 = pass); prints each mismatch by field name.
int runProbe(Device &dev, PipelineLayout &layout, const char *entryPoint, const void *srcBytes, size_t srcSize,
             const std::vector<FieldSpec> &fields)
{
    Pipeline p;
    p.init(dev, layout, layout_probe_spirv, layout_probe_spirv_words, entryPoint, entryPoint);

    HostRingBuffer ring;
    ring.init(dev, 1 << 14);

    Buffer bufSrc, bufOut;
    bufSrc.init(dev);
    bufOut.init(dev);
    bufSrc.resize(srcSize);
    bufOut.resize(fields.size() * sizeof(uint64_t));
    bufSrc.upload(srcBytes, srcSize);
    bufOut.zero();

    // Mirrors layout_probe.slang's ProbeArgs: the shared PipelineLayout only carries one
    // 8-byte push constant (uint64 argsAddr), so the two addresses this probe needs go
    // through a small ring-buffer struct instead, exactly like CollideArgs in
    // test_vk_collide.cpp.
    struct ProbeArgs
    {
        uint64_t srcAddr;
        uint64_t outAddr;
    } probeArgs{bufSrc.address(), bufOut.address()};

    CommandList cmd;
    cmd.init(dev, 4);
    VkDeviceAddress argsAddr = ring.write(&probeArgs, sizeof(probeArgs));
    cmd.begin();
    cmd.dispatch(p, 1, argsAddr);
    cmd.submit();
    cmd.wait();

    std::vector<uint64_t> out(fields.size());
    bufOut.readback(out.data(), out.size() * sizeof(uint64_t));

    int fails = 0;
    for (size_t i = 0; i < fields.size(); i++)
    {
        uint64_t got = fields[i].isPointer ? out[i] : (out[i] & 0xFFFFFFFFull);
        if (got != fields[i].expected)
        {
            printf("  MISMATCH %-24s field %-16s expected=0x%016llx got=0x%016llx\n", entryPoint, fields[i].name,
                   (unsigned long long)fields[i].expected, (unsigned long long)got);
            fails++;
        }
    }

    bufSrc.destroy();
    bufOut.destroy();
    cmd.destroy();
    ring.destroy();
    p.destroy();

    g_fail += fails;
    return fails;
}

} // namespace

int main()
{
    printf("=== avbd_test_vk_layout: GPU end-to-end probe of types.slang vs vk_args.h structs ===\n");

    Device dev;
    dev.init();
    PipelineLayout layout;
    layout.init(dev);

    // ---- FullBodySoAGpu: 18 pointers + count + _pad0, 20 fields ----
    {
        FullBodySoAGpu b{};
        b.pos = ptrSentinel(0);
        b.rot = ptrSentinel(1);
        b.initialLin = ptrSentinel(2);
        b.initialAng = ptrSentinel(3);
        b.inertialLin = ptrSentinel(4);
        b.inertialAng = ptrSentinel(5);
        b.velLin = ptrSentinel(6);
        b.velAng = ptrSentinel(7);
        b.prevVelLin = ptrSentinel(8);
        b.halfExtents = ptrSentinel(9);
        b.shapeType = ptrSentinel(10);
        b.moment = ptrSentinel(11);
        b.mass = ptrSentinel(12);
        b.friction = ptrSentinel(13);
        b.awake = ptrSentinel(14);
        b.sleepTimer = ptrSentinel(15);
        b.restPos = ptrSentinel(16);
        b.restRot = ptrSentinel(17);
        b.gyration = ptrSentinel(18);
        b.props = ptrSentinel(19);
        b.count = 0x1234u;
        std::vector<FieldSpec> f = {
            {"pos", true, ptrSentinel(0)},
            {"rot", true, ptrSentinel(1)},
            {"initialLin", true, ptrSentinel(2)},
            {"initialAng", true, ptrSentinel(3)},
            {"inertialLin", true, ptrSentinel(4)},
            {"inertialAng", true, ptrSentinel(5)},
            {"velLin", true, ptrSentinel(6)},
            {"velAng", true, ptrSentinel(7)},
            {"prevVelLin", true, ptrSentinel(8)},
            {"halfExtents", true, ptrSentinel(9)},
            {"shapeType", true, ptrSentinel(10)},
            {"moment", true, ptrSentinel(11)},
            {"mass", true, ptrSentinel(12)},
            {"friction", true, ptrSentinel(13)},
            {"awake", true, ptrSentinel(14)},
            {"sleepTimer", true, ptrSentinel(15)},
            {"restPos", true, ptrSentinel(16)},
            {"restRot", true, ptrSentinel(17)},
            {"gyration", true, ptrSentinel(18)},
            {"props", true, ptrSentinel(19)},
            {"count", false, 0x1234u},
        };
        int fails = runProbe(dev, layout, "csProbeBody", &b, sizeof(b), f);
        printf("FullBodySoAGpu : %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails, sizeof(b));
    }

    // ---- FullManifoldSoAGpu: 18 fields (no pad) ----
    {
        FullManifoldSoAGpu m{};
        m.key = ptrSentinel(0);
        m.bodyA = ptrSentinel(1);
        m.bodyB = ptrSentinel(2);
        m.basis = ptrSentinel(3);
        m.friction = ptrSentinel(4);
        m.contactOffset = ptrSentinel(5);
        m.contactCount = ptrSentinel(6);
        m.count = 0x1111;
        m.capacity = 0x2222;
        m.feature = ptrSentinel(7);
        m.rA = ptrSentinel(8);
        m.rB = ptrSentinel(9);
        m.C0 = ptrSentinel(10);
        m.penalty = ptrSentinel(11);
        m.lambda = ptrSentinel(12);
        m.stick = ptrSentinel(13);
        m.contactTotal = 0x3333;
        m.contactCapacity = 0x4444;
        m.shapeA = ptrSentinel(14);
        m.shapeB = ptrSentinel(15);
        std::vector<FieldSpec> f = {
            {"key", true, ptrSentinel(0)},          {"bodyA", true, ptrSentinel(1)},
            {"bodyB", true, ptrSentinel(2)},        {"basis", true, ptrSentinel(3)},
            {"friction", true, ptrSentinel(4)},     {"contactOffset", true, ptrSentinel(5)},
            {"contactCount", true, ptrSentinel(6)}, {"count", false, 0x1111u},
            {"capacity", false, 0x2222u},           {"feature", true, ptrSentinel(7)},
            {"rA", true, ptrSentinel(8)},           {"rB", true, ptrSentinel(9)},
            {"C0", true, ptrSentinel(10)},          {"penalty", true, ptrSentinel(11)},
            {"lambda", true, ptrSentinel(12)},      {"stick", true, ptrSentinel(13)},
            {"contactTotal", false, 0x3333u},       {"contactCapacity", false, 0x4444u},
            {"shapeA", true, ptrSentinel(14)},      {"shapeB", true, ptrSentinel(15)},
        };
        int fails = runProbe(dev, layout, "csProbeManifold", &m, sizeof(m), f);
        printf("FullManifoldSoAGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(m));
    }

    // ---- ShapeSoAGpu: 11 pointers + count + _pad0 ----
    {
        ShapeSoAGpu sh{};
        sh.body = ptrSentinel(0);
        sh.localPos = ptrSentinel(1);
        sh.localRot = ptrSentinel(2);
        sh.halfExtents = ptrSentinel(3);
        sh.shapeType = ptrSentinel(4);
        sh.friction = ptrSentinel(5);
        sh.category = ptrSentinel(6);
        sh.mask = ptrSentinel(7);
        sh.group = ptrSentinel(8);
        sh.flags = ptrSentinel(9);
        sh.geomOffset = ptrSentinel(10);
        sh.count = 0x6001u;
        sh._pad0 = 0;
        std::vector<FieldSpec> f = {
            {"body", true, ptrSentinel(0)},
            {"localPos", true, ptrSentinel(1)},
            {"localRot", true, ptrSentinel(2)},
            {"halfExtents", true, ptrSentinel(3)},
            {"shapeType", true, ptrSentinel(4)},
            {"friction", true, ptrSentinel(5)},
            {"category", true, ptrSentinel(6)},
            {"mask", true, ptrSentinel(7)},
            {"group", true, ptrSentinel(8)},
            {"flags", true, ptrSentinel(9)},
            {"geomOffset", true, ptrSentinel(10)},
            {"count", false, 0x6001u},
            {"_pad0", false, 0u},
        };
        int fails = runProbe(dev, layout, "csProbeShape", &sh, sizeof(sh), f);
        printf("ShapeSoAGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails, sizeof(sh));
    }

    // ---- FullJointSoAGpu: 22 pointers + count + _pad0, 24 fields ----
    {
        FullJointSoAGpu j{};
        j.bodyA = ptrSentinel(0);
        j.bodyB = ptrSentinel(1);
        j.rA = ptrSentinel(2);
        j.rB = ptrSentinel(3);
        j.C0Lin = ptrSentinel(4);
        j.C0Ang = ptrSentinel(5);
        j.penaltyLin = ptrSentinel(6);
        j.penaltyAng = ptrSentinel(7);
        j.lambdaLin = ptrSentinel(8);
        j.lambdaAng = ptrSentinel(9);
        j.stiffnessLin = ptrSentinel(10);
        j.stiffnessAng = ptrSentinel(11);
        j.fracture = ptrSentinel(12);
        j.torqueArm = ptrSentinel(13);
        j.restAng = ptrSentinel(14);
        j.broken = ptrSentinel(15);
        j.frameA = ptrSentinel(16);
        j.frameB = ptrSentinel(17);
        j.axisMode = ptrSentinel(18);
        j.axisLo = ptrSentinel(19);
        j.axisHi = ptrSentinel(20);
        j.generic = ptrSentinel(21);
        j.fractureAng = ptrSentinel(22);
        j.count = 0x5678;
        std::vector<FieldSpec> f = {
            {"bodyA", true, ptrSentinel(0)},
            {"bodyB", true, ptrSentinel(1)},
            {"rA", true, ptrSentinel(2)},
            {"rB", true, ptrSentinel(3)},
            {"C0Lin", true, ptrSentinel(4)},
            {"C0Ang", true, ptrSentinel(5)},
            {"penaltyLin", true, ptrSentinel(6)},
            {"penaltyAng", true, ptrSentinel(7)},
            {"lambdaLin", true, ptrSentinel(8)},
            {"lambdaAng", true, ptrSentinel(9)},
            {"stiffnessLin", true, ptrSentinel(10)},
            {"stiffnessAng", true, ptrSentinel(11)},
            {"fracture", true, ptrSentinel(12)},
            {"torqueArm", true, ptrSentinel(13)},
            {"restAng", true, ptrSentinel(14)},
            {"broken", true, ptrSentinel(15)},
            {"frameA", true, ptrSentinel(16)},
            {"frameB", true, ptrSentinel(17)},
            {"axisMode", true, ptrSentinel(18)},
            {"axisLo", true, ptrSentinel(19)},
            {"axisHi", true, ptrSentinel(20)},
            {"generic", true, ptrSentinel(21)},
            {"fractureAng", true, ptrSentinel(22)},
            {"count", false, 0x5678u},
        };
        int fails = runProbe(dev, layout, "csProbeJoint", &j, sizeof(j), f);
        printf("FullJointSoAGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails, sizeof(j));
    }

    // ---- FullSpringSoAGpu: 6 pointers + count, 7 fields (+1 implicit tail pad, not probed) ----
    {
        FullSpringSoAGpu s{};
        s.bodyA = ptrSentinel(0);
        s.bodyB = ptrSentinel(1);
        s.rA = ptrSentinel(2);
        s.rB = ptrSentinel(3);
        s.rest = ptrSentinel(4);
        s.stiffness = ptrSentinel(5);
        s.count = 0x9abc;
        std::vector<FieldSpec> f = {
            {"bodyA", true, ptrSentinel(0)}, {"bodyB", true, ptrSentinel(1)}, {"rA", true, ptrSentinel(2)},
            {"rB", true, ptrSentinel(3)},    {"rest", true, ptrSentinel(4)},  {"stiffness", true, ptrSentinel(5)},
            {"count", false, 0x9abcu},
        };
        int fails = runProbe(dev, layout, "csProbeSpring", &s, sizeof(s), f);
        printf("FullSpringSoAGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails, sizeof(s));
    }

    // ---- SolverParamsGpu: packed from the REAL SolverParams via
    // src/world/solver_params_pack.h, exercising that packer as well as
    // types.slang's declaration. ----
    {
        SolverParams p{};
        p.dt = 1.0f / 60.0f;
        p.gravity = vec3{0.0f, 0.0f, -10.0f};
        p.iterations = 8;
        p.primalLanes = 4;
        p.alpha = 0.97f;
        p.betaLin = 10000.0f;
        p.betaAng = 100.0f;
        p.gamma = 0.99f;
        p.sleepEnabled = true;
        p.sleepFrames = 12;
        p.sleepLinVel = 0.05f;
        p.sleepAngVel = 0.05f;
        p.sleepDisp = 0.01f;
        p.sleepFracture = 0.9f;
        p.autoKStart = true;
        p.dualDamping = 0.1f;
        p.dualDampVel = 0.5f;
        p.killFallenEnabled = false;
        p.killHeight = -50.0f;
        p.grabStrength = 1.0f;
        p.worldFrameInertia = true;
        avbdvk::SolverParamsGpu g = avbdvk::packSolverParams(p);

        auto fbits = [](float v)
        {
            uint32_t u;
            std::memcpy(&u, &v, 4);
            return (uint64_t)u;
        };
        std::vector<FieldSpec> f = {
            {"dt", false, fbits(g.dt)},
            {"worldFrameInertia", false, fbits(g.worldFrameInertia)},
            {"iterations", false, (uint64_t)(uint32_t)g.iterations},
            {"primalLanes", false, (uint64_t)(uint32_t)g.primalLanes},
            {"alpha", false, fbits(g.alpha)},
            {"betaLin", false, fbits(g.betaLin)},
            {"betaAng", false, fbits(g.betaAng)},
            {"gamma", false, fbits(g.gamma)},
            {"sleepEnabled", false, (uint64_t)g.sleepEnabled},
            {"sleepFrames", false, (uint64_t)(uint32_t)g.sleepFrames},
            {"sleepLinVel", false, fbits(g.sleepLinVel)},
            {"sleepAngVel", false, fbits(g.sleepAngVel)},
            {"sleepDisp", false, fbits(g.sleepDisp)},
            {"sleepFracture", false, fbits(g.sleepFracture)},
            {"autoKStart", false, (uint64_t)g.autoKStart},
            {"dualDamping", false, fbits(g.dualDamping)},
            {"dualDampVel", false, fbits(g.dualDampVel)},
            {"killFallenEnabled", false, (uint64_t)g.killFallenEnabled},
            {"killHeight", false, fbits(g.killHeight)},
            {"grabStrength", false, fbits(g.grabStrength)},
        };
        int fails = runProbe(dev, layout, "csProbeSolverParams", &g, sizeof(g), f);
        printf("SolverParamsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails, sizeof(g));
    }

    // ---- IntegrateArgs (integrate.slang's own BodySoA subset + 2-field SolverParams) ----
    {
        struct IntegrateBodySoAReal
        {
            vec3 *pos;
            quat *rot;
            vec3 *initialLin;
            quat *initialAng;
            vec3 *inertialLin;
            quat *inertialAng;
            vec3 *velLin;
            vec3 *velAng;
            vec3 *prevVelLin;
            float *mass;
            unsigned char *awake;
            uint32_t count;
            uint32_t _pad0;
        };
        struct IntegrateSolverParamsReal
        {
            float dt;
            float gravityX;
            float gravityY;
            float gravityZ;
            uint32_t spinExact;
            uint32_t _padP;
        };
        struct IntegrateArgsReal
        {
            IntegrateBodySoAReal b;
            IntegrateSolverParamsReal p;
        };

        IntegrateArgsReal a{};
        a.b.pos = (vec3 *)ptrSentinel(0);
        a.b.rot = (quat *)ptrSentinel(1);
        a.b.initialLin = (vec3 *)ptrSentinel(2);
        a.b.initialAng = (quat *)ptrSentinel(3);
        a.b.inertialLin = (vec3 *)ptrSentinel(4);
        a.b.inertialAng = (quat *)ptrSentinel(5);
        a.b.velLin = (vec3 *)ptrSentinel(6);
        a.b.velAng = (vec3 *)ptrSentinel(7);
        a.b.prevVelLin = (vec3 *)ptrSentinel(8);
        a.b.mass = (float *)ptrSentinel(9);
        a.b.awake = (unsigned char *)ptrSentinel(10);
        a.b.count = 0xdef0u;
        a.b._pad0 = 0;
        a.p.dt = 1.0f / 60.0f;
        a.p.gravityX = 0.0f;
        a.p.gravityY = 0.0f;
        a.p.gravityZ = -10.0f;
        a.p.spinExact = 1;
        a.p._padP = 0;

        auto fbits = [](float v)
        {
            uint32_t u;
            std::memcpy(&u, &v, 4);
            return (uint64_t)u;
        };
        std::vector<FieldSpec> f = {
            {"pos", true, ptrSentinel(0)},
            {"rot", true, ptrSentinel(1)},
            {"initialLin", true, ptrSentinel(2)},
            {"initialAng", true, ptrSentinel(3)},
            {"inertialLin", true, ptrSentinel(4)},
            {"inertialAng", true, ptrSentinel(5)},
            {"velLin", true, ptrSentinel(6)},
            {"velAng", true, ptrSentinel(7)},
            {"prevVelLin", true, ptrSentinel(8)},
            {"mass", true, ptrSentinel(9)},
            {"awake", true, ptrSentinel(10)},
            {"count", false, 0xdef0u},
            {"_pad0", false, 0u},
            {"dt", false, fbits(a.p.dt)},
            {"gravityX", false, fbits(a.p.gravityX)},
            {"gravityY", false, fbits(a.p.gravityY)},
            {"gravityZ", false, fbits(a.p.gravityZ)},
            {"spinExact", false, 0x1u},
            {"_padP", false, 0u},
        };
        int fails = runProbe(dev, layout, "csProbeIntegrateArgs", &a, sizeof(a), f);
        printf("IntegrateArgs  : %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails, sizeof(a));
    }

    // ---- VelocityArgs / KillFallenArgs / ExplosionGuardArgs (same IntegrateBodySoA) ----
    {
        struct IntegrateBodySoAReal
        {
            vec3 *pos;
            quat *rot;
            vec3 *initialLin;
            quat *initialAng;
            vec3 *inertialLin;
            quat *inertialAng;
            vec3 *velLin;
            vec3 *velAng;
            vec3 *prevVelLin;
            float *mass;
            unsigned char *awake;
            uint32_t count;
            uint32_t _pad0;
        };
        struct IntegrateSolverParamsReal
        {
            float dt;
            float gravityX;
            float gravityY;
            float gravityZ;
            uint32_t spinExact;
            uint32_t _padP;
        };
        auto fbits = [](float v)
        {
            uint32_t u;
            std::memcpy(&u, &v, 4);
            return (uint64_t)u;
        };

        auto makeBody = [&](IntegrateBodySoAReal &b)
        {
            b.pos = (vec3 *)ptrSentinel(0);
            b.rot = (quat *)ptrSentinel(1);
            b.initialLin = (vec3 *)ptrSentinel(2);
            b.initialAng = (quat *)ptrSentinel(3);
            b.inertialLin = (vec3 *)ptrSentinel(4);
            b.inertialAng = (quat *)ptrSentinel(5);
            b.velLin = (vec3 *)ptrSentinel(6);
            b.velAng = (vec3 *)ptrSentinel(7);
            b.prevVelLin = (vec3 *)ptrSentinel(8);
            b.mass = (float *)ptrSentinel(9);
            b.awake = (unsigned char *)ptrSentinel(10);
            b.count = 0xdef0u;
            b._pad0 = 0;
        };
        std::vector<FieldSpec> bodyFields = {
            {"pos", true, ptrSentinel(0)},
            {"rot", true, ptrSentinel(1)},
            {"initialLin", true, ptrSentinel(2)},
            {"initialAng", true, ptrSentinel(3)},
            {"inertialLin", true, ptrSentinel(4)},
            {"inertialAng", true, ptrSentinel(5)},
            {"velLin", true, ptrSentinel(6)},
            {"velAng", true, ptrSentinel(7)},
            {"prevVelLin", true, ptrSentinel(8)},
            {"mass", true, ptrSentinel(9)},
            {"awake", true, ptrSentinel(10)},
            {"count", false, 0xdef0u},
            {"_pad0", false, 0u},
        };

        {
            struct VelocityArgsReal
            {
                IntegrateBodySoAReal b;
                IntegrateSolverParamsReal p;
            };
            VelocityArgsReal a{};
            makeBody(a.b);
            a.p.dt = 1.0f / 60.0f;
            a.p.gravityX = 0.0f;
            a.p.gravityY = 0.0f;
            a.p.gravityZ = -10.0f;
            a.p.spinExact = 1;
            a.p._padP = 0;
            std::vector<FieldSpec> f = bodyFields;
            f.push_back({"dt", false, fbits(a.p.dt)});
            f.push_back({"gravityX", false, fbits(a.p.gravityX)});
            f.push_back({"gravityY", false, fbits(a.p.gravityY)});
            f.push_back({"gravityZ", false, fbits(a.p.gravityZ)});
            f.push_back({"spinExact", false, 0x1u});
            f.push_back({"_padP", false, 0u});
            int fails = runProbe(dev, layout, "csProbeVelocityArgs", &a, sizeof(a), f);
            printf("VelocityArgs   : %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
                   sizeof(a));
        }
        {
            struct KillFallenArgsReal
            {
                IntegrateBodySoAReal b;
                int *dyn;
                int dynCount;
                float killHeight, upX, upY, upZ, _padK;
                int *sleepTimer;
                vec3 *restPos;
                quat *restRot;
            };
            KillFallenArgsReal a{};
            makeBody(a.b);
            a.dyn = (int *)ptrSentinel(11);
            a.dynCount = 0x1234;
            a.killHeight = -50.0f;
            a.upX = 0.25f;
            a.upY = 0.5f;
            a.upZ = 0.75f;
            a._padK = 0.0f;
            a.sleepTimer = (int *)ptrSentinel(12);
            a.restPos = (vec3 *)ptrSentinel(13);
            a.restRot = (quat *)ptrSentinel(14);
            std::vector<FieldSpec> f = bodyFields;
            f.push_back({"dyn", true, ptrSentinel(11)});
            f.push_back({"dynCount", false, (uint64_t)(uint32_t)a.dynCount});
            f.push_back({"killHeight", false, fbits(a.killHeight)});
            f.push_back({"upX", false, fbits(a.upX)});
            f.push_back({"upY", false, fbits(a.upY)});
            f.push_back({"upZ", false, fbits(a.upZ)});
            f.push_back({"_padK", false, fbits(a._padK)});
            f.push_back({"sleepTimer", true, ptrSentinel(12)});
            f.push_back({"restPos", true, ptrSentinel(13)});
            f.push_back({"restRot", true, ptrSentinel(14)});
            int fails = runProbe(dev, layout, "csProbeKillFallenArgs", &a, sizeof(a), f);
            printf("KillFallenArgs : %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
                   sizeof(a));
        }
        {
            struct ExplosionGuardArgsReal
            {
                IntegrateBodySoAReal b;
                int *dyn;
                int dynCount;
                int *sleepTimer;
                vec3 *restPos;
                quat *restRot;
            };
            ExplosionGuardArgsReal a{};
            makeBody(a.b);
            a.dyn = (int *)ptrSentinel(11);
            a.dynCount = 0x1234;
            a.sleepTimer = (int *)ptrSentinel(12);
            a.restPos = (vec3 *)ptrSentinel(13);
            a.restRot = (quat *)ptrSentinel(14);
            std::vector<FieldSpec> f = bodyFields;
            f.push_back({"dyn", true, ptrSentinel(11)});
            f.push_back({"dynCount", false, (uint64_t)(uint32_t)a.dynCount});
            f.push_back({"sleepTimer", true, ptrSentinel(12)});
            f.push_back({"restPos", true, ptrSentinel(13)});
            f.push_back({"restRot", true, ptrSentinel(14)});
            int fails = runProbe(dev, layout, "csProbeExplosionGuardArgs", &a, sizeof(a), f);
            printf("ExplosionGuardArgs: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
                   sizeof(a));
        }
    }

    // =====================================================================
    // Args structs (src/world/vk_args.h) vs their Slang counterparts in
    // broadphase.slang, narrowphase.slang, scan.slang, compact.slang, radix.slang.
    // Closes the bug class that caused the narrowphase penalty divergence: these
    // structs are dispatched via HostRingBuffer, not imported from types.slang, so
    // nothing previously proved the Slang compiler's offsets matched vk_args.h at all.
    // =====================================================================
    using namespace avbdvk;

    // ---- broadphase.slang: AabbBinArgsGpu ----
    {
        AabbBinArgsGpu a{};
        a.b.pos = ptrSentinel(0);
        a.b.rot = ptrSentinel(1);
        a.b.halfExtents = ptrSentinel(2);
        a.b.shapeBody = ptrSentinel(900);
        a.b.localPos = ptrSentinel(901);
        a.b.localRot = ptrSentinel(902);
        a.b.flags = ptrSentinel(903);
        a.b.count = 0x1001u;
        a.b._pad0 = 0;
        a.idx = ptrSentinel(3);
        a.n = 0x1002u;
        a.invCellSize = 3.5f;
        a.aabbMin = ptrSentinel(4);
        a.aabbMax = ptrSentinel(5);
        a.cellId = ptrSentinel(6);
        a.hash = ptrSentinel(7);
        a.binBody = ptrSentinel(8);
        a.mask = 0x1003u;
        a._pad0 = 0;
        auto fbits = [](float v)
        {
            uint32_t u;
            std::memcpy(&u, &v, 4);
            return (uint64_t)u;
        };
        std::vector<FieldSpec> f = {
            {"b.pos", true, ptrSentinel(0)},
            {"b.rot", true, ptrSentinel(1)},
            {"b.shapeBody", true, ptrSentinel(900)},
            {"b.localPos", true, ptrSentinel(901)},
            {"b.localRot", true, ptrSentinel(902)},
            {"b.halfExtents", true, ptrSentinel(2)},
            {"b.flags", true, ptrSentinel(903)},
            {"b.count", false, 0x1001u},
            {"b._pad0", false, 0u},
            {"idx", true, ptrSentinel(3)},
            {"n", false, 0x1002u},
            {"invCellSize", false, fbits(3.5f)},
            {"aabbMin", true, ptrSentinel(4)},
            {"aabbMax", true, ptrSentinel(5)},
            {"cellId", true, ptrSentinel(6)},
            {"hash", true, ptrSentinel(7)},
            {"binBody", true, ptrSentinel(8)},
            {"mask", false, 0x1003u},
        };
        int fails = runProbe(dev, layout, "csProbeAabbBinArgs", &a, sizeof(a), f);
        printf("AabbBinArgsGpu : %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails, sizeof(a));
    }

    // ---- broadphase.slang: StaticBinArgsGpu ----
    {
        StaticBinArgsGpu a{};
        a.b.pos = ptrSentinel(0);
        a.b.rot = ptrSentinel(1);
        a.b.halfExtents = ptrSentinel(2);
        a.b.shapeBody = ptrSentinel(900);
        a.b.localPos = ptrSentinel(901);
        a.b.localRot = ptrSentinel(902);
        a.b.flags = ptrSentinel(903);
        a.b.count = 0x1001u;
        a.b._pad0 = 0;
        a.idx = ptrSentinel(3);
        a.n = 0x1002u;
        a.invCellSize = 3.5f;
        a.aabbMin = ptrSentinel(4);
        a.aabbMax = ptrSentinel(5);
        a.cellId = ptrSentinel(6);
        a.hash = ptrSentinel(7);
        a.binBody = ptrSentinel(8);
        a.mask = 0x1003u;
        a.cellSize = 1.75f;
        auto fbits = [](float v)
        {
            uint32_t u;
            std::memcpy(&u, &v, 4);
            return (uint64_t)u;
        };
        std::vector<FieldSpec> f = {
            {"b.pos", true, ptrSentinel(0)},
            {"b.rot", true, ptrSentinel(1)},
            {"b.shapeBody", true, ptrSentinel(900)},
            {"b.localPos", true, ptrSentinel(901)},
            {"b.localRot", true, ptrSentinel(902)},
            {"b.halfExtents", true, ptrSentinel(2)},
            {"b.flags", true, ptrSentinel(903)},
            {"b.count", false, 0x1001u},
            {"b._pad0", false, 0u},
            {"idx", true, ptrSentinel(3)},
            {"n", false, 0x1002u},
            {"invCellSize", false, fbits(3.5f)},
            {"aabbMin", true, ptrSentinel(4)},
            {"aabbMax", true, ptrSentinel(5)},
            {"cellId", true, ptrSentinel(6)},
            {"hash", true, ptrSentinel(7)},
            {"binBody", true, ptrSentinel(8)},
            {"mask", false, 0x1003u},
            {"cellSize", false, fbits(1.75f)},
        };
        int fails = runProbe(dev, layout, "csProbeStaticBinArgs", &a, sizeof(a), f);
        printf("StaticBinArgsGpu : %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- broadphase.slang: AabbStaticArgsGpu ----
    {
        AabbStaticArgsGpu a{};
        a.b.pos = ptrSentinel(0);
        a.b.rot = ptrSentinel(1);
        a.b.halfExtents = ptrSentinel(2);
        a.b.shapeBody = ptrSentinel(900);
        a.b.localPos = ptrSentinel(901);
        a.b.localRot = ptrSentinel(902);
        a.b.flags = ptrSentinel(903);
        a.b.count = 0x1101u;
        a.b._pad0 = 0;
        a.idx = ptrSentinel(3);
        a.n = 0x1102u;
        a._pad0 = 0;
        a.aabbMin = ptrSentinel(4);
        a.aabbMax = ptrSentinel(5);
        std::vector<FieldSpec> f = {
            {"b.pos", true, ptrSentinel(0)},
            {"b.rot", true, ptrSentinel(1)},
            {"b.shapeBody", true, ptrSentinel(900)},
            {"b.localPos", true, ptrSentinel(901)},
            {"b.localRot", true, ptrSentinel(902)},
            {"b.halfExtents", true, ptrSentinel(2)},
            {"b.flags", true, ptrSentinel(903)},
            {"b.count", false, 0x1101u},
            {"b._pad0", false, 0u},
            {"idx", true, ptrSentinel(3)},
            {"n", false, 0x1102u},
            {"aabbMin", true, ptrSentinel(4)},
            {"aabbMax", true, ptrSentinel(5)},
        };
        int fails = runProbe(dev, layout, "csProbeAabbStaticArgs", &a, sizeof(a), f);
        printf("AabbStaticArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- broadphase.slang: ResetRangesArgsGpu ----
    {
        ResetRangesArgsGpu a{};
        a.cellStart = ptrSentinel(0);
        a.cellEnd = ptrSentinel(1);
        a.tableSize = 0x1201u;
        a._pad0 = 0;
        std::vector<FieldSpec> f = {
            {"cellStart", true, ptrSentinel(0)},
            {"cellEnd", true, ptrSentinel(1)},
            {"tableSize", false, 0x1201u},
        };
        int fails = runProbe(dev, layout, "csProbeResetRangesArgs", &a, sizeof(a), f);
        printf("ResetRangesArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- broadphase.slang: CellRangesArgsGpu ----
    {
        CellRangesArgsGpu a{};
        a.hashSorted = ptrSentinel(0);
        a.n = 0x1301u;
        a._pad0 = 0;
        a.cellStart = ptrSentinel(1);
        a.cellEnd = ptrSentinel(2);
        std::vector<FieldSpec> f = {
            {"hashSorted", true, ptrSentinel(0)},
            {"n", false, 0x1301u},
            {"cellStart", true, ptrSentinel(1)},
            {"cellEnd", true, ptrSentinel(2)},
        };
        int fails = runProbe(dev, layout, "csProbeCellRangesArgs", &a, sizeof(a), f);
        printf("CellRangesArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- broadphase.slang: EmitPairsArgsGpu ----
    {
        EmitPairsArgsGpu a{};
        a.binBodySorted = ptrSentinel(0);
        a.n = 0x1401u;
        a._pad0 = 0;
        a.cellId = ptrSentinel(1);
        a.aabbMin = ptrSentinel(2);
        a.aabbMax = ptrSentinel(3);
        a.cellStart = ptrSentinel(4);
        a.cellEnd = ptrSentinel(5);
        a.invCellSize = 7.25f;
        a.mask = 0x1402u;
        a.pairs = ptrSentinel(6);
        a.counter = ptrSentinel(7);
        a.capacity = 0x1403;
        a._pad1 = 0;
        a.ignore = ptrSentinel(8);
        a.ignoreCount = 0x1404;
        a._pad2 = 0;
        a.shapeBody = ptrSentinel(12);
        a.sBinBodySorted = ptrSentinel(9);
        a.sCellStart = ptrSentinel(10);
        a.sCellEnd = ptrSentinel(11);
        a.sMask = 0x1405u;
        a._pad3 = 0;
        auto fbits = [](float v)
        {
            uint32_t u;
            std::memcpy(&u, &v, 4);
            return (uint64_t)u;
        };
        std::vector<FieldSpec> f = {
            {"binBodySorted", true, ptrSentinel(0)},
            {"n", false, 0x1401u},
            {"cellId", true, ptrSentinel(1)},
            {"aabbMin", true, ptrSentinel(2)},
            {"aabbMax", true, ptrSentinel(3)},
            {"cellStart", true, ptrSentinel(4)},
            {"cellEnd", true, ptrSentinel(5)},
            {"invCellSize", false, fbits(7.25f)},
            {"mask", false, 0x1402u},
            {"pairs", true, ptrSentinel(6)},
            {"counter", true, ptrSentinel(7)},
            {"capacity", false, (uint64_t)0x1403u},
            {"ignore", true, ptrSentinel(8)},
            {"ignoreCount", false, (uint64_t)0x1404u},
            {"shapeBody", true, ptrSentinel(12)},
            {"sBinBodySorted", true, ptrSentinel(9)},
            {"sCellStart", true, ptrSentinel(10)},
            {"sCellEnd", true, ptrSentinel(11)},
            {"sMask", false, (uint64_t)0x1405u},
        };
        int fails = runProbe(dev, layout, "csProbeEmitPairsArgs", &a, sizeof(a), f);
        printf("EmitPairsArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails, sizeof(a));
    }

    // ---- broadphase.slang: EmitCrossPairsArgsGpu ----
    {
        EmitCrossPairsArgsGpu a{};
        a.aIdx = ptrSentinel(0);
        a.nA = 0x1501u;
        a._pad0 = 0;
        a.bIdx = ptrSentinel(1);
        a.nB = 0x1502u;
        a._pad1 = 0;
        a.bStart = ptrSentinel(2);
        a.bEnd = ptrSentinel(3);
        a.aabbMin = ptrSentinel(4);
        a.aabbMax = ptrSentinel(5);
        a.pairs = ptrSentinel(6);
        a.counter = ptrSentinel(7);
        a.capacity = 0x1503;
        a._pad2 = 0;
        a.ignore = ptrSentinel(8);
        a.ignoreCount = 0x1504;
        a._pad3 = 0;
        a.shapeBody = ptrSentinel(9);
        std::vector<FieldSpec> f = {
            {"aIdx", true, ptrSentinel(0)},
            {"nA", false, 0x1501u},
            {"bIdx", true, ptrSentinel(1)},
            {"nB", false, 0x1502u},
            {"bStart", true, ptrSentinel(2)},
            {"bEnd", true, ptrSentinel(3)},
            {"aabbMin", true, ptrSentinel(4)},
            {"aabbMax", true, ptrSentinel(5)},
            {"pairs", true, ptrSentinel(6)},
            {"counter", true, ptrSentinel(7)},
            {"capacity", false, (uint64_t)0x1503u},
            {"ignore", true, ptrSentinel(8)},
            {"ignoreCount", false, (uint64_t)0x1504u},
            {"shapeBody", true, ptrSentinel(9)},
        };
        int fails = runProbe(dev, layout, "csProbeEmitCrossPairsArgs", &a, sizeof(a), f);
        printf("EmitCrossPairsArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- narrowphase.slang: NarrowphaseArgsGpu (the struct behind the earlier bug) ----
    {
        NarrowphaseArgsGpu a{};
        a.b.pos = ptrSentinel(0);
        a.b.rot = ptrSentinel(1);
        a.b.halfExtents = ptrSentinel(2);
        a.b.shapeBody = ptrSentinel(900);
        a.b.localPos = ptrSentinel(901);
        a.b.localRot = ptrSentinel(902);
        a.b.flags = ptrSentinel(903);
        a.b.shapeType = ptrSentinel(3);
        a.b.friction = ptrSentinel(4);
        a.b.mass = ptrSentinel(5);
        a.b.moment = ptrSentinel(6);
        a.b.velLin = ptrSentinel(7);
        a.b.velAng = ptrSentinel(8);
        a.b.props = ptrSentinel(904);
        a.b.category = ptrSentinel(905);
        a.b.mask = ptrSentinel(906);
        a.b.group = ptrSentinel(907);
        a.b.count = 0x2001u;
        a.b._pad0 = 0;
        a.pairs = ptrSentinel(9);
        a.pairCount = 0x2002u;
        a._pad0 = 0;

        auto fillManifold = [](NpManifoldSoAGpu &m, int base)
        {
            m.key = ptrSentinel(base + 0);
            m.bodyA = ptrSentinel(base + 1);
            m.bodyB = ptrSentinel(base + 2);
            m.basis = ptrSentinel(base + 3);
            m.friction = ptrSentinel(base + 4);
            m.contactOffset = ptrSentinel(base + 5);
            m.contactCount = ptrSentinel(base + 6);
            m.feature = ptrSentinel(base + 7);
            m.rA = ptrSentinel(base + 8);
            m.rB = ptrSentinel(base + 9);
            m.C0 = ptrSentinel(base + 10);
            m.penalty = ptrSentinel(base + 11);
            m.lambda = ptrSentinel(base + 12);
            m.stick = ptrSentinel(base + 13);
            m.shapeA = ptrSentinel(base + 14);
            m.shapeB = ptrSentinel(base + 15);
        };
        fillManifold(a.cur, 10);
        fillManifold(a.prev, 30);

        a.prevKeys = ptrSentinel(50);
        a.prevCount = 0x2003u;
        a._pad1 = 0;
        a.contactCounter = ptrSentinel(51);
        a.activeCounter = ptrSentinel(52);
        a.contactCapacity = 0x2004;
        a.p.dt = 1.0f / 60.0f;
        a.p.alpha = 0.97f;
        a.p.gamma = 0.99f;
        a.p.autoKStart = 1u;
        a.p.dualDamping = 0.1f;
        a.p.dualDampVel = 0.5f;
        a.livePairs = ptrSentinel(53);
        a.contactCap = 0x2005;
        a.hullList = ptrSentinel(55);
        a.hullCounter = ptrSentinel(56);
        a.meshList = ptrSentinel(57);
        a.meshCounter = ptrSentinel(58);

        auto fbits = [](float v)
        {
            uint32_t u;
            std::memcpy(&u, &v, 4);
            return (uint64_t)u;
        };
        std::vector<FieldSpec> f = {
            {"b.pos", true, ptrSentinel(0)},
            {"b.rot", true, ptrSentinel(1)},
            {"b.shapeBody", true, ptrSentinel(900)},
            {"b.localPos", true, ptrSentinel(901)},
            {"b.localRot", true, ptrSentinel(902)},
            {"b.halfExtents", true, ptrSentinel(2)},
            {"b.shapeType", true, ptrSentinel(3)},
            {"b.friction", true, ptrSentinel(4)},
            {"b.flags", true, ptrSentinel(903)},
            {"b.mass", true, ptrSentinel(5)},
            {"b.moment", true, ptrSentinel(6)},
            {"b.velLin", true, ptrSentinel(7)},
            {"b.velAng", true, ptrSentinel(8)},
            {"b.props", true, ptrSentinel(904)},
            {"b.category", true, ptrSentinel(905)},
            {"b.mask", true, ptrSentinel(906)},
            {"b.group", true, ptrSentinel(907)},
            {"b.count", false, 0x2001u},
            {"pairs", true, ptrSentinel(9)},
            {"pairCount", false, 0x2002u},
            {"cur.key", true, ptrSentinel(10)},
            {"cur.bodyA", true, ptrSentinel(11)},
            {"cur.bodyB", true, ptrSentinel(12)},
            {"cur.basis", true, ptrSentinel(13)},
            {"cur.friction", true, ptrSentinel(14)},
            {"cur.contactOffset", true, ptrSentinel(15)},
            {"cur.contactCount", true, ptrSentinel(16)},
            {"cur.feature", true, ptrSentinel(17)},
            {"cur.rA", true, ptrSentinel(18)},
            {"cur.rB", true, ptrSentinel(19)},
            {"cur.C0", true, ptrSentinel(20)},
            {"cur.penalty", true, ptrSentinel(21)},
            {"cur.lambda", true, ptrSentinel(22)},
            {"cur.stick", true, ptrSentinel(23)},
            {"cur.shapeA", true, ptrSentinel(24)},
            {"cur.shapeB", true, ptrSentinel(25)},
            {"prev.key", true, ptrSentinel(30)},
            {"prev.bodyA", true, ptrSentinel(31)},
            {"prev.bodyB", true, ptrSentinel(32)},
            {"prev.basis", true, ptrSentinel(33)},
            {"prev.friction", true, ptrSentinel(34)},
            {"prev.contactOffset", true, ptrSentinel(35)},
            {"prev.contactCount", true, ptrSentinel(36)},
            {"prev.feature", true, ptrSentinel(37)},
            {"prev.rA", true, ptrSentinel(38)},
            {"prev.rB", true, ptrSentinel(39)},
            {"prev.C0", true, ptrSentinel(40)},
            {"prev.penalty", true, ptrSentinel(41)},
            {"prev.lambda", true, ptrSentinel(42)},
            {"prev.stick", true, ptrSentinel(43)},
            {"prev.shapeA", true, ptrSentinel(44)},
            {"prev.shapeB", true, ptrSentinel(45)},
            {"prevKeys", true, ptrSentinel(50)},
            {"prevCount", false, 0x2003u},
            {"contactCounter", true, ptrSentinel(51)},
            {"activeCounter", true, ptrSentinel(52)},
            {"contactCapacity", false, (uint64_t)0x2004u},
            {"p.dt", false, fbits(a.p.dt)},
            {"p.alpha", false, fbits(0.97f)},
            {"p.gamma", false, fbits(0.99f)},
            {"p.autoKStart", false, 1u},
            {"p.dualDamping", false, fbits(0.1f)},
            {"p.dualDampVel", false, fbits(0.5f)},
            {"livePairs", true, ptrSentinel(53)},
            {"contactCap", false, (uint64_t)0x2005u},
            {"hullList", true, ptrSentinel(55)},
            {"hullCounter", true, ptrSentinel(56)},
            {"meshList", true, ptrSentinel(57)},
            {"meshCounter", true, ptrSentinel(58)},
        };
        int fails = runProbe(dev, layout, "csProbeNarrowphaseArgs", &a, sizeof(a), f);
        printf("NarrowphaseArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- constraints.slang: Cs*ArgsGpu structs (src/world/vk_args.h) ----

    // ---- CsResetColorsArgsGpu ----
    {
        CsResetColorsArgsGpu a{};
        a.color = ptrSentinel(0);
        a.dyn = ptrSentinel(1);
        a.dynCount = 0x4001u;
        a._pad0 = 0;
        a.dynCountPtr = ptrSentinel(60);
        std::vector<FieldSpec> f = {
            {"color", true, ptrSentinel(0)},
            {"dyn", true, ptrSentinel(1)},
            {"dynCount", false, 0x4001u},
            {"dynCountPtr", true, ptrSentinel(60)},
        };
        int fails = runProbe(dev, layout, "csProbeCsResetColorsArgs", &a, sizeof(a), f);
        printf("CsResetColorsArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- CsEmitManifoldArgsGpu ----
    {
        CsEmitManifoldArgsGpu a{};
        a.mBodyA = ptrSentinel(0);
        a.mBodyB = ptrSentinel(1);
        a.mContactCount = ptrSentinel(2);
        a.slots = 0x4101u;
        a._pad0 = 0;
        a.mass = ptrSentinel(3);
        a.awake = ptrSentinel(4);
        a.keys = ptrSentinel(5);
        a.isA = ptrSentinel(6);
        a.counter = ptrSentinel(7);
        a.capacity = 0x4102;
        a._pad1 = 0;
        std::vector<FieldSpec> f = {
            {"mBodyA", true, ptrSentinel(0)},        {"mBodyB", true, ptrSentinel(1)},
            {"mContactCount", true, ptrSentinel(2)}, {"slots", false, 0x4101u},
            {"mass", true, ptrSentinel(3)},          {"awake", true, ptrSentinel(4)},
            {"keys", true, ptrSentinel(5)},          {"isA", true, ptrSentinel(6)},
            {"counter", true, ptrSentinel(7)},       {"capacity", false, (uint64_t)0x4102u},
        };
        int fails = runProbe(dev, layout, "csProbeCsEmitManifoldArgs", &a, sizeof(a), f);
        printf("CsEmitManifoldArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- CsEmitForceArgsGpu ----
    {
        CsEmitForceArgsGpu a{};
        a.fBodyA = ptrSentinel(0);
        a.fBodyB = ptrSentinel(1);
        a.count = 0x4201u;
        a._pad0 = 0;
        a.broken = ptrSentinel(2);
        a.mass = ptrSentinel(3);
        a.awake = ptrSentinel(4);
        a.keys = ptrSentinel(5);
        a.isA = ptrSentinel(6);
        a.counter = ptrSentinel(7);
        a.capacity = 0x4202;
        a.type = 1;
        std::vector<FieldSpec> f = {
            {"fBodyA", true, ptrSentinel(0)},       {"fBodyB", true, ptrSentinel(1)}, {"count", false, 0x4201u},
            {"broken", true, ptrSentinel(2)},       {"mass", true, ptrSentinel(3)},   {"awake", true, ptrSentinel(4)},
            {"keys", true, ptrSentinel(5)},         {"isA", true, ptrSentinel(6)},    {"counter", true, ptrSentinel(7)},
            {"capacity", false, (uint64_t)0x4202u}, {"type", false, (uint64_t)1u},
        };
        int fails = runProbe(dev, layout, "csProbeCsEmitForceArgs", &a, sizeof(a), f);
        printf("CsEmitForceArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- CsRowsArgsGpu ----
    {
        CsRowsArgsGpu a{};
        a.keys = ptrSentinel(0);
        a.n = 0x4301u;
        a._pad0 = 0;
        a.start = ptrSentinel(1);
        a.end = ptrSentinel(2);
        std::vector<FieldSpec> f = {
            {"keys", true, ptrSentinel(0)},
            {"n", false, 0x4301u},
            {"start", true, ptrSentinel(1)},
            {"end", true, ptrSentinel(2)},
        };
        int fails = runProbe(dev, layout, "csProbeCsRowsArgs", &a, sizeof(a), f);
        printf("CsRowsArgsGpu  : %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails, sizeof(a));
    }

    // ---- CsJpRoundArgsGpu ----
    {
        CsJpRoundArgsGpu a{};
        a.gKey = ptrSentinel(0);
        a.gIsA = ptrSentinel(1);
        a.gStart = ptrSentinel(2);
        a.gEnd = ptrSentinel(3);
        a.mBodyA = ptrSentinel(4);
        a.mBodyB = ptrSentinel(5);
        a.jBodyA = ptrSentinel(6);
        a.jBodyB = ptrSentinel(7);
        a.sBodyA = ptrSentinel(8);
        a.sBodyB = ptrSentinel(9);
        a.mass = ptrSentinel(10);
        a.awake = ptrSentinel(11);
        a.dyn = ptrSentinel(12);
        a.dynCount = 0x4401u;
        a._pad0 = 0;
        a.color = ptrSentinel(13);
        a.remaining = ptrSentinel(14);
        a.dynCountPtr = ptrSentinel(61);
        std::vector<FieldSpec> f = {
            {"gKey", true, ptrSentinel(0)},         {"gIsA", true, ptrSentinel(1)},
            {"gStart", true, ptrSentinel(2)},       {"gEnd", true, ptrSentinel(3)},
            {"mBodyA", true, ptrSentinel(4)},       {"mBodyB", true, ptrSentinel(5)},
            {"jBodyA", true, ptrSentinel(6)},       {"jBodyB", true, ptrSentinel(7)},
            {"sBodyA", true, ptrSentinel(8)},       {"sBodyB", true, ptrSentinel(9)},
            {"mass", true, ptrSentinel(10)},        {"awake", true, ptrSentinel(11)},
            {"dyn", true, ptrSentinel(12)},         {"dynCount", false, 0x4401u},
            {"color", true, ptrSentinel(13)},       {"remaining", true, ptrSentinel(14)},
            {"dynCountPtr", true, ptrSentinel(61)},
        };
        int fails = runProbe(dev, layout, "csProbeCsJpRoundArgs", &a, sizeof(a), f);
        printf("CsJpRoundArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails, sizeof(a));
    }

    // ---- CsColorKeysArgsGpu ----
    {
        CsColorKeysArgsGpu a{};
        a.color = ptrSentinel(0);
        a.dyn = ptrSentinel(1);
        a.dynCount = 0x4501u;
        a._pad0 = 0;
        a.keys = ptrSentinel(2);
        a.vals = ptrSentinel(3);
        a.dynCountPtr = ptrSentinel(62);
        std::vector<FieldSpec> f = {
            {"color", true, ptrSentinel(0)}, {"dyn", true, ptrSentinel(1)},  {"dynCount", false, 0x4501u},
            {"keys", true, ptrSentinel(2)},  {"vals", true, ptrSentinel(3)}, {"dynCountPtr", true, ptrSentinel(62)},
        };
        int fails = runProbe(dev, layout, "csProbeCsColorKeysArgs", &a, sizeof(a), f);
        printf("CsColorKeysArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- CsColorRowsArgsGpu ----
    {
        CsColorRowsArgsGpu a{};
        a.keys = ptrSentinel(0);
        a.n = 0x4601u;
        a._pad0 = 0;
        a.start = ptrSentinel(1);
        a.end = ptrSentinel(2);
        a.dynCountPtr = ptrSentinel(63);
        std::vector<FieldSpec> f = {
            {"keys", true, ptrSentinel(0)},         {"n", false, 0x4601u},
            {"start", true, ptrSentinel(1)},        {"end", true, ptrSentinel(2)},
            {"dynCountPtr", true, ptrSentinel(63)},
        };
        int fails = runProbe(dev, layout, "csProbeCsColorRowsArgs", &a, sizeof(a), f);
        printf("CsColorRowsArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- primitives: scan.slang / compact.slang / radix.slang args structs ----
    // These already have static_assert(sizeof % 8) guards in vk_args.h; this closes the
    // remaining gap (per-field GPU offset, not just total size).
    {
        {
            ScanBlockArgs a{ptrSentinel(0), ptrSentinel(1), ptrSentinel(2), 0x3001u, 0};
            std::vector<FieldSpec> f = {{"in", true, ptrSentinel(0)},
                                        {"out", true, ptrSentinel(1)},
                                        {"blockSums", true, ptrSentinel(2)},
                                        {"n", false, 0x3001u}};
            int fails = runProbe(dev, layout, "csProbeScanBlockArgs", &a, sizeof(a), f);
            printf("ScanBlockArgs   : %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
                   sizeof(a));
        }
        {
            AddBlockSumsArgs a{ptrSentinel(0), ptrSentinel(1), 0x3002u, 0};
            std::vector<FieldSpec> f = {
                {"out", true, ptrSentinel(0)}, {"blockOffsets", true, ptrSentinel(1)}, {"n", false, 0x3002u}};
            int fails = runProbe(dev, layout, "csProbeAddBlockSumsArgs", &a, sizeof(a), f);
            printf("AddBlockSumsArgs: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
                   sizeof(a));
        }
        {
            FlagsToU32Args a{ptrSentinel(0), ptrSentinel(1), 0x3003u, 0};
            std::vector<FieldSpec> f = {
                {"flags", true, ptrSentinel(0)}, {"out", true, ptrSentinel(1)}, {"n", false, 0x3003u}};
            int fails = runProbe(dev, layout, "csProbeFlagsToU32Args", &a, sizeof(a), f);
            printf("FlagsToU32Args  : %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
                   sizeof(a));
        }
        {
            ScatterCompactArgs a{ptrSentinel(0), ptrSentinel(1), ptrSentinel(2), ptrSentinel(3), 0x3004u, 0};
            std::vector<FieldSpec> f = {{"items", true, ptrSentinel(0)},
                                        {"flags", true, ptrSentinel(1)},
                                        {"scanned", true, ptrSentinel(2)},
                                        {"out", true, ptrSentinel(3)},
                                        {"n", false, 0x3004u}};
            int fails = runProbe(dev, layout, "csProbeScatterCompactArgs", &a, sizeof(a), f);
            printf("ScatterCompactArgs: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
                   sizeof(a));
        }
        {
            NumSelectedArgs a{ptrSentinel(0), ptrSentinel(1), ptrSentinel(2), 0x3005u, 0};
            std::vector<FieldSpec> f = {{"scanned", true, ptrSentinel(0)},
                                        {"flags", true, ptrSentinel(1)},
                                        {"numSelected", true, ptrSentinel(2)},
                                        {"n", false, 0x3005u}};
            int fails = runProbe(dev, layout, "csProbeNumSelectedArgs", &a, sizeof(a), f);
            printf("NumSelectedArgs : %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
                   sizeof(a));
        }
        {
            HistArgsU32Pairs a{ptrSentinel(0), ptrSentinel(1), ptrSentinel(2), 0x3006u, 0x3007u, 0x3008u, 0};
            std::vector<FieldSpec> f = {{"keysIn", true, ptrSentinel(0)},   {"digitCount", true, ptrSentinel(1)},
                                        {"skipMask", true, ptrSentinel(2)}, {"n", false, 0x3006u},
                                        {"numBlocks", false, 0x3007u},      {"pass", false, 0x3008u}};
            int fails = runProbe(dev, layout, "csProbeHistArgsU32Pairs", &a, sizeof(a), f);
            printf("HistArgsU32Pairs: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
                   sizeof(a));
        }
        {
            ScatterArgsU32Pairs a{ptrSentinel(0), ptrSentinel(1), ptrSentinel(2), ptrSentinel(3), ptrSentinel(4),
                                  ptrSentinel(5), 0x3009u,        0x300au,        0x300bu,        0};
            std::vector<FieldSpec> f = {{"keysIn", true, ptrSentinel(0)},
                                        {"valsIn", true, ptrSentinel(1)},
                                        {"digitCountScanned", true, ptrSentinel(2)},
                                        {"skipMask", true, ptrSentinel(3)},
                                        {"keysOut", true, ptrSentinel(4)},
                                        {"valsOut", true, ptrSentinel(5)},
                                        {"n", false, 0x3009u},
                                        {"numBlocks", false, 0x300au},
                                        {"pass", false, 0x300bu}};
            int fails = runProbe(dev, layout, "csProbeScatterArgsU32Pairs", &a, sizeof(a), f);
            printf("ScatterArgsU32Pairs: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
                   sizeof(a));
        }
        {
            HistArgsU64Pairs a{ptrSentinel(0), ptrSentinel(1), ptrSentinel(2), 0x300cu, 0x300du, 0x300eu, 0};
            std::vector<FieldSpec> f = {{"keysIn", true, ptrSentinel(0)},   {"digitCount", true, ptrSentinel(1)},
                                        {"skipMask", true, ptrSentinel(2)}, {"n", false, 0x300cu},
                                        {"numBlocks", false, 0x300du},      {"pass", false, 0x300eu}};
            int fails = runProbe(dev, layout, "csProbeHistArgsU64Pairs", &a, sizeof(a), f);
            printf("HistArgsU64Pairs: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
                   sizeof(a));
        }
        {
            ScatterArgsU64Pairs a{ptrSentinel(0), ptrSentinel(1), ptrSentinel(2), ptrSentinel(3), ptrSentinel(4),
                                  ptrSentinel(5), 0x300fu,        0x3010u,        0x3011u,        0};
            std::vector<FieldSpec> f = {{"keysIn", true, ptrSentinel(0)},
                                        {"valsIn", true, ptrSentinel(1)},
                                        {"digitCountScanned", true, ptrSentinel(2)},
                                        {"skipMask", true, ptrSentinel(3)},
                                        {"keysOut", true, ptrSentinel(4)},
                                        {"valsOut", true, ptrSentinel(5)},
                                        {"n", false, 0x300fu},
                                        {"numBlocks", false, 0x3010u},
                                        {"pass", false, 0x3011u}};
            int fails = runProbe(dev, layout, "csProbeScatterArgsU64Pairs", &a, sizeof(a), f);
            printf("ScatterArgsU64Pairs: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
                   sizeof(a));
        }
        {
            HistArgsU64Keys a{ptrSentinel(0), ptrSentinel(1), ptrSentinel(2), 0x3012u, 0x3013u, 0x3014u, 0};
            std::vector<FieldSpec> f = {{"keysIn", true, ptrSentinel(0)},   {"digitCount", true, ptrSentinel(1)},
                                        {"skipMask", true, ptrSentinel(2)}, {"n", false, 0x3012u},
                                        {"numBlocks", false, 0x3013u},      {"pass", false, 0x3014u}};
            int fails = runProbe(dev, layout, "csProbeHistArgsU64Keys", &a, sizeof(a), f);
            printf("HistArgsU64Keys : %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
                   sizeof(a));
        }
        {
            ScatterArgsU64Keys a{ptrSentinel(0), ptrSentinel(1), ptrSentinel(2), ptrSentinel(3),
                                 0x3015u,        0x3016u,        0x3017u,        0};
            std::vector<FieldSpec> f = {{"keysIn", true, ptrSentinel(0)},
                                        {"digitCountScanned", true, ptrSentinel(1)},
                                        {"skipMask", true, ptrSentinel(2)},
                                        {"keysOut", true, ptrSentinel(3)},
                                        {"n", false, 0x3015u},
                                        {"numBlocks", false, 0x3016u},
                                        {"pass", false, 0x3017u}};
            int fails = runProbe(dev, layout, "csProbeScatterArgsU64Keys", &a, sizeof(a), f);
            printf("ScatterArgsU64Keys: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
                   sizeof(a));
        }
        {
            ReduceOrAndArgsU32 a{ptrSentinel(0), ptrSentinel(1), ptrSentinel(2), 0x3018u, 0};
            std::vector<FieldSpec> f = {{"keysIn", true, ptrSentinel(0)},
                                        {"outOr", true, ptrSentinel(1)},
                                        {"outAnd", true, ptrSentinel(2)},
                                        {"n", false, 0x3018u}};
            int fails = runProbe(dev, layout, "csProbeReduceOrAndArgsU32", &a, sizeof(a), f);
            printf("ReduceOrAndArgsU32: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
                   sizeof(a));
        }
        {
            ReduceOrAndArgsU64 a{ptrSentinel(0), ptrSentinel(1), ptrSentinel(2), 0x3019u, 0};
            std::vector<FieldSpec> f = {{"keysIn", true, ptrSentinel(0)},
                                        {"outOr", true, ptrSentinel(1)},
                                        {"outAnd", true, ptrSentinel(2)},
                                        {"n", false, 0x3019u}};
            int fails = runProbe(dev, layout, "csProbeReduceOrAndArgsU64", &a, sizeof(a), f);
            printf("ReduceOrAndArgsU64: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
                   sizeof(a));
        }
        {
            FinalizeSkipMaskArgsU32 a{ptrSentinel(0), ptrSentinel(1), ptrSentinel(2), 0x301au, 0};
            std::vector<FieldSpec> f = {{"orAddr", true, ptrSentinel(0)},
                                        {"andAddr", true, ptrSentinel(1)},
                                        {"skipMask", true, ptrSentinel(2)},
                                        {"numPasses", false, 0x301au}};
            int fails = runProbe(dev, layout, "csProbeFinalizeSkipMaskArgsU32", &a, sizeof(a), f);
            printf("FinalizeSkipMaskArgsU32: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
                   sizeof(a));
        }
        {
            FinalizeSkipMaskArgsU64 a{ptrSentinel(0), ptrSentinel(1), ptrSentinel(2), 0x301bu, 0};
            std::vector<FieldSpec> f = {{"orAddr", true, ptrSentinel(0)},
                                        {"andAddr", true, ptrSentinel(1)},
                                        {"skipMask", true, ptrSentinel(2)},
                                        {"numPasses", false, 0x301bu}};
            int fails = runProbe(dev, layout, "csProbeFinalizeSkipMaskArgsU64", &a, sizeof(a), f);
            printf("FinalizeSkipMaskArgsU64: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
                   sizeof(a));
        }
    }

    // =====================================================================
    // Phase 4b: primal.slang / dual.slang args structs (src/world/vk_args.h).
    // =====================================================================

    auto fillFullBody = [](FullBodySoAGpu &b, int base)
    {
        b.pos = ptrSentinel(base + 0);
        b.rot = ptrSentinel(base + 1);
        b.initialLin = ptrSentinel(base + 2);
        b.initialAng = ptrSentinel(base + 3);
        b.inertialLin = ptrSentinel(base + 4);
        b.inertialAng = ptrSentinel(base + 5);
        b.velLin = ptrSentinel(base + 6);
        b.velAng = ptrSentinel(base + 7);
        b.prevVelLin = ptrSentinel(base + 8);
        b.halfExtents = ptrSentinel(base + 9);
        b.shapeType = ptrSentinel(base + 10);
        b.moment = ptrSentinel(base + 11);
        b.mass = ptrSentinel(base + 12);
        b.friction = ptrSentinel(base + 13);
        b.awake = ptrSentinel(base + 14);
        b.sleepTimer = ptrSentinel(base + 15);
        b.restPos = ptrSentinel(base + 16);
        b.restRot = ptrSentinel(base + 17);
        b.gyration = ptrSentinel(base + 18);
        b.props = ptrSentinel(base + 950);
        b.count = 0x5001u;
    };
    auto fullBodyFields = [](int base, std::vector<FieldSpec> &f)
    {
        f.push_back({"b.pos", true, ptrSentinel(base + 0)});
        f.push_back({"b.rot", true, ptrSentinel(base + 1)});
        f.push_back({"b.initialLin", true, ptrSentinel(base + 2)});
        f.push_back({"b.initialAng", true, ptrSentinel(base + 3)});
        f.push_back({"b.inertialLin", true, ptrSentinel(base + 4)});
        f.push_back({"b.inertialAng", true, ptrSentinel(base + 5)});
        f.push_back({"b.velLin", true, ptrSentinel(base + 6)});
        f.push_back({"b.velAng", true, ptrSentinel(base + 7)});
        f.push_back({"b.prevVelLin", true, ptrSentinel(base + 8)});
        f.push_back({"b.halfExtents", true, ptrSentinel(base + 9)});
        f.push_back({"b.shapeType", true, ptrSentinel(base + 10)});
        f.push_back({"b.moment", true, ptrSentinel(base + 11)});
        f.push_back({"b.mass", true, ptrSentinel(base + 12)});
        f.push_back({"b.friction", true, ptrSentinel(base + 13)});
        f.push_back({"b.awake", true, ptrSentinel(base + 14)});
        f.push_back({"b.sleepTimer", true, ptrSentinel(base + 15)});
        f.push_back({"b.restPos", true, ptrSentinel(base + 16)});
        f.push_back({"b.restRot", true, ptrSentinel(base + 17)});
        f.push_back({"b.gyration", true, ptrSentinel(base + 18)});
        f.push_back({"b.props", true, ptrSentinel(base + 950)});
        f.push_back({"b.count", false, 0x5001u});
    };

    auto fillFullManifold = [](FullManifoldSoAGpu &m, int base)
    {
        m.key = ptrSentinel(base + 0);
        m.bodyA = ptrSentinel(base + 1);
        m.bodyB = ptrSentinel(base + 2);
        m.basis = ptrSentinel(base + 3);
        m.friction = ptrSentinel(base + 4);
        m.contactOffset = ptrSentinel(base + 5);
        m.contactCount = ptrSentinel(base + 6);
        m.count = 0x5111;
        m.capacity = 0x5112;
        m.feature = ptrSentinel(base + 7);
        m.rA = ptrSentinel(base + 8);
        m.rB = ptrSentinel(base + 9);
        m.C0 = ptrSentinel(base + 10);
        m.penalty = ptrSentinel(base + 11);
        m.lambda = ptrSentinel(base + 12);
        m.stick = ptrSentinel(base + 13);
        m.contactTotal = 0x5113;
        m.contactCapacity = 0x5114;
        m.shapeA = ptrSentinel(base + 14);
        m.shapeB = ptrSentinel(base + 15);
    };
    auto fullManifoldFieldsAll = [](int base, std::vector<FieldSpec> &f)
    {
        f.push_back({"m.key", true, ptrSentinel(base + 0)});
        f.push_back({"m.bodyA", true, ptrSentinel(base + 1)});
        f.push_back({"m.bodyB", true, ptrSentinel(base + 2)});
        f.push_back({"m.basis", true, ptrSentinel(base + 3)});
        f.push_back({"m.friction", true, ptrSentinel(base + 4)});
        f.push_back({"m.contactOffset", true, ptrSentinel(base + 5)});
        f.push_back({"m.contactCount", true, ptrSentinel(base + 6)});
        f.push_back({"m.count", false, 0x5111u});
        f.push_back({"m.capacity", false, 0x5112u});
        f.push_back({"m.feature", true, ptrSentinel(base + 7)});
        f.push_back({"m.rA", true, ptrSentinel(base + 8)});
        f.push_back({"m.rB", true, ptrSentinel(base + 9)});
        f.push_back({"m.C0", true, ptrSentinel(base + 10)});
        f.push_back({"m.penalty", true, ptrSentinel(base + 11)});
        f.push_back({"m.lambda", true, ptrSentinel(base + 12)});
        f.push_back({"m.stick", true, ptrSentinel(base + 13)});
        f.push_back({"m.contactTotal", false, 0x5113u});
        f.push_back({"m.contactCapacity", false, 0x5114u});
        f.push_back({"m.shapeA", true, ptrSentinel(base + 14)});
        f.push_back({"m.shapeB", true, ptrSentinel(base + 15)});
    };
    // Subset used by DualArgs probes (per those kernels' field lists).
    auto fullManifoldFieldsDual = [](int base, std::vector<FieldSpec> &f, bool withStick)
    {
        f.push_back({"m.bodyA", true, ptrSentinel(base + 1)});
        f.push_back({"m.bodyB", true, ptrSentinel(base + 2)});
        f.push_back({"m.basis", true, ptrSentinel(base + 3)});
        f.push_back({"m.friction", true, ptrSentinel(base + 4)});
        f.push_back({"m.contactOffset", true, ptrSentinel(base + 5)});
        f.push_back({"m.contactCount", true, ptrSentinel(base + 6)});
        f.push_back({"m.rA", true, ptrSentinel(base + 8)});
        f.push_back({"m.rB", true, ptrSentinel(base + 9)});
        f.push_back({"m.C0", true, ptrSentinel(base + 10)});
        f.push_back({"m.penalty", true, ptrSentinel(base + 11)});
        f.push_back({"m.lambda", true, ptrSentinel(base + 12)});
        if (withStick)
            f.push_back({"m.stick", true, ptrSentinel(base + 13)});
    };

    auto fillShape = [](ShapeSoAGpu &sh, int base)
    {
        sh.body = ptrSentinel(base + 0);
        sh.localPos = ptrSentinel(base + 1);
        sh.localRot = ptrSentinel(base + 2);
        sh.halfExtents = ptrSentinel(base + 3);
        sh.shapeType = ptrSentinel(base + 4);
        sh.friction = ptrSentinel(base + 5);
        sh.category = ptrSentinel(base + 6);
        sh.mask = ptrSentinel(base + 7);
        sh.group = ptrSentinel(base + 8);
        sh.flags = ptrSentinel(base + 9);
        sh.count = 0x6101u;
    };
    auto shapeFields = [](int base, std::vector<FieldSpec> &f)
    {
        const char *names[10] = {"s.body",     "s.localPos", "s.localRot", "s.halfExtents", "s.shapeType",
                                 "s.friction", "s.category", "s.mask",     "s.group",       "s.flags"};
        for (int i = 0; i < 10; i++)
            f.push_back({names[i], true, ptrSentinel(base + i)});
        f.push_back({"s.count", false, 0x6101u});
    };

    auto fillFullJoint = [](FullJointSoAGpu &j, int base)
    {
        j.bodyA = ptrSentinel(base + 0);
        j.bodyB = ptrSentinel(base + 1);
        j.rA = ptrSentinel(base + 2);
        j.rB = ptrSentinel(base + 3);
        j.C0Lin = ptrSentinel(base + 4);
        j.C0Ang = ptrSentinel(base + 5);
        j.penaltyLin = ptrSentinel(base + 6);
        j.penaltyAng = ptrSentinel(base + 7);
        j.lambdaLin = ptrSentinel(base + 8);
        j.lambdaAng = ptrSentinel(base + 9);
        j.stiffnessLin = ptrSentinel(base + 10);
        j.stiffnessAng = ptrSentinel(base + 11);
        j.fracture = ptrSentinel(base + 12);
        j.torqueArm = ptrSentinel(base + 13);
        j.restAng = ptrSentinel(base + 14);
        j.broken = ptrSentinel(base + 15);
        j.frameA = ptrSentinel(base + 16);
        j.frameB = ptrSentinel(base + 17);
        j.axisMode = ptrSentinel(base + 18);
        j.axisLo = ptrSentinel(base + 19);
        j.axisHi = ptrSentinel(base + 20);
        j.generic = ptrSentinel(base + 21);
        j.count = 0x5211u;
    };
    auto fullJointFieldsAll = [](int base, std::vector<FieldSpec> &f)
    {
        f.push_back({"j.bodyA", true, ptrSentinel(base + 0)});
        f.push_back({"j.bodyB", true, ptrSentinel(base + 1)});
        f.push_back({"j.rA", true, ptrSentinel(base + 2)});
        f.push_back({"j.rB", true, ptrSentinel(base + 3)});
        f.push_back({"j.C0Lin", true, ptrSentinel(base + 4)});
        f.push_back({"j.C0Ang", true, ptrSentinel(base + 5)});
        f.push_back({"j.penaltyLin", true, ptrSentinel(base + 6)});
        f.push_back({"j.penaltyAng", true, ptrSentinel(base + 7)});
        f.push_back({"j.lambdaLin", true, ptrSentinel(base + 8)});
        f.push_back({"j.lambdaAng", true, ptrSentinel(base + 9)});
        f.push_back({"j.stiffnessLin", true, ptrSentinel(base + 10)});
        f.push_back({"j.stiffnessAng", true, ptrSentinel(base + 11)});
        f.push_back({"j.fracture", true, ptrSentinel(base + 12)});
        f.push_back({"j.torqueArm", true, ptrSentinel(base + 13)});
        f.push_back({"j.restAng", true, ptrSentinel(base + 14)});
        f.push_back({"j.broken", true, ptrSentinel(base + 15)});
        f.push_back({"j.frameA", true, ptrSentinel(base + 16)});
        f.push_back({"j.frameB", true, ptrSentinel(base + 17)});
        f.push_back({"j.axisMode", true, ptrSentinel(base + 18)});
        f.push_back({"j.axisLo", true, ptrSentinel(base + 19)});
        f.push_back({"j.axisHi", true, ptrSentinel(base + 20)});
        f.push_back({"j.generic", true, ptrSentinel(base + 21)});
        f.push_back({"j.count", false, 0x5211u});
    };

    auto fillFullSpring = [](FullSpringSoAGpu &s, int base)
    {
        s.bodyA = ptrSentinel(base + 0);
        s.bodyB = ptrSentinel(base + 1);
        s.rA = ptrSentinel(base + 2);
        s.rB = ptrSentinel(base + 3);
        s.rest = ptrSentinel(base + 4);
        s.stiffness = ptrSentinel(base + 5);
        s.count = 0x5311u;
    };
    auto fullSpringFieldsAll = [](int base, std::vector<FieldSpec> &f)
    {
        f.push_back({"springs.bodyA", true, ptrSentinel(base + 0)});
        f.push_back({"springs.bodyB", true, ptrSentinel(base + 1)});
        f.push_back({"springs.rA", true, ptrSentinel(base + 2)});
        f.push_back({"springs.rB", true, ptrSentinel(base + 3)});
        f.push_back({"springs.rest", true, ptrSentinel(base + 4)});
        f.push_back({"springs.stiffness", true, ptrSentinel(base + 5)});
        f.push_back({"springs.count", false, 0x5311u});
    };

    auto fillSolverParamsFull = [](SolverParams &p)
    {
        p.dt = 1.0f / 60.0f;
        p.gravity = vec3{0.0f, 0.0f, -10.0f};
        p.iterations = 8;
        p.primalLanes = 4;
        p.alpha = 0.97f;
        p.betaLin = 10000.0f;
        p.betaAng = 100.0f;
        p.gamma = 0.99f;
        p.sleepEnabled = true;
        p.sleepFrames = 12;
        p.sleepLinVel = 0.05f;
        p.sleepAngVel = 0.05f;
        p.sleepDisp = 0.01f;
        p.sleepFracture = 0.9f;
        p.autoKStart = true;
        p.dualDamping = 0.1f;
        p.dualDampVel = 0.5f;
        p.killFallenEnabled = false;
        p.killHeight = -50.0f;
        p.grabStrength = 1.0f;
        p.worldFrameInertia = true;
    };
    auto fbitsF = [](float v)
    {
        uint32_t u;
        std::memcpy(&u, &v, 4);
        return (uint64_t)u;
    };

    // ---- PrimalArgsGpu ----
    {
        PrimalArgsGpu a{};
        fillFullBody(a.b, 0);
        a.g.key = ptrSentinel(100);
        a.g.isA = ptrSentinel(101);
        a.g.start = ptrSentinel(102);
        a.g.end = ptrSentinel(103);
        fillFullManifold(a.m, 200);
        fillFullJoint(a.joints, 300);
        fillFullSpring(a.springs, 400);
        a.colorBodies = ptrSentinel(500);
        a.listStart = 0x5401;
        a.listCount = 0x5402;
        SolverParams sp{};
        fillSolverParamsFull(sp);
        a.p = packSolverParams(sp);

        std::vector<FieldSpec> f;
        fullBodyFields(0, f);
        f.push_back({"g.key", true, ptrSentinel(100)});
        f.push_back({"g.isA", true, ptrSentinel(101)});
        f.push_back({"g.start", true, ptrSentinel(102)});
        f.push_back({"g.end", true, ptrSentinel(103)});
        fullManifoldFieldsAll(200, f);
        fullJointFieldsAll(300, f);
        fullSpringFieldsAll(400, f);
        f.push_back({"colorBodies", true, ptrSentinel(500)});
        f.push_back({"listStart", false, (uint64_t)0x5401u});
        f.push_back({"listCount", false, (uint64_t)0x5402u});
        f.push_back({"p.dt", false, fbitsF(a.p.dt)});
        f.push_back({"p.worldFrameInertia", false, fbitsF(a.p.worldFrameInertia)});
        f.push_back({"p.iterations", false, (uint64_t)(uint32_t)a.p.iterations});
        f.push_back({"p.primalLanes", false, (uint64_t)(uint32_t)a.p.primalLanes});
        f.push_back({"p.alpha", false, fbitsF(a.p.alpha)});
        f.push_back({"p.betaLin", false, fbitsF(a.p.betaLin)});
        f.push_back({"p.betaAng", false, fbitsF(a.p.betaAng)});
        f.push_back({"p.gamma", false, fbitsF(a.p.gamma)});

        int fails = runProbe(dev, layout, "csProbePrimalArgs", &a, sizeof(a), f);
        printf("PrimalArgsGpu  : %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails, sizeof(a));
    }

    // ---- CoarseArgsGpu ----
    {
        CoarseArgsGpu a{};
        a.pa.colorBodies = ptrSentinel(500);
        a.pa.colorStart = ptrSentinel(501);
        a.pa.color = 0x5403;
        a.pa.shapeRolling = ptrSentinel(507);
        a.active = ptrSentinel(600);
        a.activeCount = ptrSentinel(601);
        a.aggOf = ptrSentinel(602);
        a.keys = ptrSentinel(603);
        a.acc = ptrSentinel(604);
        a.delta = ptrSentinel(605);
        a.stats = ptrSentinel(606);
        a.maxActive = 0x6101;
        a.bodyCount = 0x6102;
        a.tableSize = 0x6103;
        a.cell = 1.25f;
        a.damping = 0.7f;
        a.withForces = 0x6104;
        a.color = 0x6105;
        std::vector<FieldSpec> f;
        f.push_back({"pa.colorBodies", true, ptrSentinel(500)});
        f.push_back({"pa.colorStart", true, ptrSentinel(501)});
        f.push_back({"pa.color", false, (uint64_t)0x5403u});
        f.push_back({"pa.shapeRolling", true, ptrSentinel(507)});
        f.push_back({"active", true, ptrSentinel(600)});
        f.push_back({"activeCount", true, ptrSentinel(601)});
        f.push_back({"aggOf", true, ptrSentinel(602)});
        f.push_back({"keys", true, ptrSentinel(603)});
        f.push_back({"acc", true, ptrSentinel(604)});
        f.push_back({"delta", true, ptrSentinel(605)});
        f.push_back({"stats", true, ptrSentinel(606)});
        f.push_back({"maxActive", false, (uint64_t)0x6101u});
        f.push_back({"bodyCount", false, (uint64_t)0x6102u});
        f.push_back({"tableSize", false, (uint64_t)0x6103u});
        f.push_back({"cell", false, fbitsF(1.25f)});
        f.push_back({"damping", false, fbitsF(0.7f)});
        f.push_back({"withForces", false, (uint64_t)0x6104u});
        f.push_back({"color", false, (uint64_t)0x6105u});
        int fails = runProbe(dev, layout, "csProbeCoarseArgs", &a, sizeof(a), f);
        printf("CoarseArgsGpu  : %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails, sizeof(a));
    }

    // ---- JointInitArgsGpu ----
    {
        JointInitArgsGpu a{};
        fillFullBody(a.b, 0);
        fillFullJoint(a.j, 300);
        SolverParams sp{};
        fillSolverParamsFull(sp);
        a.p = packSolverParams(sp);

        std::vector<FieldSpec> f;
        f.push_back({"b.pos", true, ptrSentinel(0)});
        f.push_back({"b.mass", true, ptrSentinel(12)});
        f.push_back({"b.awake", true, ptrSentinel(14)});
        f.push_back({"b.count", false, 0x5001u});
        f.push_back({"j.bodyA", true, ptrSentinel(300)});
        f.push_back({"j.penaltyLin", true, ptrSentinel(306)});
        f.push_back({"j.broken", true, ptrSentinel(315)});
        f.push_back({"j.count", false, 0x5211u});
        f.push_back({"p.dt", false, fbitsF(a.p.dt)});
        f.push_back({"p.alpha", false, fbitsF(a.p.alpha)});
        f.push_back({"p.gamma", false, fbitsF(a.p.gamma)});

        int fails = runProbe(dev, layout, "csProbeJointInitArgs", &a, sizeof(a), f);
        printf("JointInitArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails, sizeof(a));
    }

    // ---- JointDualArgsGpu ----
    {
        JointDualArgsGpu a{};
        fillFullBody(a.b, 0);
        fillFullJoint(a.j, 300);
        SolverParams sp{};
        fillSolverParamsFull(sp);
        a.p = packSolverParams(sp);

        std::vector<FieldSpec> f;
        f.push_back({"b.pos", true, ptrSentinel(0)});
        f.push_back({"b.mass", true, ptrSentinel(12)});
        f.push_back({"b.awake", true, ptrSentinel(14)});
        f.push_back({"b.count", false, 0x5001u});
        f.push_back({"j.bodyA", true, ptrSentinel(300)});
        f.push_back({"j.lambdaLin", true, ptrSentinel(308)});
        f.push_back({"j.broken", true, ptrSentinel(315)});
        f.push_back({"j.count", false, 0x5211u});
        f.push_back({"p.alpha", false, fbitsF(a.p.alpha)});
        f.push_back({"p.betaLin", false, fbitsF(a.p.betaLin)});
        f.push_back({"p.betaAng", false, fbitsF(a.p.betaAng)});

        int fails = runProbe(dev, layout, "csProbeJointDualArgs", &a, sizeof(a), f);
        printf("JointDualArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails, sizeof(a));
    }

    // ---- DualArgsGpu -- the struct that briefly had the spurious _pad0 (see its comment
    // in vk_args.h). This probe is exactly what would have caught it: p.alpha/p.betaLin
    // would land 4 bytes off if a pad were reintroduced between `slots` and `p`. ----
    {
        DualArgsGpu a{};
        fillFullBody(a.b, 0);
        fillFullManifold(a.m, 200);
        a.slots = 0x5601;
        SolverParams sp{};
        fillSolverParamsFull(sp);
        a.p = packSolverParams(sp);

        std::vector<FieldSpec> f;
        f.push_back({"b.pos", true, ptrSentinel(0)});
        f.push_back({"b.rot", true, ptrSentinel(1)});
        f.push_back({"b.initialLin", true, ptrSentinel(2)});
        f.push_back({"b.initialAng", true, ptrSentinel(3)});
        f.push_back({"b.count", false, 0x5001u});
        fullManifoldFieldsDual(200, f, true);
        f.push_back({"slots", false, (uint64_t)0x5601u});
        f.push_back({"p.alpha", false, fbitsF(a.p.alpha)});
        f.push_back({"p.betaLin", false, fbitsF(a.p.betaLin)});

        int fails = runProbe(dev, layout, "csProbeDualArgs", &a, sizeof(a), f);
        printf("DualArgsGpu    : %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails, sizeof(a));
    }

    // ---- WriteInstancesArgsGpu (src/shaders/render.slang, Phase 5C/5C-cont) ----
    {
        WriteInstancesArgsGpu a{};
        fillFullBody(a.b, 0);
        a.out = ptrSentinel(700);
        a.maxInstances = 0x5801;
        a.frustum = ptrSentinel(710);
        a.visible = ptrSentinel(711);
        a.ov.mode = 0x5802;
        a.ov.speedMax = 1.25f;
        a.ov.loadMax = 2.25f;
        a.ov.degreeMax = 3.25f;
        a.ov.invCellSize = 4.25f;
        a.ov.sleepFrames = 5.25f;
        a.ov.colorCount = 0x5803;
        a.ov.colorStatics = 0x5804;
        a.ov.graphColor = ptrSentinel(712);
        a.ov.load = ptrSentinel(713);
        a.ov.degree = ptrSentinel(714);
        a.ov.jointLoadMax = 6.25f;
        a.ov.degreeJointMax = 7.25f;
        a.ov.penetrationMax = 8.25f;
        a.ov.jointDegree = ptrSentinel(715);
        a.ov.push = ptrSentinel(716);
        a.ov.pull = ptrSentinel(717);
        a.ov.stress = ptrSentinel(718);
        a.ov.cone = ptrSentinel(719);
        a.ov.penetration = ptrSentinel(720);
        a.groupOf = ptrSentinel(721);
        a.shapeBase = ptrSentinel(722);
        a.shapeVisible = ptrSentinel(723);
        fillShape(a.s, 730);

        std::vector<FieldSpec> f;
        fullBodyFields(0, f);
        f.push_back({"out", true, ptrSentinel(700)});
        f.push_back({"maxInstances", false, (uint64_t)0x5801u});
        f.push_back({"frustum", true, ptrSentinel(710)});
        f.push_back({"visible", true, ptrSentinel(711)});
        f.push_back({"ov.mode", false, (uint64_t)0x5802u});
        f.push_back({"ov.speedMax", false, fbitsF(1.25f)});
        f.push_back({"ov.loadMax", false, fbitsF(2.25f)});
        f.push_back({"ov.degreeMax", false, fbitsF(3.25f)});
        f.push_back({"ov.invCellSize", false, fbitsF(4.25f)});
        f.push_back({"ov.sleepFrames", false, fbitsF(5.25f)});
        f.push_back({"ov.colorCount", false, (uint64_t)0x5803u});
        f.push_back({"ov.colorStatics", false, (uint64_t)0x5804u});
        f.push_back({"ov.graphColor", true, ptrSentinel(712)});
        f.push_back({"ov.load", true, ptrSentinel(713)});
        f.push_back({"ov.degree", true, ptrSentinel(714)});
        f.push_back({"ov.jointLoadMax", false, fbitsF(6.25f)});
        f.push_back({"ov.degreeJointMax", false, fbitsF(7.25f)});
        f.push_back({"ov.penetrationMax", false, fbitsF(8.25f)});
        f.push_back({"ov.jointDegree", true, ptrSentinel(715)});
        f.push_back({"ov.push", true, ptrSentinel(716)});
        f.push_back({"ov.pull", true, ptrSentinel(717)});
        f.push_back({"ov.stress", true, ptrSentinel(718)});
        f.push_back({"ov.cone", true, ptrSentinel(719)});
        f.push_back({"ov.penetration", true, ptrSentinel(720)});
        f.push_back({"groupOf", true, ptrSentinel(721)});
        f.push_back({"shapeBase", true, ptrSentinel(722)});
        f.push_back({"shapeVisible", true, ptrSentinel(723)});
        shapeFields(730, f);

        int fails = runProbe(dev, layout, "csProbeWriteInstancesArgs", &a, sizeof(a), f);
        printf("WriteInstancesArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- OverlayStatsClearArgsGpu (src/shaders/render.slang) ----
    {
        OverlayStatsClearArgsGpu a{};
        a.s.push = ptrSentinel(730);
        a.s.pull = ptrSentinel(731);
        a.s.stress = ptrSentinel(732);
        a.s.cone = ptrSentinel(733);
        a.s.penetration = ptrSentinel(734);
        a.s.jointDegree = ptrSentinel(735);
        a.n = 0x5901;

        std::vector<FieldSpec> f;
        f.push_back({"s.push", true, ptrSentinel(730)});
        f.push_back({"s.pull", true, ptrSentinel(731)});
        f.push_back({"s.stress", true, ptrSentinel(732)});
        f.push_back({"s.cone", true, ptrSentinel(733)});
        f.push_back({"s.penetration", true, ptrSentinel(734)});
        f.push_back({"s.jointDegree", true, ptrSentinel(735)});
        f.push_back({"n", false, (uint64_t)0x5901u});

        int fails = runProbe(dev, layout, "csProbeOverlayStatsClearArgs", &a, sizeof(a), f);
        printf("OverlayStatsClearArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- OverlayStatsContactsArgsGpu (src/shaders/render.slang) ----
    {
        OverlayStatsContactsArgsGpu a{};
        fillFullManifold(a.m, 200);
        a.slots = 0x5902;
        a.s.push = ptrSentinel(730);
        a.s.pull = ptrSentinel(731);
        a.s.stress = ptrSentinel(732);
        a.s.cone = ptrSentinel(733);
        a.s.penetration = ptrSentinel(734);
        a.s.jointDegree = ptrSentinel(735);

        // Full-field probe: every ManifoldSoA field plus slots/s, in probe-write order.
        std::vector<FieldSpec> f;
        fullManifoldFieldsAll(200, f);
        f.push_back({"slots", false, (uint64_t)0x5902u});
        f.push_back({"s.push", true, ptrSentinel(730)});
        f.push_back({"s.pull", true, ptrSentinel(731)});
        f.push_back({"s.stress", true, ptrSentinel(732)});
        f.push_back({"s.cone", true, ptrSentinel(733)});
        f.push_back({"s.penetration", true, ptrSentinel(734)});
        f.push_back({"s.jointDegree", true, ptrSentinel(735)});

        int fails = runProbe(dev, layout, "csProbeOverlayStatsContactsArgs", &a, sizeof(a), f);
        printf("OverlayStatsContactsArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- OverlayStatsForcesArgsGpu (src/shaders/render.slang) ----
    {
        OverlayStatsForcesArgsGpu a{};
        fillFullBody(a.b, 0);
        fillFullJoint(a.j, 300);
        fillFullSpring(a.sp, 400);
        a.jointsAuthored = 0x5903;
        a.s.push = ptrSentinel(730);
        a.s.pull = ptrSentinel(731);
        a.s.stress = ptrSentinel(732);
        a.s.cone = ptrSentinel(733);
        a.s.penetration = ptrSentinel(734);
        a.s.jointDegree = ptrSentinel(735);

        // Full-field probe: every BodySoA/JointSoA/SpringSoA field plus jointsAuthored/s.
        std::vector<FieldSpec> f;
        fullBodyFields(0, f);
        fullJointFieldsAll(300, f);
        fullSpringFieldsAll(400, f);
        f.push_back({"jointsAuthored", false, (uint64_t)0x5903u});
        f.push_back({"s.push", true, ptrSentinel(730)});
        f.push_back({"s.pull", true, ptrSentinel(731)});
        f.push_back({"s.stress", true, ptrSentinel(732)});
        f.push_back({"s.cone", true, ptrSentinel(733)});
        f.push_back({"s.penetration", true, ptrSentinel(734)});
        f.push_back({"s.jointDegree", true, ptrSentinel(735)});

        int fails = runProbe(dev, layout, "csProbeOverlayStatsForcesArgs", &a, sizeof(a), f);
        printf("OverlayStatsForcesArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- OverlayLoadClearArgsGpu (src/shaders/render.slang) ----
    {
        OverlayLoadClearArgsGpu a{};
        a.load = ptrSentinel(740);
        a.n = 0x5904;

        std::vector<FieldSpec> f;
        f.push_back({"load", true, ptrSentinel(740)});
        f.push_back({"n", false, (uint64_t)0x5904u});

        int fails = runProbe(dev, layout, "csProbeOverlayLoadClearArgs", &a, sizeof(a), f);
        printf("OverlayLoadClearArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- OverlayLoadAccumArgsGpu (src/shaders/render.slang) ----
    {
        OverlayLoadAccumArgsGpu a{};
        fillFullBody(a.b, 0);
        fillFullManifold(a.m, 200);
        a.slots = 0x5905;
        a.load = ptrSentinel(740);

        // Full-field probe: every BodySoA/ManifoldSoA field plus slots/load.
        std::vector<FieldSpec> f;
        fullBodyFields(0, f);
        fullManifoldFieldsAll(200, f);
        f.push_back({"slots", false, (uint64_t)0x5905u});
        f.push_back({"load", true, ptrSentinel(740)});

        int fails = runProbe(dev, layout, "csProbeOverlayLoadAccumArgs", &a, sizeof(a), f);
        printf("OverlayLoadAccumArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- WriteDebugInstancesArgsGpu (src/shaders/render.slang) ----
    {
        WriteDebugInstancesArgsGpu a{};
        fillFullBody(a.b, 0);
        fillFullManifold(a.m, 200);
        a.slots = 0x5906;
        a.out = ptrSentinel(750);
        a.maxInstances = 0x5907;
        a.counter = ptrSentinel(751);
        a.d.contactPoints = 1;
        a.d.contactNormals = 1;
        a.d.manifoldLinks = 0;
        a.d.tint = 2;
        a.d.pointSize = 0.1f;
        a.d.normalLength = 0.2f;
        a.d.forceMax = 100.0f;

        // Full-field probe: every BodySoA/ManifoldSoA field plus slots/out/etc/d.*.
        std::vector<FieldSpec> f;
        fullBodyFields(0, f);
        fullManifoldFieldsAll(200, f);
        f.push_back({"slots", false, (uint64_t)0x5906u});
        f.push_back({"out", true, ptrSentinel(750)});
        f.push_back({"maxInstances", false, (uint64_t)0x5907u});
        f.push_back({"counter", true, ptrSentinel(751)});
        f.push_back({"d.contactPoints", false, (uint64_t)1u});
        f.push_back({"d.contactNormals", false, (uint64_t)1u});
        f.push_back({"d.manifoldLinks", false, (uint64_t)0u});
        f.push_back({"d.tint", false, (uint64_t)2u});
        f.push_back({"d.pointSize", false, fbitsF(0.1f)});
        f.push_back({"d.normalLength", false, fbitsF(0.2f)});
        f.push_back({"d.forceMax", false, fbitsF(100.0f)});

        int fails = runProbe(dev, layout, "csProbeWriteDebugInstancesArgs", &a, sizeof(a), f);
        printf("WriteDebugInstancesArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- DragUpdateArgsGpu (src/shaders/drag.slang, Phase 5C) ----
    {
        DragUpdateArgsGpu a{};
        fillFullBody(a.b, 0);
        fillFullJoint(a.j, 300);
        a.slot0 = 0x5A01;
        a.count = 0x5A02;
        a.bodies = ptrSentinel(703);
        a.targets = ptrSentinel(704);
        a.locals = ptrSentinel(705);
        a.stiffness = ptrSentinel(706);
        a.grab = 1;

        std::vector<FieldSpec> f;
        f.push_back({"b.pos", true, ptrSentinel(0)});
        f.push_back({"b.count", false, 0x5001u});
        f.push_back({"j.bodyA", true, ptrSentinel(300)});
        f.push_back({"j.bodyB", true, ptrSentinel(301)});
        f.push_back({"j.rA", true, ptrSentinel(302)});
        f.push_back({"j.count", false, 0x5211u});
        f.push_back({"slot0", false, (uint64_t)0x5A01u});
        f.push_back({"count", false, (uint64_t)0x5A02u});
        f.push_back({"bodies", true, ptrSentinel(703)});
        f.push_back({"targets", true, ptrSentinel(704)});
        f.push_back({"locals", true, ptrSentinel(705)});
        f.push_back({"stiffness", true, ptrSentinel(706)});
        f.push_back({"grab", false, (uint64_t)1u});

        int fails = runProbe(dev, layout, "csProbeDragUpdateArgs", &a, sizeof(a), f);
        printf("DragUpdateArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- DragReleaseArgsGpu (src/shaders/drag.slang, Phase 5C) ----
    {
        DragReleaseArgsGpu a{};
        fillFullJoint(a.j, 300);
        a.slot0 = 0x5B01;
        a.count = 0x5B02;

        std::vector<FieldSpec> f;
        f.push_back({"j.bodyA", true, ptrSentinel(300)});
        f.push_back({"j.count", false, 0x5211u});
        f.push_back({"slot0", false, (uint64_t)0x5B01u});
        f.push_back({"count", false, (uint64_t)0x5B02u});

        int fails = runProbe(dev, layout, "csProbeDragReleaseArgs", &a, sizeof(a), f);
        printf("DragReleaseArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // =====================================================================
    // Phase 5B: src/shaders/{sleep,wake,mutate}.slang args structs
    // (src/world/vk_args.h). Full-field coverage per struct, not just the
    // fields avbd_test_vk_sleep_wake's three scenarios happen to touch -- that parity
    // test proved outputs match; this proves every field's OFFSET matches, including
    // ones no scenario exercises.
    // =====================================================================

    // ---- SleepBlockFractureArgsGpu ----
    {
        SleepBlockFractureArgsGpu a{};
        fillFullBody(a.b, 0);
        fillFullJoint(a.j, 300);
        a.ratio = 0.97f;
        a.block = ptrSentinel(800);

        std::vector<FieldSpec> f;
        fullBodyFields(0, f);
        fullJointFieldsAll(300, f);
        f.push_back({"ratio", false, fbitsF(a.ratio)});
        f.push_back({"block", true, ptrSentinel(800)});

        int fails = runProbe(dev, layout, "csProbeSleepBlockFractureArgs", &a, sizeof(a), f);
        printf("SleepBlockFractureArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- SleepBlockDragArgsGpu ----
    {
        SleepBlockDragArgsGpu a{};
        a.dragIdx = ptrSentinel(801);
        a.dragCount = 0x5C01;
        a.block = ptrSentinel(802);

        std::vector<FieldSpec> f;
        f.push_back({"dragIdx", true, ptrSentinel(801)});
        f.push_back({"dragCount", false, (uint64_t)0x5C01u});
        f.push_back({"block", true, ptrSentinel(802)});

        int fails = runProbe(dev, layout, "csProbeSleepBlockDragArgs", &a, sizeof(a), f);
        printf("SleepBlockDragArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- SleepMarkArgsGpu ----
    {
        SleepMarkArgsGpu a{};
        fillFullBody(a.b, 0);
        a.dyn = ptrSentinel(803);
        a.dynCount = 0x5C02;
        a.candidate = ptrSentinel(804);
        a.frames = 0x5C03;
        a.linVel = 0.05f;
        a.angVel = 0.05f;
        a.disp = 0.01f;
        a.block = ptrSentinel(805);
        a.hasBlock = 1u;
        a.leak = 0.75f;

        std::vector<FieldSpec> f;
        fullBodyFields(0, f);
        f.push_back({"dyn", true, ptrSentinel(803)});
        f.push_back({"dynCount", false, (uint64_t)0x5C02u});
        f.push_back({"candidate", true, ptrSentinel(804)});
        f.push_back({"frames", false, (uint64_t)0x5C03u});
        f.push_back({"linVel", false, fbitsF(a.linVel)});
        f.push_back({"angVel", false, fbitsF(a.angVel)});
        f.push_back({"disp", false, fbitsF(a.disp)});
        f.push_back({"block", true, ptrSentinel(805)});
        f.push_back({"hasBlock", false, (uint64_t)1u});
        f.push_back({"leak", false, fbitsF(a.leak)});

        int fails = runProbe(dev, layout, "csProbeSleepMarkArgs", &a, sizeof(a), f);
        printf("SleepMarkArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails, sizeof(a));
    }

    // ---- SleepCommitArgsGpu ----
    {
        SleepCommitArgsGpu a{};
        fillFullBody(a.b, 0);
        a.g.key = ptrSentinel(100);
        a.g.isA = ptrSentinel(101);
        a.g.start = ptrSentinel(102);
        a.g.end = ptrSentinel(103);
        fillFullManifold(a.m, 200);
        fillFullJoint(a.j, 300);
        fillFullSpring(a.sp, 400);
        a.active = ptrSentinel(806);
        a.activeCount = 0x5C04;
        a.candidate = ptrSentinel(807);
        a.touchCount = ptrSentinel(808);
        a.touchHash = ptrSentinel(809);
        a.snapCount = ptrSentinel(810);
        a.snapHash = ptrSentinel(811);
        a.activeCountPtr = ptrSentinel(812);

        std::vector<FieldSpec> f;
        fullBodyFields(0, f);
        f.push_back({"g.key", true, ptrSentinel(100)});
        f.push_back({"g.isA", true, ptrSentinel(101)});
        f.push_back({"g.start", true, ptrSentinel(102)});
        f.push_back({"g.end", true, ptrSentinel(103)});
        fullManifoldFieldsAll(200, f);
        fullJointFieldsAll(300, f);
        fullSpringFieldsAll(400, f);
        f.push_back({"active", true, ptrSentinel(806)});
        f.push_back({"activeCount", false, (uint64_t)0x5C04u});
        f.push_back({"candidate", true, ptrSentinel(807)});
        f.push_back({"touchCount", true, ptrSentinel(808)});
        f.push_back({"touchHash", true, ptrSentinel(809)});
        f.push_back({"snapCount", true, ptrSentinel(810)});
        f.push_back({"snapHash", true, ptrSentinel(811)});
        f.push_back({"activeCountPtr", true, ptrSentinel(812)});

        int fails = runProbe(dev, layout, "csProbeSleepCommitArgs", &a, sizeof(a), f);
        printf("SleepCommitArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- ActiveFlagsArgsGpu ----
    {
        ActiveFlagsArgsGpu a{};
        fillFullBody(a.b, 0);
        a.dyn = ptrSentinel(812);
        a.dynCount = 0x5C05;
        a.flags = ptrSentinel(813);

        std::vector<FieldSpec> f;
        fullBodyFields(0, f);
        f.push_back({"dyn", true, ptrSentinel(812)});
        f.push_back({"dynCount", false, (uint64_t)0x5C05u});
        f.push_back({"flags", true, ptrSentinel(813)});

        int fails = runProbe(dev, layout, "csProbeActiveFlagsArgs", &a, sizeof(a), f);
        printf("ActiveFlagsArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- WakeRequestArgsGpu ----
    {
        WakeRequestArgsGpu a{};
        fillFullBody(a.b, 0);
        fillFullManifold(a.m, 200);
        a.slots = 0x5D01;
        a.wakeLinVel = 0.05f;
        a.wakeAngVel = 0.05f;
        a.request = ptrSentinel(814);

        std::vector<FieldSpec> f;
        fullBodyFields(0, f);
        fullManifoldFieldsAll(200, f);
        f.push_back({"slots", false, (uint64_t)0x5D01u});
        f.push_back({"wakeLinVel", false, fbitsF(a.wakeLinVel)});
        f.push_back({"wakeAngVel", false, fbitsF(a.wakeAngVel)});
        f.push_back({"request", true, ptrSentinel(814)});

        int fails = runProbe(dev, layout, "csProbeWakeRequestArgs", &a, sizeof(a), f);
        printf("WakeRequestArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- TouchClearArgsGpu ----
    {
        TouchClearArgsGpu a{};
        a.dyn = ptrSentinel(815);
        a.dynCount = 0x5D02;
        a.count = ptrSentinel(816);
        a.hash = ptrSentinel(817);

        std::vector<FieldSpec> f;
        f.push_back({"dyn", true, ptrSentinel(815)});
        f.push_back({"dynCount", false, (uint64_t)0x5D02u});
        f.push_back({"count", true, ptrSentinel(816)});
        f.push_back({"hash", true, ptrSentinel(817)});

        int fails = runProbe(dev, layout, "csProbeTouchClearArgs", &a, sizeof(a), f);
        printf("TouchClearArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- TouchAccumArgsGpu ----
    {
        TouchAccumArgsGpu a{};
        fillFullBody(a.b, 0);
        fillFullManifold(a.m, 200);
        a.slots = 0x5D03;
        a.count = ptrSentinel(818);
        a.hash = ptrSentinel(819);

        std::vector<FieldSpec> f;
        fullBodyFields(0, f);
        fullManifoldFieldsAll(200, f);
        f.push_back({"slots", false, (uint64_t)0x5D03u});
        f.push_back({"count", true, ptrSentinel(818)});
        f.push_back({"hash", true, ptrSentinel(819)});

        int fails = runProbe(dev, layout, "csProbeTouchAccumArgs", &a, sizeof(a), f);
        printf("TouchAccumArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- WakeLostSupportArgsGpu ----
    {
        WakeLostSupportArgsGpu a{};
        fillFullBody(a.b, 0);
        a.dyn = ptrSentinel(820);
        a.dynCount = 0x5D04;
        a.count = ptrSentinel(821);
        a.hash = ptrSentinel(822);
        a.snapCount = ptrSentinel(823);
        a.snapHash = ptrSentinel(824);
        a.request = ptrSentinel(825);

        std::vector<FieldSpec> f;
        fullBodyFields(0, f);
        f.push_back({"dyn", true, ptrSentinel(820)});
        f.push_back({"dynCount", false, (uint64_t)0x5D04u});
        f.push_back({"count", true, ptrSentinel(821)});
        f.push_back({"hash", true, ptrSentinel(822)});
        f.push_back({"snapCount", true, ptrSentinel(823)});
        f.push_back({"snapHash", true, ptrSentinel(824)});
        f.push_back({"request", true, ptrSentinel(825)});

        int fails = runProbe(dev, layout, "csProbeWakeLostSupportArgs", &a, sizeof(a), f);
        printf("WakeLostSupportArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- WakeApplyArgsGpu ----
    {
        WakeApplyArgsGpu a{};
        fillFullBody(a.b, 0);
        a.dyn = ptrSentinel(826);
        a.dynCount = 0x5D05;
        a.request = ptrSentinel(827);

        std::vector<FieldSpec> f;
        fullBodyFields(0, f);
        f.push_back({"dyn", true, ptrSentinel(826)});
        f.push_back({"dynCount", false, (uint64_t)0x5D05u});
        f.push_back({"request", true, ptrSentinel(827)});

        int fails = runProbe(dev, layout, "csProbeWakeApplyArgs", &a, sizeof(a), f);
        printf("WakeApplyArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails, sizeof(a));
    }

    // ---- TouchAccumJointsArgsGpu ----
    {
        TouchAccumJointsArgsGpu a{};
        fillFullBody(a.b, 0);
        fillFullJoint(a.j, 300);
        fillFullSpring(a.sp, 400);
        a.count = ptrSentinel(828);
        a.hash = ptrSentinel(829);

        std::vector<FieldSpec> f;
        fullBodyFields(0, f);
        fullJointFieldsAll(300, f);
        fullSpringFieldsAll(400, f);
        f.push_back({"count", true, ptrSentinel(828)});
        f.push_back({"hash", true, ptrSentinel(829)});

        int fails = runProbe(dev, layout, "csProbeTouchAccumJointsArgs", &a, sizeof(a), f);
        printf("TouchAccumJointsArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- WakeRequestJointsArgsGpu ----
    {
        WakeRequestJointsArgsGpu a{};
        fillFullBody(a.b, 0);
        fillFullJoint(a.j, 300);
        fillFullSpring(a.sp, 400);
        a.request = ptrSentinel(830);

        std::vector<FieldSpec> f;
        fullBodyFields(0, f);
        fullJointFieldsAll(300, f);
        fullSpringFieldsAll(400, f);
        f.push_back({"request", true, ptrSentinel(830)});

        int fails = runProbe(dev, layout, "csProbeWakeRequestJointsArgs", &a, sizeof(a), f);
        printf("WakeRequestJointsArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- WakeIslandSeedArgsGpu ----
    {
        WakeIslandSeedArgsGpu a{};
        fillFullBody(a.b, 0);
        a.dyn = ptrSentinel(831);
        a.dynCount = 0x5D06;
        a.request = ptrSentinel(832);
        a.seed = ptrSentinel(833);
        a.any = ptrSentinel(834);

        std::vector<FieldSpec> f;
        fullBodyFields(0, f);
        f.push_back({"dyn", true, ptrSentinel(831)});
        f.push_back({"dynCount", false, (uint64_t)0x5D06u});
        f.push_back({"request", true, ptrSentinel(832)});
        f.push_back({"seed", true, ptrSentinel(833)});
        f.push_back({"any", true, ptrSentinel(834)});

        int fails = runProbe(dev, layout, "csProbeWakeIslandSeedArgs", &a, sizeof(a), f);
        printf("WakeIslandSeedArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- WakeAllArgsGpu ----
    {
        WakeAllArgsGpu a{};
        fillFullBody(a.b, 0);
        a.dyn = ptrSentinel(835);
        a.dynCount = 0x5D07;

        std::vector<FieldSpec> f;
        fullBodyFields(0, f);
        f.push_back({"dyn", true, ptrSentinel(835)});
        f.push_back({"dynCount", false, (uint64_t)0x5D07u});

        int fails = runProbe(dev, layout, "csProbeWakeAllArgs", &a, sizeof(a), f);
        printf("WakeAllArgsGpu : %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails, sizeof(a));
    }

    // ---- MutateApplyArgsGpu ----
    {
        MutateApplyArgsGpu a{};
        fillFullBody(a.b, 0);
        a.type = 0x5E01;
        a.idx = ptrSentinel(836);
        a.data = ptrSentinel(837);
        a.count = 0x5E02;
        a.dt = 1.0f / 60.0f;
        a.seed = ptrSentinel(838);

        std::vector<FieldSpec> f;
        fullBodyFields(0, f);
        f.push_back({"type", false, (uint64_t)0x5E01u});
        f.push_back({"idx", true, ptrSentinel(836)});
        f.push_back({"data", true, ptrSentinel(837)});
        f.push_back({"count", false, (uint64_t)0x5E02u});
        f.push_back({"dt", false, fbitsF(a.dt)});
        f.push_back({"seed", true, ptrSentinel(838)});

        int fails = runProbe(dev, layout, "csProbeMutateApplyArgs", &a, sizeof(a), f);
        printf("MutateApplyArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- WakeSeedRegionArgsGpu ----
    {
        WakeSeedRegionArgsGpu a{};
        fillFullBody(a.b, 0);
        a.dyn = ptrSentinel(839);
        a.dynCount = 0x5E03;
        a.loX = -1.0f;
        a.loY = -2.0f;
        a.loZ = -3.0f;
        a.hiX = 1.0f;
        a.hiY = 2.0f;
        a.hiZ = 3.0f;
        a.seed = ptrSentinel(840);
        a.changed = ptrSentinel(841);

        std::vector<FieldSpec> f;
        fullBodyFields(0, f);
        f.push_back({"dyn", true, ptrSentinel(839)});
        f.push_back({"dynCount", false, (uint64_t)0x5E03u});
        f.push_back({"loX", false, fbitsF(a.loX)});
        f.push_back({"loY", false, fbitsF(a.loY)});
        f.push_back({"loZ", false, fbitsF(a.loZ)});
        f.push_back({"hiX", false, fbitsF(a.hiX)});
        f.push_back({"hiY", false, fbitsF(a.hiY)});
        f.push_back({"hiZ", false, fbitsF(a.hiZ)});
        f.push_back({"seed", true, ptrSentinel(840)});
        f.push_back({"changed", true, ptrSentinel(841)});

        int fails = runProbe(dev, layout, "csProbeWakeSeedRegionArgs", &a, sizeof(a), f);
        printf("WakeSeedRegionArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- WakeFloodContactsArgsGpu ----
    {
        WakeFloodContactsArgsGpu a{};
        fillFullBody(a.b, 0);
        fillFullManifold(a.m, 200);
        a.slots = 0x5E04;
        a.seed = ptrSentinel(842);
        a.changed = ptrSentinel(843);

        std::vector<FieldSpec> f;
        fullBodyFields(0, f);
        fullManifoldFieldsAll(200, f);
        f.push_back({"slots", false, (uint64_t)0x5E04u});
        f.push_back({"seed", true, ptrSentinel(842)});
        f.push_back({"changed", true, ptrSentinel(843)});

        int fails = runProbe(dev, layout, "csProbeWakeFloodContactsArgs", &a, sizeof(a), f);
        printf("WakeFloodContactsArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- WakeFloodJointsArgsGpu ----
    {
        WakeFloodJointsArgsGpu a{};
        fillFullBody(a.b, 0);
        fillFullJoint(a.j, 300);
        fillFullSpring(a.sp, 400);
        a.seed = ptrSentinel(844);
        a.changed = ptrSentinel(845);

        std::vector<FieldSpec> f;
        fullBodyFields(0, f);
        fullJointFieldsAll(300, f);
        fullSpringFieldsAll(400, f);
        f.push_back({"seed", true, ptrSentinel(844)});
        f.push_back({"changed", true, ptrSentinel(845)});

        int fails = runProbe(dev, layout, "csProbeWakeFloodJointsArgs", &a, sizeof(a), f);
        printf("WakeFloodJointsArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- WakeSeedMergeArgsGpu ----
    {
        WakeSeedMergeArgsGpu a{};
        a.dyn = ptrSentinel(846);
        a.dynCount = 0x5E05;
        a.seed = ptrSentinel(847);
        a.request = ptrSentinel(848);

        std::vector<FieldSpec> f;
        f.push_back({"dyn", true, ptrSentinel(846)});
        f.push_back({"dynCount", false, (uint64_t)0x5E05u});
        f.push_back({"seed", true, ptrSentinel(847)});
        f.push_back({"request", true, ptrSentinel(848)});

        int fails = runProbe(dev, layout, "csProbeWakeSeedMergeArgs", &a, sizeof(a), f);
        printf("WakeSeedMergeArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- BodyGatherArgsGpu ----
    {
        BodyGatherArgsGpu a{};
        fillFullBody(a.b, 0);
        a.type = 0x5E06;
        a.idx = ptrSentinel(849);
        a.out = ptrSentinel(850);
        a.count = 0x5E07;

        std::vector<FieldSpec> f;
        fullBodyFields(0, f);
        f.push_back({"type", false, (uint64_t)0x5E06u});
        f.push_back({"idx", true, ptrSentinel(849)});
        f.push_back({"out", true, ptrSentinel(850)});
        f.push_back({"count", false, (uint64_t)0x5E07u});

        int fails = runProbe(dev, layout, "csProbeBodyGatherArgs", &a, sizeof(a), f);
        printf("BodyGatherArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- GateDispatchArgsGpu ----
    {
        GateDispatchArgsGpu a{};
        a.src = ptrSentinel(851);
        a.dst = ptrSentinel(852);
        a.flag = ptrSentinel(853);
        a.count = 0x5E08;

        std::vector<FieldSpec> f;
        f.push_back({"src", true, ptrSentinel(851)});
        f.push_back({"dst", true, ptrSentinel(852)});
        f.push_back({"flag", true, ptrSentinel(853)});
        f.push_back({"count", false, (uint64_t)0x5E08u});

        int fails = runProbe(dev, layout, "csProbeGateDispatchArgs", &a, sizeof(a), f);
        printf("GateDispatchArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- StepFlagIfSetArgsGpu ----
    {
        StepFlagIfSetArgsGpu a{};
        a.src = ptrSentinel(854);
        a.flag = ptrSentinel(855);

        std::vector<FieldSpec> f;
        f.push_back({"src", true, ptrSentinel(854)});
        f.push_back({"flag", true, ptrSentinel(855)});

        int fails = runProbe(dev, layout, "csProbeStepFlagIfSet", &a, sizeof(a), f);
        printf("StepFlagIfSetArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- StepFlagIfGreaterArgsGpu ----
    {
        StepFlagIfGreaterArgsGpu a{};
        a.count = ptrSentinel(859);
        a.flag = ptrSentinel(860);
        a.cap = 0x5E0B;

        std::vector<FieldSpec> f;
        f.push_back({"count", true, ptrSentinel(859)});
        f.push_back({"flag", true, ptrSentinel(860)});
        f.push_back({"cap", false, (uint64_t)0x5E0Bu});

        int fails = runProbe(dev, layout, "csProbeStepFlagIfGreater", &a, sizeof(a), f);
        printf("StepFlagIfGreaterArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    // ---- StepColorBoundArgsGpu ----
    {
        StepColorBoundArgsGpu a{};
        a.start = ptrSentinel(856);
        a.end = ptrSentinel(857);
        a.flag = ptrSentinel(858);
        a.bound = 0x5E09;
        a.maxColors = 0x5E0A;

        std::vector<FieldSpec> f;
        f.push_back({"start", true, ptrSentinel(856)});
        f.push_back({"end", true, ptrSentinel(857)});
        f.push_back({"flag", true, ptrSentinel(858)});
        f.push_back({"bound", false, (uint64_t)0x5E09u});
        f.push_back({"maxColors", false, (uint64_t)0x5E0Au});

        int fails = runProbe(dev, layout, "csProbeStepColorBound", &a, sizeof(a), f);
        printf("StepColorBoundArgsGpu: %s (%d field mismatch(es), sizeof=%zu)\n", fails ? "FAIL" : "PASS", fails,
               sizeof(a));
    }

    layout.destroy();

    if (g_fail)
    {
        printf("%d field mismatch(es) total\n", g_fail);
        printf("FAIL\n");
        return 1;
    }
    printf("All probed structs match types.slang field-for-field through an actual GPU dispatch.\n");
    printf("PASS\n");
    return 0;
}
