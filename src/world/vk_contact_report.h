// vk_contact_report.h -- the contact-report pass (contact_report.slang, csContactReport) and the
// restitution passes (restitution.slang).
// Owned by VkWorld; the api3d layer reads the records through VkWorld::contactReportView().
#pragma once

#include "vk_args.h"
#include "vk_buffer.h"
#include "vk_cmd.h"
#include "vk_device.h"
#include "vk_pipeline.h"

namespace avbdvk
{

struct ContactReportPipelines
{
    Pipeline report;

    void init(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t spirvWords);
    void destroy();
};

// One dispatch over the manifold slots. Records are appended at most `args.capacity` deep.
void launchContactReport(CommandList &cmd, HostRingBuffer &ring, ContactReportPipelines &pl,
                         const ContactReportArgsGpu &args);

// Restitution (restitution.slang): recorded only while some body has restitution > 0.
struct RestitutionPipelines
{
    Pipeline pre, impulse, apply;

    void init(Device &dev, PipelineLayout &layout, const uint32_t *spirv, size_t spirvWords);
    void destroy();
};

// csRestitutionPre / csRestitutionImpulse over args.slots, csRestitutionApply over args.dynCount.
void launchRestitutionPre(CommandList &cmd, HostRingBuffer &ring, RestitutionPipelines &pl,
                          const RestitutionArgsGpu &args);
void launchRestitutionImpulse(CommandList &cmd, HostRingBuffer &ring, RestitutionPipelines &pl,
                              const RestitutionArgsGpu &args);
void launchRestitutionApply(CommandList &cmd, HostRingBuffer &ring, RestitutionPipelines &pl,
                            const RestitutionArgsGpu &args);

} // namespace avbdvk
