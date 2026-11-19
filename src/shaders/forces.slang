// The persistent-force (joint/spring) stamp and dual update, split by kind: joint_weld.slang,
// joint_generic.slang and spring.slang. Kernels import this umbrella and call
// jointInitialize/stampJoint/jointDual and the spring/cloth entry points.

module forces;

__exported import joint_weld;
__exported import joint_generic;
__exported import spring;
