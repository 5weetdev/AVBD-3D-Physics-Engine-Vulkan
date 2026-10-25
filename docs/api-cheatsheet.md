# API cheatsheet

**What you get:** the calls grouped by task. Exact signatures are in `include/avbd3d/*.h`. Every name starts with `avbd3d_`, and every call returns `Avbd3dResult` except `abi_version`, `result_string` and the `default_*` fillers.

## Library and world

| Task | Calls |
| :--- | :--- |
| Version, errors | `abi_version`, `result_string` |
| Device info, counters | `get_capabilities`, `get_stats` |
| Lifetime | `default_world_def`, `create_world`, `destroy_world` |
| Stepping | `step`, `step_begin`, `step_end` |
| Gravity, switches | `world_set_gravity`, `world_enable_sleeping`, `world_enable_continuous` |
| Solver settings | `default_solver_def`, `world_set_solver_def` |
| Misc | `world_explode`, `world_draw` |

## Bodies

| Task | Calls |
| :--- | :--- |
| Lifetime | `default_body_def`, `create_body`, `destroy_body` |
| Pose | `body_set_transform`, `body_get_position`, `body_get_rotation`, `body_set_target_transform` |
| Velocity | `body_set_linear_velocity`, `body_set_angular_velocity`, `body_get_linear_velocity`, `body_get_angular_velocity` |
| Forces | `body_apply_force`, `body_apply_torque`, `body_apply_linear_impulse`, `body_apply_angular_impulse` |
| Mass | `body_get_mass`, `body_get_mass_data`, `body_set_mass_data`, `body_apply_mass_from_shapes` |
| Settings | `body_set_type`, `body_set_linear_damping`, `body_set_angular_damping`, `body_set_gravity_scale`, `body_set_bullet`, `body_set_motion_locks`, `body_set_awake` |
| Topology | `body_get_shapes`, `body_get_joints`, `body_get_contact_data` |
| Queries on one body | `body_cast_ray`, `body_cast_shape`, `body_overlap_shape`, `body_get_closest_point` |
| Batched read or write | `world_get_body_data`, `world_set_body_data` |

## Shapes

| Task | Calls |
| :--- | :--- |
| Create | `default_shape_def`, `create_sphere_shape`, `create_capsule_shape`, `create_hull_shape`, `create_mesh_shape`, `create_height_field_shape`, `create_baked_compound_shape` |
| Destroy | `destroy_shape` |
| Material, filter | `shape_set_density`, `shape_set_friction`, `shape_set_restitution`, `shape_set_surface_material`, `shape_set_filter` |
| Events | `shape_enable_contact_events`, `shape_enable_hit_events`, `shape_enable_sensor_events` |
| Read | `shape_get_type`, `shape_get_body`, `shape_get_aabb`, `shape_ray_cast` |

## Joints

| Task | Calls |
| :--- | :--- |
| Create | `create_distance_joint`, `create_motor_joint`, `create_prismatic_joint`, `create_revolute_joint`, `create_spherical_joint`, `create_weld_joint`, `create_wheel_joint`, `create_parallel_joint`, `create_filter_joint`, `create_generic_joint`, `create_spring` |
| Destroy | `destroy_joint` (with `wake_bodies`) |
| Common | `joint_get_type`, `joint_get_body_a`, `joint_get_body_b`, `joint_get_constraint_force`, `joint_get_constraint_torque`, `joint_set_force_threshold`, `joint_set_torque_threshold` |

Per-type calls are `<type>_joint_*`. See [shapes and joints](shapes-and-joints.md).

## Queries

| Task | Calls |
| :--- | :--- |
| Ray | `world_cast_ray`, `world_cast_ray_closest` |
| Sweep | `world_cast_shape` |
| Overlap | `world_overlap_aabb`, `world_overlap_shape` |
| Character mover | `world_cast_mover`, `world_collide_mover`, `solve_planes`, `clip_vector` |

## Events

| Task | Calls |
| :--- | :--- |
| Bodies that moved | `world_get_body_events` |
| Contacts: begin, end, hit | `world_get_contact_events` |
| Sensors | `world_get_sensor_events` |
| Joints over threshold or broken | `world_get_joint_events` |

## Grab

| Task | Calls |
| :--- | :--- |
| Mouse-style drag | `grab_begin`, `grab_move`, `grab_end`, `grab_survivors` |

## Vulkan and rendering (`avbd3d_vulkan.h`)

| Task | Calls |
| :--- | :--- |
| Shared device | `default_gpu_def`, `create_gpu`, `destroy_gpu`, `create_world_on_gpu` |
| Handles for your renderer | `gpu_get_vulkan_context` |
| Render groups | `body_set_render_group`, `world_get_render_group_count` |
| Instance records | `world_write_instances`, `world_get_instance_buffer` |
| Contact markers | `world_write_contact_markers`, `world_get_contact_marker_buffer` |
| Validation | `get_validation_message_count` |

## Geometry (`avbd3d_geometry.h`)

| Task | Calls |
| :--- | :--- |
| Hull | `create_hull`, `destroy_hull`, `make_box_hull`, `box_hull_data` |
| Mesh | `create_mesh`, `destroy_mesh` |
| Height field | `create_height_field`, `destroy_height_field` |
| Compound | `create_compound`, `destroy_compound` |
