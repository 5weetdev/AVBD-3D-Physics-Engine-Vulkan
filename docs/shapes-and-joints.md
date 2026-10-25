# Shapes and joints

**What you get:** which shapes collide with which, and what each joint type can do.

Names below are the public calls in `include/avbd3d/*.h`. Verify exact signatures there.

## Shapes

| Shape | Create call | Static body | Dynamic body | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Sphere | `avbd3d_create_sphere_shape` | yes | yes | |
| Capsule | `avbd3d_create_capsule_shape` | yes | yes | |
| Convex hull (boxes are hulls) | `avbd3d_create_hull_shape` | yes | yes | Box hull from `avbd3d_make_box_hull` |
| Mesh | `avbd3d_create_mesh_shape` | yes | closed meshes only | Open meshes are refused on dynamic bodies |
| Height field | `avbd3d_create_height_field_shape` | yes | no | |
| Compound | `avbd3d_create_baked_compound_shape` | yes | convex children only | |

Hulls, meshes, height fields and compounds are built with `avbd3d_geometry.h`. The caller owns the geometry and keeps it alive while shapes use it.

## Collision pairs

| | Sphere | Capsule | Hull | Mesh | Height field |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sphere** | yes | yes | yes | yes | yes |
| **Capsule** | yes | yes | yes | yes | yes |
| **Hull** | yes | yes | yes | yes | yes |
| **Mesh** | yes | yes | yes | no | no |
| **Height field** | yes | yes | yes | no | no |

A compound is tested child by child, so the rows above apply to each child. Mesh against mesh is never tested (see [limitations](limitations.md)).

## Joints

| Type | Create call | Limit | Motor | Spring |
| :--- | :--- | :--- | :--- | :--- |
| Distance | `avbd3d_create_distance_joint` | length range | yes | yes |
| Motor | `avbd3d_create_motor_joint` | - | linear and angular | yes |
| Prismatic | `avbd3d_create_prismatic_joint` | translation | yes | yes |
| Revolute | `avbd3d_create_revolute_joint` | angle | yes | yes |
| Spherical | `avbd3d_create_spherical_joint` | cone, twist | 3 axes | yes |
| Weld | `avbd3d_create_weld_joint` | - | - | linear and angular |
| Wheel | `avbd3d_create_wheel_joint` | suspension, steering | spin | suspension, steering |
| Parallel | `avbd3d_create_parallel_joint` | - | - | yes |
| Filter | `avbd3d_create_filter_joint` | - | - | - |

Each type has `set`/`get`/`enable` calls named `avbd3d_<type>_joint_*`. State readout (angle, translation, speed, motor force or torque, cone and twist angle, current length) is available where the type has it.

Every joint supports `avbd3d_joint_get_constraint_force` and `avbd3d_joint_get_constraint_torque`, and force and torque thresholds that raise joint events.

## AVBD extensions

| Type | Create call | What it is |
| :--- | :--- | :--- |
| Generic | `avbd3d_create_generic_joint` | Six-axis joint. Each axis is free, locked, limited, motor or spring. Optional fracture force and torque make it breakable |
| Spring | `avbd3d_create_spring` | Linear spring with stiffness and rest length |

## Authoring rules

| Rule | Why |
| :--- | :--- |
| Anchors must coincide in world space at creation | A gap is amplified by the penalty ramp and can blow a structure apart |
| Cables use zero angular stiffness | Angular stiffness on a chain link makes it rigid |
| Crossing braces must not collide | Use `avbd3d_joint_set_collide_connected` or a filter joint |
| `avbd3d_destroy_joint(joint, wake_bodies)` | With `wake_bodies` 0 the bodies are left asleep |
