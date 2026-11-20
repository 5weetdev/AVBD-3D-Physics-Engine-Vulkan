cmake_minimum_required(VERSION 3.20)

project(avbd3d VERSION 1.0.0 LANGUAGES C CXX)

option(AVBD3D_BUILD_DEMO     "Build the SDL2 + ImGui demo" ON)
option(AVBD3D_BUILD_TESTS    "Build the tests and register them with CTest" ON)
option(AVBD3D_BUILD_EXAMPLES "Build examples/hello.c" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    message(FATAL_ERROR "avbd3d needs desktop Vulkan; there is no WebAssembly target.")
endif()

# Executables and the shared library land in the build folder (build/release, build/debug).
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Release builds carry a PDB so a crash can be traced to a function and line. /Z7 puts the
# debug info in each object (no shared compile-time PDB under parallel Ninja); /OPT:REF,ICF
# restore what /DEBUG turns off. Code generation is unchanged.
if(MSVC)
    add_compile_options($<$<CONFIG:Release>:/Z7>)
    add_link_options($<$<CONFIG:Release>:/DEBUG> $<$<CONFIG:Release>:/OPT:REF> $<$<CONFIG:Release>:/OPT:ICF>)
endif()

# ---------------------------------------------------------------------------
# Toolchain: the Vulkan SDK (loader + slangc) and Python (SPIR-V embedding, checkers).
# ---------------------------------------------------------------------------
find_package(Vulkan QUIET)
if(NOT Vulkan_FOUND AND DEFINED ENV{VULKAN_SDK})
    find_package(Vulkan QUIET HINTS "$ENV{VULKAN_SDK}")
endif()
if(NOT Vulkan_FOUND)
    find_package(Vulkan QUIET HINTS "C:/VulkanSDK/1.4.357.0")
endif()
if(NOT Vulkan_FOUND)
    message(FATAL_ERROR "The Vulkan SDK was not found. "
        "Set VULKAN_SDK or pass -DVulkan_ROOT=<path to SDK>.")
endif()

# slangc ships in the SDK's Bin directory next to the Vulkan loader; Vulkan::Vulkan does not
# expose it, so it is found by hand.
get_filename_component(AVBD_VULKAN_SDK_BIN "${Vulkan_INCLUDE_DIR}/../Bin" ABSOLUTE)
find_program(AVBD_SLANGC slangc
    HINTS "${AVBD_VULKAN_SDK_BIN}" "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin" "C:/VulkanSDK/1.4.357.0/Bin")
if(NOT AVBD_SLANGC)
    message(FATAL_ERROR "slangc was not found next to the Vulkan SDK.")
endif()
message(STATUS "avbd3d Vulkan: SDK ${Vulkan_INCLUDE_DIR}, slangc ${AVBD_SLANGC}")

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(AVBD_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ---------------------------------------------------------------------------
# Compute shaders: <dir>/<name>.slang -> SPIR-V -> an embedded C array
# (<gen>/<name>_spirv.h exposing `<name>_spirv` / `<name>_spirv_words`).
#
# Modules import each other and slangc has no depfile output, so every SPIR-V depends on
# every .slang under its directory: an edit to an imported module must recompile its
# importers, or the binary silently keeps the old code.
# ---------------------------------------------------------------------------
function(avbd_compile_slang dir gen name out_header)
    file(GLOB_RECURSE all_slang CONFIGURE_DEPENDS ${dir}/*.slang)
    set(src ${dir}/${name}.slang)
    set(spv ${gen}/${name}.spv)
    set(hdr ${gen}/${name}_spirv.h)

    add_custom_command(
        OUTPUT ${spv}
        COMMAND ${AVBD_SLANGC} ${src}
                -target spirv -profile spirv_1_5
                -fvk-use-scalar-layout
                -fvk-use-entrypoint-name
                -emit-spirv-directly
                -O2
                -I ${dir}
                -o ${spv}
        DEPENDS ${all_slang}
        COMMENT "slangc: ${name}.slang -> SPIR-V"
        VERBATIM)

    add_custom_command(
        OUTPUT ${hdr}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spirv.py
                ${spv} ${hdr} ${name}_spirv
        DEPENDS ${spv} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spirv.py
        COMMENT "embed: ${name}_spirv.h"
        VERBATIM)

    set(${out_header} ${hdr} PARENT_SCOPE)
endfunction()

set(AVBD_SHADER_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders_gen)
file(MAKE_DIRECTORY ${AVBD_SHADER_GEN_DIR})
set(AVBD_SPIRV_HEADERS)
foreach(module
        integrate collide_test scan compact radix sort_small layout_probe broadphase narrowphase
        constraints primal step_control coarse dual render render_overlay render_debug drag sleep wake mutate narrowphase_hull
        narrowphase_mesh contact_report restitution ccd)
    avbd_compile_slang(${AVBD_SRC}/shaders ${AVBD_SHADER_GEN_DIR} ${module} hdr)
    list(APPEND AVBD_SPIRV_HEADERS ${hdr})
endforeach()

# ---------------------------------------------------------------------------
# avbd3d_box3d -- Box3D's host code (hull/mesh/height-field/compound builders, mass, dynamic
# tree, casts, GJK/TOI, mover, id pool), compiled unmodified from external/box3d (see its
# VENDOR.md). Internal b3 symbols get hidden visibility so a game that also links real Box3D
# does not clash. The whole src/ set is compiled; a static library only pulls in what is
# referenced, and narrowing the glob does not link (mesh.c and height_field.c need shape.c,
# which needs the body, broad phase, contact and id-pool sources).
# ---------------------------------------------------------------------------
file(GLOB AVBD3D_BOX3D_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/external/box3d/src/*.c)
add_library(avbd3d_box3d STATIC ${AVBD3D_BOX3D_SRC})
target_include_directories(avbd3d_box3d
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/external/box3d/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/box3d/src)
set_target_properties(avbd3d_box3d PROPERTIES
    C_STANDARD 17 C_STANDARD_REQUIRED YES C_EXTENSIONS YES
    C_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    target_compile_options(avbd3d_box3d PRIVATE /W3 /wd4820 /wd5045 /wd4061 /wd4711 /wd4710)
endif()
if(UNIX AND NOT APPLE)
    target_link_libraries(avbd3d_box3d PUBLIC m)
endif()

# ---------------------------------------------------------------------------
# avbd3d_core_static -- everything below the C API: scene authoring, the Vulkan layer
# (src/gpu), VkWorld and its phases (src/world) and the embedded kernels. Not installed.
# The shared library and the tests that look at solver internals both use it.
# src/ never sees demo/: that directory is on no src/ target's include path.
# ---------------------------------------------------------------------------
add_library(avbd3d_core_static STATIC
    src/scene/scene_builder.cpp
    src/scene/shape_mass_b3.cpp
    src/gpu/vk_device.cpp
    src/gpu/vk_buffer.cpp
    src/gpu/vk_pipeline.cpp
    src/gpu/vk_cmd.cpp
    src/gpu/vk_primitives.cpp
    src/gpu/vk_gate.cpp
    src/world/vk_broadphase.cpp
    src/world/vk_narrowphase.cpp
    src/world/vk_constraints.cpp
    src/world/vk_solver.cpp
    src/world/vk_integrate.cpp
    src/world/vk_render.cpp
    src/world/vk_interact.cpp
    src/world/vk_sleep.cpp
    src/world/vk_contact_report.cpp
    src/world/vk_ccd.cpp
    src/world/vk_world.cpp
    src/world/vk_world_upload.cpp
    src/world/vk_world_snapshot.cpp
    src/world/vk_world_step.cpp
    src/world/vk_world_finish.cpp
    src/world/vk_world_render.cpp
    src/world/vk_world_mutate.cpp
    src/world/vk_world_readback.cpp
    ${AVBD_SPIRV_HEADERS})
target_include_directories(avbd3d_core_static PUBLIC
    ${AVBD_SRC}/core ${AVBD_SRC}/scene ${AVBD_SRC}/gpu ${AVBD_SRC}/world ${AVBD_SHADER_GEN_DIR})
target_link_libraries(avbd3d_core_static PUBLIC Vulkan::Vulkan avbd3d_box3d)
set_target_properties(avbd3d_core_static PROPERTIES
    POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)

# ---------------------------------------------------------------------------
# avbd3d -- the public C API, one shared library. include/avbd3d/*.h are the contract; only
# AVBD3D_API symbols are exported. Consumers get AVBD3D_USE_SHARED (dllimport) from the
# interface definition.
# ---------------------------------------------------------------------------
add_library(avbd3d SHARED
    src/api/avbd3d_world.cpp src/api/avbd3d_api.cpp src/api/avbd3d_query.cpp
    src/api/avbd3d_collision.cpp src/api/avbd3d_props.cpp src/api/avbd3d_interact.cpp
    src/api/avbd3d_vulkan.cpp src/api/avbd3d_capi_gen.cpp src/api/avbd3d_capi_special.cpp
    src/api/avbd3d_geometry.cpp)
target_include_directories(avbd3d
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
    PRIVATE ${AVBD_SRC}/api)
target_link_libraries(avbd3d PRIVATE avbd3d_core_static)
target_compile_definitions(avbd3d PRIVATE AVBD3D_BUILD_SHARED INTERFACE AVBD3D_USE_SHARED)
set_target_properties(avbd3d PROPERTIES
    VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR}
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)

# ---------------------------------------------------------------------------
# Install: the library and its headers, and the CMake package avbd3d::avbd3d.
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS avbd3d EXPORT avbd3dTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/avbd3d DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h")
install(EXPORT avbd3dTargets NAMESPACE avbd3d:: FILE avbd3dTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/avbd3d)
configure_package_config_file(cmake/avbd3dConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/avbd3dConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/avbd3d)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/avbd3dConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/avbd3dConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/avbd3dConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/avbd3d)

# ---------------------------------------------------------------------------
# Example: a C99 program that uses only the public headers.
# ---------------------------------------------------------------------------
if(AVBD3D_BUILD_EXAMPLES)
    add_executable(avbd3d_hello examples/hello.c)
    set_target_properties(avbd3d_hello PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
    target_link_libraries(avbd3d_hello PRIVATE avbd3d)
endif()

# ---------------------------------------------------------------------------
# Demo: SDL2 + ImGui (vendored submodules). No library target links either.
# ---------------------------------------------------------------------------
if(AVBD3D_BUILD_DEMO)
    add_subdirectory(external/SDL)

    file(GLOB IMGUI_SRC
        external/imgui/*.cpp
        external/imgui/backends/imgui_impl_sdl2.cpp)
    add_library(imgui STATIC ${IMGUI_SRC})
    target_include_directories(imgui PUBLIC
        external/imgui
        external/imgui/backends
        external/SDL/include)
    target_link_libraries(imgui PUBLIC SDL2::SDL2 SDL2::SDL2main)

    set(AVBD_DEMO_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/demo_shaders_gen)
    file(MAKE_DIRECTORY ${AVBD_DEMO_GEN_DIR})
    avbd_compile_slang(${CMAKE_CURRENT_SOURCE_DIR}/demo/shaders ${AVBD_DEMO_GEN_DIR} scene scene_hdr)

    add_executable(avbd3d_demo
        demo/main.cpp
        demo/vk_renderer.cpp
        demo/demo_app.cpp
        demo/scene_import.cpp
        demo/demo_log.cpp
        demo/render_options.cpp
        external/imgui/backends/imgui_impl_vulkan.cpp
        ${scene_hdr})
    # The demo reaches the solver only through the public C API; it shares the scene authoring
    # headers (src/core, src/scene) with the library.
    target_include_directories(avbd3d_demo PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/demo
        ${AVBD_SRC}/core ${AVBD_SRC}/scene
        ${AVBD_DEMO_GEN_DIR})
    target_link_libraries(avbd3d_demo PRIVATE avbd3d avbd3d_core_static imgui SDL2::SDL2 SDL2::SDL2main Vulkan::Vulkan)
    if(WIN32)
        # demo/demo_log.cpp: the crash handler resolves symbols with DbgHelp.
        target_link_libraries(avbd3d_demo PRIVATE dbghelp)
    endif()
    # SDL2 is built as a shared library; avbd3d_demo needs it beside the executable.
    add_custom_command(TARGET avbd3d_demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:SDL2::SDL2> $<TARGET_FILE_DIR:avbd3d_demo>
        COMMENT "Copying SDL2 next to avbd3d_demo")
endif()

# ---------------------------------------------------------------------------
# Tests (tests/). Every test is a CTest entry named avbd3d_test_<x>.
# ---------------------------------------------------------------------------
if(AVBD3D_BUILD_TESTS)
    enable_testing()

    # Tests of solver internals link avbd3d_core_static (and see src/ headers). onesubmit and
    # overflow also use the demo's scene catalogue (demo/scenes.h).
    function(avbd3d_add_internal_test name)
        add_executable(avbd3d_test_${name} tests/${name}.cpp)
        target_link_libraries(avbd3d_test_${name} PRIVATE avbd3d_core_static)
        add_test(NAME avbd3d_test_${name} COMMAND avbd3d_test_${name})
    endfunction()

    # Host structs (vk_args.h) against what the shaders read (layout_probe.slang).
    avbd3d_add_internal_test(layout)
    # Exactly one vkQueueSubmit per non-rerun step, no allocations after warm-up.
    avbd3d_add_internal_test(onesubmit)
    target_include_directories(avbd3d_test_onesubmit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/demo)
    # Forced tiny capacities recover bit-identically; a hard cap reports a StepResult.
    avbd3d_add_internal_test(overflow)
    target_include_directories(avbd3d_test_overflow PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/demo)
    # A Vulkan failure (injected device loss) is an error result, never an abort.
    avbd3d_add_internal_test(device_error)
    target_link_libraries(avbd3d_test_device_error PRIVATE avbd3d)

    # The remaining tests use only the public C API.
    foreach(name api_world api_shapes api_joints api_events api_query api_filter_material api_sleep_wake)
        add_executable(avbd3d_test_${name} tests/${name}.cpp)
        target_link_libraries(avbd3d_test_${name} PRIVATE avbd3d)
        add_test(NAME avbd3d_test_${name} COMMAND avbd3d_test_${name})
    endforeach()

    # Every function the public headers declare has a definition in src/api/*.cpp.
    add_test(NAME avbd3d_test_abi_defined
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/abi_defined.py)

    # The public headers compile as C99 on their own, with only include/ on the path.
    add_executable(avbd3d_test_c99_headers tests/c99_headers.c)
    set_target_properties(avbd3d_test_c99_headers PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
    target_include_directories(avbd3d_test_c99_headers PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME avbd3d_test_c99_headers COMMAND avbd3d_test_c99_headers)

    if(AVBD3D_BUILD_EXAMPLES)
        add_test(NAME avbd3d_test_hello COMMAND avbd3d_hello)
    endif()
    if(AVBD3D_BUILD_DEMO)
        # The demo starts, runs frames and exits cleanly with the Khronos validation layer on.
        add_test(NAME avbd3d_test_demo_smoke COMMAND avbd3d_demo --smoke)
        set_tests_properties(avbd3d_test_demo_smoke PROPERTIES ENVIRONMENT "AVBD_VK_VALIDATION=1")
    endif()
endif()
