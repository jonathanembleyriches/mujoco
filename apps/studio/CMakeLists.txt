cmake_minimum_required(VERSION 3.20)
project(protospec_studio CXX)

# Standalone: builds protospec_studio.exe on its own (`cmake -S apps/studio`).
# It does NOT depend on cpp/CMakeLists.txt; like cpp/python it compiles the
# protospec object-model / IO / validate / bridge sources directly, then adds a
# vendored thin shell (SDL2 window + Dear ImGui + classic mjr renderer + step
# control + pick math) and the plugin host. MuJoCo is a prebuilt lib via
# MUJOCO_ROOT; the vendored MuJoCo src/ dir is on the include path for the pick
# math's private engine headers (same situation as cpp/harness).

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(CPP_ROOT "${REPO_ROOT}/cpp")

# --- MuJoCo (vendored prebuilt) -------------------------------------------- #
set(MUJOCO_ROOT
    "C:/Users/jonat/Documents/Unreal Projects/url_proj/Plugins/UnrealRoboticsLab/third_party/MuJoCo/src"
    CACHE PATH "Root of the vendored prebuilt MuJoCo (contains include/, src/ and build/)")
set(MUJOCO_INCLUDE_DIR "${MUJOCO_ROOT}/include")
set(MUJOCO_SRC_DIR     "${MUJOCO_ROOT}/src")
set(MUJOCO_IMPORT_LIB  "${MUJOCO_ROOT}/build/lib/Release/mujoco.lib")
set(MUJOCO_DLL         "${MUJOCO_ROOT}/build/bin/Release/mujoco.dll")

# Asset-pipeline deps prebuilt alongside MuJoCo (same static libs it links);
# the lifted mesh pipeline links qhull + tinyobjloader (as in cpp/CMakeLists).
set(MUJOCO_DEPS "${MUJOCO_ROOT}/build/_deps")
set(QHULL_INCLUDE_DIR   "${MUJOCO_DEPS}/qhull-src/src")
set(QHULL_LIB           "${MUJOCO_ROOT}/build/lib/Release/qhullstatic_r.lib")
set(TINYOBJ_INCLUDE_DIR "${MUJOCO_DEPS}/tinyobjloader-src")
set(TINYOBJ_LIB         "${MUJOCO_ROOT}/build/lib/Release/tinyobjloader.lib")
set(LODEPNG_INCLUDE_DIR "${MUJOCO_DEPS}/lodepng-src")
set(LODEPNG_LIB         "${MUJOCO_ROOT}/build/lib/Release/lodepng.lib")
if(NOT EXISTS "${MUJOCO_INCLUDE_DIR}/mujoco/mujoco.h")
  message(FATAL_ERROR "MuJoCo headers not found under ${MUJOCO_INCLUDE_DIR}; set -DMUJOCO_ROOT=...")
endif()
if(NOT EXISTS "${MUJOCO_IMPORT_LIB}")
  message(FATAL_ERROR "mujoco.lib not found at ${MUJOCO_IMPORT_LIB}; set -DMUJOCO_ROOT=...")
endif()

# --- Third-party UI deps via FetchContent ---------------------------------- #
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# SDL2 (static). Pinned to a release tag; window/context/input backend.
set(SDL_SHARED OFF CACHE BOOL "" FORCE)
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_TEST OFF CACHE BOOL "" FORCE)
set(SDL2_DISABLE_INSTALL ON CACHE BOOL "" FORCE)
FetchContent_Declare(
  SDL2
  GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
  GIT_TAG        release-2.30.9   # SDL 2.30.9
  GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(SDL2)

# Dear ImGui (docking branch) at the exact commit MuJoCo Studio pins.
FetchContent_Declare(
  dear_imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG        3109131a882daec56a530aff540416983c240443
  GIT_SHALLOW    FALSE)
FetchContent_MakeAvailable(dear_imgui)

add_library(dear_imgui STATIC
  ${dear_imgui_SOURCE_DIR}/imgui.cpp
  ${dear_imgui_SOURCE_DIR}/imgui_draw.cpp
  ${dear_imgui_SOURCE_DIR}/imgui_tables.cpp
  ${dear_imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${dear_imgui_SOURCE_DIR}/imgui_demo.cpp
  ${dear_imgui_SOURCE_DIR}/backends/imgui_impl_sdl2.cpp
  ${dear_imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
  ${dear_imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp
)
target_include_directories(dear_imgui PUBLIC
  ${dear_imgui_SOURCE_DIR}
  ${dear_imgui_SOURCE_DIR}/backends
  ${dear_imgui_SOURCE_DIR}/misc/cpp
)
target_link_libraries(dear_imgui PUBLIC SDL2::SDL2-static)

# --- ProtoSpec sources (compiled directly, mirroring cpp/python) ----------- #
file(GLOB PS_GENERATED_SRC CONFIGURE_DEPENDS "${CPP_ROOT}/generated/*.cc")
set(PROTOSPEC_SRC
  ${PS_GENERATED_SRC}
  ${CPP_ROOT}/core/resolve.cc
  ${CPP_ROOT}/io/numeric.cc
  ${CPP_ROOT}/io/include.cc
  ${CPP_ROOT}/io/defaults.cc
  ${CPP_ROOT}/io/mjcf_reader.cc
  ${CPP_ROOT}/io/mjcf_writer.cc
  ${CPP_ROOT}/validate/validate.cc
  ${CPP_ROOT}/validate/sizes.cc
  ${CPP_ROOT}/bridge/compile.cc
  ${CPP_ROOT}/bridge/binding.cc
  ${CPP_ROOT}/bridge/pose.cc
  ${CPP_ROOT}/compile/native.cc
  ${CPP_ROOT}/compile/build.cc
  ${CPP_ROOT}/compile/lifted/make_model.cc
  ${CPP_ROOT}/compile/lifted/mjuu_util.cc
  ${CPP_ROOT}/compile/lifted/mesh_pipeline.cc
  ${CPP_ROOT}/compile/lifted/image_pipeline.cc
  ${CPP_ROOT}/compile/lifted/builtin_mesh.cc
  ${CPP_ROOT}/compile/lifted/marching_cube.cc
  ${CPP_ROOT}/third_party/tinyxml2/tinyxml2.cpp
)

add_library(protospec_studio_core STATIC ${PROTOSPEC_SRC})
target_include_directories(protospec_studio_core PUBLIC
  ${CPP_ROOT}/include
  ${CPP_ROOT}/generated
  ${CPP_ROOT}/core
  ${CPP_ROOT}/io
  ${CPP_ROOT}/validate
  ${CPP_ROOT}/bridge
  ${CPP_ROOT}/compile
  ${CPP_ROOT}/compile/lifted
  ${CPP_ROOT}/sdk
  ${CPP_ROOT}/sdk/protospec
  ${CPP_ROOT}/third_party/tinyxml2
  ${MUJOCO_INCLUDE_DIR}
  ${QHULL_INCLUDE_DIR}
  ${TINYOBJ_INCLUDE_DIR}
  ${LODEPNG_INCLUDE_DIR}
)
target_link_libraries(protospec_studio_core PUBLIC
  "${MUJOCO_IMPORT_LIB}" "${QHULL_LIB}" "${TINYOBJ_LIB}" "${LODEPNG_LIB}")
if(MSVC)
  target_compile_options(protospec_studio_core PRIVATE /bigobj)
endif()

# --- Vendored thin shell + plugin host ------------------------------------- #
add_library(protospec_studio_shell STATIC
  platform/hal/window.cc
  platform/hal/renderer.cc
  platform/sim/step_control.cc
  platform/ux/interaction.cc
  platform/ux/registry.cc
  platform/ux/ps_plugin_ext.cc
  host/app.cc
  editor/editor_ops.cc
  editor/undo.cc
  editor/hierarchy_icons.cc
  editor/hierarchy_model.cc
  editor/hierarchy_panel.cc
  editor/protospec_editor.cc
  editor/panels.cc
  # SE2 viewport editor: gizmo interaction, screen-space drawing, delta math.
  editor/viewport_plugin.cc
  editor/gizmo.cc
  editor/gizmo_math.cc
  editor/joint_overlay.cc
  editor/transform_math.cc
  # SE1b (generated Details panel) adds its source here:
  editor/details_panel.cc
  # SE3 authoring: add primitives, mesh import, duplicate, reparent, new model.
  editor/authoring_ops.cc
  editor/asset_import.cc
)
target_include_directories(protospec_studio_shell PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${MUJOCO_INCLUDE_DIR}
  ${MUJOCO_SRC_DIR}     # engine/engine_vis_*.h for the pick math
)
target_link_libraries(protospec_studio_shell PUBLIC
  protospec_studio_core
  dear_imgui
  SDL2::SDL2-static
  "${MUJOCO_IMPORT_LIB}"
)
if(MSVC)
  target_compile_options(protospec_studio_shell PRIVATE /bigobj)
endif()

# --- Executable ------------------------------------------------------------ #
add_executable(protospec_studio main.cc)
target_link_libraries(protospec_studio PRIVATE protospec_studio_shell)
if(WIN32)
  target_link_libraries(protospec_studio PRIVATE opengl32)
  # SDL2main is not used (we own main); SDL2-static pulls the needed system libs.
endif()

add_custom_command(TARGET protospec_studio POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${MUJOCO_DLL}" "$<TARGET_FILE_DIR:protospec_studio>"
  VERBATIM)

# --- Windowless logic tests (ctest) ---------------------------------------- #
enable_testing()
add_executable(protospec_studio_tests
  test/test_studio.cc
  editor/editor_ops.cc
  editor/undo.cc
  editor/hierarchy_icons.cc
  editor/hierarchy_model.cc
)
target_link_libraries(protospec_studio_tests PRIVATE
  protospec_studio_core
  "${MUJOCO_IMPORT_LIB}"
)
target_include_directories(protospec_studio_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${MUJOCO_INCLUDE_DIR}
)
if(MSVC)
  target_compile_options(protospec_studio_tests PRIVATE /bigobj)
endif()
add_custom_command(TARGET protospec_studio_tests POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${MUJOCO_DLL}" "$<TARGET_FILE_DIR:protospec_studio_tests>"
  VERBATIM)
add_test(NAME protospec_studio_tests COMMAND protospec_studio_tests)

# --- SE2 gizmo / delta-math windowless tests (own main) -------------------- #
# The DR-S6 delta rule + gizmo projection/hit-testing math are pure (no window),
# so they link the core plus the transform/gizmo math directly.
add_executable(protospec_studio_gizmo_tests
  test/test_gizmo.cc
  editor/transform_math.cc
  editor/gizmo_math.cc
  editor/joint_overlay.cc
  editor/editor_ops.cc
  editor/undo.cc
)
target_link_libraries(protospec_studio_gizmo_tests PRIVATE
  protospec_studio_core
  "${MUJOCO_IMPORT_LIB}"
)
target_include_directories(protospec_studio_gizmo_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${MUJOCO_INCLUDE_DIR}
)
if(MSVC)
  target_compile_options(protospec_studio_gizmo_tests PRIVATE /bigobj)
endif()
add_custom_command(TARGET protospec_studio_gizmo_tests POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${MUJOCO_DLL}" "$<TARGET_FILE_DIR:protospec_studio_gizmo_tests>"
  VERBATIM)
add_test(NAME protospec_studio_gizmo_tests COMMAND protospec_studio_gizmo_tests)

# --- SE2 movability audit windowless tests (own main) ---------------------- #
# The class-catching test: drive a translate through the gizmo core path over
# every body/geom/site/camera/light in a corpus (inline fixtures + vendored
# models) and assert each element's compiled world pose moved. Links the core
# plus the transform/undo helpers, like the gizmo delta battery.
add_executable(protospec_studio_movability_tests
  test/test_movability.cc
  editor/transform_math.cc
  editor/undo.cc
)
target_link_libraries(protospec_studio_movability_tests PRIVATE
  protospec_studio_core
  "${MUJOCO_IMPORT_LIB}"
)
target_include_directories(protospec_studio_movability_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${MUJOCO_INCLUDE_DIR}
)
if(MSVC)
  target_compile_options(protospec_studio_movability_tests PRIVATE /bigobj)
endif()
add_custom_command(TARGET protospec_studio_movability_tests POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${MUJOCO_DLL}" "$<TARGET_FILE_DIR:protospec_studio_movability_tests>"
  VERBATIM)
add_test(NAME protospec_studio_movability_tests
         COMMAND protospec_studio_movability_tests)

# --- Host-wiring windowless tests (own main) ------------------------------- #
# G1 Play/Stop state discard, G4 Save As + VFS externalization, G7 delete-confirm
# referrer flow: the host reactions are reproduced over the same windowless ops.
add_executable(protospec_studio_host_tests
  test/test_host.cc
  editor/authoring_ops.cc
  editor/asset_import.cc
  editor/editor_ops.cc
  editor/undo.cc
  editor/transform_math.cc
)
target_link_libraries(protospec_studio_host_tests PRIVATE
  protospec_studio_core
  "${MUJOCO_IMPORT_LIB}"
)
target_include_directories(protospec_studio_host_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${MUJOCO_INCLUDE_DIR}
)
if(MSVC)
  target_compile_options(protospec_studio_host_tests PRIVATE /bigobj)
endif()
add_custom_command(TARGET protospec_studio_host_tests POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${MUJOCO_DLL}" "$<TARGET_FILE_DIR:protospec_studio_host_tests>"
  VERBATIM)
add_test(NAME protospec_studio_host_tests COMMAND protospec_studio_host_tests)

# --- Host-UI tests (headless ImGui, no window) ----------------------------- #
# G2 key-routing gate + G3 pick / gizmo priority run the real plugin vtables over
# a headless ImGui context. They link the shell (viewport editor + registry +
# gizmo) and dear_imgui, but never open a window.
add_executable(protospec_studio_hostui_tests
  test/test_hostui.cc
)
target_link_libraries(protospec_studio_hostui_tests PRIVATE
  protospec_studio_shell
  protospec_studio_core
  dear_imgui
  "${MUJOCO_IMPORT_LIB}"
)
if(WIN32)
  target_link_libraries(protospec_studio_hostui_tests PRIVATE opengl32)
endif()
target_include_directories(protospec_studio_hostui_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${MUJOCO_INCLUDE_DIR}
)
if(MSVC)
  target_compile_options(protospec_studio_hostui_tests PRIVATE /bigobj)
endif()
add_custom_command(TARGET protospec_studio_hostui_tests POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${MUJOCO_DLL}" "$<TARGET_FILE_DIR:protospec_studio_hostui_tests>"
  VERBATIM)
add_test(NAME protospec_studio_hostui_tests COMMAND protospec_studio_hostui_tests)

# --- SE1b generated-Details windowless tests (own main) -------------------- #
# details_panel.h is windowless (no ImGui); these tests link only the core.
add_executable(protospec_studio_details_tests
  test/test_details.cc
)
target_link_libraries(protospec_studio_details_tests PRIVATE
  protospec_studio_core
  "${MUJOCO_IMPORT_LIB}"
)
target_include_directories(protospec_studio_details_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${MUJOCO_INCLUDE_DIR}
)
if(MSVC)
  target_compile_options(protospec_studio_details_tests PRIVATE /bigobj)
endif()
add_custom_command(TARGET protospec_studio_details_tests POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${MUJOCO_DLL}" "$<TARGET_FILE_DIR:protospec_studio_details_tests>"
  VERBATIM)
add_test(NAME protospec_studio_details_tests
         COMMAND protospec_studio_details_tests)

# --- SE3 authoring windowless tests (own main) ----------------------------- #
# authoring_ops / asset_import are windowless (no ImGui); they exercise the
# add / duplicate / reparent / mesh-import / new-model ops end to end, compiling
# after each op. transform_math is linked for the reparent keep-world-pose math.
add_executable(protospec_studio_authoring_tests
  test/test_authoring.cc
  editor/authoring_ops.cc
  editor/asset_import.cc
  editor/editor_ops.cc
  editor/undo.cc
  editor/transform_math.cc
)
target_link_libraries(protospec_studio_authoring_tests PRIVATE
  protospec_studio_core
  "${MUJOCO_IMPORT_LIB}"
)
target_include_directories(protospec_studio_authoring_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${MUJOCO_INCLUDE_DIR}
)
if(MSVC)
  target_compile_options(protospec_studio_authoring_tests PRIVATE /bigobj)
endif()
add_custom_command(TARGET protospec_studio_authoring_tests POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${MUJOCO_DLL}" "$<TARGET_FILE_DIR:protospec_studio_authoring_tests>"
  VERBATIM)
add_test(NAME protospec_studio_authoring_tests
         COMMAND protospec_studio_authoring_tests)

add_executable(protospec_studio_se5_tests
  test/test_se5_assets.cc
  editor/authoring_ops.cc
  editor/asset_import.cc
  editor/editor_ops.cc
  editor/undo.cc
  editor/transform_math.cc
)
target_link_libraries(protospec_studio_se5_tests PRIVATE
  protospec_studio_core
  "${MUJOCO_IMPORT_LIB}"
)
target_include_directories(protospec_studio_se5_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${MUJOCO_INCLUDE_DIR}
)
if(MSVC)
  target_compile_options(protospec_studio_se5_tests PRIVATE /bigobj)
endif()
add_custom_command(TARGET protospec_studio_se5_tests POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${MUJOCO_DLL}" "$<TARGET_FILE_DIR:protospec_studio_se5_tests>"
  VERBATIM)
add_test(NAME protospec_studio_se5_tests
         COMMAND protospec_studio_se5_tests)
