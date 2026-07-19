# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ProtoSpec editor for MuJoCo Studio — host-mount build (studio/glue).
#
# This directory IS part of the ProtoSpec repository; the MuJoCo Studio fork
# mounts it with add_subdirectory(${PROTOSPEC_ROOT}/studio/glue ...) so that
# every line of build logic, the plugin-API shim (platform/ux/plugin.h), and
# the registration TU (register_editor.cc, compiled by the fork directly into
# the mujoco_studio executable) live with the single source. The fork carries
# only the ~10-line mount block in its studio/CMakeLists.txt.
#
# It compiles the editor cluster from ../editor together with the ProtoSpec
# object-model / IO / validate / compile-bridge core from ../../protospec/lib.
# Third-party deps (qhull, tinyobjloader, lodepng, dear_imgui, SDL2) are the
# host build's own FetchContent trees — this file only reads their properties.
#
# The editor's own test batteries live and run with the single source
# (windowless: protospec/tests/test_plugin_windowless.py); the fork does not
# duplicate them.

get_filename_component(PROTOSPEC_GLUE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(PROTOSPEC_CPP "${PROTOSPEC_GLUE_ROOT}/protospec/lib")
set(PROTOSPEC_ATTIC "${PROTOSPEC_GLUE_ROOT}/attic")
set(PROTOSPEC_EDITOR "${PROTOSPEC_GLUE_ROOT}/studio/editor")

# The native compiler is parked in attic/ and excluded from the default build.
# Turn ON to compile it in and define PROTOSPEC_NATIVE. See attic/ATTIC.md.
option(PROTOSPEC_BUILD_NATIVE "Build the atticked native compiler (attic/compile)" OFF)

include(FetchContent)
FetchContent_GetProperties(qhull)
FetchContent_GetProperties(tinyobjloader)
FetchContent_GetProperties(lodepng)
FetchContent_GetProperties(marchingcubecpp)
FetchContent_GetProperties(dear_imgui)

# --- ProtoSpec core: generated object model + IO + validation + bridge ------ #
# The source list mirrors the standalone studio core exactly, so this build
# carries the full native compiler + lifted asset pipeline (texture / builtin
# mesh / marching cube) rather than the reduced set the frozen snapshot held.
file(GLOB PROTOSPEC_GENERATED_SRC CONFIGURE_DEPENDS
     "${PROTOSPEC_CPP}/generated/*.cc")

add_library(protospec_core STATIC
  ${PROTOSPEC_GENERATED_SRC}
  ${PROTOSPEC_CPP}/core/resolve.cc
  ${PROTOSPEC_CPP}/io/numeric.cc
  ${PROTOSPEC_CPP}/io/include.cc
  ${PROTOSPEC_CPP}/io/defaults.cc
  ${PROTOSPEC_CPP}/io/mjcf_reader.cc
  ${PROTOSPEC_CPP}/io/mjcf_writer.cc
  ${PROTOSPEC_CPP}/validate/validate.cc
  ${PROTOSPEC_CPP}/validate/sizes.cc
  ${PROTOSPEC_CPP}/compile/compile.cc
  ${PROTOSPEC_CPP}/compile/binding.cc
  ${PROTOSPEC_CPP}/compile/pose.cc
  ${PROTOSPEC_CPP}/compile/mjs_builder.cc
  ${PROTOSPEC_CPP}/third_party/tinyxml2/tinyxml2.cpp
  # lodepng (PNG decode for the texture wave). MuJoCo keeps its own copy internal
  # to mujoco.dll (not exported), so a private copy here does not clash at link.
  ${lodepng_SOURCE_DIR}/lodepng.cpp
)

# Native compiler + lifted asset pipeline (parked in attic/). Opt-in only; the
# default build excludes it and the compile bridge's native dispatch stays
# #ifdef PROTOSPEC_NATIVE-gated (off).
if(PROTOSPEC_BUILD_NATIVE)
  target_sources(protospec_core PRIVATE
    ${PROTOSPEC_ATTIC}/compile/native.cc
    ${PROTOSPEC_ATTIC}/compile/build.cc
    ${PROTOSPEC_ATTIC}/compile/lifted/make_model.cc
    ${PROTOSPEC_ATTIC}/compile/lifted/mjuu_util.cc
    ${PROTOSPEC_ATTIC}/compile/lifted/mesh_pipeline.cc
    ${PROTOSPEC_ATTIC}/compile/lifted/image_pipeline.cc
    ${PROTOSPEC_ATTIC}/compile/lifted/builtin_mesh.cc
    ${PROTOSPEC_ATTIC}/compile/lifted/marching_cube.cc
  )
  target_compile_definitions(protospec_core PRIVATE PROTOSPEC_NATIVE)
endif()

target_include_directories(protospec_core PUBLIC
  ${PROTOSPEC_CPP}/include
  ${PROTOSPEC_CPP}/generated
  ${PROTOSPEC_CPP}/core
  ${PROTOSPEC_CPP}/io
  ${PROTOSPEC_CPP}/validate
  ${PROTOSPEC_CPP}/compile
  ${PROTOSPEC_CPP}/sdk
  ${PROTOSPEC_CPP}/sdk/protospec
  ${PROTOSPEC_CPP}/third_party/tinyxml2
)
if(PROTOSPEC_BUILD_NATIVE)
  target_include_directories(protospec_core PUBLIC
    ${PROTOSPEC_ATTIC}/compile
    ${PROTOSPEC_ATTIC}/compile/lifted)
endif()
target_include_directories(protospec_core PRIVATE
  ${qhull_SOURCE_DIR}/src
  ${tinyobjloader_SOURCE_DIR}
  ${lodepng_SOURCE_DIR}
  ${marchingcubecpp_SOURCE_DIR}
)
target_link_libraries(protospec_core PUBLIC
  mujoco::mujoco
  qhullstatic_r
  tinyobjloader
)
if(MSVC)
  target_compile_options(protospec_core PRIVATE /bigobj)
endif()

# --- Editor plugin cluster (compiled live from the single source) ----------- #
add_library(protospec_editor STATIC
  ${PROTOSPEC_EDITOR}/asset_import.cc
  ${PROTOSPEC_EDITOR}/authoring_ops.cc
  ${PROTOSPEC_EDITOR}/details_panel.cc
  ${PROTOSPEC_EDITOR}/dock_layout.cc
  ${PROTOSPEC_EDITOR}/editor_ops.cc
  ${PROTOSPEC_EDITOR}/gizmo.cc
  ${PROTOSPEC_EDITOR}/gizmo_math.cc
  ${PROTOSPEC_EDITOR}/hierarchy_icons.cc
  ${PROTOSPEC_EDITOR}/hierarchy_model.cc
  ${PROTOSPEC_EDITOR}/hierarchy_panel.cc
  ${PROTOSPEC_EDITOR}/joint_overlay.cc
  ${PROTOSPEC_EDITOR}/layers.cc
  ${PROTOSPEC_EDITOR}/panels.cc
  ${PROTOSPEC_EDITOR}/placement.cc
  ${PROTOSPEC_EDITOR}/protospec_editor.cc
  ${PROTOSPEC_EDITOR}/screenshot_service.cc
  ${PROTOSPEC_EDITOR}/transform_math.cc
  ${PROTOSPEC_EDITOR}/undo.cc
  ${PROTOSPEC_EDITOR}/viewport_plugin.cc
  # R1: host/shell.cc is gone -- the File/Edit menus, transform toolbar, Edit/Play
  # mode and native file dialog collapsed into the editor's own plugins (panels.cc),
  # so the glue registers only the upstream plugin types.
)

# Include order matters: the glue directory is searched FIRST so its
# platform/ux/plugin.h shim (mapping the ps::studio plugin names onto the real
# mujoco::platform registry) is the copy the editor's
# #include "platform/ux/plugin.h" resolves to. The studio root then supplies
# the editor/*.h headers, and the host's src tree supplies the real
# experimental/platform/ux headers the shim forwards to.
target_include_directories(protospec_editor PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROTOSPEC_GLUE_ROOT}/studio
  ${PROJECT_SOURCE_DIR}/src
)
target_include_directories(protospec_editor PRIVATE
  ${dear_imgui_SOURCE_DIR}/misc/cpp
  # screenshot_service.cc PNG-encodes with the host build's lodepng (the same
  # copy protospec_core compiles in).
  ${lodepng_SOURCE_DIR}
)
target_link_libraries(protospec_editor PUBLIC
  protospec_core
  dear_imgui
  dear_imgui_stdlib
)
# screenshot_service.cc reaches the classic backend's GL through the live SDL
# GL context (SDL_GL_GetProcAddress) and requests app exit via SDL_PushEvent;
# the host app links SDL2 anyway, so this adds headers, not a new runtime dep.
target_link_libraries(protospec_editor PRIVATE SDL2::SDL2-static)
if(MSVC)
  target_compile_options(protospec_editor PRIVATE /bigobj)
endif()

add_library(mujoco::protospec_editor ALIAS protospec_editor)

set_target_properties(protospec_core protospec_editor PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)
