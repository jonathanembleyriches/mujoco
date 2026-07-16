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

# ProtoSpec editor for MuJoCo Studio.
#
# There is ONE source of truth for the editor cluster: the ProtoSpec repository
# at PROTOSPEC_ROOT. This fork does NOT hold a copy of the editor sources. It
# compiles them live from ${PROTOSPEC_ROOT}/apps/studio/editor together with the
# ProtoSpec object-model / IO / validate / compile-bridge core from
# ${PROTOSPEC_ROOT}/cpp. The only sources that live here are genuinely host-side
# glue: the platform/ux plugin-API shims (which map the ps::studio plugin names
# onto the real mujoco::platform registry) and host/shell.cc (the SE4 File/Edit
# menu, transform toolbar, and Play/Stop bridge, which bind to host-only plugin
# types the standalone apps/studio host does not have).
#
# The editor's own test batteries live and run with that single source, in the
# standalone apps/studio build (the CI / ASan surface); this fork does not
# duplicate them.

set(PROTOSPEC_ROOT "" CACHE PATH "Root of the ProtoSpec repository")
if(NOT EXISTS "${PROTOSPEC_ROOT}/cpp/include")
  message(FATAL_ERROR
    "PROTOSPEC_ROOT does not point at a ProtoSpec checkout: ${PROTOSPEC_ROOT}")
endif()
set(PROTOSPEC_CPP "${PROTOSPEC_ROOT}/cpp")

# The editor cluster single source. A separate cache var so the two roots can be
# split later if the studio tree ever moves out of the core repo.
set(PROTOSPEC_STUDIO_ROOT "${PROTOSPEC_ROOT}/apps/studio" CACHE PATH
    "Root of the ProtoSpec Studio editor cluster (single source)")
if(NOT EXISTS "${PROTOSPEC_STUDIO_ROOT}/editor/plugins.h")
  message(FATAL_ERROR
    "PROTOSPEC_STUDIO_ROOT does not contain the editor cluster: "
    "${PROTOSPEC_STUDIO_ROOT}")
endif()
set(PROTOSPEC_EDITOR "${PROTOSPEC_STUDIO_ROOT}/editor")

include(FetchContent)
FetchContent_GetProperties(qhull)
FetchContent_GetProperties(tinyobjloader)
FetchContent_GetProperties(lodepng)
FetchContent_GetProperties(marchingcubecpp)
FetchContent_GetProperties(dear_imgui)

# --- ProtoSpec core: generated object model + IO + validation + bridge ------ #
# The source list mirrors the standalone apps/studio core exactly, so the fork
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
  ${PROTOSPEC_CPP}/bridge/compile.cc
  ${PROTOSPEC_CPP}/bridge/binding.cc
  ${PROTOSPEC_CPP}/bridge/pose.cc
  ${PROTOSPEC_CPP}/compile/native.cc
  ${PROTOSPEC_CPP}/compile/build.cc
  ${PROTOSPEC_CPP}/compile/lifted/make_model.cc
  ${PROTOSPEC_CPP}/compile/lifted/mjuu_util.cc
  ${PROTOSPEC_CPP}/compile/lifted/mesh_pipeline.cc
  ${PROTOSPEC_CPP}/compile/lifted/image_pipeline.cc
  ${PROTOSPEC_CPP}/compile/lifted/builtin_mesh.cc
  ${PROTOSPEC_CPP}/compile/lifted/marching_cube.cc
  ${PROTOSPEC_CPP}/third_party/tinyxml2/tinyxml2.cpp
  # lodepng (PNG decode for the texture wave). MuJoCo keeps its own copy internal
  # to mujoco.dll (not exported), so a private copy here does not clash at link.
  ${lodepng_SOURCE_DIR}/lodepng.cpp
)

target_include_directories(protospec_core PUBLIC
  ${PROTOSPEC_CPP}/include
  ${PROTOSPEC_CPP}/generated
  ${PROTOSPEC_CPP}/core
  ${PROTOSPEC_CPP}/io
  ${PROTOSPEC_CPP}/validate
  ${PROTOSPEC_CPP}/bridge
  ${PROTOSPEC_CPP}/compile
  ${PROTOSPEC_CPP}/compile/lifted
  ${PROTOSPEC_CPP}/sdk
  ${PROTOSPEC_CPP}/sdk/protospec
  ${PROTOSPEC_CPP}/third_party/tinyxml2
)
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
# The editor .cc live in the ProtoSpec repo; host/shell.cc is the only fork-side
# source in this library.
add_library(protospec_editor STATIC
  ${PROTOSPEC_EDITOR}/asset_import.cc
  ${PROTOSPEC_EDITOR}/authoring_ops.cc
  ${PROTOSPEC_EDITOR}/details_panel.cc
  ${PROTOSPEC_EDITOR}/editor_ops.cc
  ${PROTOSPEC_EDITOR}/gizmo.cc
  ${PROTOSPEC_EDITOR}/gizmo_math.cc
  ${PROTOSPEC_EDITOR}/hierarchy_icons.cc
  ${PROTOSPEC_EDITOR}/hierarchy_model.cc
  ${PROTOSPEC_EDITOR}/hierarchy_panel.cc
  ${PROTOSPEC_EDITOR}/joint_overlay.cc
  ${PROTOSPEC_EDITOR}/panels.cc
  ${PROTOSPEC_EDITOR}/protospec_editor.cc
  ${PROTOSPEC_EDITOR}/transform_math.cc
  ${PROTOSPEC_EDITOR}/undo.cc
  ${PROTOSPEC_EDITOR}/viewport_plugin.cc
  host/shell.cc
)

# Include order matters: this fork directory is searched FIRST so the platform
# plugin-API shims here (platform/ux/plugin.h, ps_plugin_ext.h) win over the
# standalone struct definitions of the same relative path in the studio root.
# The studio root then supplies the editor/*.h headers, and the Studio src tree
# supplies the real experimental/platform/ux headers the shims forward to.
target_include_directories(protospec_editor PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROTOSPEC_STUDIO_ROOT}
  ${PROJECT_SOURCE_DIR}/src
)
target_include_directories(protospec_editor PRIVATE
  ${dear_imgui_SOURCE_DIR}/misc/cpp
)
target_link_libraries(protospec_editor PUBLIC
  protospec_core
  dear_imgui
  dear_imgui_stdlib
)
if(MSVC)
  target_compile_options(protospec_editor PRIVATE /bigobj)
endif()

add_library(mujoco::protospec_editor ALIAS protospec_editor)

set_target_properties(protospec_core protospec_editor PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)
