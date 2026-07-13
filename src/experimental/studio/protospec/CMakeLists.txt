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

# ProtoSpec editor for MuJoCo Studio. The object model / IO / validation /
# compile bridge sources are compiled directly from the ProtoSpec repository
# (PROTOSPEC_ROOT); the editor plugin cluster in this directory registers
# against the platform plugin registry.

set(PROTOSPEC_ROOT "" CACHE PATH "Root of the ProtoSpec repository")
if(NOT EXISTS "${PROTOSPEC_ROOT}/cpp/include")
  message(FATAL_ERROR
    "PROTOSPEC_ROOT does not point at a ProtoSpec checkout: ${PROTOSPEC_ROOT}")
endif()
set(PROTOSPEC_CPP "${PROTOSPEC_ROOT}/cpp")

include(FetchContent)
FetchContent_GetProperties(qhull)
FetchContent_GetProperties(tinyobjloader)
FetchContent_GetProperties(dear_imgui)

# --- ProtoSpec core: generated object model + IO + validation + bridge ------ #
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
  ${PROTOSPEC_CPP}/compile/native.cc
  ${PROTOSPEC_CPP}/compile/build.cc
  ${PROTOSPEC_CPP}/compile/lifted/make_model.cc
  ${PROTOSPEC_CPP}/compile/lifted/mjuu_util.cc
  ${PROTOSPEC_CPP}/compile/lifted/mesh_pipeline.cc
  ${PROTOSPEC_CPP}/third_party/tinyxml2/tinyxml2.cpp
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
)
target_link_libraries(protospec_core PUBLIC
  mujoco::mujoco
  qhullstatic_r
  tinyobjloader
)
if(MSVC)
  target_compile_options(protospec_core PRIVATE /bigobj)
endif()

# --- Editor plugin cluster --------------------------------------------------- #
add_library(protospec_editor STATIC
  editor/asset_import.cc
  editor/authoring_ops.cc
  editor/details_panel.cc
  editor/editor_ops.cc
  editor/gizmo.cc
  editor/gizmo_math.cc
  editor/hierarchy_model.cc
  editor/hierarchy_panel.cc
  editor/joint_overlay.cc
  editor/panels.cc
  editor/protospec_editor.cc
  editor/shell.cc
  editor/transform_math.cc
  editor/undo.cc
  editor/viewport_plugin.cc
)

target_include_directories(protospec_editor PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
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

# --- Windowless editor tests (own mains, no GUI) ----------------------------- #
enable_testing()

function(protospec_add_test NAME)
  add_executable(${NAME} ${ARGN})
  target_include_directories(${NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${NAME} PRIVATE protospec_core)
  if(MSVC)
    target_compile_options(${NAME} PRIVATE /bigobj)
  endif()
  add_test(NAME ${NAME} COMMAND ${NAME})
  set_target_properties(${NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
  )
endfunction()

protospec_add_test(protospec_studio_tests
  test/test_studio.cc
  editor/editor_ops.cc
  editor/undo.cc
  editor/hierarchy_model.cc
)
protospec_add_test(protospec_studio_gizmo_tests
  test/test_gizmo.cc
  editor/transform_math.cc
  editor/gizmo_math.cc
  editor/joint_overlay.cc
  editor/editor_ops.cc
  editor/undo.cc
)
protospec_add_test(protospec_studio_movability_tests
  test/test_movability.cc
  editor/transform_math.cc
  editor/undo.cc
)
protospec_add_test(protospec_studio_details_tests
  test/test_details.cc
)
protospec_add_test(protospec_studio_authoring_tests
  test/test_authoring.cc
  editor/authoring_ops.cc
  editor/asset_import.cc
  editor/editor_ops.cc
  editor/undo.cc
  editor/transform_math.cc
)
protospec_add_test(protospec_studio_host_tests
  test/test_host.cc
  editor/authoring_ops.cc
  editor/asset_import.cc
  editor/editor_ops.cc
  editor/undo.cc
  editor/transform_math.cc
)

# --- Host-UI tests (headless ImGui, no window) ------------------------------- #
# G2 key-routing gate + G3 pick / gizmo priority run the real plugin vtables over
# a headless ImGui context. They compile the viewport/gizmo sources directly and
# supply their own minimal plugin registry, so they need dear_imgui but not the
# SDL/Filament platform library.
add_executable(protospec_studio_hostui_tests
  test/test_hostui.cc
  editor/viewport_plugin.cc
  editor/gizmo.cc
  editor/gizmo_math.cc
  editor/transform_math.cc
  editor/joint_overlay.cc
  editor/editor_ops.cc
  editor/authoring_ops.cc
  editor/asset_import.cc
  editor/undo.cc
)
target_include_directories(protospec_studio_hostui_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/src
  ${dear_imgui_SOURCE_DIR}/misc/cpp
)
target_link_libraries(protospec_studio_hostui_tests PRIVATE
  protospec_core
  dear_imgui
)
set_target_properties(protospec_studio_hostui_tests PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)
if(MSVC)
  target_compile_options(protospec_studio_hostui_tests PRIVATE /bigobj)
endif()
add_test(NAME protospec_studio_hostui_tests COMMAND protospec_studio_hostui_tests)

set_target_properties(protospec_core protospec_editor PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)
