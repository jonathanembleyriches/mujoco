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
  ${CPP_ROOT}/io/numeric.cc
  ${CPP_ROOT}/io/include.cc
  ${CPP_ROOT}/io/defaults.cc
  ${CPP_ROOT}/io/mjcf_reader.cc
  ${CPP_ROOT}/io/mjcf_writer.cc
  ${CPP_ROOT}/validate/validate.cc
  ${CPP_ROOT}/validate/sizes.cc
  ${CPP_ROOT}/bridge/compile.cc
  ${CPP_ROOT}/bridge/binding.cc
  ${CPP_ROOT}/compile/native.cc
  ${CPP_ROOT}/compile/build.cc
  ${CPP_ROOT}/compile/lifted/make_model.cc
  ${CPP_ROOT}/compile/lifted/mjuu_util.cc
  ${CPP_ROOT}/compile/lifted/mesh_pipeline.cc
  ${CPP_ROOT}/third_party/tinyxml2/tinyxml2.cpp
)

add_library(protospec_studio_core STATIC ${PROTOSPEC_SRC})
target_include_directories(protospec_studio_core PUBLIC
  ${CPP_ROOT}/include
  ${CPP_ROOT}/generated
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
)
target_link_libraries(protospec_studio_core PUBLIC
  "${MUJOCO_IMPORT_LIB}" "${QHULL_LIB}" "${TINYOBJ_LIB}")
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
  editor/protospec_editor.cc
  editor/panels.cc
  editor/pick_logger.cc
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
