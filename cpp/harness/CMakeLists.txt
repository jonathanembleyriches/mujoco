cmake_minimum_required(VERSION 3.20)

# Buildable both standalone (`cmake -S cpp/harness`) and as a subdirectory of
# the root project (which does add_subdirectory(harness) once this file exists).
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(protospec_harness CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

set(MUJOCO_ROOT
    "C:/Users/jonat/Documents/Unreal Projects/url_proj/Plugins/UnrealRoboticsLab/third_party/MuJoCo/src"
    CACHE PATH "Root of the vendored prebuilt MuJoCo (contains include/ and build/)")

set(MUJOCO_INCLUDE_DIR "${MUJOCO_ROOT}/include")
set(MUJOCO_IMPORT_LIB  "${MUJOCO_ROOT}/build/lib/Release/mujoco.lib")
set(MUJOCO_DLL         "${MUJOCO_ROOT}/build/bin/Release/mujoco.dll")

if(NOT EXISTS "${MUJOCO_INCLUDE_DIR}/mujoco/mjxmacro.h")
  message(FATAL_ERROR
    "MuJoCo headers not found under ${MUJOCO_INCLUDE_DIR}; set -DMUJOCO_ROOT=...")
endif()
if(NOT EXISTS "${MUJOCO_IMPORT_LIB}")
  message(FATAL_ERROR
    "mujoco.lib not found at ${MUJOCO_IMPORT_LIB}; set -DMUJOCO_ROOT=...")
endif()

add_executable(mj_model_diff mj_model_diff.cc)
target_include_directories(mj_model_diff PRIVATE "${MUJOCO_INCLUDE_DIR}")
target_link_libraries(mj_model_diff PRIVATE "${MUJOCO_IMPORT_LIB}")

if(MSVC)
  target_compile_options(mj_model_diff PRIVATE /W4 /permissive-)
else()
  target_compile_options(mj_model_diff PRIVATE -Wall -Wextra)
endif()

# The runtime DLL must sit next to the exe so it loads without PATH tweaks.
add_custom_command(TARGET mj_model_diff POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${MUJOCO_DLL}" "$<TARGET_FILE_DIR:mj_model_diff>"
  VERBATIM)
