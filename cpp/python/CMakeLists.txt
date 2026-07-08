cmake_minimum_required(VERSION 3.20)
project(protospec_python CXX)

# Standalone: build ONLY the `protospec` Python extension. It does not depend on
# cpp/CMakeLists.txt (a concurrent agent owns that); it compiles the object
# model, MJCF IO, validation and the MuJoCo bridge directly from source, exactly
# as cpp/harness/ builds its one tool standalone. Configure + build with:
#
#   cmake -S cpp/python -B cpp/python/build
#   cmake --build cpp/python/build --config Release
#
# Python + pybind11 come from the repo's uv venv by default; override with
# -DPS_PYTHON=/path/to/python. MuJoCo comes from MUJOCO_ROOT (same cache var the
# rest of the project uses).

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

get_filename_component(CPP_ROOT  "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
get_filename_component(REPO_ROOT "${CPP_ROOT}/.." ABSOLUTE)

# --- Python interpreter (default: the repo uv venv) ------------------------ #
if(NOT DEFINED PS_PYTHON)
  if(WIN32)
    set(PS_PYTHON "${REPO_ROOT}/.venv/Scripts/python.exe")
  else()
    set(PS_PYTHON "${REPO_ROOT}/.venv/bin/python")
  endif()
endif()
if(NOT EXISTS "${PS_PYTHON}")
  message(FATAL_ERROR
    "Python not found at ${PS_PYTHON}; run `uv sync` or pass -DPS_PYTHON=...")
endif()
set(Python_EXECUTABLE  "${PS_PYTHON}")
set(Python3_EXECUTABLE "${PS_PYTHON}")

# Force pybind11 onto the modern FindPython (which honors Python_EXECUTABLE), so
# it links against the uv venv interpreter rather than whatever legacy
# FindPythonInterp discovers on PATH/registry -- a version mismatch there yields
# a .pyd the venv cannot import.
set(PYBIND11_FINDPYTHON ON CACHE BOOL "" FORCE)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# --- pybind11 (installed into the venv via `uv add --dev pybind11`) -------- #
execute_process(
  COMMAND "${PS_PYTHON}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKEDIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE PYBIND11_QUERY)
if(NOT PYBIND11_QUERY EQUAL 0)
  message(FATAL_ERROR
    "pybind11 is not installed in ${PS_PYTHON}. Run: uv add --dev pybind11 numpy")
endif()
find_package(pybind11 CONFIG REQUIRED
             PATHS "${PYBIND11_CMAKEDIR}" NO_DEFAULT_PATH)

# --- MuJoCo (vendored prebuilt; the sim surface owns an mjData) ------------ #
set(MUJOCO_ROOT
    "C:/Users/jonat/Documents/Unreal Projects/url_proj/Plugins/UnrealRoboticsLab/third_party/MuJoCo/src"
    CACHE PATH "Root of the vendored prebuilt MuJoCo (contains include/ and build/)")
set(MUJOCO_INCLUDE_DIR "${MUJOCO_ROOT}/include")
set(MUJOCO_IMPORT_LIB  "${MUJOCO_ROOT}/build/lib/Release/mujoco.lib")
set(MUJOCO_DLL         "${MUJOCO_ROOT}/build/bin/Release/mujoco.dll")
if(NOT EXISTS "${MUJOCO_INCLUDE_DIR}/mujoco/mujoco.h")
  message(FATAL_ERROR "MuJoCo headers not found under ${MUJOCO_INCLUDE_DIR}; set -DMUJOCO_ROOT=...")
endif()
if(NOT EXISTS "${MUJOCO_IMPORT_LIB}")
  message(FATAL_ERROR "mujoco.lib not found at ${MUJOCO_IMPORT_LIB}; set -DMUJOCO_ROOT=...")
endif()

# --- Sources: object model + IO + validate + bridge + bindings ------------- #
file(GLOB PY_GENERATED_SRC CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/generated/*.cc")

set(PROTOSPEC_SRC
  ${CPP_ROOT}/generated/types.cc
  ${CPP_ROOT}/generated/reflect.cc
  ${CPP_ROOT}/generated/keywords.cc
  ${CPP_ROOT}/generated/defaults.cc
  ${CPP_ROOT}/generated/xml_binding.cc
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
  ${CPP_ROOT}/compile/lifted/make_model.cc
  ${CPP_ROOT}/third_party/tinyxml2/tinyxml2.cpp
)

pybind11_add_module(protospec
  ${CMAKE_CURRENT_SOURCE_DIR}/module.cc
  ${PY_GENERATED_SRC}
  ${PROTOSPEC_SRC}
)

target_include_directories(protospec PRIVATE
  ${CPP_ROOT}/include
  ${CPP_ROOT}/generated
  ${CPP_ROOT}/io
  ${CPP_ROOT}/validate
  ${CPP_ROOT}/bridge
  ${CPP_ROOT}/compile
  ${CPP_ROOT}/compile/lifted
  ${CPP_ROOT}/sdk
  ${CPP_ROOT}/third_party/tinyxml2
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${MUJOCO_INCLUDE_DIR}
)

target_link_libraries(protospec PRIVATE "${MUJOCO_IMPORT_LIB}")

if(MSVC)
  # The generated binding TUs instantiate thousands of templates: /bigobj is
  # required. Third-party headers (pybind11, tinyxml2, mujoco) are noisy under
  # /W4, so this target keeps default warnings.
  target_compile_options(protospec PRIVATE /bigobj)
endif()

# The MuJoCo runtime DLL must sit next to the .pyd so `import protospec` loads.
add_custom_command(TARGET protospec POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${MUJOCO_DLL}" "$<TARGET_FILE_DIR:protospec>"
  VERBATIM)
