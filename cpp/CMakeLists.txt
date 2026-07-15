cmake_minimum_required(VERSION 3.20)
project(protospec CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# MuJoCo-free static library: the generated object model + handwritten core.
add_library(protospec STATIC
  generated/types.cc
  generated/reflect.cc
  generated/keywords.cc
  generated/defaults.cc
  generated/xml_binding.cc
)
target_include_directories(protospec PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/generated
)

if(MSVC)
  target_compile_options(protospec PRIVATE /W4 /permissive-)
else()
  target_compile_options(protospec PRIVATE -Wall -Wextra)
endif()

# Vendored tinyxml2 (zlib license). Third-party: built with default warnings so
# the project's /W4 does not flag it. Only protospec_io links it; the core stays
# dependency-free.
add_library(tinyxml2 STATIC third_party/tinyxml2/tinyxml2.cpp)
target_include_directories(tinyxml2 PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinyxml2
)

# Core resolvers: orientation + inertia canonicalization (Q-ORIENT/Q-INERTIA),
# a MuJoCo-free translation unit of lifted math (plan.md Section 3 core/resolve).
# The reader folds authored orientation/inertia spellings into their canonical
# quat/diaginertia forms at parse end through this library, so protospec_io stays
# MuJoCo-free (no mujoco.h, no mujoco.dll runtime dependency for ps_roundtrip).
add_library(protospec_core STATIC core/resolve.cc)
target_include_directories(protospec_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core)
target_link_libraries(protospec_core PUBLIC protospec)
if(MSVC)
  target_compile_options(protospec_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(protospec_core PRIVATE -Wall -Wextra)
endif()

# MJCF IO: reader (tinyxml2 -> Model) + writer (Model -> MJCF), table-driven by
# the generated xml_binding, with handwritten quirk handlers. Links tinyxml2, the
# core object model, and the MuJoCo-free canonicalization resolvers.
add_library(protospec_io STATIC
  io/numeric.cc
  io/include.cc
  io/defaults.cc
  io/mjcf_reader.cc
  io/mjcf_writer.cc
)
target_include_directories(protospec_io PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/io
)
target_link_libraries(protospec_io PUBLIC protospec protospec_core PRIVATE tinyxml2)
if(MSVC)
  target_compile_options(protospec_io PRIVATE /W4 /permissive-)
else()
  target_compile_options(protospec_io PRIVATE -Wall -Wextra)
endif()

# Validation: three-tier structural/referential/semantic checks over a const
# Model (plan Section 9). MuJoCo-free and tinyxml2-free -- it only needs the
# generated tables in the core library.
add_library(protospec_validate STATIC
  validate/validate.cc
  validate/sizes.cc
)
target_include_directories(protospec_validate PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/validate
)
target_link_libraries(protospec_validate PUBLIC protospec)
if(MSVC)
  target_compile_options(protospec_validate PRIVATE /W4 /permissive-)
else()
  target_compile_options(protospec_validate PRIVATE -Wall -Wextra)
endif()

# Round-trip tool: the harness contract (ps_roundtrip <in.xml>).
add_executable(ps_roundtrip tools/ps_roundtrip.cc)
target_link_libraries(ps_roundtrip PRIVATE protospec_io)

# Validation CLI: the corpus-sanity harness contract (ps_validate <in.xml>).
add_executable(ps_validate tools/ps_validate.cc)
target_link_libraries(ps_validate PRIVATE protospec_io protospec_validate)

enable_testing()
add_executable(protospec_tests test/test_model.cc)
target_link_libraries(protospec_tests PRIVATE protospec)
if(MSVC)
  target_compile_options(protospec_tests PRIVATE /W4 /permissive-)
else()
  target_compile_options(protospec_tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME protospec_tests COMMAND protospec_tests)

add_executable(protospec_io_tests test/test_io.cc)
target_link_libraries(protospec_io_tests PRIVATE protospec_io)
if(MSVC)
  target_compile_options(protospec_io_tests PRIVATE /W4 /permissive-)
else()
  target_compile_options(protospec_io_tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME protospec_io_tests COMMAND protospec_io_tests)

add_executable(protospec_validate_tests test/test_validate.cc)
target_link_libraries(protospec_validate_tests PRIVATE protospec_validate protospec_io)
if(MSVC)
  target_compile_options(protospec_validate_tests PRIVATE /W4 /permissive-)
else()
  target_compile_options(protospec_validate_tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME protospec_validate_tests COMMAND protospec_validate_tests)

# --------------------------------------------------------------------------- #
# MuJoCo-dependent targets: the compile bridge (milestone 5) + its tools/tests. #
# All gated on MuJoCo being found so the MuJoCo-free core above still builds     #
# standalone. MUJOCO_ROOT is the same cache variable the harness uses.           #
# --------------------------------------------------------------------------- #
set(MUJOCO_ROOT
    "C:/Users/jonat/Documents/Unreal Projects/url_proj/Plugins/UnrealRoboticsLab/third_party/MuJoCo/src"
    CACHE PATH "Root of the vendored prebuilt MuJoCo (contains include/ and build/)")

set(MUJOCO_INCLUDE_DIR "${MUJOCO_ROOT}/include")
set(MUJOCO_IMPORT_LIB  "${MUJOCO_ROOT}/build/lib/Release/mujoco.lib")
set(MUJOCO_DLL         "${MUJOCO_ROOT}/build/bin/Release/mujoco.dll")

# NC3 asset-pipeline third-party deps, prebuilt alongside the vendored MuJoCo
# (the same static libs MuJoCo itself links). qhull (convex-hull graph) and
# tinyobjloader (OBJ parsing) are linked into the lifted TU; lodepng (PNG decode)
# is reserved for the file-texture/hfield wave. Their symbols are internal to
# mujoco.dll (not exported), so our own copies do not clash at link.
set(MUJOCO_DEPS "${MUJOCO_ROOT}/build/_deps")
set(QHULL_INCLUDE_DIR "${MUJOCO_DEPS}/qhull-src/src")
set(QHULL_LIB         "${MUJOCO_ROOT}/build/lib/Release/qhullstatic_r.lib")
set(TINYOBJ_INCLUDE_DIR "${MUJOCO_DEPS}/tinyobjloader-src")
set(TINYOBJ_LIB         "${MUJOCO_ROOT}/build/lib/Release/tinyobjloader.lib")
set(LODEPNG_INCLUDE_DIR "${MUJOCO_DEPS}/lodepng-src")
set(LODEPNG_LIB         "${MUJOCO_ROOT}/build/lib/Release/lodepng.lib")

# The differential/corpus harness lives in cpp/harness/ (owned separately); wire
# it in before the MuJoCo bridge tools below so its plugin_registry / model_diff
# targets exist when ps_compile and ps_native_diff link them.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/harness/CMakeLists.txt)
  add_subdirectory(harness)
endif()

if(EXISTS "${MUJOCO_INCLUDE_DIR}/mujoco/mujoco.h" AND EXISTS "${MUJOCO_IMPORT_LIB}")
  # Lifted MuJoCo code (CDR-3): copied-with-provenance algorithms, tracked in
  # snapshots/lifted_code.json. Its own TU so the conformance mode (/permissive-)
  # that the handwritten compiler builds under does not reject verbatim upstream
  # C++; still /W4. Symbols live in ps::mjcf::compile::lifted.
  add_library(protospec_lifted STATIC
    compile/lifted/make_model.cc
    compile/lifted/mjuu_util.cc
    compile/lifted/mesh_pipeline.cc
    compile/lifted/image_pipeline.cc
  )
  target_include_directories(protospec_lifted
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/compile/lifted
    PRIVATE ${MUJOCO_INCLUDE_DIR}
            ${QHULL_INCLUDE_DIR}
            ${TINYOBJ_INCLUDE_DIR}
            ${LODEPNG_INCLUDE_DIR}
  )
  target_link_libraries(protospec_lifted PRIVATE "${MUJOCO_IMPORT_LIB}"
                        "${QHULL_LIB}" "${TINYOBJ_LIB}" "${LODEPNG_LIB}")
  if(MSVC)
    target_compile_options(protospec_lifted PRIVATE /W4)
  else()
    target_compile_options(protospec_lifted PRIVATE -Wall -Wextra)
  endif()

  # The native compiler (impl-plan NC0+): ProtoSpec tree -> mjModel directly.
  # Jointly with bridge/ this is the MuJoCo quarantine zone. Today it is
  # scaffolding (the feature gate admits nothing, so it always reports
  # UnsupportedNatively); it plugs in behind the bridge dispatch below.
  add_library(protospec_compile STATIC
    compile/native.cc
    compile/build.cc
  )
  target_include_directories(protospec_compile
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/compile
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bridge
            ${CMAKE_CURRENT_SOURCE_DIR}/sdk
            ${CMAKE_CURRENT_SOURCE_DIR}/sdk/protospec
            ${MUJOCO_INCLUDE_DIR}
  )
  target_link_libraries(protospec_compile PUBLIC protospec protospec_io protospec_lifted
                                          PRIVATE "${MUJOCO_IMPORT_LIB}")
  if(MSVC)
    target_compile_options(protospec_compile PRIVATE /W4 /permissive-)
  else()
    target_compile_options(protospec_compile PRIVATE -Wall -Wextra)
  endif()

  # The one quarantine zone for mujoco.h besides the harness: Model -> mjModel +
  # Binding + Recompile (plan DR-5/DR-10/DR-11). Links the object model, the MJCF
  # writer (XML path), the native compiler, and MuJoCo. /W4 like every target.
  add_library(protospec_bridge STATIC
    bridge/compile.cc
    bridge/binding.cc
  )
  target_include_directories(protospec_bridge
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/bridge
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/compile
            ${MUJOCO_INCLUDE_DIR}
  )
  target_link_libraries(protospec_bridge PUBLIC protospec protospec_io protospec_compile
                                         PRIVATE "${MUJOCO_IMPORT_LIB}")
  if(MSVC)
    target_compile_options(protospec_bridge PRIVATE /W4 /permissive-)
  else()
    target_compile_options(protospec_bridge PRIVATE -Wall -Wextra)
  endif()

  # Copies mujoco.dll next to any exe that links the bridge.
  function(ps_copy_mujoco_dll target)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${MUJOCO_DLL}" "$<TARGET_FILE_DIR:${target}>"
      VERBATIM)
  endfunction()

  # Corpus compile CLI (the harness contract: ps_compile <in.xml>). Links the
  # shared plugin_registry so the bridge's mj_loadXML resolves plugin models.
  add_executable(ps_compile tools/ps_compile.cc)
  target_include_directories(ps_compile PRIVATE
    ${MUJOCO_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/harness)
  target_link_libraries(ps_compile PRIVATE protospec_bridge plugin_registry "${MUJOCO_IMPORT_LIB}")
  ps_copy_mujoco_dll(ps_compile)

  add_executable(protospec_bridge_tests test/test_bridge.cc)
  target_include_directories(protospec_bridge_tests PRIVATE ${MUJOCO_INCLUDE_DIR})
  target_link_libraries(protospec_bridge_tests PRIVATE protospec_bridge "${MUJOCO_IMPORT_LIB}")
  if(MSVC)
    target_compile_options(protospec_bridge_tests PRIVATE /W4 /permissive-)
  else()
    target_compile_options(protospec_bridge_tests PRIVATE -Wall -Wextra)
  endif()
  ps_copy_mujoco_dll(protospec_bridge_tests)
  add_test(NAME protospec_bridge_tests COMMAND protospec_bridge_tests)
endif()

# SDK: the ergonomic authoring layer over the generated object model. Header-
# only (templated over the generated Visit/reflect hooks), MuJoCo-free. Consumers
# link the core (types/reflect/defaults) plus this interface for the includes.
add_library(protospec_sdk INTERFACE)
target_include_directories(protospec_sdk INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/sdk
)
target_link_libraries(protospec_sdk INTERFACE protospec)

# SDK save surface (protospec/save.h): the one compiled piece of the SDK. It
# reaches to disk and pulls in the MJCF writer, so it is a small static library
# kept separate from the header-only pure-tree SDK. Also carries the single
# shared asset-externalization implementation the studio editor delegates to.
add_library(protospec_sdk_io STATIC sdk/save.cc)
target_link_libraries(protospec_sdk_io PUBLIC protospec_sdk protospec_io)
if(MSVC)
  target_compile_options(protospec_sdk_io PRIVATE /W4 /permissive-)
else()
  target_compile_options(protospec_sdk_io PRIVATE -Wall -Wextra)
endif()

add_executable(protospec_sdk_tests test/test_sdk.cc)
target_link_libraries(protospec_sdk_tests PRIVATE protospec_sdk protospec_sdk_io protospec_io)
if(MSVC)
  # /bigobj: the SDK test TU (heavily templated builders + the added save cases)
  # exceeds the COFF section limit without it.
  target_compile_options(protospec_sdk_tests PRIVATE /W4 /permissive- /bigobj)
else()
  target_compile_options(protospec_sdk_tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME protospec_sdk_tests COMMAND protospec_sdk_tests)

# --------------------------------------------------------------------------- #
# Native-compiler harness + tests (impl-plan NC0). Placed after the harness    #
# subdirectory so model_diff_lib exists. Guarded on both the bridge and the    #
# comparison lib being available.                                              #
# --------------------------------------------------------------------------- #
if(TARGET protospec_bridge AND TARGET model_diff_lib)
  # Three-way differential driver (leg B XML path vs leg C native path).
  add_executable(ps_native_diff harness/ps_native_diff.cc)
  target_include_directories(ps_native_diff PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bridge ${CMAKE_CURRENT_SOURCE_DIR}/harness
    ${MUJOCO_INCLUDE_DIR})
  target_link_libraries(ps_native_diff
    PRIVATE protospec_bridge model_diff_lib plugin_registry "${MUJOCO_IMPORT_LIB}")
  if(MSVC)
    target_compile_options(ps_native_diff PRIVATE /W4 /permissive-)
  else()
    target_compile_options(ps_native_diff PRIVATE -Wall -Wextra)
  endif()
  ps_copy_mujoco_dll(ps_native_diff)

  # Native-compiler ctest: purity gate, feature-gate correctness, report shape
  # (T0.4) + the mjModel allocator round-trip spike (T0.5) + mjuu_* lifted
  # smoke goldens (T0.1).
  add_executable(protospec_compile_tests test/test_native.cc)
  target_include_directories(protospec_compile_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bridge
    ${CMAKE_CURRENT_SOURCE_DIR}/compile
    ${CMAKE_CURRENT_SOURCE_DIR}/compile/lifted
    ${CMAKE_CURRENT_SOURCE_DIR}/harness
    ${MUJOCO_INCLUDE_DIR})
  target_link_libraries(protospec_compile_tests
    PRIVATE protospec_bridge protospec_compile model_diff_lib "${MUJOCO_IMPORT_LIB}")
  if(MSVC)
    target_compile_options(protospec_compile_tests PRIVATE /W4 /permissive-)
  else()
    target_compile_options(protospec_compile_tests PRIVATE -Wall -Wextra)
  endif()
  ps_copy_mujoco_dll(protospec_compile_tests)
  add_test(NAME protospec_compile_tests COMMAND protospec_compile_tests)
endif()
