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

# MJCF IO: reader (tinyxml2 -> Model) + writer (Model -> MJCF), table-driven by
# the generated xml_binding, with handwritten quirk handlers. Links tinyxml2 and
# the core object model.
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
target_link_libraries(protospec_io PUBLIC protospec PRIVATE tinyxml2)
if(MSVC)
  target_compile_options(protospec_io PRIVATE /W4 /permissive-)
else()
  target_compile_options(protospec_io PRIVATE -Wall -Wextra)
endif()

# Round-trip tool: the harness contract (ps_roundtrip <in.xml>).
add_executable(ps_roundtrip tools/ps_roundtrip.cc)
target_link_libraries(ps_roundtrip PRIVATE protospec_io)

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

# The differential/corpus harness lives in cpp/harness/ and is owned by a
# separate agent; wire it in only when present.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/harness/CMakeLists.txt)
  add_subdirectory(harness)
endif()
