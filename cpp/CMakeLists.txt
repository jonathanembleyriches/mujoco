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

enable_testing()
add_executable(protospec_tests test/test_model.cc)
target_link_libraries(protospec_tests PRIVATE protospec)
if(MSVC)
  target_compile_options(protospec_tests PRIVATE /W4 /permissive-)
else()
  target_compile_options(protospec_tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME protospec_tests COMMAND protospec_tests)
