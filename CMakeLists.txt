cmake_minimum_required(VERSION 3.9.0)
project(hignn)

set(CMAKE_BUILD_TYPE RELEASE)

find_package(PythonInterp 3.8 REQUIRED)
find_package(PythonLibs 3.8 REQUIRED)
find_package(Torch REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

set(Kokkos_DIR ${PROJECT_SOURCE_DIR}/../kokkos/lib/cmake/Kokkos)
find_package(Kokkos REQUIRED)

include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/eigen/")
include_directories("${PROJECT_SOURCE_DIR}/pybind11/include/")

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/)

add_compile_options(-O3 -w -DNDEBUG -fPIC)

file(GLOB_RECURSE sources ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_executable(hignn ${sources})
set_target_properties(hignn PROPERTIES PREFIX "")

target_include_directories(hignn SYSTEM PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(hignn PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(hignn PRIVATE Python::Python)
target_link_libraries(hignn PRIVATE Kokkos::kokkos)
target_compile_options(hignn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --ftz=false
    --prec-div=true
    --prec-sqrt=true
    >)