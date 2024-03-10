cmake_minimum_required(VERSION 3.16)
project(hignn LANGUAGES C CXX)

# find dependencies
find_package(PythonInterp REQUIRED)
find_package(Torch REQUIRED)
find_package(Kokkos REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(TBB REQUIRED)
find_package(MPI REQUIRED)

include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/extern/eigen/)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${PROJECT_SOURCE_DIR}/extern/eigen/")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS off)

include(CheckIPOSupported)
check_ipo_supported(RESULT result)

add_subdirectory(./pybind11)

include_directories(
    include
    ${MPI_INCLUDE_PATH}
)
include_directories("${PROJECT_SOURCE_DIR}/eigen")

add_compile_options(-O3 -Werror -Wall -Wextra -DNDEBUG --expt-relaxed-constexpr -Wno-div-by-zero)

# link/include
file(GLOB_RECURSE sources ${PROJECT_SOURCE_DIR}/src/*.cpp)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# pybind11_add_module(hignn interface/HignnInterface.cpp ${sources})
add_library(hignn SHARED ${sources} interface/HignnInterface.cpp)

set_target_properties(hignn PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

target_include_directories(hignn SYSTEM PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(hignn PRIVATE Kokkos::kokkos)
target_link_libraries(hignn PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(hignn PRIVATE "${MPI_CXX_LIBRARIES}")
target_link_libraries(hignn PRIVATE Python::Python)
target_link_libraries(hignn PRIVATE pybind11::pybind11)
target_link_libraries(hignn PRIVATE TBB::tbb)