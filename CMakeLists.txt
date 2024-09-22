cmake_minimum_required(VERSION 3.16)
project(hignn LANGUAGES C CXX)

# find dependencies
set (PYTHON_EXECUTABLE "/usr/bin/python3")

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
find_package(Kokkos REQUIRED)
find_package(TBB REQUIRED)
find_package(MPI REQUIRED)

find_package(VTK REQUIRED)
include_directories(SYSTEM ${VTK_INCLUDE_DIRS})

include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/extern/eigen/)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${PROJECT_SOURCE_DIR}/extern/eigen/")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS off)

include(CheckIPOSupported)
check_ipo_supported(RESULT result)

add_subdirectory(./extern/pybind11)

include_directories(
    include
    ${MPI_INCLUDE_PATH}
)

if (USE_GPU)
    add_compile_definitions(USE_GPU)
    add_compile_options(-O3 -Werror -Wall -Wextra -DNDEBUG --expt-relaxed-constexpr)
else()
    add_compile_options(-O3 -Werror -Wall -Wextra -DNDEBUG)
endif()

# link/include
file(GLOB_RECURSE sources ${PROJECT_SOURCE_DIR}/src/*.cpp)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# pybind11_add_module(hignn interface/HignnInterface.cpp ${sources})
add_library(hignn SHARED ${sources} interface/HignnInterface.cpp)

set_target_properties(hignn PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

target_include_directories(hignn SYSTEM PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(hignn "${VTK_LIBRARIES}")
target_link_libraries(hignn Kokkos::kokkos)
target_link_libraries(hignn "${TORCH_LIBRARIES}")
target_link_libraries(hignn "${MPI_CXX_LIBRARIES}")
target_link_libraries(hignn pybind11::pybind11)
target_link_libraries(hignn TBB::tbb)