cmake_minimum_required(VERSION 3.16)
project(hignn LANGUAGES C CXX)

# find dependencies
find_package(Torch REQUIRED)
find_package(Kokkos REQUIRED)
# find_package(Python REQUIRED COMPONENTS Interpreter Development)
# find_package(TBB REQUIRED)
find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 17)

# add_subdirectory(pybind11)

include_directories(
    include
    ${MPI_INCLUDE_PATH}
)
include_directories("${PROJECT_SOURCE_DIR}/eigen")

add_compile_options(-O3 -w -fPIC -fopenmp -Wno-deprecated-gpu-targets)

# link/include
file(GLOB_RECURSE sources ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_executable(hignn hignn.cpp ${sources})

target_compile_definitions(hignn PRIVATE -DCUSTOM_SYS_PATH="${PROJECT_SOURCE_DIR}/include")

target_include_directories(hignn SYSTEM PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(hignn PRIVATE Kokkos::kokkos)
target_link_libraries(hignn PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(hignn PRIVATE "${MPI_CXX_LIBRARIES}")
# target_link_libraries(hignn PRIVATE Python::Python)
# target_link_libraries(hignn PRIVATE pybind11::pybind11)
# target_link_libraries(hignn PRIVATE TBB::tbb)