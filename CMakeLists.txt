cmake_minimum_required(VERSION 3.16)
project(hignn LANGUAGES C CXX)

# find dependencies
find_package(Kokkos REQUIRED)
find_package(Torch REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# targets
file(GLOB SOURCES "src/*.cpp")
add_executable(hignn ${SOURCES})

target_compile_features(hignn PUBLIC cxx_std_17)
set_target_properties(hignn PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
)

add_compile_options(-O3 -w -fPIC -fopenmp -Wno-deprecated-gpu-targets)

# link/include
target_compile_definitions(hignn PRIVATE -DCUSTOM_SYS_PATH="${PROJECT_SOURCE_DIR}/include")
target_include_directories(hignn SYSTEM PRIVATE ${PYTHON_INCLUDE_DIRS})
target_include_directories(hignn PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(hignn PRIVATE Kokkos::kokkos)
target_link_libraries(hignn PRIVATE "${TORCH_LIBRARIES}")