# Readme

Before running the edge information builder, pybind11, Trilinos and compadre need to be installed.

## [Trilinos](https://github.com/trilinos/Trilinos)

To install Trilinos, please follow the steps below. Start from the root directory of Trilinos

```[bash]
make build
cd build
cmake -D CMAKE_INSTALL_PREFIX:PATH=$PREFIX_PATH -D Trilinos_ENABLE_Fortran:BOOL=OFF -D BUILD_SHARED_LIBS=ON -D TPL_ENABLE_MPI:BOOL=OFF -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF -D Trilinos_ENABLE_OpenMP=ON ../
make -j
make install
```

It is noted that PREFIX_PATH should be previously set as an evironment variable.

## [Compadre](https://github.com/zishengye/compadre)

After installing Trilinos, the compadre package could be installed.  Make a new directory named _build_ and get into the new directory.  Build a new bash script file filled with following stuff.

```[bash]
#!/bin/bash
# 
#
# Script for invoking CMake using the CMakeLists.txt file in this directory. 


# Serial on CPU via Kokkos
# No Python interface
# Standalone Kokkos

# following lines for build directory cleanup
find . ! -name '*.sh' -type f -exec rm -f {} +
find . -mindepth 1 -type d -exec rm -rf {} +

# pick your favorite c++ compiler
MY_CXX_COMPILER=`which g++`

# this will install in your build directory in a folder called install by default
INSTALL_PREFIX="/opt/compadre/"
MY_TRILINOS_PREFIX="/opt/trilinos/"

cmake \
    -D CMAKE_CXX_COMPILER="$MY_CXX_COMPILER" \
    -D CMAKE_CXX_FLAGS="-O3" \
    -D CMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -D Compadre_USE_PYTHON:BOOL=OFF \
    -D Trilinos_PREFIX="$MY_TRILINOS_PREFIX" \
    -D Compadre_DEBUG:BOOL=OFF \
    -D Compadre_TESTS:BOOL=OFF \
    -D Compadre_USE_OpenMP:BOOL=ON \
    \
    ..

make -j
make install
```

You could change INSTALL_PREFIX to wherever you want.  MY_TRILINOS_PREFIX should be specified as the directory where you install your Trilinos.

## Running the python script

First, change the directory on Line 5, 40, 45 to the exact directory where you install Trilinos and Compadre.  Then follow the general cmake commands as follows to build the python module

```[bash]
cmake ./
make
```

After this, you could play with the test_edge_info.py file.
