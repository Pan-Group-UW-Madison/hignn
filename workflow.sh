#!/usr/bin/bash

if [ $1 == "clean" ]; then
    make clean

    rm CMakeCache.txt
    rm -rf CMakeFiles/
    
    exit
fi

export PATH=$PATH:/opt/openmpi/bin

Kokkos_PATH=/opt/kokkos
LibTorch_PATH=/opt/libtorch

if [ $1 == "rebuild" ]; then
    make clean

    rm CMakeCache.txt
    rm -rf CMakeFiles/

    export CXX=$Kokkos_PATH/bin/nvcc_wrapper
    cmake -DCMAKE_PREFIX_PATH="$LibTorch_PATH/share/cmake/;$Kokkos_PATH" -DCMAKE_CXX_EXTENSIONS=Off ./
fi

cmake_result=$?
if [ $cmake_result -ne 0 ]; then
    echo -e "${RED}cmake fails${NC}"
    exit
else
    echo -e "${Green}cmake suceeds${NC}"
fi

export OMPI_CXX=$Kokkos_PATH/bin/nvcc_wrapper
export NVCC_WRAPPER_DEFAULT_COMPILER=/usr/bin/g++
export CUDA_LAUNCH_BLOCKING=1

make hignn -j

compile_result=$?
if [ $compile_result -ne 0 ]; then
    echo -e "${RED}make fails${NC}"
    exit
else
    echo "make suceeds"
fi

export OMP_NUM_THREADS=10
export OMP_PLACES=sockets
export OMP_PROC_BIND=spread

mpirun -n 4 --allow-run-as-root ./hignn > output.txt 2>&1