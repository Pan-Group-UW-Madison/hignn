if [ $1 == "rebuild" ]; then
    make clean

    rm CMakeCache.txt
    rm -rf CMakeFiles/

    export CXX=~/kokkos/bin/nvcc_wrapper
    cmake -DCMAKE_PREFIX_PATH="`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`;~/kokkos/" -DCMAKE_CXX_EXTENSIONS=Off ./
fi

cmake_result=$?
if [ $cmake_result -ne 0 ]; then
    echo -e "${RED}cmake fails${NC}"
else
    echo -e "${Green}cmake suceeds${NC}"
fi

export OMPI_CXX=~/kokkos/bin/nvcc_wrapper
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

export OMP_NUM_THREADS=30
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export CUDA_VISIBLE_DEVICES=0

./hignn