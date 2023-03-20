if [ $1 == "rebuild" ]; then
    rm CMakeCache.txt
    rm -rf CMakeFiles/

    cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ./
fi

cmake_result=$?
if [ $cmake_result -ne 0 ]; then
    echo -e "${RED}cmake fails${NC}"
else
    echo -e "${Green}cmake suceeds${NC}"
fi

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