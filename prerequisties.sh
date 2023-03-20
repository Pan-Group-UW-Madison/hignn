#!/bin/bash

# install cuda driver

# install pytorch
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get install libcudnn8=8.5.0.*-1+cuda11.7
sudo apt-get install libcudnn8-dev=8.5.0.*-1+cuda11.7

pip install torch torchvision torchaudio

# install kokkos
cd ~
mkdir package
cd package
git clone https://github.com/kokkos/kokkos.git
cd kokkos
export OMPI_CXX=./bin/nvcc_wrapper
mkdir build
cd build
cmake -D CMAKE_INSTALL_PREFIX:PATH=~/kokkos/ -D Kokkos_ENABLE_OPENMP:BOOL=ON -D Kokkos_ENABLE_SERIAL:BOOL=OFF -D Kokkos_ENABLE_CUDA:BOOL=ON -D Kokkos_ENABLE_CUDA_UVM:BOOL=ON -D Kokkos_ENABLE_CUDA_LAMBDA:BOOL=ON -D Kokkos_ARCH_BDW:BOOL=ON -D Kokkos_ARCH_AMPERE86:BOOL=ON ../
make install -j

# prepare hignn directory
cd ~
cd hignn

git clone https://github.com/pybind/pybind11.git
git clone https://gitlab.com/libeigen/eigen.git