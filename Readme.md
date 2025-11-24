# H-HIGNN Toolkit

## Introduction

$\mathcal{H}$-HIGNN is a framework designed for efficient and scalable simulation of large-scale particulate suspensions. It effectively captures both short- and long-range HIs and their many-body effects and enables substantial computational acceleration by harvesting the power of machine learning and hierarchical matrix.

## Prerequisites
[Nvidia driver](https://www.nvidia.com/en-us/drivers/) (currently, it requires >= 11.8), [docker engine](https://docs.docker.com/engine/install/) and [Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Prepare the docker environment
One can build the docker image locally
```shell
git clone https://github.com/Pan-Group-UW-Madison/hignn/tree/main --recursive
cd hignn/script
docker build --rm -f Dockerfile.hignn.Ampere -t hignn .
```
or pull the image from the docker hub. By default, it assumes the Ampere architecture
```shell
docker pull panlabuwmadison/hignn:latest
```
On Ada Lovelace architecture
```shell
docker pull panlabuwmadison/hignn:adalovelace
```
If using CPU only, one can pull the image via
```
docker pull panlabuwmadison/hignn:cpu
```

## Initialize
Rename the image with
```shell
docker image tag panlabuwmadison/hignn:{ARCH} hignn
```
where {ARCH} can be latest/ampere/adalovelace/cpu.

On Linux
```shell
docker run --privileged -it --rm -v $PWD:/local -w /local --entrypoint /bin/bash --gpus=all --shm-size=4g --hostname hignn hignn
```
On Windows
```shell
docker run --privileged -it --rm -v ${PWD}:/local -w /local --entrypoint /bin/bash --gpus=all --shm-size=4g --hostname hignn hignn
```
If using cpu only, please use
```shell
docker run --privileged -it --rm -v $PWD:/local -w /local --entrypoint /bin/bash --shm-size=4g --hostname hignn hignn:cpu
```

## Compile the code

After entering the docker, you can run the Python script at the root directory as follows:
```shell
python3 python/compile.py --rebuild
```

The above command only need once, the argument ``--rebuild`` is no more needed after the first time.  One only needs
```shell
python3 python/compile.py
```
Also, re-entering to the docker environment won't need to compile the code again if the code is unchanged.

## Run the code

```shell
python3 python/engine.py $config_file $new_cmd
```

One can select ``$new_cmd`` from one of ``--generate``, ``--simulate`` and ``--visualize``, which can be used to generate the initial configuration of particles, perform the simulation and post-processing the data, respectively.

## Test

The test can be launched via
```
pytest test/test_engine.py
```
within the docker image.

## Documentation

HIGNN uses [Doxygen](https://www.doxygen.nl/index.html) for the generation of API documentation. To generate the documentation, one only needs to run the folllowing command:
```[bash]
doxygen Doxyfile
```