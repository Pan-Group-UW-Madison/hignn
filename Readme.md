# Prerequisites
[Nvidia driver](https://www.nvidia.com/en-us/drivers/), [docker engine](https://docs.docker.com/engine/install/) and [Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

# Prepare the docker environment
One can build the docker image locally
```shell
cd script
docker build --rm -f Dockerfile.hignn -t hignn .
```
or pull the image from the docker hub. On Ampere architecture
```shell
docker push zishengy/hignn:latest
```
On Ada Lovelace architecture
```shell
docker pull zishengy/hignn:adalovelace
```

# Initialize

On Linux
```shell
docker run --privileged -it --rm -v $PWD:/local -w /local --entrypoint /bin/bash --gpus=all --shm-size=4g --hostname hignn hignn
```
On Windows
```shell
docker run --privileged -it --rm -v ${PWD}:/local -w /local --entrypoint /bin/bash --gpus=all --shm-size=4g --hostname hignn hignn
```

# Compile the code

After entering the docker, you can run the Python script at the root directory as follows:
```shell
python3 python/init.py --rebuild
```

The above command only need once, the argument ``--rebuild`` is no more needed after the first time.  Also, re-entering to the docker environment won't need to compile the code again if the code is unchanged.

# Run the code

```shell
python3 python/example.py
```

To initialize the mpirun running, one can use the following command:
```shell
mpirun -n $n python3 python/example.py
```
For ``$n``, you can change it to whatever number of GPUs you want to use.

# Manage model
For multiple GPUs, one have to convert the models into duplicates in ``nn`` directory.  To initiate it, you should use the ``init.py`` with the ``--model`` argument.