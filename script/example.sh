#!/bin/bash
#SBATCH --job-name=example           # Job name
#SBATCH --output=example_%j.out       # Output file (%j will be replaced with job ID)
#SBATCH --error=example_%j.err        # Error file (%j will be replaced with job ID)
#SBATCH --ntasks=1                       # Number of tasks (processes)
#SBATCH --cpus-per-task=40                # Number of CPU cores per task
#SBATCH --gres=gpu:4                     # Request 1 GPU
#SBATCH --mem=16G                        # Memory per node
#SBATCH --time=01:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=hignn         # Partition to submit to

DOCKER_IMAGE="hignn"
DOCKER_CMD="mpirun -n 4 python3 python/example.py"

srun --gres=gpu:4 --mem=15G docker run --gpus all --rm -v $PWD:/local -w /local --entrypoint /bin/bash --shm-size=4g -e TERM=xterm $DOCKER_IMAGE -c "$DOCKER_CMD"