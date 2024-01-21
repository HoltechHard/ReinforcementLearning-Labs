#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J jupyter
#SBATCH -t 0-12

module load python/3.10

export JUPYTER_RUNTIME_DIR=/tmp
jupyter notebook --ip=$(hostname) --no-browser --port=8888