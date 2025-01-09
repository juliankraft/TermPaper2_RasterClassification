#!/usr/bin/env bash

#SBATCH --job-name=pytorch-test

#SBATCH --mail-type=fail,end

#SBATCH --time=00-01:00:00
#SBATCH --partition=earth-4
#SBATCH --constraint=rhel8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
# #SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:l40s:1

# shellcheck disable=SC1091
source load_env.sh

hostname
# ## get GPU info
nvidia-smi

echo
echo "#########################################   Tensorflow Info"
echo

micromamba run -n pytorch python pytorch_info.py

# echo
# echo "#########################################   DL part"
# echo

# micromamba run -n pytorch python mnist.py
