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

# ## strict bash mode
set -eEuo pipefail

# module reset
module purge
module load DefaultModules

# ## init micromamba
export MAMBA_ROOT_PREFIX="/cfs/earth/scratch/${USER}/.conda/"
eval "$("/cfs/earth/scratch/${USER}/bin/micromamba" shell hook -s posix)"

hostname
# ## get GPU info
nvidia-smi

echo
echo "#########################################   Tensorflow Info"
echo

micromamba run -n sa2 python pytorch_info.py

# echo
# echo "#########################################   DL part"
# echo

# micromamba run -n pytorch python mnist.py
