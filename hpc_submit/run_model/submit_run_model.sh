#!/usr/bin/env bash

#SBATCH --job-name=run_model

#SBATCH --mail-type=fail,end

#SBATCH --time=01-00:00:00
#SBATCH --partition=earth-4
#SBATCH --constraint=rhel8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
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
echo "#########################################   Run Model"
echo
echo "canging wd"
cd /cfs/earth/scratch/kraftjul/sa2/code
pwd
echo
echo "#########################################   Lets go!"
echo

micromamba run -n sa2 python run_model.py --device=gpu --num_workers=6 --learning_rate=0.001 --patience=20 --batch_size=256 --overwrite
