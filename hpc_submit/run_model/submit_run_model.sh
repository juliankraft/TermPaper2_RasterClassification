#!/usr/bin/env bash

#SBATCH --job-name=run_model

#SBATCH --mail-type=fail,end

#SBATCH --time=01-00:00:00
#SBATCH --partition=earth-4
#SBATCH --constraint=rhel8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=42G
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
echo "#########################################   changing wd"
echo
echo "changing wd"
cd /cfs/earth/scratch/kraftjul/sa2/code
pwd
echo
echo "#########################################   defining drguments"
echo

# Define the arguments for the Python script in a list
PYTHON_ARGS=(
    --device=gpu
    --batch_size=256
    --num_workers=12
    # --cutout_size=51
    # --output_patch_size=5
    --learning_rate=0.001
    # --weight_decay=0.0
    # --use_data_augmentation
    --patience=10
    --overwrite
    # --dev_run
    --use_class_weights
    # --disable_progress_bar
    # --output_path=/path/to/output
    --label_type=sealed
)

echo "${PYTHON_ARGS[@]}"

echo
echo "#########################################   start model"

# Pass the arguments to the Python script
micromamba run -n sa2 python run_model.py "${PYTHON_ARGS[@]}"