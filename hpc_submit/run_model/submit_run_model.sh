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

echo '#########################################################################################'
echo '### Host info: ##########################################################################'
echo
echo 'Running on host:'
hostname
echo
nvidia-smi
echo
echo 'Working directory:'
cd /cfs/earth/scratch/kraftjul/sa2/code
pwd
echo
echo '#########################################################################################'
echo
echo '#########################################################################################'
echo '### Arguments: ##########################################################################'
echo
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
    # --overwrite
    # --dev_run
    --use_class_weights
    # --disable_progress_bar
    # --output_path=/cfs/earth/scratch/kraftjul/sa2/runs/
    --label_type=sealed_simple
)

for arg in "${PYTHON_ARGS[@]}"; do
    echo "$arg"
done
echo
echo '#########################################################################################'
echo '### Running skript ######################################################################'
echo '#########################################################################################'
echo

micromamba run -n sa2 python run_model.py "${PYTHON_ARGS[@]}"

echo
echo '#########################################################################################'
echo '### Completed skript ####################################################################'
echo '#########################################################################################'