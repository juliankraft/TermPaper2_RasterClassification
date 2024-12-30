#!/bin/bash
#SBATCH --job-name=dataprocessing_sa2_kraftjul
#SBATCH --mail-type=all
#SBATCH --nodes=7
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --partition=earth-1

module load USS/2020

module load gcc/7.3.0
module load miniconda3/4.8.2
module load lsfm-init-miniconda/1.0.0
conda activate processing_env

python dataprocessing.py > dataprocessing.out