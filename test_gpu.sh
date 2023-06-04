#!/bin/bash

#SBATCH --job-name=FNO
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --account=research-ceg-gse
#SBATCH --mem=32G
##add log file
#SBATCH --output=slurm-%x-%j.out
##error file
#SBATCH --error=err_slurm-%x-%j.err



# Load modules:
module load 2022r2
module load openmpi
module load miniconda3

python3 fouier_3d_GPU_enabled.py
