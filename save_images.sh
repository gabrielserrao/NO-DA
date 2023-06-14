#!/bin/bash

#SBATCH --job-name=DA-FNO
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --account=research-ceg-gse
#SBATCH --mem=150G
##add log file
#request only 1 cpu core
#SBATCH --cpus-per-task=1

#SBATCH --output=slurm-%x-%j.out
##error file
#SBATCH --error=err_slurm-%x-%j.err



# Load modules:
module load 2022r2
module load openmpi
module load miniconda3

python3 DataAssimilation.py