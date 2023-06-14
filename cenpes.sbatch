#!/bin/bash
#SBATCH --account=td
##SBATCH --chdir=/nethome/reservatorio01/usuarios/ur7y/CMG
##partition=gpu
#SBATCH --partition=gpu
#SBATCH --job-name=serrao
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=40
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=1
##SBATCH --output= %a_base_case_log.txt
##SBATCH --error= %a_base_case_err.txt
#SBATCH --exclusive

 


##SBATCH -J $1

 


source /nethome/drp/microtom/init.sh

 

 

python fouier_3d_cenpes.py