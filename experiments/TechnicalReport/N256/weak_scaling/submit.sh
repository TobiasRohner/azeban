#!/bin/bash -l
#SBATCH --job-name=tg_N256_weak_1
#SBATCH --account="s1069"
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --switches=1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/TechnicalReport/N256/weak_scaling/config.json
