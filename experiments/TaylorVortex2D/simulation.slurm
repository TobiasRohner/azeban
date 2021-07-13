#!/bin/bash -l
#SBATCH --job-name=taylor_vortex
#SBATCH --account="s1069"
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/TaylorVortex2D/taylor_vortex_N16.json
srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/TaylorVortex2D/taylor_vortex_N32.json
srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/TaylorVortex2D/taylor_vortex_N64.json
srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/TaylorVortex2D/taylor_vortex_N128.json
