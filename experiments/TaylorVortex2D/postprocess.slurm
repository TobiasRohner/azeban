#!/bin/bash -l
#SBATCH --job-name=taylor_vortex_post
#SBATCH --account="s1069"
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun /users/trohner/azeban/experiments/TaylorVortex2D/postprocessing.py None Step Smooth Quadratic taylor_vortex_2D_errors.pkl
