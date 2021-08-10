#!/bin/bash -l
#SBATCH --job-name=postprocess
#SBATCH --account="s1069"
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun /users/trohner/azeban/experiments/DiscontinuousVortexPatch/postprocessing.py Step Smooth Quadratic discontinuous_vortex_patch_2D_errors.pkl
