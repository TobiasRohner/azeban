#!/bin/bash -l
#SBATCH --job-name=shear_tube
#SBATCH --account="s1069"
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/Profiling/discontinuous_shear_tube_N256_rho0.1_delta0.025_mpi.json
