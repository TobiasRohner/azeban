#!/bin/bash -l
#SBATCH --job-name=brownian_bridge
#SBATCH --account="s1069"
#SBATCH --time=01:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/BrownianBridge3D/brownian_bridge_N64_H0.75.json
