#!/bin/bash -l
#SBATCH --job-name=discontinuous_vortex_patch
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

srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/DiscontinuousVortexPatch/discontinuous_vortex_patch_N128_Step.json
srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/DiscontinuousVortexPatch/discontinuous_vortex_patch_N256_Step.json
srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/DiscontinuousVortexPatch/discontinuous_vortex_patch_N512_Step.json
srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/DiscontinuousVortexPatch/discontinuous_vortex_patch_N1024_Step.json
srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/DiscontinuousVortexPatch/discontinuous_vortex_patch_N128_Smooth.json
srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/DiscontinuousVortexPatch/discontinuous_vortex_patch_N256_Smooth.json
srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/DiscontinuousVortexPatch/discontinuous_vortex_patch_N512_Smooth.json
srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/DiscontinuousVortexPatch/discontinuous_vortex_patch_N1024_Smooth.json
srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/DiscontinuousVortexPatch/discontinuous_vortex_patch_N128_Quadratic.json
srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/DiscontinuousVortexPatch/discontinuous_vortex_patch_N256_Quadratic.json
srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/DiscontinuousVortexPatch/discontinuous_vortex_patch_N512_Quadratic.json
srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/DiscontinuousVortexPatch/discontinuous_vortex_patch_N1024_Quadratic.json
