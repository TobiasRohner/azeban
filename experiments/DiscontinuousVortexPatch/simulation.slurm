#!/bin/bash -l
#SBATCH --job-name=discontinuous_vortex_patch
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

srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/DiscontinuousVortexPatch/discontinuous_vortex_patch_N16_Step.json
srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/DiscontinuousVortexPatch/discontinuous_vortex_patch_N32_Step.json
srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/DiscontinuousVortexPatch/discontinuous_vortex_patch_N64_Step.json
srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/DiscontinuousVortexPatch/discontinuous_vortex_patch_N128_Step.json
srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/DiscontinuousVortexPatch/discontinuous_vortex_patch_N16_Smooth.json
srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/DiscontinuousVortexPatch/discontinuous_vortex_patch_N32_Smooth.json
srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/DiscontinuousVortexPatch/discontinuous_vortex_patch_N64_Smooth.json
srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/DiscontinuousVortexPatch/discontinuous_vortex_patch_N128_Smooth.json
srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/DiscontinuousVortexPatch/discontinuous_vortex_patch_N16_Quadratic.json
srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/DiscontinuousVortexPatch/discontinuous_vortex_patch_N32_Quadratic.json
srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/DiscontinuousVortexPatch/discontinuous_vortex_patch_N64_Quadratic.json
srun /users/trohner/azeban/build_profile/azeban /users/trohner/azeban/experiments/DiscontinuousVortexPatch/discontinuous_vortex_patch_N128_Quadratic.json
