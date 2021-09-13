#!/bin/bash -l
#SBATCH --job-name=structure_function
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

srun /users/trohner/alsvinn_build/library_examples/alsuq/structure_standalone/build/structure_standalone --platform cuda --p 2 --input-filename taylor_green_3D_N32_perturb_T5.0 --output-filename taylor_green_3D_N32_T5.0 --samples 32 --number-of-h 16 --nx 32 --ny 32 --nz 32 --time 5.000000
srun /users/trohner/alsvinn_build/library_examples/alsuq/structure_standalone/build/structure_standalone --platform cuda --p 2 --input-filename taylor_green_3D_N64_perturb_T5.0 --output-filename taylor_green_3D_N64_T5.0 --samples 64 --number-of-h 32 --nx 64 --ny 64 --nz 64 --time 5.000000
