#!/bin/bash -l
#SBATCH --job-name=structure_function
#SBATCH --account="s1069"
#SBATCH --time=24:00:00
#SBATCH --nodes=256
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun /users/trohner/alsvinn_build/library_examples/alsuq/structure_standalone/build/structure_standalone --platform cuda --p 2 --input-filename taylor_green_3D_N256_perturb_T5.0 --output-filename taylor_green_3D_N256_T5.0 --samples 256 --number-of-h 128 --nx 256 --ny 256 --nz 256 --time 5.000000
