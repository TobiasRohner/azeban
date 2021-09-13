#!/bin/bash -l
#SBATCH --job-name=structure_function
#SBATCH --account="s1069"
#SBATCH --time=8:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun /users/trohner/alsvinn_build/library_examples/alsuq/structure_standalone/build/structure_standalone --platform cuda --p 2 --input-filename taylor_green_3D_N128_perturb_T5.0 --output-filename taylor_green_3D_N128_T5.0 --samples 128 --number-of-h 64 --nx 128 --ny 128 --nz 128 --time 5.000000
