#!/bin/bash -l
#SBATCH --job-name=structure_function
#SBATCH --account="s1069"
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun /users/trohner/alsvinn_build/library_examples/alsuq/structure_standalone/build/structure_standalone --platform cuda --p 2 --input-filename brownian_bridge_2D_N128_H0.15_T1.0 --output-filename brownian_bridge_2D_N128_H0.15_T0.0 --samples 128 --number-of-h 64 --nx 128 --ny 128 --nz 1 --time 0.000000
srun /users/trohner/alsvinn_build/library_examples/alsuq/structure_standalone/build/structure_standalone --platform cuda --p 2 --input-filename brownian_bridge_2D_N256_H0.15_T1.0 --output-filename brownian_bridge_2D_N256_H0.15_T0.0 --samples 256 --number-of-h 128 --nx 256 --ny 256 --nz 1 --time 0.000000
srun /users/trohner/alsvinn_build/library_examples/alsuq/structure_standalone/build/structure_standalone --platform cuda --p 2 --input-filename brownian_bridge_2D_N512_H0.15_T1.0 --output-filename brownian_bridge_2D_N512_H0.15_T0.0 --samples 512 --number-of-h 256 --nx 512 --ny 512 --nz 1 --time 0.000000
srun /users/trohner/alsvinn_build/library_examples/alsuq/structure_standalone/build/structure_standalone --platform cuda --p 2 --input-filename brownian_bridge_2D_N128_H0.5_T1.0 --output-filename brownian_bridge_2D_N128_H0.5_T0.0 --samples 128 --number-of-h 64 --nx 128 --ny 128 --nz 1 --time 0.000000
srun /users/trohner/alsvinn_build/library_examples/alsuq/structure_standalone/build/structure_standalone --platform cuda --p 2 --input-filename brownian_bridge_2D_N256_H0.5_T1.0 --output-filename brownian_bridge_2D_N256_H0.5_T0.0 --samples 256 --number-of-h 128 --nx 256 --ny 256 --nz 1 --time 0.000000
srun /users/trohner/alsvinn_build/library_examples/alsuq/structure_standalone/build/structure_standalone --platform cuda --p 2 --input-filename brownian_bridge_2D_N512_H0.5_T1.0 --output-filename brownian_bridge_2D_N512_H0.5_T0.0 --samples 512 --number-of-h 256 --nx 512 --ny 512 --nz 1 --time 0.000000
srun /users/trohner/alsvinn_build/library_examples/alsuq/structure_standalone/build/structure_standalone --platform cuda --p 2 --input-filename brownian_bridge_2D_N128_H0.75_T1.0 --output-filename brownian_bridge_2D_N128_H0.75_T0.0 --samples 128 --number-of-h 64 --nx 128 --ny 128 --nz 1 --time 0.000000
srun /users/trohner/alsvinn_build/library_examples/alsuq/structure_standalone/build/structure_standalone --platform cuda --p 2 --input-filename brownian_bridge_2D_N256_H0.75_T1.0 --output-filename brownian_bridge_2D_N256_H0.75_T0.0 --samples 256 --number-of-h 128 --nx 256 --ny 256 --nz 1 --time 0.000000
srun /users/trohner/alsvinn_build/library_examples/alsuq/structure_standalone/build/structure_standalone --platform cuda --p 2 --input-filename brownian_bridge_2D_N512_H0.75_T1.0 --output-filename brownian_bridge_2D_N512_H0.75_T0.0 --samples 512 --number-of-h 256 --nx 512 --ny 512 --nz 1 --time 0.000000
