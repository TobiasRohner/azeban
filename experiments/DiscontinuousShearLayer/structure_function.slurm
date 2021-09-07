#!/bin/bash -l
#SBATCH --job-name=structure_function
#SBATCH --account="s1069"
#SBATCH --time=04:00:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun /users/trohner/alsvinn_build/library_examples/alsuq/structure_standalone/build/structure_standalone --platform cuda --p 2 --input-filename discontinuous_shear_layer_2D_N128_rho0.1_delta0.025_T1.0 --output-filename discontinuous_shear_layer_2D_N128_rho0.1_delta0.025_T0.4 --samples 128 --number-of-h 64 --nx 128 --ny 128 --nz 1 --time 0.400000
srun /users/trohner/alsvinn_build/library_examples/alsuq/structure_standalone/build/structure_standalone --platform cuda --p 2 --input-filename discontinuous_shear_layer_2D_N256_rho0.1_delta0.025_T1.0 --output-filename discontinuous_shear_layer_2D_N256_rho0.1_delta0.025_T0.4 --samples 256 --number-of-h 128 --nx 256 --ny 256 --nz 1 --time 0.400000
srun /users/trohner/alsvinn_build/library_examples/alsuq/structure_standalone/build/structure_standalone --platform cuda --p 2 --input-filename discontinuous_shear_layer_2D_N512_rho0.1_delta0.025_T1.0 --output-filename discontinuous_shear_layer_2D_N512_rho0.1_delta0.025_T0.4 --samples 512 --number-of-h 256 --nx 512 --ny 512 --nz 1 --time 0.400000
srun /users/trohner/alsvinn_build/library_examples/alsuq/structure_standalone/build/structure_standalone --platform cuda --p 2 --input-filename discontinuous_shear_layer_2D_N1024_rho0.1_delta0.025_T1.0 --output-filename discontinuous_shear_layer_2D_N1024_rho0.1_delta0.025_T0.4 --samples 1024 --number-of-h 512 --nx 1024 --ny 1024 --nz 1 --time 0.400000
srun /users/trohner/alsvinn_build/library_examples/alsuq/structure_standalone/build/structure_standalone --platform cuda --p 2 --input-filename discontinuous_shear_layer_2D_N128_rho0_delta0.025_T1.0 --output-filename discontinuous_shear_layer_2D_N128_rho0_delta0.025_T0.4 --samples 128 --number-of-h 64 --nx 128 --ny 128 --nz 1 --time 0.400000
srun /users/trohner/alsvinn_build/library_examples/alsuq/structure_standalone/build/structure_standalone --platform cuda --p 2 --input-filename discontinuous_shear_layer_2D_N256_rho0_delta0.025_T1.0 --output-filename discontinuous_shear_layer_2D_N256_rho0_delta0.025_T0.4 --samples 256 --number-of-h 128 --nx 256 --ny 256 --nz 1 --time 0.400000
srun /users/trohner/alsvinn_build/library_examples/alsuq/structure_standalone/build/structure_standalone --platform cuda --p 2 --input-filename discontinuous_shear_layer_2D_N512_rho0_delta0.025_T1.0 --output-filename discontinuous_shear_layer_2D_N512_rho0_delta0.025_T0.4 --samples 512 --number-of-h 256 --nx 512 --ny 512 --nz 1 --time 0.400000
srun /users/trohner/alsvinn_build/library_examples/alsuq/structure_standalone/build/structure_standalone --platform cuda --p 2 --input-filename discontinuous_shear_layer_2D_N1024_rho0_delta0.025_T1.0 --output-filename discontinuous_shear_layer_2D_N1024_rho0_delta0.025_T0.4 --samples 1024 --number-of-h 512 --nx 1024 --ny 1024 --nz 1 --time 0.400000
