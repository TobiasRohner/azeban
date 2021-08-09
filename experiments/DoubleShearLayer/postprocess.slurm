#!/bin/bash -l
#SBATCH --job-name=postprocess
#SBATCH --account="s1069"
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun /users/trohner/azeban/scripts/structure_postprocess.py --input double_shear_layer_2D_N128_rho0.1_delta0.025_T0.4_structure_cube_2.000000_0.nc --output double_shear_layer_2D_N128_rho0.1_delta0.025_T0.4_structure_cube_2.000000_0_postprocess.nc --nx 128 --ny 128 --nz 1 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input double_shear_layer_2D_N256_rho0.1_delta0.025_T0.4_structure_cube_2.000000_0.nc --output double_shear_layer_2D_N256_rho0.1_delta0.025_T0.4_structure_cube_2.000000_0_postprocess.nc --nx 256 --ny 256 --nz 1 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input double_shear_layer_2D_N512_rho0.1_delta0.025_T0.4_structure_cube_2.000000_0.nc --output double_shear_layer_2D_N512_rho0.1_delta0.025_T0.4_structure_cube_2.000000_0_postprocess.nc --nx 512 --ny 512 --nz 1 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input double_shear_layer_2D_N1024_rho0.1_delta0.025_T0.4_structure_cube_2.000000_0.nc --output double_shear_layer_2D_N1024_rho0.1_delta0.025_T0.4_structure_cube_2.000000_0_postprocess.nc --nx 1024 --ny 1024 --nz 1 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input double_shear_layer_2D_N128_rho0_delta0.025_T0.4_structure_cube_2.000000_0.nc --output double_shear_layer_2D_N128_rho0_delta0.025_T0.4_structure_cube_2.000000_0_postprocess.nc --nx 128 --ny 128 --nz 1 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input double_shear_layer_2D_N256_rho0_delta0.025_T0.4_structure_cube_2.000000_0.nc --output double_shear_layer_2D_N256_rho0_delta0.025_T0.4_structure_cube_2.000000_0_postprocess.nc --nx 256 --ny 256 --nz 1 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input double_shear_layer_2D_N512_rho0_delta0.025_T0.4_structure_cube_2.000000_0.nc --output double_shear_layer_2D_N512_rho0_delta0.025_T0.4_structure_cube_2.000000_0_postprocess.nc --nx 512 --ny 512 --nz 1 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input double_shear_layer_2D_N1024_rho0_delta0.025_T0.4_structure_cube_2.000000_0.nc --output double_shear_layer_2D_N1024_rho0_delta0.025_T0.4_structure_cube_2.000000_0_postprocess.nc --nx 1024 --ny 1024 --nz 1 -p 2
