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

srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_3D_N32_H0.15_T1.0_structure_cube_2.000000_0.nc --output brownian_bridge_3D_N32_H0.15_T1.0_structure_cube_2.000000_0_postprocess.nc --nx 32 --ny 32 --nz 32 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_3D_N32_H0.5_T1.0_structure_cube_2.000000_0.nc --output brownian_bridge_3D_N32_H0.5_T1.0_structure_cube_2.000000_0_postprocess.nc --nx 32 --ny 32 --nz 32 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_3D_N32_H0.75_T1.0_structure_cube_2.000000_0.nc --output brownian_bridge_3D_N32_H0.75_T1.0_structure_cube_2.000000_0_postprocess.nc --nx 32 --ny 32 --nz 32 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_3D_N64_H0.15_T1.0_structure_cube_2.000000_0.nc --output brownian_bridge_3D_N64_H0.15_T1.0_structure_cube_2.000000_0_postprocess.nc --nx 64 --ny 64 --nz 64 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_3D_N64_H0.5_T1.0_structure_cube_2.000000_0.nc --output brownian_bridge_3D_N64_H0.5_T1.0_structure_cube_2.000000_0_postprocess.nc --nx 64 --ny 64 --nz 64 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_3D_N64_H0.75_T1.0_structure_cube_2.000000_0.nc --output brownian_bridge_3D_N64_H0.75_T1.0_structure_cube_2.000000_0_postprocess.nc --nx 64 --ny 64 --nz 64 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_3D_N128_H0.15_T1.0_structure_cube_2.000000_0.nc --output brownian_bridge_3D_N128_H0.15_T1.0_structure_cube_2.000000_0_postprocess.nc --nx 128 --ny 128 --nz 128 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_3D_N128_H0.5_T1.0_structure_cube_2.000000_0.nc --output brownian_bridge_3D_N128_H0.5_T1.0_structure_cube_2.000000_0_postprocess.nc --nx 128 --ny 128 --nz 128 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_3D_N128_H0.75_T1.0_structure_cube_2.000000_0.nc --output brownian_bridge_3D_N128_H0.75_T1.0_structure_cube_2.000000_0_postprocess.nc --nx 128 --ny 128 --nz 128 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_3D_N256_H0.75_T1.0_structure_cube_2.000000_0.nc --output brownian_bridge_3D_N256_H0.75_T1.0_structure_cube_2.000000_0_postprocess.nc --nx 256 --ny 256 --nz 256 -p 2
