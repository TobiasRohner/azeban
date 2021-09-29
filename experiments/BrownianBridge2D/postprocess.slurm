#!/bin/bash -l
#SBATCH --job-name=postprocess
#SBATCH --account="s1069"
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_2D_N128_H0.15_T0.0_structure_cube_2.000000_0.nc --output brownian_bridge_2D_N128_H0.15_T0.0_structure_cube_2.000000_0_postprocess.nc --nx 128 --ny 128 --nz 1 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_2D_N256_H0.15_T0.0_structure_cube_2.000000_0.nc --output brownian_bridge_2D_N256_H0.15_T0.0_structure_cube_2.000000_0_postprocess.nc --nx 256 --ny 256 --nz 1 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_2D_N512_H0.15_T0.0_structure_cube_2.000000_0.nc --output brownian_bridge_2D_N512_H0.15_T0.0_structure_cube_2.000000_0_postprocess.nc --nx 512 --ny 512 --nz 1 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_2D_N128_H0.5_T0.0_structure_cube_2.000000_0.nc --output brownian_bridge_2D_N128_H0.5_T0.0_structure_cube_2.000000_0_postprocess.nc --nx 128 --ny 128 --nz 1 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_2D_N256_H0.5_T0.0_structure_cube_2.000000_0.nc --output brownian_bridge_2D_N256_H0.5_T0.0_structure_cube_2.000000_0_postprocess.nc --nx 256 --ny 256 --nz 1 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_2D_N512_H0.5_T0.0_structure_cube_2.000000_0.nc --output brownian_bridge_2D_N512_H0.5_T0.0_structure_cube_2.000000_0_postprocess.nc --nx 512 --ny 512 --nz 1 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_2D_N128_H0.75_T0.0_structure_cube_2.000000_0.nc --output brownian_bridge_2D_N128_H0.75_T0.0_structure_cube_2.000000_0_postprocess.nc --nx 128 --ny 128 --nz 1 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_2D_N256_H0.75_T0.0_structure_cube_2.000000_0.nc --output brownian_bridge_2D_N256_H0.75_T0.0_structure_cube_2.000000_0_postprocess.nc --nx 256 --ny 256 --nz 1 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_2D_N512_H0.75_T0.0_structure_cube_2.000000_0.nc --output brownian_bridge_2D_N512_H0.75_T0.0_structure_cube_2.000000_0_postprocess.nc --nx 512 --ny 512 --nz 1 -p 2

srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_2D_N128_H0.15_T1.0_structure_cube_2.000000_0.nc --output brownian_bridge_2D_N128_H0.15_T1.0_structure_cube_2.000000_0_postprocess.nc --nx 128 --ny 128 --nz 1 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_2D_N256_H0.15_T1.0_structure_cube_2.000000_0.nc --output brownian_bridge_2D_N256_H0.15_T1.0_structure_cube_2.000000_0_postprocess.nc --nx 256 --ny 256 --nz 1 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_2D_N512_H0.15_T1.0_structure_cube_2.000000_0.nc --output brownian_bridge_2D_N512_H0.15_T1.0_structure_cube_2.000000_0_postprocess.nc --nx 512 --ny 512 --nz 1 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_2D_N128_H0.5_T1.0_structure_cube_2.000000_0.nc --output brownian_bridge_2D_N128_H0.5_T1.0_structure_cube_2.000000_0_postprocess.nc --nx 128 --ny 128 --nz 1 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_2D_N256_H0.5_T1.0_structure_cube_2.000000_0.nc --output brownian_bridge_2D_N256_H0.5_T1.0_structure_cube_2.000000_0_postprocess.nc --nx 256 --ny 256 --nz 1 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_2D_N512_H0.5_T1.0_structure_cube_2.000000_0.nc --output brownian_bridge_2D_N512_H0.5_T1.0_structure_cube_2.000000_0_postprocess.nc --nx 512 --ny 512 --nz 1 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_2D_N128_H0.75_T1.0_structure_cube_2.000000_0.nc --output brownian_bridge_2D_N128_H0.75_T1.0_structure_cube_2.000000_0_postprocess.nc --nx 128 --ny 128 --nz 1 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_2D_N256_H0.75_T1.0_structure_cube_2.000000_0.nc --output brownian_bridge_2D_N256_H0.75_T1.0_structure_cube_2.000000_0_postprocess.nc --nx 256 --ny 256 --nz 1 -p 2
srun /users/trohner/azeban/scripts/structure_postprocess.py --input brownian_bridge_2D_N512_H0.75_T1.0_structure_cube_2.000000_0.nc --output brownian_bridge_2D_N512_H0.75_T1.0_structure_cube_2.000000_0_postprocess.nc --nx 512 --ny 512 --nz 1 -p 2
