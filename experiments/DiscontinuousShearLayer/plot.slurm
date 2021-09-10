#!/bin/bash -l
#SBATCH --job-name=plot
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

srun /users/trohner/azeban/experiments/plot_snapshot.py discontinuous_shear_layer_2D_N128_rho0.1_delta0.025_T0.0 discontinuous_shear_layer_2D_N128_rho0.1_delta0.025_T1.0/sample_*_time_0.000000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py discontinuous_shear_layer_2D_N256_rho0.1_delta0.025_T0.0 discontinuous_shear_layer_2D_N256_rho0.1_delta0.025_T1.0/sample_*_time_0.000000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py discontinuous_shear_layer_2D_N512_rho0.1_delta0.025_T0.0 discontinuous_shear_layer_2D_N512_rho0.1_delta0.025_T1.0/sample_*_time_0.000000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py discontinuous_shear_layer_2D_N1024_rho0.1_delta0.025_T0.0 discontinuous_shear_layer_2D_N1024_rho0.1_delta0.025_T1.0/sample_*_time_0.000000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py discontinuous_shear_layer_2D_N128_rho0_delta0.025_T0.0 discontinuous_shear_layer_2D_N128_rho0_delta0.025_T1.0/sample_*_time_0.000000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py discontinuous_shear_layer_2D_N256_rho0_delta0.025_T0.0 discontinuous_shear_layer_2D_N256_rho0_delta0.025_T1.0/sample_*_time_0.000000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py discontinuous_shear_layer_2D_N512_rho0_delta0.025_T0.0 discontinuous_shear_layer_2D_N512_rho0_delta0.025_T1.0/sample_*_time_0.000000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py discontinuous_shear_layer_2D_N1024_rho0_delta0.025_T0.0 discontinuous_shear_layer_2D_N1024_rho0_delta0.025_T1.0/sample_*_time_0.000000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py discontinuous_shear_layer_2D_N128_rho0.1_delta0.025_T0.4 discontinuous_shear_layer_2D_N128_rho0.1_delta0.025_T1.0/sample_*_time_0.400000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py discontinuous_shear_layer_2D_N256_rho0.1_delta0.025_T0.4 discontinuous_shear_layer_2D_N256_rho0.1_delta0.025_T1.0/sample_*_time_0.400000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py discontinuous_shear_layer_2D_N512_rho0.1_delta0.025_T0.4 discontinuous_shear_layer_2D_N512_rho0.1_delta0.025_T1.0/sample_*_time_0.400000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py discontinuous_shear_layer_2D_N1024_rho0.1_delta0.025_T0.4 discontinuous_shear_layer_2D_N1024_rho0.1_delta0.025_T1.0/sample_*_time_0.400000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py discontinuous_shear_layer_2D_N128_rho0_delta0.025_T0.4 discontinuous_shear_layer_2D_N128_rho0_delta0.025_T1.0/sample_*_time_0.400000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py discontinuous_shear_layer_2D_N256_rho0_delta0.025_T0.4 discontinuous_shear_layer_2D_N256_rho0_delta0.025_T1.0/sample_*_time_0.400000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py discontinuous_shear_layer_2D_N512_rho0_delta0.025_T0.4 discontinuous_shear_layer_2D_N512_rho0_delta0.025_T1.0/sample_*_time_0.400000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py discontinuous_shear_layer_2D_N1024_rho0_delta0.025_T0.4 discontinuous_shear_layer_2D_N1024_rho0_delta0.025_T1.0/sample_*_time_0.400000.nc
