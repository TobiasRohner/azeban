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

srun /users/trohner/azeban/experiments/plot_snapshot.py brownian_bridge_2D_N128_H0.15_T0.0 brownian_bridge_2D_N128_H0.15_T1.0/sample_*_time_0.000000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py brownian_bridge_2D_N128_H0.5_T0.0 brownian_bridge_2D_N128_H0.5_T1.0/sample_*_time_0.000000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py brownian_bridge_2D_N128_H0.75_T0.0 brownian_bridge_2D_N128_H0.75_T1.0/sample_*_time_0.000000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py brownian_bridge_2D_N256_H0.15_T0.0 brownian_bridge_2D_N256_H0.15_T1.0/sample_*_time_0.000000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py brownian_bridge_2D_N256_H0.5_T0.0 brownian_bridge_2D_N256_H0.5_T1.0/sample_*_time_0.000000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py brownian_bridge_2D_N256_H0.75_T0.0 brownian_bridge_2D_N256_H0.75_T1.0/sample_*_time_0.000000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py brownian_bridge_2D_N512_H0.15_T0.0 brownian_bridge_2D_N512_H0.15_T1.0/sample_*_time_0.000000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py brownian_bridge_2D_N512_H0.5_T0.0 brownian_bridge_2D_N512_H0.5_T1.0/sample_*_time_0.000000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py brownian_bridge_2D_N512_H0.75_T0.0 brownian_bridge_2D_N512_H0.75_T1.0/sample_*_time_0.000000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py brownian_bridge_2D_N128_H0.15_T1.0 brownian_bridge_2D_N128_H0.15_T1.0/sample_*_time_1.000000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py brownian_bridge_2D_N128_H0.5_T1.0 brownian_bridge_2D_N128_H0.5_T1.0/sample_*_time_1.000000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py brownian_bridge_2D_N128_H0.75_T1.0 brownian_bridge_2D_N128_H0.75_T1.0/sample_*_time_1.000000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py brownian_bridge_2D_N256_H0.15_T1.0 brownian_bridge_2D_N256_H0.15_T1.0/sample_*_time_1.000000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py brownian_bridge_2D_N256_H0.5_T1.0 brownian_bridge_2D_N256_H0.5_T1.0/sample_*_time_1.000000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py brownian_bridge_2D_N256_H0.75_T1.0 brownian_bridge_2D_N256_H0.75_T1.0/sample_*_time_1.000000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py brownian_bridge_2D_N512_H0.15_T1.0 brownian_bridge_2D_N512_H0.15_T1.0/sample_*_time_1.000000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py brownian_bridge_2D_N512_H0.5_T1.0 brownian_bridge_2D_N512_H0.5_T1.0/sample_*_time_1.000000.nc
srun /users/trohner/azeban/experiments/plot_snapshot.py brownian_bridge_2D_N512_H0.75_T1.0 brownian_bridge_2D_N512_H0.75_T1.0/sample_*_time_1.000000.nc
