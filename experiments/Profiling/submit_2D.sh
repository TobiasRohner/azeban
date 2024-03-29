#!/bin/bash

mkdir profiling
cd profiling
mkdir discontinuous_shear_layer_N128_cpu
sbatch --chdir=discontinuous_shear_layer_N128_cpu /users/trohner/azeban/experiments/Profiling/simulation_2D_N128_rho0.1_delta0.025_cpu.slurm
mkdir discontinuous_shear_layer_N256_cpu
sbatch --chdir=discontinuous_shear_layer_N256_cpu /users/trohner/azeban/experiments/Profiling/simulation_2D_N256_rho0.1_delta0.025_cpu.slurm
mkdir discontinuous_shear_layer_N512_cpu
sbatch --chdir=discontinuous_shear_layer_N512_cpu /users/trohner/azeban/experiments/Profiling/simulation_2D_N512_rho0.1_delta0.025_cpu.slurm
mkdir discontinuous_shear_layer_N1024_cpu
sbatch --chdir=discontinuous_shear_layer_N1024_cpu /users/trohner/azeban/experiments/Profiling/simulation_2D_N1024_rho0.1_delta0.025_cpu.slurm
mkdir discontinuous_shear_layer_N128_gpu
sbatch --chdir=discontinuous_shear_layer_N128_gpu /users/trohner/azeban/experiments/Profiling/simulation_2D_N128_rho0.1_delta0.025_gpu.slurm
mkdir discontinuous_shear_layer_N256_gpu
sbatch --chdir=discontinuous_shear_layer_N256_gpu /users/trohner/azeban/experiments/Profiling/simulation_2D_N256_rho0.1_delta0.025_gpu.slurm
mkdir discontinuous_shear_layer_N512_gpu
sbatch --chdir=discontinuous_shear_layer_N512_gpu /users/trohner/azeban/experiments/Profiling/simulation_2D_N512_rho0.1_delta0.025_gpu.slurm
mkdir discontinuous_shear_layer_N1024_gpu
sbatch --chdir=discontinuous_shear_layer_N1024_gpu /users/trohner/azeban/experiments/Profiling/simulation_2D_N1024_rho0.1_delta0.025_gpu.slurm
