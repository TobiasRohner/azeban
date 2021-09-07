#!/bin/bash

path="$(realpath "$(dirname "$(readlink -f "$0")")")"
sbatch ${path}/simulation_N128_rho0.1_delta0.025.slurm
sbatch ${path}/simulation_N256_rho0.1_delta0.025.slurm
sbatch ${path}/simulation_N512_rho0.1_delta0.025.slurm
sbatch ${path}/simulation_N1024_rho0.1_delta0.025.slurm
sbatch ${path}/simulation_N128_rho0_delta0.025.slurm
sbatch ${path}/simulation_N256_rho0_delta0.025.slurm
sbatch ${path}/simulation_N512_rho0_delta0.025.slurm
sbatch ${path}/simulation_N1024_rho0_delta0.025.slurm
