#!/bin/bash

path="$(realpath "$(dirname "$(readlink -f "$0")")")"
ID1=$(sbatch --parsable ${path}/simulation_N32_rho0.1_delta0.025.slurm)
ID2=$(sbatch --parsable ${path}/simulation_N64_rho0.1_delta0.025.slurm)
ID3=$(sbatch --parsable ${path}/simulation_N128_rho0.1_delta0.025.slurm)
ID4=$(sbatch --parsable ${path}/simulation_N256_rho0.1_delta0.025.slurm)
sbatch --dependency=afterok:${ID1},afterok:${ID2} ${path}/structure_function_N32_N64.slurm
sbatch --dependency=afterok:${ID3} ${path}/structure_function_N128.slurm
sbatch --dependency=afterok:${ID4} ${path}/structure_function_N256.slurm
