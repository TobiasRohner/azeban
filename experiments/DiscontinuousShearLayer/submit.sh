#!/bin/bash

path="$(realpath "$(dirname "$(readlink -f "$0")")")"
ID1=$(sbatch --parsable ${path}/simulation_N128_rho0.1_delta0.025.slurm)
ID2=$(sbatch --parsable ${path}/simulation_N256_rho0.1_delta0.025.slurm)
ID3=$(sbatch --parsable ${path}/simulation_N512_rho0.1_delta0.025.slurm)
ID4=$(sbatch --parsable ${path}/simulation_N1024_rho0.1_delta0.025.slurm)
ID5=$(sbatch --parsable ${path}/simulation_N128_rho0_delta0.025.slurm)
ID6=$(sbatch --parsable ${path}/simulation_N256_rho0_delta0.025.slurm)
ID7=$(sbatch --parsable ${path}/simulation_N512_rho0_delta0.025.slurm)
ID8=$(sbatch --parsable ${path}/simulation_N1024_rho0_delta0.025.slurm)
ID9=$(sbatch --parsable --dependency=afterok:${ID1},afterok:${ID2},afterok:${ID3},afterok:${ID4},afterok:${ID5},afterok:${ID6},afterok:${ID7},afterok:${ID8} ${path}/structure_function.slurm)
sbatch --dependency=afterok:${ID9} ${path}/postprocess.slurm
sbatch --parsable --dependency=afterok:${ID1},afterok:${ID2},afterok:${ID3},afterok:${ID4},afterok:${ID5},afterok:${ID6},afterok:${ID7},afterok:${ID8} ${path}/plot.slurm
