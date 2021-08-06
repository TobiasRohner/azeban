#!/bin/bash

path="$(realpath "$(dirname "$(readlink -f "$0")")")"
ID1=$(sbatch --parsable ${path}/simulation_N128_rho0.1_delta0.025.slurm)
ID2=$(sbatch --parsable ${path}/simulation_N256_rho0.1_delta0.025.slurm)
ID3=$(sbatch --parsable ${path}/simulation_N512_rho0.1_delta0.025.slurm)
ID4=$(sbatch --parsable ${path}/simulation_N1024_rho0.1_delta0.025.slurm)
ID5=$(sbatch --parsable --dependency=afterok:${ID1},afterok:${ID2},afterok:${ID3},afterok:${ID4} ${path}/structure_function.slurm)
sbatch --dependency=afterok:${ID5} ${path}/postprocess.slurm
