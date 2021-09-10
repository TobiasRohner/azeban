#!/bin/bash

path="$(realpath "$(dirname "$(readlink -f "$0")")")"
ID1=$(sbatch --parsable ${path}/simulation_N128_H0.15.slurm)
ID2=$(sbatch --parsable ${path}/simulation_N256_H0.15.slurm)
ID3=$(sbatch --parsable ${path}/simulation_N512_H0.15.slurm)
ID4=$(sbatch --parsable ${path}/simulation_N128_H0.5.slurm)
ID5=$(sbatch --parsable ${path}/simulation_N256_H0.5.slurm)
ID6=$(sbatch --parsable ${path}/simulation_N512_H0.5.slurm)
ID7=$(sbatch --parsable ${path}/simulation_N128_H0.75.slurm)
ID8=$(sbatch --parsable ${path}/simulation_N256_H0.75.slurm)
ID9=$(sbatch --parsable ${path}/simulation_N512_H0.75.slurm)
ID10=$(sbatch --parsable --dependency=afterok:${ID1},afterok:${ID2},afterok:${ID3},afterok:${ID4},afterok:${ID5},afterok:${ID6},afterok:${ID7},afterok:${ID8},afterok:${ID9} ${path}/structure_function.slurm)
sbatch --dependency=afterok:${ID10} ${path}/postprocess.slurm
sbatch --parsable --dependency=afterok:${ID1},afterok:${ID2},afterok:${ID3},afterok:${ID4},afterok:${ID5},afterok:${ID6},afterok:${ID7},afterok:${ID8},afterok:${ID9} ${path}/plot.slurm
