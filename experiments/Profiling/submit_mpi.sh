#!/bin/bash

mkdir profiling
cd profiling
for N in 32 64 128 256
do
  for r in 2 4 8
  do
    mkdir discontinuous_shear_tube_N${N}_mpi_rank${r}
    sbatch --chdir=discontinuous_shear_tube_N${N}_mpi_rank${r} /users/trohner/azeban/experiments/Profiling/simulation_3D_N${N}_rho0.1_delta0.025_mpi_rank${r}.slurm
  done
done
