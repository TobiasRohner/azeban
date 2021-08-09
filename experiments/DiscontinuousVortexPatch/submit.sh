#!/bin/bash

path="$(realpath "$(dirname "$(readlink -f "$0")")")"
ID=$(sbatch --parsable ${path}/simulation.slurm)
sbatch --dependency=afterok:${ID} ${path}/postprocess.slurm
