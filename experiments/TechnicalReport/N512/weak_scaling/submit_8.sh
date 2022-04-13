#!/bin/bash -l
#SBATCH --job-name=tg_N512_weak_8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobias.rohner@math.ethz.ch
#SBATCH --account="s1069"
#SBATCH --time=24:00:00
#SBATCH --nodes=32
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --output=slurm-%x.%j.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun /users/trohner/azeban/build_profile/azeban --ranks-per-sample=4 /users/trohner/azeban/experiments/TechnicalReport/N512/weak_scaling/config_8.json
