#!/bin/bash -l
#SBATCH --job-name=tg_N1024_ss_8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobias.rohner@math.ethz.ch
#SBATCH --account="s1069"
#SBATCH --time=03:00:00
#SBATCH --nodes=256
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --output=slurm-%x.%j.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun /users/trohner/azeban/build_profile/azeban --ranks-per-sample=256 config_8.json
