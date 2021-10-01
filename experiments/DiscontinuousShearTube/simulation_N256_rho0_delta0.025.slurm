#!/bin/bash -l
#SBATCH --job-name=shear_tube
#SBATCH --account="s1069"
#SBATCH --time=24:00:00
#SBATCH --array=0-255
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

sis=`echo "${SLURM_ARRAY_TASK_ID}" | bc`
seed=`echo "${SLURM_ARRAY_TASK_ID} + 1" | bc`
cp /users/trohner/azeban/experiments/DiscontinuousShearTube/discontinuous_shear_tube_N256_rho0_delta0.025.json discontinuous_shear_tube_N256_rho0_delta0.025_T1.0_${SLURM_ARRAY_TASK_ID}.json
sed -i "s/<sis>/${sis}/" discontinuous_shear_tube_N256_rho0_delta0.025_T1.0_${SLURM_ARRAY_TASK_ID}.json
sed -i "s/<seed>/${seed}/" discontinuous_shear_tube_N256_rho0_delta0.025_T1.0_${SLURM_ARRAY_TASK_ID}.json
srun /users/trohner/azeban/build_profile/azeban discontinuous_shear_tube_N256_rho0_delta0.025_T1.0_${SLURM_ARRAY_TASK_ID}.json
