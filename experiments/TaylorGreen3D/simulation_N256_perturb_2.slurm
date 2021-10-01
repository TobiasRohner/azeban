#!/bin/bash -l
#SBATCH --job-name=taylor_green
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

sis=`echo "1 * ${SLURM_ARRAY_TASK_ID}" | bc`
seed=`echo "${SLURM_ARRAY_TASK_ID} + 1" | bc`
cp /users/trohner/azeban/experiments/TaylorGreen3D/taylor_green_N256_perturb_2.json taylor_green_3D_N256_perturb_T5.0_2_${SLURM_ARRAY_TASK_ID}.json
sed -i "s/<sis>/${sis}/" taylor_green_3D_N256_perturb_T5.0_2_${SLURM_ARRAY_TASK_ID}.json
sed -i "s/<seed>/${seed}/" taylor_green_3D_N256_perturb_T5.0_2_${SLURM_ARRAY_TASK_ID}.json
srun /users/trohner/azeban/build_profile/azeban taylor_green_3D_N256_perturb_T5.0_2_${SLURM_ARRAY_TASK_ID}.json
