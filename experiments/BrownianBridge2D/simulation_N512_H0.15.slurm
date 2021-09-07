#!/bin/bash -l
#SBATCH --job-name=brownian_bridge
#SBATCH --account="s1069"
#SBATCH --time=03:00:00
#SBATCH --array=0-31
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

sis=`echo "16 * ${SLURM_ARRAY_TASK_ID}" | bc`
seed=`echo "${SLURM_ARRAY_TASK_ID} + 1" | bc`
cp /users/trohner/azeban/experiments/BrownianBridge2D/brownian_bridge_N512_H0.15.json brownian_bridge_2D_N512_H0.15_T1.0_${SLURM_ARRAY_TASK_ID}.json
sed -i "s/<sis>/${sis}/" brownian_bridge_2D_N512_H0.15_T1.0_${SLURM_ARRAY_TASK_ID}.json
sed -i "s/<seed>/${seed}/" brownian_bridge_2D_N512_H0.15_T1.0_${SLURM_ARRAY_TASK_ID}.json
srun /users/trohner/azeban/build_profile/azeban brownian_bridge_2D_N512_H0.15_T1.0_${SLURM_ARRAY_TASK_ID}.json
