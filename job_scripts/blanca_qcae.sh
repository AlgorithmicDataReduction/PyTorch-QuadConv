#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --qos=preemptable
#SBATCH --job-name=ignition_qcae
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --no-requeue

#ntasks per node should be num_workers*num_gpus

TEST=ignition/qcae_pool_uniform.yaml
TIME=00:11:50:00

module purge
module load anaconda

conda activate compression

python main.py --experiment $TEST --max_time $TIME
