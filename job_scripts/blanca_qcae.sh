#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --qos=preemptable
#SBATCH --job-name=ignition_qcae
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --no-requeue

#ntasks per node should be num_workers*num_gpus

TEST=ignition/skip_qcae.yaml
TIME=00:23:50:00

module purge
module load anaconda

conda activate compression

python main.py --experiment $TEST --max_time $TIME
