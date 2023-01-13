#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --qos=preemptable
#SBATCH --job-name=flow_qcae
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --no-requeue

#ntasks per node should be num_workers*num_gpus

TEST=flow_mesh/test/qcae_skip_small.yaml
TIME=00:09:45:00

module purge
module load anaconda

conda activate compression

python main.py --experiment $TEST --max_time $TIME
