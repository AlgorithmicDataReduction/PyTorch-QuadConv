#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --qos=preemptable
#SBATCH --job-name=ignition_pae
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --no-requeue

#ntasks per node should be num_workers*num_gpus

TEST=ignition_mesh/test/vae.yaml
TIME=00:03:30:00

module purge
module load anaconda

conda activate compression

python main.py --experiment $TEST --max_time $TIME
