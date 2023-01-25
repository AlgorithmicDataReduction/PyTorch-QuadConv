#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --qos=preemptable
#SBATCH --job-name=flow_qcae
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue

#ntasks per node should be num_workers*num_gpus

TEST=flow_mesh/test/qcae_skip.yaml
TIME=00:14:45:00
PYTHON=/projects/cosi1728/software/anaconda/envs/compression/bin/python

module purge
module load anaconda

conda activate compression

srun $PYTHON main.py --experiment $TEST --max_time $TIME
