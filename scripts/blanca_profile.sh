#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --qos=preemptable
#SBATCH --job-name=qcnn_profile
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --no-requeue

#ntasks per node should be num_workers*num_gpus

ROOT=/projects/cosi1728/QuadConv
TEST=ignition_qcnn_full_profile
DATA=data/ignition_full/

module purge
module load anaconda

conda activate compression

python $ROOT/main.py --experiment $TEST --default_root_dir $ROOT --data_dir $ROOT/$DATA
