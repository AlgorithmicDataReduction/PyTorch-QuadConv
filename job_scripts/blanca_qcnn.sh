#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --qos=preemptable
#SBATCH --job-name=ignition_qcnn
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --no-requeue

#ntasks per node should be num_workers*num_gpus

ROOT=/projects/cosi1728/QuadConv
TEST=ignition/qcnn_unstructured_center.yaml
DATA=data/unstructured_ignition_center_cut
TIME=00:04:45:00

module purge
module load anaconda

conda activate compression

python $ROOT/main.py --experiment $TEST --default_root_dir $ROOT/lightning_logs --data_dir $ROOT/$DATA --max_time $TIME
