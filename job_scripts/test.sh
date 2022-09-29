#!/bin/bash

ROOT=/projects/cosi1728/QuadConv
TEST=ignition_qcnn_update
DATA=data/ignition_center_cut
TIME=00:04:45:00
GPUS=1

#SBATCH --job-name=$TEST
#SBATCH --qos=preemptable
#SBATCH --time=$TIME
#SBATCH --gres=gpu:$GPUS
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=$((4*GPUS))
#SBATCH --no-requeue
#SBATCH --output=./

module purge
module load anaconda

conda activate compression

python $ROOT/main.py --experiment $TEST --default_root_dir $ROOT/lightning_logs --data_dir $ROOT/$DATA --max_time $TIME
