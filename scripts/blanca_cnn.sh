#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --qos=preemptable
#SBATCH --job-name=ignition_cnn
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --no-requeue

#ntasks per node should be num_workers*num_gpus

ROOT=/projects/cosi1728/QuadConv
TEST=ignition_cnn_center
DATA=data/ignition_center_cut

module purge
module load anaconda

conda activate compression

#remove old logs
rm -r $ROOT/lightning_logs/$TEST/*

python $ROOT/main.py --experiment $TEST --default_root_dir $ROOT --data_dir $ROOT/$DATA
