#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --qos=preemptable
#SBATCH --job-name=ignition_qcnn
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --no-requeue

#ntasks per node should be num_workers*num_gpus

ROOT=/projects/cosi1728/QuadConv
TEST=ignition_qcnn_full
DATA=data/ignition_square/train.npy

module purge
module load anaconda

conda activate compression

#copy dataset to scratch
cp $ROOT/$DATA $SLURM_SCRATCH/

#remove old logs
rm -r $ROOT/lightning_logs/$TEST/*

python $ROOT/main.py --experiment $TEST --default_root_dir $ROOT --data_dir $SLURM_SCRATCH
