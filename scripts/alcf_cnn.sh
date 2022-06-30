#!/bin/bash
#COBALT -n 1
#COBALT -t 2:00:00
#COBALT -q single-gpu
#COBALT -A <project name>

ROOT=/projects/cosi1728/QuadConv
TEST=blanca_ignition_qcnn_full
DATA=data/ignition_square/train.npy
SCRATCH=/raid/scratch

NPROC_PER_NODE=4
NPROC=$((NPROC_PER_NODE*COBALT_JOBSIZE))

#load anaconda and acitvate env
module load conda/2021-06-26
conda activate /home/coopersimpson/compression

#copy training data to scratch and make log dir
cp $ROOT/$DATA $SCRATCH
mkdir $SCRATCH/lightning_logs

#run experiment
aprun -n $NPROC -N $NPROC_PER_NODE python $ROOT/main.py --experiment $TEST --default_root_dir $SCRATCH --data_dir $SCRATCH

#remove old logs and copy from scratch
rm -r $ROOT/lightning_logs/$TEST
cp -r $SCRATCH/lightning_logs/$TEST $ROOT/lightning_logs/
