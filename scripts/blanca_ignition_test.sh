#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --qos=preemptable
#SBATCH --job-name=ignition_test
#SBATCH --gres=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4

ROOT=/projects/cosi1728/QuadConv
TEST=blanca_ignition_cnn
DATA=data/ignition_square/train.npy

module purge
module load anaconda

conda activate compression

#copy dataset to scratch
cp $ROOT/$DATA $SLURM_SCRATCH/

mkdir $SLURM_SCRATCH/lightning_logs

python $ROOT/main.py --experiment $TEST --default_root_dir $SLURM_SCRATCH --data_dir $SLURM_SCRATCH

#copy logs from scratch
cp -R $SLURM_SCRATCH/lightning_logs/version_0 $ROOT/lightning_logs/$TEST
