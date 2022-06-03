#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --qos=blanca-appm-student
#SBATCH --partition=blanca-appm
#SBATCH --account=blanca-appm
#SBATCH --job-name=ignition_test
#SBATCH --gres=gpu

ROOT=/projects/cosi1728/QuadConv
TEST=blanca_ignition_test

module purge
module load anaconda

conda activate compression

#copy dataset to scratch
cp $ROOT/data/ignition_center_cut/train.npy $SLURM_SCRATCH/

python main.py --experiment $TEST

#copy logs from scratch
cp -R $SLURM_SCRATCH/lightning_logs/version_0 $ROOT/lightning_logs/$TEST
