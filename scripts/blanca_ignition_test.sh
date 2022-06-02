#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --qos=blanca-appm-student
#SBATCH --partition=blanca-appm
#SBATCH --account=blanca-appm
#SBATCH --job-name=ignition_test
#SBATCH --gres=gpu

module purge
module load anaconda

conda activate compression

python main.py --experiment blanca_ignition_test
