#!/bin/bash

    ### LSF syntax
    #BSUB -B
    #BSUB -N
    #BSUB -eo mystderr_%J.txt
    #BSUB -cwd
    #BSUB -nnodes 2                   #number of nodes
    #BSUB -W 120                      #walltime in minutes
    #BSUB -G doherty8                 #account
    #BSUB -e myerrors.txt             #stderr
    #BSUB -o myoutput.txt             #stdout
    #BSUB -J qcnn                    #name of job
    #BSUB -q pbatch                   #queue to use

    ### Shell scripting
    date; hostname
    echo -n 'JobID is '; echo $LSB_JOBID

    REPO=~/QuadConv
    SAVE=/usr/workspace/doherty8
    TEST=ignition_maxskip_qcnn
    DATA=data/ignition_center_cut

    module load cuda/11.3.0

    conda activate compression
    conda info

    #Confirm pytorch detects GPUs
    python -c "import torch; print(f'GPUs available: {torch.cuda.is_available()}\nNumber of CUDA devices: {torch.cuda.device_count()}\n')"

    #remove old logs
    #rm -r $REPO/lightning_logs/$TEST/*

    lrun ~/.conda/envs/compression/bin/python $REPO/main.py --experiment $TEST --default_root_dir $SAVE --data_dir $SAVE/$DATA