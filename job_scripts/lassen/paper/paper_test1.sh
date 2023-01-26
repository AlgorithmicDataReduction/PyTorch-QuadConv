#!/bin/bash

	### LSF syntax
    #BSUB -env "all"
	#BSUB -e mystderr_test1.txt
	#BSUB -o mystdout_test1.txt
	#BSUB -nnodes 1                  #number of nodes
	#BSUB -W 12:00                    #walltime in minutes
	#BSUB -J paper_test1          #name of job
	#BSUB -q pbatch                  #queue to use
	#BSUB -G uco

    ### Shell scripting
    date; hostname
    echo -n 'JobID is '; echo $LSB_JOBID

    SAVE=/usr/workspace/doherty8/lightning_logs
    IG_GRID_DATA=/usr/workspace/doherty8/data/ignition_grid
    TEST1=ignition_grid/paper/standard_cae_pool.yaml

    module load cuda/11.3.0
    conda activate torch
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/torch/bin
    conda info

    echo "=== STARTING JOB ==="    

    jsrun -n 1 -r 1 -a 4 -c 40 -g 4 ~/.conda/envs/torch/bin/python ~/QuadConv/main.py --experiment $TEST1 --default_root_dir $SAVE --data_dir $IG_GRID_DATA