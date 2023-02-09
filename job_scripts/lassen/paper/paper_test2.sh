#!/bin/bash
    ### LSF syntax
    #BSUB -env "all"
	#BSUB -e mystderr_test2.txt
	#BSUB -o mystdout_test2.txt
	#BSUB -nnodes 1                  #number of nodes
	#BSUB -W 12:00                    #walltime in minutes
	#BSUB -J paper_test2          #name of job
	#BSUB -q pbatch                  #queue to use
	#BSUB -G uco

    module load cuda/11.3.0
    conda activate torch
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/torch/bin
    conda info

    TEST2=ignition_grid/paper/qcae_pool.yaml
    SAVE=/usr/workspace/doherty8/lightning_logs
    IG_GRID_DATA=/usr/workspace/doherty8/data/ignition_grid

    echo "=== STARTING JOB ==="  
    jsrun -n 1 -r 1 -a 4 -c 40 -g 4 ~/.conda/envs/torch/bin/python ~/QuadConv/main.py --experiment $TEST2 --default_root_dir $SAVE --data_dir $IG_GRID_DATA
