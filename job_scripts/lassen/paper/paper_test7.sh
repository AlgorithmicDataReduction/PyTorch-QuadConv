#!/bin/bash

    ### LSF syntax
    #BSUB -env "all"
	#BSUB -e mystderr_test7.txt
	#BSUB -o mystdout_test7.txt
	#BSUB -nnodes 1                  #number of nodes
	#BSUB -W 12:00                    #walltime in minutes
	#BSUB -J paper_test7          #name of job
	#BSUB -q pbatch                  #queue to use
	#BSUB -G uco

    module load cuda/11.3.0
    conda activate torch
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/torch/bin
    conda info

    TEST7=flow_mesh/paper/qcae_pool.yaml
    SAVE=/usr/workspace/doherty8/lightning_logs
    FLOW_DATA=/usr/workspace/doherty8/data/FlowCylinder

    echo "=== STARTING JOB ==="  
    jsrun -n 1 -r 1 -a 1 -c 40 -g 4 ~/.conda/envs/torch/bin/python ~/QuadConv/main.py --experiment $TEST7 --default_root_dir $SAVE --data_dir $FLOW_DATA
