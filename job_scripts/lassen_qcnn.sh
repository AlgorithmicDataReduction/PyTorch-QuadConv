#!/bin/bash

	### LSF syntax
	#BSUB -e mystderr.txt
	#BSUB -o mystdout.txt
	#BSUB -nnodes 1                   #number of nodes
	#BSUB -W 12:00                      #walltime in minutes
	#BSUB -J qcnn                     #name of job
	#BSUB -q pbatch                   #queue to use
	#BSUB -G uco

    ### Shell scripting
    date; hostname
    echo -n 'JobID is '; echo $LSB_JOBID

    #REPO=~/QuadConv
    #SAVE=/usr/workspace/doherty8
    #TEST=ignition_maxskip_qcnn
    DATA=data/ignition_center_cut

    module load cuda/11.3.0

    conda activate torch
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/torch/bin
    conda info

    #Confirm pytorch detects GPUs
    #~/.conda/envs/compression/bin/python -c "import torch; print(f'GPUs available: {torch.cuda.is_available()}\nNumber of CUDA devices: {torch.cuda.device_count()}\n')"

    #remove old logs
    #rm -r $REPO/lightning_logs/$TEST/*


    echo "=== STARTING JOB ==="    

    jsrun -n 1 -r 1 -a 4 -c 40 -g 4 ~/.conda/envs/torch/bin/python ~/QuadConv/main.py --experiment ignition_maxskip_qcnn --default_root_dir ~/QuadConv/lightning_logs --data_dir /usr/workspace/doherty8/data/ignition_center_cut
