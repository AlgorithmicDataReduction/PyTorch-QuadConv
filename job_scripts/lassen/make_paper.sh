#!/bin/bash

	### LSF syntax
    #BSUB -env "all"
	#BSUB -e mystderr.txt
	#BSUB -o mystdout.txt
	#BSUB -nnodes 8                  #number of nodes
	#BSUB -W 6:00                    #walltime in minutes
	#BSUB -J quadconv_paper          #name of job
	#BSUB -q pbatch                  #queue to use
	#BSUB -G uco

    ### Shell scripting
    date; hostname
    echo -n 'JobID is '; echo $LSB_JOBID

    SAVE=/usr/workspace/doherty8/lightning_logs
    IG_GRID_DATA=/usr/workspace/doherty8/data/ignition_center_cut
    IG_MESH_DATA=/usr/workspace/doherty8/data/ignition_center_cut
    FLOW_DATA=/usr/workspace/doherty8/data/ignition_center_cut

    TEST1 = ignition_grid/paper/cae_pool.yaml
    TEST2 = ignition_grid/paper/qcae_pool.yaml
    TEST3 = ignition_grid/paper/qcae_pool_learn_weights.yaml

    TEST4 = ignition_mesh/paper/qcae_pool.yaml
    TEST5 = ignition_mesh/paper/pvcae.yaml
    TEST6 = ignition_mesh/paper/vcae.yaml

    TEST7 = flow_mesh/paper/qcae_pool.yaml
    TEST8 = flow_mesh/paper/qcae_skip.yaml

    module load cuda/11.3.0
    conda activate torch
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/torch/bin
    conda info

    #Confirm pytorch detects GPUs
    #~/.conda/envs/compression/bin/python -c "import torch; print(f'GPUs available: {torch.cuda.is_available()}\nNumber of CUDA devices: {torch.cuda.device_count()}\n')"

    echo "=== STARTING JOB ==="    

    echo "=== RUNNING IGNITION GRID TESTS ==="
    lrun -N1 -T1 ~/.conda/envs/torch/bin/python ~/QuadConv/main.py --experiment $TEST1 --default_root_dir $SAVE --data_dir $IG_GRID_DATA &
    lrun -N1 -T1 ~/.conda/envs/torch/bin/python ~/QuadConv/main.py --experiment $TEST2 --default_root_dir $SAVE --data_dir $IG_GRID_DATA &
    lrun -N1 -T1 ~/.conda/envs/torch/bin/python ~/QuadConv/main.py --experiment $TEST3 --default_root_dir $SAVE --data_dir $IG_GRID_DATA &

    echo "=== RUNNING IGNITION MESH TESTS ==="
    lrun -N1 -T1 ~/.conda/envs/torch/bin/python ~/QuadConv/main.py --experiment $TEST4 --default_root_dir $SAVE --data_dir $IG_MESH_DATA &
    lrun -N1 -T1 ~/.conda/envs/torch/bin/python ~/QuadConv/main.py --experiment $TEST5 --default_root_dir $SAVE --data_dir $IG_MESH_DATA &
    lrun -N1 -T1 ~/.conda/envs/torch/bin/python ~/QuadConv/main.py --experiment $TEST6 --default_root_dir $SAVE --data_dir $IG_MESH_DATA &

    echo "=== RUNNING FLOW TESTS ==="
    lrun -N1 -T1 ~/.conda/envs/torch/bin/python ~/QuadConv/main.py --experiment $TEST7 --default_root_dir $SAVE --data_dir $FLOW_DATA &
    lrun -N1 -T1 ~/.conda/envs/torch/bin/python ~/QuadConv/main.py --experiment $TEST8 --default_root_dir $SAVE --data_dir $FLOW_DATA &

    wait

    
