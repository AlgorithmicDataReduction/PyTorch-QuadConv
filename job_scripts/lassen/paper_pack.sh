#!/bin/bash

	### LSF syntax
    #BSUB -env "all"
	#BSUB -e mystderr_test1.txt
	#BSUB -o mystdout_test1.txt
	#BSUB -nnodes 1                  #number of nodes
	#BSUB -W 10:00                    #walltime in minutes
	#BSUB -J paper_test1          #name of job
	#BSUB -q pbatch                  #queue to use
	#BSUB -G uco

    ### Shell scripting
    date; hostname
    echo -n 'JobID is '; echo $LSB_JOBID

    SAVE=/usr/workspace/doherty8/lightning_logs/paper
    IG_GRID_DATA=/usr/workspace/doherty8/data/ignition_grid
    TEST1=ignition_grid/paper/standard_cae_pool.yaml

    source ~/.bashrc
    module load cuda/11.3.0
    conda activate torch
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/torch/bin
    conda info

    echo "=== STARTING JOB ==="    

    lrun -N1 -T1 ~/.conda/envs/torch/bin/python ~/QuadConv/main.py --experiment $TEST1 --default_root_dir $SAVE --data_dir $IG_GRID_DATA &


    ### LSF syntax
    #BSUB -env "all"
	#BSUB -e mystderr_test2.txt
	#BSUB -o mystdout_test2.txt
	#BSUB -nnodes 1                  #number of nodes
	#BSUB -W 10:00                    #walltime in minutes
	#BSUB -J paper_test2          #name of job
	#BSUB -q pbatch                  #queue to use
	#BSUB -G uco

    source ~/.bashrc
    module load cuda/11.3.0
    conda activate torch
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/torch/bin
    conda info

    TEST2=ignition_grid/paper/qcae_pool.yaml
    SAVE=/usr/workspace/doherty8/lightning_logs/paper
    IG_GRID_DATA=/usr/workspace/doherty8/data/ignition_grid

    echo "=== STARTING JOB ==="  
    lrun -N1 -T1 ~/.conda/envs/torch/bin/python ~/QuadConv/main.py --experiment $TEST2 --default_root_dir $SAVE --data_dir $IG_GRID_DATA &


    ### LSF syntax
    #BSUB -env "all"
	#BSUB -e mystderr_test3.txt
	#BSUB -o mystdout_test3.txt
	#BSUB -nnodes 1                  #number of nodes
	#BSUB -W 10:00                    #walltime in minutes
	#BSUB -J paper_test3          #name of job
	#BSUB -q pbatch                  #queue to use
	#BSUB -G uco

    source ~/.bashrc
    module load cuda/11.3.0
    conda activate torch
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/torch/bin
    conda info

    TEST3=ignition_grid/paper/qcae_pool_learn_weights.yaml
    SAVE=/usr/workspace/doherty8/lightning_logs/paper
    IG_GRID_DATA=/usr/workspace/doherty8/data/ignition_grid

    echo "=== STARTING JOB ==="  
    lrun -N1 -T1 ~/.conda/envs/torch/bin/python ~/QuadConv/main.py --experiment $TEST3 --default_root_dir $SAVE --data_dir $IG_GRID_DATA &


    ### LSF syntax
    #BSUB -env "all"
	#BSUB -e mystderr_test4.txt
	#BSUB -o mystdout_test4.txt
	#BSUB -nnodes 1                  #number of nodes
	#BSUB -W 10:00                    #walltime in minutes
	#BSUB -J paper_test4          #name of job
	#BSUB -q pbatch                  #queue to use
	#BSUB -G uco

    source ~/.bashrc
    module load cuda/11.3.0
    conda activate torch
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/torch/bin
    conda info

    TEST4=ignition_mesh/paper/qcae_pool.yaml
    SAVE=/usr/workspace/doherty8/lightning_logs/paper
    IG_MESH_DATA=/usr/workspace/doherty8/data/ignition_mesh

    echo "=== STARTING JOB ==="  
    lrun -N1 -T1 ~/.conda/envs/torch/bin/python ~/QuadConv/main.py --experiment $TEST4 --default_root_dir $SAVE --data_dir $IG_MESH_DATA &

    ### LSF syntax
    #BSUB -env "all"
	#BSUB -e mystderr_test5.txt
	#BSUB -o mystdout_test5.txt
	#BSUB -nnodes 1                  #number of nodes
	#BSUB -W 10:00                    #walltime in minutes
	#BSUB -J paper_test5          #name of job
	#BSUB -q pbatch                  #queue to use
	#BSUB -G uco

    source ~/.bashrc
    module load cuda/11.3.0
    conda activate torch
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/torch/bin
    conda info

    TEST5=ignition_mesh/paper/pvcae.yaml
    SAVE=/usr/workspace/doherty8/lightning_logs/paper
    IG_MESH_DATA=/usr/workspace/doherty8/data/ignition_mesh

    echo "=== STARTING JOB ==="  
    lrun -N1 -T1 ~/.conda/envs/torch/bin/python ~/QuadConv/main.py --experiment $TEST5 --default_root_dir $SAVE --data_dir $IG_MESH_DATA &


    ### LSF syntax
    #BSUB -env "all"
	#BSUB -e mystderr_test6.txt
	#BSUB -o mystdout_test6.txt
	#BSUB -nnodes 1                  #number of nodes
	#BSUB -W 10:00                    #walltime in minutes
	#BSUB -J paper_test6          #name of job
	#BSUB -q pbatch                  #queue to use
	#BSUB -G uco

    source ~/.bashrc
    module load cuda/11.3.0
    conda activate torch
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/torch/bin
    conda info

    TEST6=ignition_mesh/paper/vcae.yaml
    SAVE=/usr/workspace/doherty8/lightning_logs/paper
    IG_MESH_DATA=/usr/workspace/doherty8/data/ignition_mesh

    echo "=== STARTING JOB ==="  
    lrun -N1 -T1 ~/.conda/envs/torch/bin/python ~/QuadConv/main.py --experiment $TEST6 --default_root_dir $SAVE --data_dir $IG_MESH_DATA &

    ### LSF syntax
    #BSUB -env "all"
	#BSUB -e mystderr_test7.txt
	#BSUB -o mystdout_test7.txt
	#BSUB -nnodes 1                  #number of nodes
	#BSUB -W 10:00                    #walltime in minutes
	#BSUB -J paper_test7          #name of job
	#BSUB -q pbatch                  #queue to use
	#BSUB -G uco

    source ~/.bashrc
    module load cuda/11.3.0
    conda activate torch
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/torch/bin
    conda info

    TEST7=flow_mesh/paper/qcae_pool.yaml
    SAVE=/usr/workspace/doherty8/lightning_logs/paper
    FLOW_DATA=/usr/workspace/doherty8/data/FlowCylinder

    echo "=== STARTING JOB ==="  
    lrun -N1 -T1 ~/.conda/envs/torch/bin/python ~/QuadConv/main.py --experiment $TEST7 --default_root_dir $SAVE --data_dir $FLOW_DATA &

    ### LSF syntax
    #BSUB -env "all"
	#BSUB -e mystderr_test8.txt
	#BSUB -o mystdout_test8.txt
	#BSUB -nnodes 1                  #number of nodes
	#BSUB -W 10:00                    #walltime in minutes
	#BSUB -J paper_test8          #name of job
	#BSUB -q pbatch                  #queue to use
	#BSUB -G uco

    source ~/.bashrc
    module load cuda/11.3.0
    conda activate torch
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/torch/bin
    conda info

    TEST8=flow_mesh/paper/qcae_skip.yaml
    SAVE=/usr/workspace/doherty8/lightning_logs/paper
    FLOW_DATA=/usr/workspace/doherty8/data/FlowCylinder

    echo "=== STARTING JOB ==="  
    lrun -N1 -T1 ~/.conda/envs/torch/bin/python ~/QuadConv/main.py --experiment $TEST8 --default_root_dir $SAVE --data_dir $FLOW_DATA &