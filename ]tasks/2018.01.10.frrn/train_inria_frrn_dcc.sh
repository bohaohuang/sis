#!/bin/bash
#SBATCH -e slurm.err
#SBATCH --mem=20G
#SBATCH -c 6
#SBATCH --exclude=dcc-gpu-[31-32]
#SBATCH --exclude=dcc-collinslab-gpu-[02,04]
#SBATCH -p collinslab --gres=gpu:1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/apps/rhel7/cudnn/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/dscrhome/bh163/code/uab
python train_inria_frrn_dcc.py --run-id=1 --learning-rate=1e-3