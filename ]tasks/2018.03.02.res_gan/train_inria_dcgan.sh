#!/bin/bash
#SBATCH -e slurm.err
#SBATCH --mem=20G
#SBATCH -c 6
##SBATCH --exclude=dcc-gpu-[31-32]
##SBATCH --exclude=dcc-collinslab-gpu-[02,04]
#SBATCH -p collinslab --gres=gpu:1
module load Anaconda3/3.5.2
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/apps/rhel7/cudnn/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/dscrhome/bh163/code/uab
export PYTHONPATH=$PYTHONPATH:/dscrhome/bh163/code/sis
python train_inria_dcgan.py --learning-rate=1e-4 --decay-rate=0.1 --decay-step=100 --z-dim=1000