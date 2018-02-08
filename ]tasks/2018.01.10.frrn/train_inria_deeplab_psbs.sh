#!/bin/bash
#SBATCH -e slurm.err
#SBATCH --mem=20G
#SBATCH -c 6
#SBATCH --exclude=dcc-gpu-[31-32]
#SBATCH --exclude=dcc-collinslab-gpu-[02,04]
#SBATCH -p gpu-common --gres=gpu:1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/apps/rhel7/cudnn/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/dscrhome/bh163/code/uab
python train_inria_deeplab_psbs.py --run-id=0 --batch-size=1 --input-size=736 --n-train=1600 --n-valid=200 ----res-dir=/dscrhome/bh163/resnet_v1_101.ckpt