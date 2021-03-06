#!/bin/bash
#SBATCH -e slurm.err
#SBATCH --mem=20G
#SBATCH -c 6
#SBATCH --exclude=dcc-gpu-[31-32]
#SBATCH --exclude=dcc-collinslab-gpu-[02,04]
#SBATCH -p gpu-common --gres=gpu:1
module load Anaconda3/3.5.2
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/apps/rhel7/cudnn/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/dscrhome/bh163/code/uab
python train_spca_deeplab_psbs.py --run-id=0 --batch-size=10 --input-size=232 --n-train=16000 --n-valid=2000 --res-dir=/dscrhome/bh163/resnet_v1_101.ckpt