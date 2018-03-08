#!/bin/bash
#SBATCH -e slurm.err
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node 7
#SBATCH -t 35:00:00
module load cuda
source activate tf-aml
export PYTHONPATH=$PYTHONPATH:/home/bohaohua/code/uab
python train_spca_deeplab_psbs.py --run-id=0 --batch-size=10 --input-size=232 --n-train=16000 --n-valid=2000 --res-dir=/home/bohaohua/resnet_v1_101.ckpt
