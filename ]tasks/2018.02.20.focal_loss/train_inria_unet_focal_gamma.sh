#!/bin/bash
#SBATCH -e slurm.err
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node 14
#SBATCH -t 30:00:00
source activate tf-aml
module load cuda
export PYTHONPATH=$PYTHONPATH:/home/bohaohua/code/uab
python train_inria_unet_focal_gamma.py --run-id=0 --gamma=0.2 --GPU=0