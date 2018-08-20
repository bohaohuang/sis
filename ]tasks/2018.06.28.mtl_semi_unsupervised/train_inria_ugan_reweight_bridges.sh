#!/bin/bash
#SBATCH -e slurm.err
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node 7
#SBATCH -t 35:00:00
source activate tf-aml
module load cuda
export PYTHONPATH=$PYTHONPATH:/home/bohaohua/code/uab
python train_inria_unet_gan_weight.py --learning-rate=1e-4,1e-5,1e-6 --decay-step=30,10,30 --finetune-city=0 --GPU=0 --pred-model-dir=/pylon5/ac5fp5p/bohaohua/misc/unet_loo/inria_leave_\{\} --llh-file-dir=/pylon5/ac5fp5p/bohaohua/misc/unet_loo_mmd_target_\{\}_5050.npy