#!/bin/bash
#SBATCH -e slurm.err
#SBATCH --mem=20G
#SBATCH -c 6
##SBATCH --exclude=dcc-gpu-[31-32]
##SBATCH --exclude=dcc-collinslab-gpu-[02,03,04]
#SBATCH -p collinslab --gres=gpu:1
module load Python-GPU/3.6.5
export PYTHONPATH=$PYTHONPATH:/dscrhome/bh163/code/uab
export PYTHONPATH=$PYTHONPATH:/dscrhome/bh163/code/sis
python train_inria_ugan_v3shrink_control.py --learning-rate=1e-4,1e-6,1e-6 --finetune-city=1 --GPU=0 --pred-model-dir=/dscrhome/bh163/misc/unet_loo/inria_leave_\{\} --llh-file=/dscrhome/bh163/misc/temp_files/unet_loo_mmd_target_\{\}_5050.npy