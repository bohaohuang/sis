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
python finetune_inria_deeplab_loo_mmd.py --learning-rate=1e-5 --finetune-city=0 --model-name=inria_distance_loo_5050_\{\}_\{\} --llh-file=/dscrhome/bh163/misc/temp_files/deeplab_loo_distance_target_\{\}_5050.npy --pred-model-dir=/dscrhome/bh163/misc/unet_loo/inria_leave_\{\}