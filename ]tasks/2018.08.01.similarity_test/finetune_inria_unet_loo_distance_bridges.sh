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
python finetune_inria_unet_loo_mmd.py --learning-rate=1e-6 --model-name=inria_distance_loo_5050_\{\}_\{\} --finetune-city=1 --llh-file=/pylon5/ac5fp5p/bohaohua/misc/unet_loo_mmd_target_\{\}_5050.npy --pred-model-dir=/pylon5/ac5fp5p/bohaohua/misc/unet_loo/inria_leave_\{\}