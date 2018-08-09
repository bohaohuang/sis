#!/bin/bash
#SBATCH -e slurm.err
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node 7
#SBATCH -t 35:00:00
module load cuda
export PYTHONPATH=$PYTHONPATH:/home/bohaohua/code/uab
python finetune_unet_gmm_prior.py --t=10000 --train-city=tyrol-w --llh-file=/pylon5/ac5fp5p/bohaohua/misc/llh_unet_inria_n50.npy --pred-model-dir=/dscrhome/bh163/misc/temp_files/unet_xregion