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
export PYTHONPATH=$PYTHONPATH:/dscrhome/bh163/code/ersa
python finetune_inria_mmd_aioi.py --city-name=Arlington \
                                  --pred-model-dir=/work/bh163/Models/Inria_decay/UnetCrop_inria_decay_0_PS\(572, 572\)_BS5_EP100_LR0.0001_DS60.0_DR0.1_SFN32 \
                                  --llh-file-dir=/work/bh163/misc/domain_selection/unet_xregion_mmd_target_\{\}_5050.npy