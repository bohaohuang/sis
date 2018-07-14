#!/bin/bash
#SBATCH -e slurm.err
#SBATCH --mem=20G
#SBATCH -c 6
##SBATCH --exclude=dcc-gpu-[31-32]
##SBATCH --exclude=dcc-collinslab-gpu-[02,03,04]
#SBATCH -p collinslab --gres=gpu:1
module load Anaconda3/3.5.2
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/apps/rhel7/cudnn/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/dscrhome/bh163/code/uab
export PYTHONPATH=$PYTHONPATH:/dscrhome/bh163/code/sis
python finetune_inria_unet_loo_mtl_res50.py --leave-city=1 --pred-file-dir=/dscrhome/bh163/misc/temp_files --finetune-dir=/work/bh163/Models/Inria_Domain_LOO/UnetCrop_inria_aug_leave_{}_0_PS\(572,\ 572\)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32