#!/bin/bash
#SBATCH -e slurm.err
#SBATCH --mem=20G
#SBATCH -c 6
##SBATCH --exclude=dcc-gpu-[31-32]
##SBATCH --exclude=dcc-collinslab-gpu-[02,03,04]
#SBATCH -p collinslab --gres=gpu:1
module load Python-GPU/3.6.5
export PYTHONPATH=$PYTHONPATH:/dscrhome/bh163/code/ersa
export PYTHONPATH=$PYTHONPATH:/dscrhome/bh163/code/sis
python train_pspnet_inria.py --res-dir=/dscrhome/bh163/misc/pspnet101 --data-dir=/work/bh163/uab_datasets/inria/data/Original_Tiles