#!/bin/bash
#SBATCH -e slurm.err
#SBATCH --mem=20G
#SBATCH -c 6
##SBATCH --exclude=dcc-gpu-[31-32]
##SBATCH --exclude=dcc-collinslab-gpu-[02,03,04]
#SBATCH -p gpu-common --gres=gpu:1
module load Python-GPU/3.6.5
export PYTHONPATH=$PYTHONPATH:/dscrhome/bh163/code/uab
export PYTHONPATH=$PYTHONPATH:/dscrhome/bh163/code/sis
export PYTHONPATH=$PYTHONPATH:/dscrhome/bh163/code/ersa
python train_uab_line_oracle_towers2.py --pos-weight=1 --leave-city=0 --patch-dir=/work/bh163/misc/lines_patches