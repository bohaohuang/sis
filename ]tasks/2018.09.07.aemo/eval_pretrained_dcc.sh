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
python eval_pretrained_dcc.py