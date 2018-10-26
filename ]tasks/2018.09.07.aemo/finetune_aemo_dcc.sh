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
python finetune_aemo_dcc.py --from-scratch=True --par-dir=aemo_align --ds-name=aemo_align --learn-rate=1e-3,1e-4 --start-layer=10 --model-dir=/work/bh163/misc/unet_reweight --data-dir=/work/bh163/data/aemo_align