#!/bin/bash
#SBATCH -e slurm.err
#SBATCH --mem=20G
#SBATCH -c 6
#SBATCH -p collinslab --gres=gpu:1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/apps/rhel7/cudnn/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/dscrhome/bh163/code/sis
export PYTHONPATH=$PYTHONPATH:/dscrhome/bh163/code/rsr
cd ../
python train_inria.py \
                --train-data-dir=bohao_inria_train \
                --valid-data-dir=bohao_inria_valid \
                --rsr-data-dir=/work/bh163/data/remote_sensing_data \
                --patch-dir=/work/bh163/data/iai \
                --train-patch-appendix=train_noaug \
                --valid-patch-appendix=valid_noaug \
                --epochs=100 \
                --n-train=8000 \
                --decay-step=60 \
                --batch-size=10 \
                --city-name=chicago,kitsap,tyrol-w,vienna \
                --valid-size=1000 \
                --model=UNET_austin_no_random