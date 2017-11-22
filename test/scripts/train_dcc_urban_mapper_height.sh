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
python train_urban_mapper_height.py \
                --GPU=2 \
                --train-data-dir=dcc_urban_mapper_height_train \
                --valid-data-dir=dcc_urban_mapper_height_valid \
                --rsr-data-dir=/work/bh163/data/remote_sensing_data \
                --patch-dir=/work/bh163/data/iai \
                --pre-trained-model=~/code/sis/test/models/UNET_PS-224__BS-10__E-100__NT-8000__DS-60__CT-__no_random \
                --layers-to-keep=1,2,3,4,5,6,7 \
                --train-patch-appendix=train_augfr_um_npy \
                --valid-patch-appendix=valid_augfr_um_npy \
                --epochs=100 \
                --n-train=8000 \
                --decay-step=60 \
                --batch-size=5 \
                --city-name=JAX,TAM \
                --height-mode=all \
                --model=unet_origin_scratch_um_augfr_5