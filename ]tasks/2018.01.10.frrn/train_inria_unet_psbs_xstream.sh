#!/bin/bash
#SBATCH -e slurm.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH -t 110:00:00
ml tensorflow/1.1.0-cp36
export PYTHONPATH=$PYTHONPATH:/home/xsede/users/xs-bohaohua/code/uab
python train_inria_unet_psbs.py --run-id=0 --batch-size=1 --input-size=1052 --n-train=1600 --n-valid=200 --learning-rate=1e-4