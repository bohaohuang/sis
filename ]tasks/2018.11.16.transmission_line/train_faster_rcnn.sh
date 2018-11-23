#!/bin/bash
#SBATCH -e slurm.err
#SBATCH --mem=20G
#SBATCH -c 6
##SBATCH --exclude=dcc-gpu-[31-32]
##SBATCH --exclude=dcc-collinslab-gpu-[02,03,04]
#SBATCH -p collinslab --gres=gpu:1
module load Python-GPU/3.6.5

TIME_STAMP=`date +%Y-%m-%d_%H-%M-%S`
PIPELIN_CONFIG_PATH=/work/bh163/misc/object_detection/data
MODEL_DIR=/work/bh163/misc/object_detection/models/faster_rcnn_${TIME_STAMP}
python /dscrhome/bh163/code/models/research/object_detection/model_main.py \
    --pipeline_config_path=${PIPELIN_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --alsologtostderr