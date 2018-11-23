#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
TIME_STAMP=`date +%Y-%m-%d_%H-%M-%S`
PIPELIN_CONFIG_PATH=/home/lab/Documents/bohao/data/transmission_line/ssd_resnet50.config
MODEL_DIR=/home/lab/Documents/bohao/data/transmission_line/models/ssd_resnet50_${TIME_STAMP}
python /home/lab/Documents/bohao/code/third_party/models/research/object_detection/model_main.py \
    --pipeline_config_path=${PIPELIN_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --alsologtostderr