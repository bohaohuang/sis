#!/bin/bash
#SBATCH -e slurm.err
#SBATCH --mem=20G
#SBATCH -c 6
##SBATCH --exclude=dcc-gpu-[31-32]
##SBATCH --exclude=dcc-collinslab-gpu-[02,03,04]
#SBATCH -p gpu-common --gres=gpu:1
module load Darknet/0.1
cd /dscrhome/bh163/code/darknet
./darknet detector train /dscrhome/bh163/code/darknet/build/darknet/x64/data/obj.data yolo-obj.cfg /dscrhome/bh163/code/darknet/build/darknet/x64darknet53.conv.74 -dont_show
