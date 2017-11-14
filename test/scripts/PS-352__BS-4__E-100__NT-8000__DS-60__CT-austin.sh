
#!/bin/bash
#SBATCH -e PS-352__BS-4__E-100__NT-8000__DS-60__CT-austin.err
#SBATCH --mem=20G
#SBATCH -c 6
#SBATCH -p collinslab --gres=gpu:1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/apps/rhel7/cudnn/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/dscrhome/bh163/code/sis
export PYTHONPATH=$PYTHONPATH:/dscrhome/bh163/code/rsr
cd ../
python train_inria.py --GPU=0 --train-data-dir=dcc_inria_train --valid-data-dir=dcc_inria_valid --rsr-data-dir=/work/bh163/data/remote_sensing_data --patch-dir=/work/bh163/data/iai train-patch-appendix=train_noaug_dcc valid-patch-appendix=valid_noaug_dcc epochs=100 n-train=8000 decay-step=60 batch-size=4 city-name=chicago,kitsap,tyrol-w,vienna valid-size=1000 --model=UNET_PS-352__BS-4__E-100__NT-8000__DS-60__CT-austin__no_random