#!/bin/bash                                                                                                                                                                                                  
#SBATCH -e slurm.err                                                                                                                                                                                         
#SBATCH -N 1                                                                                                                                                                                                 
#SBATCH -p GPU-shared                                                                                                                                                                                        
#SBATCH --gres=gpu:p100:1                                                                                                                                                                                    
#SBATCH --ntasks-per-node 7                                                                                                                                                                                  
#SBATCH -t 35:00:00                                                                                                                                                                                          
module load cuda
export PYTHONPATH=$PYTHONPATH:/home/bohaohua/code/uab
python train_inria_deeplab_psbs.py --run-id=0 --batch-size=1 --input-size=736 --n-train=1600 --n-valid=200 --res-dir=/home/hohaohua/resnet_v1_101.ckpt
