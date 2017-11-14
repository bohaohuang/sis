def make_train_scripts(gpu_group='collinslab',
                       gpu=0,
                       patch_size=224,
                       batch_size=10,
                       epochs=100,
                       n_train=8000,
                       decay_step=60,
                       leave_one_city='austin'):
    uniq_name = 'PS-{}__BS-{}__E-{}__NT-{}__DS-{}__CT-{}'.\
        format(patch_size, batch_size, epochs, n_train, decay_step, leave_one_city)
    file_str = """#!/bin/bash
#SBATCH -e {}.err
#SBATCH --mem=20G
#SBATCH -c 6
#SBATCH -p {} --gres=gpu:1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/apps/rhel7/cudnn/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/dscrhome/bh163/code/sis
export PYTHONPATH=$PYTHONPATH:/dscrhome/bh163/code/rsr
cd ../
""".format(uniq_name, gpu_group)
    final_cmd = 'python train_inria.py '
    final_cmd += '--GPU={} '.format(gpu)
    final_cmd += '--train-data-dir=dcc_inria_train '
    final_cmd += '--valid-data-dir=dcc_inria_valid '
    final_cmd += '--rsr-data-dir=/work/bh163/data/remote_sensing_data '
    final_cmd += '--patch-dir=/work/bh163/data/iai '
    final_cmd += '--train-patch-appendix=train_noaug_dcc '
    final_cmd += '--valid-patch-appendix=valid_noaug_dcc '
    final_cmd += '--epochs={} '.format(epochs)
    final_cmd += '--n-train={} '.format(n_train)
    final_cmd += '--decay-step={} '.format(decay_step)
    final_cmd += '--batch-size={} '.format(batch_size)
    city_list = ['austin','chicago','kitsap','tyrol-w','vienna']
    city_final = [city for city in city_list if city != leave_one_city]
    final_cmd += '--city-name={} '.format(','.join(city_final))
    final_cmd += '--valid-size=1000 '
    final_cmd += '--model=UNET_{}__no_random'.format(uniq_name)

    script_name = '{}.sh'.format(uniq_name)
    with open(script_name, 'w') as f:
        f.write(file_str+final_cmd)


if __name__ == '__main__':
    make_train_scripts(gpu_group='collinslab',
                       gpu=2,
                       patch_size=224,
                       batch_size=1,
                       epochs=100,
                       n_train=8000,
                       decay_step=60,
                       leave_one_city='')
