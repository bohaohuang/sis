import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import sis_utils
import ersa_utils
import uab_collectionFunctions
from collection import collectionMaker as cm
from bohaoCustom import uabMakeNetwork_DeepLabV2


if __name__ == '__main__':
    force_run = False
    y = 500
    x = 500
    patch_size = 321
    stride = 1
    block_size = 100
    pretrained_model_dir = r'/hdd6/Models/Inria_decay/DeeplabV3_inria_decay_0_PS(321, 321)_BS5_' \
                           r'EP100_LR1e-05_DS40.0_DR0.1_SFN32'
    field_name = 'chicago'
    field_id = '1'
    tf.reset_default_graph()
    record_matrix = []

    img_dir, task_dir = sis_utils.get_task_img_folder()
    save_file_name = os.path.join(task_dir, 'corr_{}{}_ps{}_bs{}.npy'.
                                  format(field_name, field_id, patch_size, block_size))
    record_matrix = np.load(save_file_name)
    print('loaded {}'.format(save_file_name))

    cov = np.corrcoef(record_matrix)
    print(cov.shape)
