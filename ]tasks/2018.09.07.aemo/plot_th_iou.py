import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import sis_utils
import ersa_utils
import processBlock
from nn import nn_utils
from collection import collectionMaker

img_dir, task_dir = sis_utils.get_task_img_folder()
model_name = 'unet_aemo_hist_0_hist_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1'


def get_ious():
    conf_dir = os.path.join(task_dir, 'conf_map_{}'.format(model_name))
    conf_files = sorted(glob(os.path.join(conf_dir, '*.npy')))

    cm = collectionMaker.read_collection('aemo_pad')
    truth_files = cm.load_files(field_name='aus50', field_id='', field_ext='.*gt_d255')
    truth_files = [f[0] for f in truth_files[:2]]

    '''uniq_vals = []
    for conf, truth in zip(conf_files, truth_files):
        c = ersa_utils.load_file(conf)

        uniq_vals.append(np.unique(c.flatten()))
    uniq_vals = np.sort(np.unique(np.concatenate(uniq_vals)))

    ious_a = np.zeros(len(uniq_vals))
    ious_b = np.zeros(len(uniq_vals))'''

    uniq_vals = np.linspace(0, 1, 1000)
    ious_a = np.zeros(len(uniq_vals))
    ious_b = np.zeros(len(uniq_vals))

    for conf, truth in zip(conf_files, truth_files):
        c = ersa_utils.load_file(conf)
        t = ersa_utils.load_file(truth)

        for cnt, th in enumerate(tqdm(uniq_vals)):
            c_th = (c > th).astype(np.int)

            a, b = nn_utils.iou_metric(c_th, t, truth_val=1, divide_flag=True)
            ious_a[cnt] = a
            ious_b[cnt] = b
    return np.stack([uniq_vals, ious_a, ious_b], axis=0)

save_file = os.path.join(task_dir, 'iou_vary_th_{}_2.npy'.format(model_name))
iou = processBlock.ValueComputeProcess('iou_vary_th_{}_2'.format(model_name), task_dir, save_file, get_ious).run().val

ious = iou[1, :] / iou[2, :]
vals = iou[0, :]
plt.plot(vals, ious)
plt.show()
