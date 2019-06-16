"""

"""


# Built-in
import os

# Libs
import numpy as np
import matplotlib.pyplot as plt

# Own modules
import sis_utils
from rst_utils import misc_utils


def parse_results(results):
    def parse_line(line):
        line = line.strip()
        city, ab = line.split('(')
        a, b = ab.split(',')
        b = int(b.split(')')[0])
        return city.strip(), int(a), b

    a_total = 0
    b_total = 0
    for line in results[:-1]:
        city, a, b = parse_line(line)
        if 'Norfolk' not in city:
            a_total += a
            b_total += b
    iou = a_total / b_total * 100
    return iou


def load_data(model_name, ds_name, patch_size):
    assert model_name in ['unet', 'deeplab']
    if model_name == 'unet':
        model_dir = 'UnetCrop_inria_decay_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60.0_DR0.1_SFN32'
    else:
        model_dir = 'DeeplabV3_inria_decay_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40.0_DR0.1_SFN32'
    result_file = os.path.join(r'/hdd/Results/size', model_dir, '{}_{}'.format(ds_name, patch_size), 'result.txt')
    results = misc_utils.load_file(result_file)
    return parse_results(results)


if __name__ == '__main__':
    img_dir, task_dir = sis_utils.get_task_img_folder()
    ds_name = 'aioi'

    patch_sizes = {
        'unet': [572, 828, 1084, 1340, 1596, 1852, 2092, 2332],
        'deeplab': [520, 736, 832, 1088, 1344, 1600, 1856, 2096],
    }

    model_name2show = ['U-Net', 'DeepLabV2']
    plt.rcParams.update({'font.size': 14})
    plt.rc('grid', linestyle='--')
    plt.figure(figsize=(8, 4))

    for cnt_m, model_name in enumerate(['unet', 'deeplab']):
        ious = np.zeros(len(patch_sizes[model_name]))
        for cnt, patch_size in enumerate(patch_sizes[model_name]):
            ious[cnt] = load_data(model_name, ds_name, patch_size)

        plt.plot(patch_sizes[model_name], ious, '-o', label=model_name2show[cnt_m])

    plt.title('D1->D3')
    plt.grid()
    plt.ylabel('IoU')
    plt.legend(loc='center right')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'unseen_results.png'))
    plt.show()
