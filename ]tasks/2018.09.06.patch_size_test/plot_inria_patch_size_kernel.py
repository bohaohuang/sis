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


def parse_results(results, exclude_city=None, include_city=None):
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
        update_flag = True
        if exclude_city:
            if exclude_city in city:
                update_flag = False
        if include_city:
            if include_city not in city:
                update_flag = False
        if update_flag:
            a_total += a
            b_total += b
    iou = a_total / b_total * 100
    return iou


def load_data(model_name, ds_name, patch_size, include_city=None):
    assert model_name in ['unet', 'deeplab']
    if ds_name == 'spca':
        if model_name == 'unet':
            model_dir = 'UnetCrop_spca_aug_grid_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'
        else:
            model_dir = 'DeeplabV3_spca_aug_grid_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32'
    else:
        if model_name == 'unet':
            model_dir = 'UnetCrop_inria_decay_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60.0_DR0.1_SFN32'
        else:
            model_dir = 'DeeplabV3_inria_decay_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40.0_DR0.1_SFN32'
    result_file = os.path.join(r'/hdd/Results/size', model_dir, '{}_{}_kernel'.format(ds_name, patch_size), 'result.txt')
    results = misc_utils.load_file(result_file)
    return parse_results(results, include_city=include_city)


if __name__ == '__main__':
    img_dir, task_dir = sis_utils.get_task_img_folder()
    save_dir = os.path.join(r'/media/ei-edl01/user/bh163/tasks/2018.01.23.score_results', 'train_patch')
    colors = misc_utils.get_default_colors()

    ds_name = 'spca'
    include_city = None
    patch_sizes = {
        'unet': [572, 828, 1084, 1340, 1596, 1852, 2092, 2332, 2636],
        'deeplab': [520, 736, 832, 1088, 1344, 1600, 1856, 2096, 2640],
    }
    model_name2show = ['U-Net', 'DeepLabV2']
    plt.rcParams.update({'font.size': 14})
    plt.rc('grid', linestyle='--')
    plt.figure(figsize=(8, 4))

    for cnt_m, model_name in enumerate(['unet', 'deeplab']):
        ious = np.zeros(len(patch_sizes[model_name]))
        for cnt, patch_size in enumerate(patch_sizes[model_name]):
            ious[cnt] = load_data(model_name, ds_name, patch_size, include_city)

        plt.plot(patch_sizes[model_name], ious, '-o', color=colors[cnt_m], label=model_name2show[cnt_m]+' Fuse')

        save_name = '{}_{}_test_patch.npy'.format(ds_name, model_name)
        [sizes, iou_record, duration_record] = np.load(os.path.join(save_dir, save_name))

        if model_name == 'deeplab':
            print(iou_record[0][1:])
            plt.plot(sizes[1:], iou_record[0][1:], '--o', color=colors[cnt_m], label=model_name2show[cnt_m])
        else:
            print(iou_record[0])
            plt.plot(sizes, iou_record[0], '--o', color=colors[cnt_m], label=model_name2show[cnt_m])

    plt.title('D1')
    plt.grid()
    plt.ylabel('IoU')
    plt.legend(loc='upper left')
    plt.tight_layout()
    #plt.savefig(os.path.join(img_dir, 'kernel_{}_{}.png'.format(ds_name, include_city)))
    plt.show()
