import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from tqdm import tqdm
import sis_utils


def get_fn_object(gt, pred):
    lbl = measure.label(gt)
    building_idx = np.unique(lbl)
    fn_cnt = 0
    fn_percent = np.zeros(building_idx.shape[0])
    for cnt, idx in enumerate(tqdm(building_idx)):
        on_target = np.sum(pred[np.where(lbl == idx)])
        if on_target == 0:
            fn_cnt += 1
        fn_percent[cnt] = on_target/np.count_nonzero(lbl == idx)
    return fn_cnt, fn_percent


cnn_name = 'unet'
top_patch_check = 15
window_size = 300
stride = 200
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
img_dir, task_dir = sis_utils.get_task_img_folder()
force_run = False

truth_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles'
cnn_base_dir = r'/hdd/Results/domain_selection/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_' \
               r'EP100_LR0.0001_DS60_DR0.1_SFN32/inria/pred'
deeplab_base_dir = r'/hdd/Results/domain_selection/DeeplabV3_inria_aug_grid_0_PS(321, 321)_BS5_' \
                   r'EP100_LR1e-05_DS40_DR0.1_SFN32/inria/pred'

for city_num in range(5):
    result_save_dir = os.path.join(task_dir, 'unet_deeplab_cmp', city_list[city_num])
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)

    unet_fn_cnt_all = 0
    deeplab_fn_fnt_all = 0
    unet_fn_percent_all = []
    deeplab_fn_percent_all = []
    for val_img_cnt in range(1, 6):
        if force_run:
            rgb = imageio.imread(os.path.join(truth_dir, '{}{}_RGB.tif'.format(city_list[city_num], val_img_cnt)))

            truth_img_name = os.path.join(truth_dir, '{}{}_GT.tif'.format(city_list[city_num], val_img_cnt))
            cnn_base_img_name = os.path.join(cnn_base_dir, '{}{}.png'.format(city_list[city_num], val_img_cnt))
            deeplab_base_img_name = os.path.join(deeplab_base_dir, '{}{}.png'.format(city_list[city_num], val_img_cnt))

            truth, unet_base, deeplab_base = imageio.imread(truth_img_name), imageio.imread(cnn_base_img_name), \
                                             imageio.imread(deeplab_base_img_name)

            truth = truth / 255
            unet_base = unet_base / 255
            deeplab_base = deeplab_base / 255

            assert np.all(np.unique(truth) == np.array([0, 1])) and np.all(np.unique(unet_base) == np.array([0, 1])) \
                   and np.all(np.unique(deeplab_base) == np.array([0, 1]))

            unet_fn_cnt, unet_fn_percent = get_fn_object(truth, unet_base)
            deeplab_fn_cnt, deeplab_fn_percent = get_fn_object(truth, deeplab_base)

            np.save(os.path.join(result_save_dir, '{}{}_unet_cnt.npy'.format(city_list[city_num], val_img_cnt)),
                    unet_fn_cnt)
            np.save(os.path.join(result_save_dir, '{}{}_unet_percent.npy'.format(city_list[city_num], val_img_cnt)),
                    unet_fn_percent)
            np.save(os.path.join(result_save_dir, '{}{}_deeplab_cnt.npy'.format(city_list[city_num], val_img_cnt)),
                    deeplab_fn_cnt)
            np.save(os.path.join(result_save_dir, '{}{}_deeplab_percent.npy'.format(city_list[city_num], val_img_cnt)),
                    deeplab_fn_percent)
        else:
            unet_fn_cnt = np.load(os.path.join(result_save_dir, '{}{}_unet_cnt.npy'.format(city_list[city_num],
                                                                                           val_img_cnt)))
            unet_fn_percent = np.load(os.path.join(result_save_dir, '{}{}_unet_percent.npy'.format(city_list[city_num],
                                                                                                   val_img_cnt)))
            deeplab_fn_cnt = np.load(os.path.join(result_save_dir, '{}{}_deeplab_cnt.npy'.format(city_list[city_num],
                                                                                                 val_img_cnt)))
            deeplab_fn_percent = np.load(os.path.join(
                result_save_dir, '{}{}_deeplab_percent.npy'.format(city_list[city_num], val_img_cnt)))

            unet_fn_cnt_all += unet_fn_cnt
            deeplab_fn_fnt_all += deeplab_fn_cnt
            unet_fn_percent_all.append(unet_fn_percent)
            deeplab_fn_percent_all.append(deeplab_fn_percent)

    plt.figure(figsize=(8, 8))
    ax1 = plt.subplot(211)
    plt.hist(np.concatenate(unet_fn_percent_all), bins=200, label='U-Net')
    plt.title('{} U-Net FN={}'.format(city_list[city_num], unet_fn_cnt_all))
    ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
    plt.hist(np.concatenate(deeplab_fn_percent_all), bins=200, label='DeepLab')
    plt.title('{} DeepLab FN={}'.format(city_list[city_num], deeplab_fn_fnt_all))
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '{}_unet_deeplab_object-wise_fn.png'.format(city_list[city_num])))
    plt.show()
