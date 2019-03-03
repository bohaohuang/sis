import os
import imageio
import numpy as np
from tqdm import tqdm
import sis_utils


def get_error_mask(pred, gt, truth_val=255):
    pred = pred/truth_val
    gt = gt/truth_val
    return pred-gt


def get_high_error_region(mask, size):
    x_max, y_max = mask.shape
    error_record = []
    maxIm0 = x_max - size - 1
    maxIm1 = y_max - size - 1
    # this is to extract with as little overlap as possible
    DS0 = np.ceil(x_max / size)
    DS1 = np.ceil(y_max / size)
    patchGridY = np.floor(np.linspace(0, maxIm0, DS0))
    patchGridX = np.floor(np.linspace(0, maxIm1, DS1))
    Y, X = np.meshgrid(patchGridY, patchGridX)

    for coordList in list(zip(Y.flatten(),X.flatten())):
        x = int(coordList[0])
        y = int(coordList[1])
        box = mask[x:x+size, y:y+size]
        error = np.sum(np.abs(box))
        error_record.append(np.array([error, x, y]))
    error_record = np.vstack(error_record)
    return error_record[error_record[:, 0].argsort()]


def get_file_list(error_record, n, city_name):
    file_list = []
    for i in range(n):
        _, x, y = error_record[-1-i, :]
        x = int(x)
        y = int(y)

        # get each row
        file_list_row = []
        for c in range(3):
            file_name = '{}_y{}x{}_RGB{}.jpg'.format(city_name, x, y, c)
            file_list_row.append(file_name)
        file_name = '{}_y{}x{}_GT_Divide.png'.format(city_name, x, y)
        file_list_row.append(file_name)
        file_list.append(file_list_row)
    return file_list


def get_file_list_finetune(size, portion):
    pred_dir_loo = r'/hdd/Results/DeeplabV3_inria_aug_leave_0_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32/default/pred'
    gt_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles'

    file_list = []
    for city_cnt in tqdm(range(6, 37)):
        city_name = 'austin{}'.format(city_cnt)
        pred_file_loo_name = os.path.join(pred_dir_loo, '{}.png'.format(city_name))
        gt_file_name = os.path.join(gt_dir, '{}_GT.tif'.format(city_name))

        pred_img_loo = imageio.imread(pred_file_loo_name)
        gt_img = imageio.imread(gt_file_name)

        mask_loo = get_error_mask(pred_img_loo, gt_img)
        error_record = get_high_error_region(mask_loo, size)
        file_list += get_file_list(error_record, int(len(error_record)*portion), city_name)
    return file_list
