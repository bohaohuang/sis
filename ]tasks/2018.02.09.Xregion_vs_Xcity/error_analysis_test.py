import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from util_functions import add_mask


def get_error_mask(pred, gt, truth_val=255):
    pred = pred/truth_val
    gt = gt/truth_val
    return pred-gt


def get_high_error_region(mask, size, step):
    x_max, y_max = mask.shape
    error_record = []
    for x in range(0, x_max-size, step):
        for y in range(0, y_max-size, step):
            box = mask[x:x+size, y:y+size]
            error = np.sum(np.abs(box))
            error_record.append(np.array([error, x, y]))
    error_record = np.vstack(error_record)
    return error_record[error_record[:, 0].argsort()]


def visulize_with_mask(img, gt, mask):
    img = add_mask(img, gt, [None, None, None], mask_1=255)
    img = add_mask(img, mask, [255, None, None], mask_1=1)
    img = add_mask(img, mask, [None, None, 255], mask_1=-1)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(gt)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    pred_dir = r'/hdd/Results/fix_pixel/FRRN_inria_aug_psbs_0_PS(160, 160)_BS10_EP100_LR0.0001_DS60_DR0.1_SFN32/inria/pred'
    gt_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles'

    city_name = 'austin1'
    pred_file_name = os.path.join(pred_dir, '{}.png'.format(city_name))
    gt_file_name = os.path.join(gt_dir, '{}_GT.tif'.format(city_name))
    rgb_file_name = os.path.join(gt_dir, '{}_RGB.tif'.format(city_name))

    pred_img = imageio.imread(pred_file_name)
    gt_img = imageio.imread(gt_file_name)
    rgb_img = imageio.imread(rgb_file_name)

    mask = get_error_mask(pred_img, gt_img)
    error_record = get_high_error_region(mask, 500, 400)

    region_num = 5
    for i in range(region_num):
        error, x, y = error_record[-1-i, :]
        x = int(x)
        y = int(y)
        visulize_with_mask(rgb_img[x:x+500, y:y+500, :], gt_img[x:x+500, y:y+500], mask[x:x+500, y:y+500])
