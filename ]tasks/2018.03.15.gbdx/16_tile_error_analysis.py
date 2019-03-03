import os
import imageio
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import sis_utils
from util_functions import add_mask, iou_metric


def get_error_mask(pred, gt, truth_val=1):
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


def get_error_mask_img(img, gt, mask):
    img = add_mask(img, gt, [None, None, None], mask_1=255)
    img = add_mask(img, mask, [255, None, None], mask_1=1)
    img = add_mask(img, mask, [None, None, 255], mask_1=-1)

    return img


if __name__ == '__main__':
    img_dir, task_dir = sis_utils.get_task_img_folder()

    pred_dir_base = r'/hdd/Results/gbdx_cmp/]shared_models/sp/pred'
    gt_dir = r'/media/ei-edl01/data/uab_datasets/sp/DATA_BUILDING_AND_PANEL'
    tile_list = [os.path.basename(x).split('.')[0] for x in glob(os.path.join(pred_dir_base, '*.png'))]
    error_size = 300
    error_step = 100

    for city_name in tqdm(tile_list):
        pred_file_base_name = os.path.join(pred_dir_base, '{}.png'.format(city_name))
        gt_file_name = os.path.join(gt_dir, '{}_GT.png'.format(city_name))
        rgb_file_name = os.path.join(gt_dir, '{}_RGB.jpg'.format(city_name))

        pred_img_base = imageio.imread(pred_file_base_name)
        gt_img = imageio.imread(gt_file_name)
        rgb_img = imageio.imread(rgb_file_name)
        rgb_img_copy = np.copy(rgb_img)

        mask_base = get_error_mask(pred_img_base, gt_img)
        error_record_loo = get_high_error_region(mask_base, error_size, error_step)

        region_num = 5
        for i in range(region_num):
            error, x, y = error_record_loo[-1-i, :]
            x = int(x)
            y = int(y)
            emi_base = get_error_mask_img(rgb_img[x:x+error_size, y:y+error_size, :],
                                          gt_img[x:x+error_size, y:y+error_size],
                                          mask_base[x:x+error_size, y:y+error_size])

            fig = plt.figure(figsize=(5, 5.5))
            #plt.subplot(121)
            plt.imshow(emi_base)
            plt.axis('off')
            '''plt.subplot(122)
            plt.imshow(rgb_img_copy[x:x+error_size, y:y+error_size, :])
            plt.axis('off')'''
            title_str = 'IoU={:.3f}'.format(iou_metric(gt_img[x:x+error_size, y:y+error_size],
                                                       pred_img_base[x:x+error_size, y:y+error_size], truth_val=1)*100)
            title_str += ' {}:({},{})'.format(city_name, x, y)
            plt.title(title_str)
            plt.tight_layout()
            plt.savefig(os.path.join(img_dir, '{}_{}_{}_boning.png'.format(city_name, x, y)))
            plt.close(fig)
            # plt.show()
