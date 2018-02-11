import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import utils
from util_functions import add_mask, iou_metric


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


def get_error_mask_img(img, gt, mask):
    img = add_mask(img, gt, [None, None, None], mask_1=255)
    img = add_mask(img, mask, [255, None, None], mask_1=1)
    img = add_mask(img, mask, [None, None, 255], mask_1=-1)

    return img


if __name__ == '__main__':
    img_dir, task_dir = utils.get_task_img_folder()

    #pred_dir_loo = r'/hdd/Results/UnetCrop_inria_aug_leave_0_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32/default/pred'
    #pred_dir_xr = r'/hdd/Results/grid_vs_random/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32/inria/pred'
    pred_dir_loo = r'/hdd/Results/DeeplabV3_inria_aug_leave_0_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32/default/pred'
    pred_dir_xr = r'/hdd/Results/DeeplabV3_res101_inria_aug_grid_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32/default/pred'
    gt_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles'

    city_name = 'austin1'
    pred_file_loo_name = os.path.join(pred_dir_loo, '{}.png'.format(city_name))
    pred_file_xr_name = os.path.join(pred_dir_xr, '{}.png'.format(city_name))
    gt_file_name = os.path.join(gt_dir, '{}_GT.tif'.format(city_name))
    rgb_file_name = os.path.join(gt_dir, '{}_RGB.tif'.format(city_name))

    pred_img_loo = imageio.imread(pred_file_loo_name)
    pred_img_xr = imageio.imread(pred_file_xr_name)
    gt_img = imageio.imread(gt_file_name)
    rgb_img = imageio.imread(rgb_file_name)
    rgb_img_copy = np.copy(rgb_img)

    mask_loo = get_error_mask(pred_img_loo, gt_img)
    mask_xr = get_error_mask(pred_img_xr, gt_img)
    error_record_loo = get_high_error_region(mask_loo, 500, 400)

    region_num = 5
    for i in range(region_num):
        error, x, y = error_record_loo[-1-i, :]
        x = 1600 #int(x)
        y = 4400 #int(y)
        emi_loo = get_error_mask_img(rgb_img[x:x+500, y:y+500, :], gt_img[x:x+500, y:y+500], mask_loo[x:x+500, y:y+500])
        emi_xr = get_error_mask_img(rgb_img_copy[x:x+500, y:y+500, :], gt_img[x:x+500, y:y+500], mask_xr[x:x+500, y:y+500])

        plt.figure(figsize=(16, 6))
        plt.subplot(131)
        plt.imshow(emi_loo)
        plt.axis('off')
        plt.title('Leave-one out IoU={:.3f}'.format(iou_metric(gt_img[x:x+500, y:y+500], pred_img_loo[x:x+500, y:y+500], truth_val=255)*100))
        plt.subplot(132)
        plt.imshow(emi_xr)
        plt.axis('off')
        plt.title('Cross region IoU={:.3f}'.format(iou_metric(gt_img[x:x+500, y:y+500], pred_img_xr[x:x+500, y:y+500], truth_val=255)*100))
        plt.subplot(133)
        plt.imshow(gt_img[x:x+500, y:y+500])
        plt.axis('off')
        plt.title('Ground truth {}({},{})'.format(city_name, x, y))
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, '{}_{}_{}_deeplab.png'.format(city_name, x, y)))
        plt.show()
