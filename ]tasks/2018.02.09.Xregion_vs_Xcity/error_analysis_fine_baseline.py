import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import sis_utils
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
    img_dir, task_dir = sis_utils.get_task_img_folder()

    #pred_dir_loo = r'/hdd/Results/UnetCrop_inria_aug_leave_1_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32/default/pred'
    #pred_dir_xr = r'/hdd/Results/grid_vs_random/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32/inria/pred'
    pred_dir_loo = r'/hdd/Results/DeeplabV3_inria_aug_leave_0_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32/default/pred'
    pred_dir_xr = r'/hdd/Results/DeeplabV3_res101_inria_aug_grid_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32/default/pred'
    pred_dir_base = r'/hdd/Results/DeeplabV3_inria_aug_leave_0_0_0_PS(321, 321)_BS5_EP20_LR1e-07_DS10_DR0.1_SFN32/default/pred'
    pred_dir_auto = r'/hdd/Results/DeeplabV3_inria_aug_leave_auto_0_0_PS(321, 321)_BS5_EP20_LR1e-07_DS40_DR0.1_SFN32/default/pred'
    gt_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles'

    city_name = 'austin5'
    pred_file_loo_name = os.path.join(pred_dir_loo, '{}.png'.format(city_name))
    pred_file_xr_name = os.path.join(pred_dir_xr, '{}.png'.format(city_name))
    pred_file_base_name = os.path.join(pred_dir_base, '{}.png'.format(city_name))
    pred_file_auto_name = os.path.join(pred_dir_auto, '{}.png'.format(city_name))
    gt_file_name = os.path.join(gt_dir, '{}_GT.tif'.format(city_name))
    rgb_file_name = os.path.join(gt_dir, '{}_RGB.tif'.format(city_name))

    pred_img_loo = imageio.imread(pred_file_loo_name)
    pred_img_xr = imageio.imread(pred_file_xr_name)
    pred_img_base = imageio.imread(pred_file_base_name)
    pred_img_auto = imageio.imread(pred_file_auto_name)
    gt_img = imageio.imread(gt_file_name)
    rgb_img = imageio.imread(rgb_file_name)
    rgb_img_copy = np.copy(rgb_img)
    rgb_img_copy2 = np.copy(rgb_img)
    rgb_img_copy3 = np.copy(rgb_img)

    mask_loo = get_error_mask(pred_img_loo, gt_img)
    mask_xr = get_error_mask(pred_img_xr, gt_img)
    mask_base = get_error_mask(pred_img_base, gt_img)
    mask_auto = get_error_mask(pred_img_auto, gt_img)
    error_record_loo = get_high_error_region(mask_loo, 500, 400)

    region_num = 5
    for i in range(region_num):
        error, x, y = error_record_loo[-1-i, :]
        x = int(x)
        y = int(y)
        emi_loo = get_error_mask_img(rgb_img[x:x+500, y:y+500, :], gt_img[x:x+500, y:y+500], mask_loo[x:x+500, y:y+500])
        emi_xr = get_error_mask_img(rgb_img_copy[x:x+500, y:y+500, :], gt_img[x:x+500, y:y+500], mask_xr[x:x+500, y:y+500])
        emi_base = get_error_mask_img(rgb_img_copy2[x:x+500, y:y+500, :], gt_img[x:x+500, y:y+500], mask_base[x:x+500, y:y+500])
        emi_auto = get_error_mask_img(rgb_img_copy3[x:x+500, y:y+500, :], gt_img[x:x+500, y:y+500], mask_auto[x:x+500, y:y+500])

        plt.figure(figsize=(15, 5.5))
        plt.subplot(131)
        plt.imshow(emi_xr)
        plt.axis('off')
        plt.title('XRegion IoU={:.3f}'.format(iou_metric(gt_img[x:x+500, y:y+500], pred_img_xr[x:x+500, y:y+500], truth_val=255)*100))
        plt.subplot(132)
        plt.imshow(emi_auto)
        plt.axis('off')
        plt.title('Auto IoU={:.3f}'.format(iou_metric(gt_img[x:x+500, y:y+500], pred_img_auto[x:x+500, y:y+500], truth_val=255)*100))
        plt.subplot(133)
        plt.imshow(emi_base)
        plt.axis('off')
        plt.title('Finetune IoU={:.3f}'.format(iou_metric(gt_img[x:x+500, y:y+500], pred_img_base[x:x+500, y:y+500], truth_val=255)*100))
        plt.suptitle('{}:({},{}) ({})'.format(city_name, x, y, 'DeepLab'), color='g')
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, '{}_{}_{}_deeplab_auto.png'.format(city_name, x, y)))
        plt.show()
