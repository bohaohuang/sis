import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from rdp import rdp
from glob import glob
from tqdm import tqdm
from skimage import measure
import utils
import ersa_utils
from visualize import visualize_utils


def get_contour(image, contour_length=5):
    _, contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = np.zeros(image.shape, np.uint8)
    cv2.drawContours(contour, contours, -1, 1, contour_length)
    return contour


def get_object_id(image):
    lbl = measure.label(image)
    building_idx = np.unique(lbl)
    return lbl, building_idx


if __name__ == '__main__':
    model_dirs = [r'UnetCrop_inria_aug_leave_0_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32',
                 r'UnetCrop_inria_aug_leave_1_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32',
                 r'UnetCrop_inria_aug_leave_2_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32',
                 r'UnetCrop_inria_aug_leave_3_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32',
                 r'UnetCrop_inria_aug_leave_4_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32',
                 r'UnetGAN_V3Shrink_inria_gan_loo_0_0_PS(572, 572)_BS20_EP30_LR0.0001_1e-06_1e-06_DS15.0_30.0_30.0_DR0.1_0.1_0.1',
                 r'UnetGAN_V3Shrink_inria_gan_loo_1_0_PS(572, 572)_BS20_EP30_LR0.0001_1e-06_1e-06_DS15.0_30.0_30.0_DR0.1_0.1_0.1',
                 r'UnetGAN_V3Shrink_inria_gan_loo_2_0_PS(572, 572)_BS20_EP30_LR0.0001_1e-06_1e-06_DS15.0_30.0_30.0_DR0.1_0.1_0.1',
                 r'UnetGAN_V3Shrink_inria_gan_loo_3_0_PS(572, 572)_BS20_EP30_LR0.0001_1e-06_1e-06_DS15.0_30.0_30.0_DR0.1_0.1_0.1',
                 r'UnetGAN_V3Shrink_inria_gan_loo_4_0_PS(572, 572)_BS20_EP30_LR0.0001_1e-06_1e-06_DS15.0_30.0_30.0_DR0.1_0.1_0.1',]
    img_dir, task_dir = utils.get_task_img_folder()

    for cnt, model_dir in enumerate(model_dirs):
        pred_files = sorted(glob(os.path.join(r'/hdd/Results/ugan', model_dir, 'inria', 'pred', '*.png')))
        save_file = os.path.join(task_dir, '{}.txt'.format(model_dir))
        f = open(save_file, 'w')

        for pred_file in pred_files:
            city_name = pred_file.split('/')[-1].split('.')[0]
            pred = ersa_utils.load_file(pred_file)

            contour = get_contour(pred)
            lbl, idx = get_object_id(pred)
        
            obj_map = contour * lbl
            vert = 0
            pbar = tqdm(idx[1:])

            for i in pbar:
                pbar.set_description('{}/10 {}'.format(cnt, city_name))
                obj_points = np.where(obj_map == i)
                obj_points = np.array([obj_points[0], obj_points[1]])
                obj_points = np.transpose(obj_points)

                vert += rdp(obj_points).shape[0]
            f.write('{} {}\n'.format(city_name, vert))

        f.close()
