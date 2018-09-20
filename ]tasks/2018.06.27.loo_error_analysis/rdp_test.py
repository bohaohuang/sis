import os
import cv2
import argparse
import numpy as np
from rdp import rdp
from glob import glob
from tqdm import tqdm
from skimage import measure
import utils
import ersa_utils


TEST_LIST = '0,5'
MIN_SIZE = 400


def get_contour(image, contour_length=5):
    _, contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = np.zeros(image.shape, np.uint8)
    cv2.drawContours(contour, contours, -1, 1, contour_length)
    return contour


def get_object_id(image):
    lbl = measure.label(image)
    building_idx = np.unique(lbl)
    return lbl, building_idx


def remove_small_objects(image, min_size=400):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    img2 = np.zeros(output.shape, np.uint8)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 1

    return img2


def read_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-list', default=TEST_LIST, type=str, help='test list')
    parser.add_argument('--min-size', default=MIN_SIZE, type=int, help='minimum size to remove')

    flags = parser.parse_args()
    return flags


if __name__ == '__main__':
    flags = read_flag()
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
    model_list = ersa_utils.str2list(flags.test_list)
    model_dirs = [model_dirs[x] for x in model_list]
    img_dir, task_dir = utils.get_task_img_folder()

    for cnt, model_dir in enumerate(model_dirs):
        pred_files = sorted(glob(os.path.join(r'/hdd/Results/ugan', model_dir, 'inria', 'pred', '*.png')))
        save_file = os.path.join(task_dir, '{}_wolverine.txt'.format(model_dir))
        with open(save_file, 'w'):
            pass

        for pred_file in pred_files:
            city_name = pred_file.split('/')[-1].split('.')[0]
            pred = ersa_utils.load_file(pred_file)
            pred = remove_small_objects(pred, flags.min_size)

            lbl, idx = get_object_id(pred)
            contour = get_contour(pred)
        
            obj_map = contour * lbl
            vert = 0
            pbar = tqdm(idx[1:])

            for i in pbar:
                pbar.set_description('{}/10 {}'.format(model_list[cnt], city_name))
                obj_points = np.where(obj_map == i)
                obj_points = np.array([obj_points[0], obj_points[1]])
                obj_points = np.transpose(obj_points)

                vert += rdp(obj_points).shape[0]
            with open(save_file, 'w+') as f:
                f.write('{} {}\n'.format(city_name, vert))
