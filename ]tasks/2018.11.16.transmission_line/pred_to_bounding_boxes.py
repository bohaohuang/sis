import os
import numpy as np
from glob import glob
from tqdm import tqdm
from skimage import measure
from natsort import natsorted
import sis_utils
import ersa_utils


def extract_grids(img, patch_size_h, patch_size_w):
    """
    Get patch grids for given image
    :param img:
    :param patch_size_h:
    :param patch_size_w:
    :return:
    """
    h, w = img.shape[:2]
    if h % patch_size_h == 0:
        h_steps = np.arange(0, h, patch_size_h).astype(int)
    else:
        h_steps = np.append(np.arange(0, h-patch_size_h, patch_size_h).astype(int), h-patch_size_h)
    if w % patch_size_w == 0:
        w_steps = np.arange(0, w, patch_size_w).astype(int)
    else:
        w_steps = np.append(np.arange(0, w-patch_size_w, patch_size_w).astype(int), w-patch_size_w)
    return h_steps, w_steps


def get_test_images(patch_dir):
    gt_list = natsorted(glob(os.path.join(patch_dir, '*[1|2|3]_GT.png')))
    return gt_list


def get_bounding_boxes(pred, iou_th=0.5):
    pred_binary = (pred > iou_th).astype(np.int)
    obj = measure.label(pred_binary)
    obj_idx = np.unique(obj)[1:]

    for cnt, idx in enumerate(obj_idx):
        confidence = np.mean(pred[np.where(obj == idx)])
        coords = np.where(obj == idx)
        ymin, ymax = int(np.min(coords[0])), int(np.max(coords[0]))
        xmin, xmax = int(np.min(coords[1])), int(np.max(coords[1]))

        yield confidence, ymin, ymax, xmin, xmax


# settings
model_dir = r'/hdd/Results/towers'
weight_range = [30]
iou_list = np.zeros(len(weight_range))
IMAGE_SIZE = (500, 500)
PATH_TO_TEST_IMAGES_DIR = r'/media/ei-edl01/data/uab_datasets/towers/data/Original_Tiles'
test_gt = get_test_images(PATH_TO_TEST_IMAGES_DIR)
img_dir, task_dir = sis_utils.get_task_img_folder()
IOU_TH = 0.5

for cnt, weight in enumerate(tqdm(weight_range)):
    model_name = 'confmap_uab_UnetCrop_towers_pw{}_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'.format(weight)
    predicted_dir = os.path.join(task_dir, 'predicted{}'.format(model_name))
    ground_truth_dir = os.path.join(task_dir, 'ground-truth{}'.format(model_name))
    ersa_utils.make_dir_if_not_exist(predicted_dir)
    ersa_utils.make_dir_if_not_exist(ground_truth_dir)
    pred_file_names = natsorted(glob(os.path.join(task_dir, model_name, '*.npy')))

    for pred_file_name, gt_file_name in zip(pred_file_names, test_gt):
        pred = ersa_utils.load_file(pred_file_name)
        pred = pred
        gt = ersa_utils.load_file(gt_file_name)
        h_steps, w_steps = extract_grids(pred, IMAGE_SIZE[0], IMAGE_SIZE[1])

        patch_cnt = 0
        for h_cnt, h in enumerate(h_steps):
            for w_cnt, w in enumerate(w_steps):
                pred_patch = pred[h:h + IMAGE_SIZE[0], w: w + IMAGE_SIZE[1]]
                gt_patch = gt[h:h + IMAGE_SIZE[0], w:w + IMAGE_SIZE[1]]
                check_sum = np.sum(gt_patch[np.where(gt_patch == 1)]) + np.sum(gt_patch[np.where(gt_patch == 3)])
                if check_sum > 0:
                    patch_cnt += 1
                    text_file_name = '{}_{}'.format(patch_cnt, os.path.basename(gt_file_name)[:-3] + 'txt')

                    with open(os.path.join(predicted_dir, text_file_name), 'w+') as f:
                        for confidence, ymin, ymax, xmin, xmax in get_bounding_boxes(pred_patch, iou_th=IOU_TH):
                            f.write('T {} {} {} {} {}\n'.format(confidence, xmin, ymin, xmax, ymax))

                    with open(os.path.join(ground_truth_dir, text_file_name), 'w+') as f:
                        for confidence, ymin, ymax, xmin, xmax in get_bounding_boxes(gt_patch, iou_th=0.5):
                            f.write('T {} {} {} {}\n'.format(xmin, ymin, xmax, ymax))
