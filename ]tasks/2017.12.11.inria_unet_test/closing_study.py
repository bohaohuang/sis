import os
import re
import time
import argparse
import numpy as np
import tensorflow as tf
import scipy.misc
import skimage.measure
import scipy.ndimage
import sklearn.metrics
import matplotlib.pyplot as plt
import utils
from network import unet
from dataReader import image_reader, patch_extractor
from rsrClassData import rsrClassData

TEST_DATA_DIR = 'dcc_inria_valid'
CITY_NAME = 'austin,chicago,kitsap,tyrol-w,vienna'
#CITY_NAME = 'tyrol-w'
RSR_DATA_DIR = r'/media/ei-edl01/data/remote_sensing_data'
PATCH_DIR = r'/media/ei-edl01/user/bh163/data/iai'
TEST_PATCH_APPENDIX = 'valid_noaug_dcc'
TEST_TILE_NAMES = ','.join(['{}'.format(i) for i in range(1, 6)])
RANDOM_SEED = 1234
BATCH_SIZE = 5
INPUT_SIZE = 572
CKDIR = r'/home/lab/Documents/bohao/code/sis/test/models/UnetInria_fr_mean_reduced'
MODEL_NAME = 'UnetInria_fr_mean_reduced_EP-100_DS-40.0_LR-0.001'
NUM_CLASS = 2
GPU = '0'
IMG_MEAN = np.array((109.629784946, 114.94964751, 102.778073453), dtype=np.float32)


def read_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-data-dir', default=TEST_DATA_DIR, help='path to release folder')
    parser.add_argument('--rsr-data-dir', default=RSR_DATA_DIR, help='path to rsrClassData folder')
    parser.add_argument('--patch-dir', default=PATCH_DIR, help='path to patch directory')
    parser.add_argument('--test-patch-appendix', default=TEST_PATCH_APPENDIX, help='valid patch appendix')
    parser.add_argument('--test-tile-names', default=TEST_TILE_NAMES, help='image tiles')
    parser.add_argument('--city-name', type=str, default=CITY_NAME, help='city name (default austin)')
    parser.add_argument('--random-seed', type=int, default=RANDOM_SEED, help='tf random seed')
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (10)')
    parser.add_argument('--input-size', default=INPUT_SIZE, type=int, help='input size 224')
    parser.add_argument('--ckdir', default=CKDIR, help='ckpt dir (models)')
    parser.add_argument('--num-classes', type=int, default=NUM_CLASS, help='# classes (including background)')
    parser.add_argument('--GPU', type=str, default=GPU, help="GPU used for computation.")
    parser.add_argument('--model-name', type=str, default=MODEL_NAME, help='Model name')

    flags = parser.parse_args()
    flags.input_size = (flags.input_size, flags.input_size)
    return flags


def make_pred_map(model_dirs, ids, p_dir):
    model_dirs = [model_dirs[a] for a in ids]
    _, task_dir = utils.get_task_img_folder(local_dir=True)
    task_dir = os.path.join(task_dir, 'fuse_{}_{}'.format(len(model_dirs), '+'.join([str(idx) for idx in ids])))
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
    else:
        return task_dir

    model_name = [a.split('/')[-1] for a in model_dirs]
    print('Evaluating using {}...'.format('+'.join(model_name)))

    # data prepare step
    Data = rsrClassData(flags.rsr_data_dir)
    (collect_files_test, meta_test) = Data.getCollectionByName(flags.test_data_dir)
    for (image_name, label_name) in collect_files_test:
        c_names = flags.city_name.split(',')
        for c_name in c_names:
            if c_name in image_name:
                city_name = re.findall('[a-z\-]*(?=[0-9]+\.)', image_name)[0]
                tile_id = re.findall('[0-9]+(?=\.tif)', image_name)[0]

                preds = np.zeros((5000, 5000, 2))
                for dir in model_dirs:
                    dir = os.path.join(p_dir, 'temp_save', dir)
                    preds += np.load(os.path.join(dir, '{}_{}.npy'.format(city_name, tile_id)))
                pred_labels = utils.get_pred_labels(preds)
                scipy.misc.imsave(os.path.join(task_dir, '{}_{}.png'.format(city_name, tile_id)), pred_labels)

                print('{}_{}.png saved in {}'.format(city_name, tile_id, task_dir))

    return task_dir


def img_closing(thresh, img):
    img[img != 0] = 1
    concomp, maxNum = skimage.measure.label(img, connectivity=1, return_num=True)
    all_labels = concomp.flatten()
    N, _ = scipy.histogram(all_labels, bins=range(0, maxNum+1, 1))
    NabTr = np.where(N > thresh)

    rr = np.zeros(all_labels.shape)
    for ind in NabTr[0][1:]:
        rr[all_labels == ind] = 1

    rr = np.reshape(rr, (img.shape))
    im_close = scipy.ndimage.morphology.binary_closing(rr)
    im_close = im_close.astype(np.uint8)
    im_close[im_close != 0] = 1
    return im_close


def test_closing(dir, thresh):
    print('Evaluating at {}...'.format(thresh))

    # data prepare step
    Data = rsrClassData(flags.rsr_data_dir)
    (collect_files_test, meta_test) = Data.getCollectionByName(flags.test_data_dir)
    iou = []

    for (image_name, label_name) in collect_files_test:
        c_names = flags.city_name.split(',')
        for c_name in c_names:
            if c_name in image_name:
                city_name = re.findall('[a-z\-]*(?=[0-9]+\.)', image_name)[0]
                tile_id = re.findall('[0-9]+(?=\.tif)', image_name)[0]
                pred_labels = scipy.misc.imread(os.path.join(dir, '{}_{}.png'.format(city_name, tile_id)))
                pred_labels = img_closing(thresh, pred_labels) * 255

                # evaluate
                truth_label_img = scipy.misc.imread(os.path.join(flags.rsr_data_dir, label_name))
                iou.append(utils.iou_metric(truth_label_img, pred_labels))
                print('{}_{} th={} iou={}'.format(city_name, tile_id, thresh, iou[-1]))

    print(np.mean(iou))


if __name__ == '__main__':
    flags = read_flag()
    img_dir, task_dir = utils.get_task_img_folder()
    iou_record = []

    model_names = ['UnetInria_fr_mean_reduced_appendix_large_EP-100_DS-60.0_LR-0.0001',
                   'UnetInria_fr_mean_reduced_large_EP-100_DS-60.0_LR-0.0001',
                   'UnetInria_fr_mean_reduced_appendix_EP-100_DS-60.0_LR-0.0001',
                   'UnetInria_fr_mean_reduced_EP-100_DS-60.0_LR-0.0001',
                   'UnetInria_fr_mean_reduced_EP-100_DS-60.0_LR-0.0005',
                   'UnetInria_fr_mean_reduced_EP-100_DS-60_LR-0.001',
                   'ResUnetCrop Inria_fr_resample_mean_reduced_EP-100_DS-60_LR-0.0001'
                  ]

    dir = make_pred_map(model_names, [1, -1], task_dir)

    for th in range(50, 155, 15):
        test_closing(dir, th)
