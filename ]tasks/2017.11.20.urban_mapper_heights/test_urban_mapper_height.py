import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import utils

TEST_DATA_DIR = 'dcc_urban_mapper_height_valid'
CITY_NAME = 'JAX,TAM'
RSR_DATA_DIR = r'/media/ei-edl01/data/remote_sensing_data'
PATCH_DIR = r'/home/lab/Documents/bohao/data/urban_mapper'
TEST_PATCH_APPENDIX = 'valid_augfr_um_npy'
TEST_TILE_NAMES = ','.join(['{}'.format(i) for i in range(0, 16)])
RANDOM_SEED = 1234
BATCH_SIZE = 5
INPUT_SIZE = 572
CKDIR = r'/home/lab/Documents/bohao/code/sis/test/models/UrbanMapper_Height_npy'
MODEL_NAME = 'unet_origin_scratch_um_augfr_5'
NUM_CLASS = 2
GPU = '0'


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


def evaluate_results(flags, model_name, height_mode):
    result = utils.test_authentic_unet_height(flags.rsr_data_dir,
                                              flags.test_data_dir,
                                              flags.input_size,
                                              model_name,
                                              flags.num_classes,
                                              flags.ckdir,
                                              flags.city_name,
                                              flags.batch_size,
                                              ds_name='urban_mapper',
                                              height_mode=height_mode)

    print(result)

    _, task_dir = utils.get_task_img_folder()
    np.save(os.path.join(task_dir, '{}.npy'.format(model_name)), result)

if __name__ == '__main__':
    '''flags = read_flag()

    model_name = 'unet_origin_fix_scratch_um_augfr_4'
    if int(model_name[-1] == 4):
        height_mode = 'subtract'
    elif int(model_name[-1] == 5):
        height_mode = 'all'
    else:
        height_mode = 'subtract_all'

    evaluate_results(flags, model_name, height_mode)

    _, task_dir = utils.get_task_img_folder()
    result = dict(np.load(os.path.join(task_dir, '{}.npy'.format(model_name))).tolist())

    iou = []
    for k, v in result.items():
        iou.append(v)
    result_mean = np.mean(iou)
    print(result_mean)'''

    data_dir = r'/media/ei-edl01/user/bh163/tasks/2017.11.20.urban_mapper_heights'
    model_name = 'UnetInria_Origin_fr_resample_EP-25_DS-20_DR-0.1_LY-6_LR-0.5-1.0e-04'
    iou = dict(np.load(os.path.join(data_dir, model_name+'.npy')).tolist())
    print(iou['JAX004'])
    print(iou['JAX005'])
    print(iou['TAM000'])
    print(iou['TAM001'])
    print((iou['JAX004']+iou['JAX005']+iou['TAM000']+iou['TAM001'])/4)

    print()
    data_dir = r'/media/ei-edl01/user/bh163/tasks/2017.11.15.grid_experiments'
    model_name = 'UNET_um_no_random_scratch.npy'
    iou = dict(np.load(os.path.join(data_dir, model_name)).tolist())
    print(iou['JAX004'])
    print(iou['JAX005'])
    print(iou['TAM000'])
    print(iou['TAM001'])
    print((iou['JAX004'] + iou['JAX005'] + iou['TAM000'] + iou['TAM001']) / 4)
