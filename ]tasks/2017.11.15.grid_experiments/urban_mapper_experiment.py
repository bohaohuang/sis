import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sis_utils

TEST_DATA_DIR = 'dcc_urban_mapper_height_valid'
CITY_NAME = 'JAX,TAM'
RSR_DATA_DIR = r'/media/ei-edl01/data/remote_sensing_data'
PATCH_DIR = r'/home/lab/Documents/bohao/data/urban_mapper'
TEST_PATCH_APPENDIX = 'valid_augfr_um_npy'
TEST_TILE_NAMES = ','.join(['{}'.format(i) for i in range(0, 16)])
RANDOM_SEED = 1234
BATCH_SIZE = 1
INPUT_SIZE = 224
CKDIR = r'/home/lab/Documents/bohao/code/sis/test/models/UrbanMapper_Height_npy'
MODEL_NAME = 'unet_origin_scratch_um_augfr_4'
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


def evaluate_a_result(flags, model_name, input_size):
    result = sis_utils.test_authentic_unet(flags.rsr_data_dir,
                                           flags.test_data_dir,
                                           input_size,
                                           model_name,
                                           flags.num_classes,
                                           flags.ckdir,
                                           flags.city_name,
                                           flags.batch_size,
                                           ds_name='urban_mapper')

    print(result)

    _, task_dir = sis_utils.get_task_img_folder()
    np.save(os.path.join(task_dir, '{}_resampled.npy'.format(model_name)), result)


def evaluate_results(flags):
    for layer_id in [6, 7, 8, 9]:
        model_name = 'UNET_um_no_random_resampled_{}'.format(layer_id)

        result = sis_utils.test_unet(flags.rsr_data_dir,
                                     flags.test_data_dir,
                                     flags.input_size,
                                     model_name,
                                     flags.num_classes,
                                     flags.ckdir,
                                     flags.city_name,
                                     flags.batch_size,
                                     ds_name='urban_mapper')

        print(result)

        _, task_dir = sis_utils.get_task_img_folder()
        np.save(os.path.join(task_dir, '{}_resampled.npy'.format(model_name)), result)


def evaluate_no_train(flags):
    model_name = 'UnetInria_no_aug_resampled'

    result = sis_utils.test_unet(flags.rsr_data_dir,
                                 flags.test_data_dir,
                                 flags.input_size,
                                 model_name,
                                 flags.num_classes,
                                 flags.ckdir,
                                 flags.city_name,
                                 flags.batch_size,
                                 ds_name='urban_mapper')

    print(result)

    _, task_dir = sis_utils.get_task_img_folder()
    np.save(os.path.join(task_dir, '{}_resampled.npy'.format(model_name)), result)


def evaluate_scratch(flags):
    model_name = 'UNET_um_no_random_scratch'

    result = sis_utils.test_unet(flags.rsr_data_dir,
                                 flags.test_data_dir,
                                 flags.input_size,
                                 model_name,
                                 flags.num_classes,
                                 flags.ckdir,
                                 flags.city_name,
                                 flags.batch_size,
                                 ds_name='urban_mapper')

    print(result)

    _, task_dir = sis_utils.get_task_img_folder()
    np.save(os.path.join(task_dir, '{}.npy'.format(model_name)), result)


if __name__ == '__main__':
    flags = read_flag()
    #evaluate_results(flags)
    #evaluate_no_train(flags)
    #evaluate_scratch(flags)

    _, task_dir = sis_utils.get_task_img_folder()

    #evaluate_a_result(flags, 'UNET_new_um_7', (572, 572))
    result = dict(np.load(os.path.join(task_dir, 'UNET_new_um_7_resampled.npy')).tolist())

    mean_iou = []

    for k, v in result.items():
        mean_iou.append(v)

    print(np.mean(mean_iou))

    '''result_mean = np.zeros(6)
    result_std = np.zeros(6)

    img_dir, task_dir = utils.get_task_img_folder()

    # scratch
    iou = []
    model_name = 'UNET_um_no_random_scratch'
    result = dict(np.load(os.path.join(task_dir, '{}.npy'.format(model_name))).tolist())

    for k, v in result.items():
        iou.append(v)
    result_mean[0] = np.mean(iou)
    result_std[0] = np.std(iou)

    # fine tune
    cnt = 0
    for layer_id in [6, 7, 8, 9]:
        iou = []
        cnt += 1
        model_name = 'UNET_um_no_random_resampled_{}'.format(layer_id)
        result = dict(np.load(os.path.join(task_dir, '{}_resampled.npy'.format(model_name))).tolist())

        for k, v in result.items():
            iou.append(v)
        result_mean[cnt] = np.mean(iou)
        result_std[cnt] = np.std(iou)

    # no train
    iou = []
    model_name = 'UnetInria_no_aug_resampled'
    result = dict(np.load(os.path.join(task_dir, '{}_resampled.npy'.format(model_name))).tolist())

    for k, v in result.items():
        iou.append(v)
    result_mean[-1] = np.mean(iou)
    result_std[-1] = np.std(iou)

    # show figure
    N = 6
    ind = np.arange(N)

    fig = plt.figure(figsize=(12, 4))
    ax = plt.subplot(121)
    rect1 = ax.bar(ind, result_mean, 0.35, color='g', yerr=result_std)
    plt.xticks(ind, np.array([0, 6, 7, 8, 9, 10]))
    plt.xlabel('Fine Tune Scheme')
    plt.ylabel('mean IoU')
    utils.barplot_autolabel(ax, rect1, margin=0.01)
    plt.title('Fine Tune Scheme Comparison')

    ax = plt.subplot(122)
    ax.bar(ind[:-1], result_mean[:-1], 0.35, color='g')
    plt.xticks(ind[:-1], np.array([0, 6, 7, 8, 9]))
    plt.xlabel('Fine Tune Scheme')
    plt.ylabel('mean IoU')
    #plt.ylim(ymin=0.68, ymax=0.72)
    plt.title('Fine Tune Scheme Comparison')

    plt.savefig(os.path.join(img_dir, 'ps_vs_iou_urban_mapper.png'))
    plt.show()'''
