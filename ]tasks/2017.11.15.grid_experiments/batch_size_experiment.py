import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import utils

TEST_DATA_DIR = 'dcc_inria_valid'
CITY_NAME = 'austin'
RSR_DATA_DIR = r'/media/ei-edl01/data/remote_sensing_data'
PATCH_DIR = r'/media/ei-edl01/user/bh163/data/iai'
TEST_PATCH_APPENDIX = 'valid_noaug_dcc'
TEST_TILE_NAMES = ','.join(['{}'.format(i) for i in range(1, 6)])
RANDOM_SEED = 1234
BATCH_SIZE = 1
INPUT_SIZE = 224
CKDIR = r'/home/lab/Documents/bohao/code/sis/test/models/GridExp'
MODEL_NAME = 'UnetInria_no_aug'
NUM_CLASS = 2
GPU = '0'
IMG_SAVE_DIR = r'/media/ei-edl01/user/bh163/figs'


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


def get_ious(flags):
    for cnt, batch_size in enumerate([1]):
        model_name = 'UNET_PS-{}__BS-{}__E-100__NT-8000__DS-60__CT-__no_random'.format(flags.input_size[0], batch_size)
        print(model_name)

        result = utils.test_unet(flags.rsr_data_dir,
                                 flags.test_data_dir,
                                 flags.input_size,
                                 model_name,
                                 flags.num_classes,
                                 flags.ckdir,
                                 flags.city_name,
                                 flags.batch_size)
        print(result)
        _, task_dir = utils.get_task_img_folder()
        np.save(os.path.join(task_dir, '{}_{}.npy'.format(model_name, flags.input_size)), result)


def get_ious_patch_size(flags, patch_size_list):
    for patch_size in patch_size_list:
        model_name = 'UNET_PS-{}__BS-{}__E-100__NT-8000__DS-60__CT-__no_random'.format(patch_size, 1)
        print(model_name)

        result = utils.test_unet(flags.rsr_data_dir,
                                 flags.test_data_dir,
                                 (patch_size, patch_size),
                                 model_name,
                                 flags.num_classes,
                                 flags.ckdir,
                                 flags.city_name,
                                 flags.batch_size)
        print(result)
        _, task_dir = utils.get_task_img_folder()
        np.save(os.path.join(task_dir, '{}_{}.npy'.format(model_name, patch_size)), result)


def get_ious_psfixed(flags, patch_size_list):
    for patch_size in patch_size_list:
        model_name = 'UNET_PS-224__BS-10__E-100__NT-8000__DS-60__CT-__no_random'
        print(model_name)

        result = utils.test_unet(flags.rsr_data_dir,
                                 flags.test_data_dir,
                                 (patch_size, patch_size),
                                 model_name,
                                 flags.num_classes,
                                 flags.ckdir,
                                 flags.city_name,
                                 flags.batch_size)
        print(result)
        _, task_dir = utils.get_task_img_folder()
        np.save(os.path.join(task_dir, '{}_fixed_{}.npy'.format(model_name, patch_size)), result)


def get_ious_patch_size_224(flags, patch_size_list):
    for patch_size in patch_size_list:
        model_name = 'UNET_PS-{}__BS-{}__E-100__NT-8000__DS-60__CT-__no_random'.format(patch_size, 1)
        print(model_name)

        result = utils.test_unet(flags.rsr_data_dir,
                                 flags.test_data_dir,
                                 (224, 224),
                                 model_name,
                                 flags.num_classes,
                                 flags.ckdir,
                                 flags.city_name,
                                 flags.batch_size)
        print(result)
        _, task_dir = utils.get_task_img_folder()
        np.save(os.path.join(task_dir, '{}_{}.npy'.format(model_name, 224)), result)


if __name__ == '__main__':
    flags = read_flag()
    img_dir, task_dir = utils.get_task_img_folder()
    #get_ious(flags)

    #patch_size_list = np.array([352, 1632])
    #get_ious_patch_size(flags, patch_size_list)
    #get_ious_patch_size_224(flags, np.array([352, 480, 608, 1632]))

    #get_ious(flags)
    #get_ious_psfixed(flags, np.array([224, 352, 480, 608, 1632]))

    '''ious = np.zeros((5, 5))
    for cnt, batch_size in enumerate([1, 2, 5, 10, 16]):
        model_name = 'UNET_PS-{}__BS-{}__E-100__NT-8000__DS-60__CT-__no_random'.format(flags.input_size[0], batch_size)

        result = dict(np.load(os.path.join(task_dir, '{}.npy'.format(model_name))).tolist())

        for i in range(5):
            ious[cnt, i] = result['austin{}'.format(i+1)]

    iou_mean = np.mean(ious, axis=1)
    iou_std = np.std(ious, axis=1)

    N = 5
    ind = np.arange(N)

    fig, ax = plt.subplots()
    ax.bar(ind, iou_mean, 0.35, color='g', yerr=iou_std)
    plt.xticks(ind, np.array([1, 2, 5, 10, 16]))
    plt.xlabel('Batch Size')
    plt.ylabel('mean IoU')
    plt.title('Batch Size Comparison')
    plt.savefig(os.path.join(img_dir, 'bs_vs_iou.png'))
    plt.show()'''

    '''ious = np.zeros((4, 5))
    for cnt, patch_size in enumerate([352, 480, 608, 1632]):
        model_name = 'UNET_PS-{}__BS-1__E-100__NT-8000__DS-60__CT-__no_random'.format(patch_size)

        result = dict(np.load(os.path.join(task_dir, '{}_{}.npy'.format(model_name, patch_size))).tolist())

        for i in range(5):
            ious[cnt, i] = result['austin{}'.format(i + 1)]

    iou_mean = np.mean(ious, axis=1)
    iou_std = np.std(ious, axis=1)

    N = 4
    ind = np.arange(N)

    fig, ax = plt.subplots()
    ax.bar(ind, iou_mean, 0.35, color='g', yerr=iou_std)
    plt.xticks(ind, np.array([352, 480, 608, 1632]))
    plt.xlabel('Patch Size')
    plt.ylabel('mean IoU')
    plt.ylim(ymax=0.8)
    plt.title('Patch Size Comparison (Evaluated at Their Sizes)')
    plt.savefig(os.path.join(img_dir, 'ps_vs_iou_adaptive.png'))
    plt.show()'''

    ious = np.zeros((5, 5))
    for cnt, patch_size in enumerate([224, 352, 480, 608, 1632]):
        model_name = 'UNET_PS-224__BS-10__E-100__NT-8000__DS-60__CT-__no_random'

        result = dict(np.load(os.path.join(task_dir, '{}_fixed_{}.npy'.format(model_name, patch_size))).tolist())

        for i in range(5):
            ious[cnt, i] = result['austin{}'.format(i + 1)]

    iou_mean = np.mean(ious, axis=1)
    iou_std = np.std(ious, axis=1)

    N = 5
    ind = np.arange(N)

    fig, ax = plt.subplots()
    ax.bar(ind, iou_mean, 0.35, color='g', yerr=iou_std)
    plt.xticks(ind, np.array([224, 352, 480, 608, 1632]))
    plt.xlabel('Patch Size')
    plt.ylabel('mean IoU')
    plt.ylim(ymin=0.7)
    plt.title('Patch Size Comparison (Trained at 224)')
    plt.savefig(os.path.join(img_dir, 'ps_vs_iou_train_224.png'))
    plt.show()
