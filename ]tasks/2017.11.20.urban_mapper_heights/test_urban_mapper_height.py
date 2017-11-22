import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import utils

TEST_DATA_DIR = 'dcc_urban_mapper_height_valid'
CITY_NAME = 'JAX,TAM'
RSR_DATA_DIR = r'/media/ei-edl01/data/remote_sensing_data'
PATCH_DIR = r'/home/lab/Documents/bohao/data/urban_mapper'
TEST_PATCH_APPENDIX = 'valid_augfr_um'
TEST_TILE_NAMES = ','.join(['{}'.format(i) for i in range(0, 16)])
RANDOM_SEED = 1234
BATCH_SIZE = 5
INPUT_SIZE = 572
CKDIR = r'/home/lab/Documents/bohao/code/sis/test/models/UrbanMapper_Height'
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


def evaluate_results(flags, model_name, height_model):
    result = utils.test_authentic_unet_height(flags.rsr_data_dir,
                                              flags.test_data_dir,
                                              flags.input_size,
                                              model_name,
                                              flags.num_classes,
                                              flags.ckdir,
                                              flags.city_name,
                                              flags.batch_size,
                                              ds_name='urban_mapper',
                                              height_mode=height_model)

    print(result)

    _, task_dir = utils.get_task_img_folder()
    np.save(os.path.join(task_dir, '{}.npy'.format(model_name)), result)

if __name__ == '__main__':
    '''flags = read_flag()
    evaluate_results(flags, 'unet_origin_scratch_um_augfr_4', 'subtract')

    _, task_dir = utils.get_task_img_folder()
    result = dict(np.load(os.path.join(task_dir, '{}.npy'.format('unet_origin_scratch_um_augfr_4'))).tolist())

    iou = []
    for k, v in result.items():
        iou.append(v)
    result_mean = np.mean(iou)
    print(result_mean)'''

    '''import matplotlib.pyplot as plt
    plt.plot(np.array([0.327, 0.664, 0.703]), np.array([266773.43, 733870.19, 749905.74]))
    plt.show()'''

    dir = r'/home/lab/Documents/bohao/data/urban_mapper/PS_(572, 572)-OL_0-AF_valid_augfr_um'
    file_name = 'JAX006_dsm_00004.png'
    import scipy.misc
    import os
    img = scipy.misc.imread(os.path.join(dir, file_name))
    print(img[:10, :10])
    print(img[-10:, -10:])

    dir2 = r'/media/ei-edl01/data/remote_sensing_data/urban_mapper/image'
    file_name2 = 'JAX_Tile_006_DSM.tif'
    img2 = scipy.misc.imread(os.path.join(dir2, file_name2))
    print(img2[:10, :10])
    print(img2[-10:, -10:])

    np.save('JAX_Tile_006_DSM_small1.npy', img2[:10, :10])
    np.save('JAX_Tile_006_DSM_small2.npy', img2[-10:, -10:])
    img3 = np.load('JAX_Tile_006_DSM_small1.npy')
    print(img3)
    img4 = np.load('JAX_Tile_006_DSM_small2.npy')
    print(img4)

    '''scipy.misc.imsave('fig_test.png', img2)
    img3 = scipy.misc.imread('fig_test.png')
    print(img3[:10, :10])
    print(img3[-10:, -10:])

    scipy.misc.imsave('fig_test.jpg', img2)
    img4 = scipy.misc.imread('fig_test.jpg')
    print(img4[:10, :10])
    print(img4[-10:, -10:])

    plt.subplot(121)
    plt.imshow(img3[250:, :])
    #plt.subplot(122)
    #plt.imshow(img4[250:, :])
    plt.show()'''
