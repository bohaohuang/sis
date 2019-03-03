import os
import re
import time
import argparse
import numpy as np
import tensorflow as tf
import scipy.misc
import matplotlib.pyplot as plt
import sis_utils
from network import unet
from dataReader import image_reader, patch_extractor
from rsrClassData import rsrClassData

TEST_DATA_DIR = 'dcc_inria_valid'
CITY_NAME = 'austin'
RSR_DATA_DIR = r'/media/ei-edl01/data/remote_sensing_data'
PATCH_DIR = r'/media/ei-edl01/user/bh163/data/iai'
TEST_PATCH_APPENDIX = 'valid_noaug_dcc'
TEST_TILE_NAMES = ','.join(['{}'.format(i) for i in range(1, 6)])
RANDOM_SEED = 1234
BATCH_SIZE = 1
INPUT_SIZE = 576
CKDIR = r'/home/lab/Documents/bohao/code/sis/test/models'
MODEL_NAME = 'UnetInria_no_aug'
NUM_CLASS = 2
GPU = '1'

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


def evaluate_on_patch(result, label_img, tile_dim, patch_size):
    label_img_block = np.expand_dims(label_img, axis=2)
    label_patches = patch_extractor.patchify(label_img_block, tile_dim, patch_size, overlap=0)
    error_map = np.zeros(patch_size)
    for cnt, label in enumerate(label_patches):
        pred = sis_utils.get_pred_labels(result[cnt, :, :, :])
        error_map[np.where(pred != label[:, :, 0])] += 1
    return error_map


def error_vs_dist(error_map):
    dist = []
    error = []
    h, w = error_map.shape
    for i in range(h):
        for j in range(w):
            coordinate = np.array([i, j]) - np.array([h/2, w/2])
            dist.append(np.sqrt(coordinate[0]**2 + coordinate[1]**2))
            error.append(error_map[i, j])

    dist = np.array(dist)
    error = np.array(error)
    dist_uniq = []
    error_uniq = []
    for d in np.unique(dist):
        dist_uniq.append(d)
        e = np.sum(error[np.where(dist == d)])/len(error[np.where(dist == d)])
        error_uniq.append(e)
    return dist_uniq, error_uniq


def main(flags):
    # set gpu
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = flags.GPU
    # environment settings
    np.random.seed(flags.random_seed)
    tf.set_random_seed(flags.random_seed)

    # data prepare step
    Data = rsrClassData(flags.rsr_data_dir)
    (collect_files_test, meta_test) = Data.getCollectionByName(flags.test_data_dir)

    # image reader
    coord = tf.train.Coordinator()

    # define place holder
    X = tf.placeholder(tf.float32, shape=[None, flags.input_size[0], flags.input_size[1], 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, flags.input_size[0], flags.input_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')

    # initialize model
    model = unet.UnetModel({'X':X, 'Y':y}, trainable=mode, model_name=flags.model_name, input_size=flags.input_size)
    model.create_graph('X', flags.num_classes)
    model.make_update_ops('X', 'Y')
    # set ckdir
    model.make_ckdir(flags.ckdir)
    # set up graph and initialize
    config = tf.ConfigProto()

    # run training
    start_time = time.time()
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)

        if os.path.exists(model.ckdir) and tf.train.get_checkpoint_state(model.ckdir):
            print('loading model from {}'.format(model.ckdir))
            latest_check_point = tf.train.latest_checkpoint(model.ckdir)
            saver.restore(sess, latest_check_point)

        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        tile_cnt = 0
        error_map = np.zeros(flags.input_size)
        fig = plt.figure(figsize=(20.0, 12.0))
        try:
            for (image_name, label_name) in collect_files_test:
                if flags.city_name in image_name:
                    tile_cnt += 1

                    city_name = re.findall('[a-z\-]*(?=[0-9]+\.)', image_name)[0]
                    tile_id = re.findall('[0-9]+(?=\.tif)', image_name)[0]
                    print('Evaluating {}_{}.tif'.format(city_name, tile_id))

                    # load reader
                    iterator_test = image_reader.image_label_iterator(
                        os.path.join(flags.rsr_data_dir, image_name),
                        batch_size=flags.batch_size,
                        tile_dim=meta_test['dim_image'][:2],
                        patch_size=flags.input_size,
                        overlap=0)
                    # run
                    result = model.test('X', sess, iterator_test)

                    # evaluate
                    label_img = scipy.misc.imread(os.path.join(flags.rsr_data_dir, label_name))
                    error_map += evaluate_on_patch(result, label_img, meta_test['dim_image'][:2], flags.input_size)

                    plt.subplot(230 + tile_cnt)
                    plt.imshow(error_map/np.max(error_map), cmap='hot')
                    plt.colorbar()
                    plt.title('Error Map {}_{}'.format(city_name, tile_id))

            plt.suptitle('Patch Size = {}'.format(flags.input_size))
            dist, error = error_vs_dist(error_map)
            plt.subplot(236)
            plt.plot(dist, error, 'r.')
            plt.title('Error VS Dist ({})'.format(city_name))
            plt.xlabel('Dist to center of patch')
            plt.ylabel('#Errors')
            save_dir = sis_utils.make_task_img_folder(IMG_SAVE_DIR)
            plt.savefig(os.path.join(save_dir, 'M-{}_PS-{}'.format(flags.model_name, flags.input_size[0])),
                        bbox_inches='tight')
            plt.show()
        finally:
            coord.request_stop()
            coord.join(threads)

    duration = time.time() - start_time
    print('duration {:.2f} minutes'.format(duration/60))


if __name__ == '__main__':
    flags = read_flag()
    main(flags)