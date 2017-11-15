import os
import re
import time
import argparse
import numpy as np
import tensorflow as tf
import shutil
import scipy.misc
import matplotlib.pyplot as plt
import utils
from glob import glob
from network import unet
from dataReader import image_reader, patch_extractor
from rsrClassData import rsrClassData

TEST_DATA_DIR = 'dcc_inria_test'
CITY_NAME = 'bellingham'
RSR_DATA_DIR = r'/media/ei-edl01/data/remote_sensing_data'
PATCH_DIR = r'/media/ei-edl01/user/bh163/data/iai'
TEST_TILE_NAMES = ','.join(['{}'.format(i) for i in range(1, 5)])
RANDOM_SEED = 1234
BATCH_SIZE = 10
INPUT_SIZE = 224
CKDIR = r'/home/lab/Documents/bohao/code/sis/test/models'
MODEL_NAME = ['UNET_austin_no_random', 'UNET_chicago_no_random',
              'UNET_kitsap_no_random', 'UNET_tyrol-w_no_random',
              'UNET_vienna_no_random']
NUM_CLASS = 2
GPU = '0'
FIG_SAVE_DIR = r'/media/ei-edl01/user/bh163/Inria_submission/raw'


def read_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-data-dir', default=TEST_DATA_DIR, help='path to release folder')
    parser.add_argument('--rsr-data-dir', default=RSR_DATA_DIR, help='path to rsrClassData folder')
    parser.add_argument('--patch-dir', default=PATCH_DIR, help='path to patch directory')
    parser.add_argument('--test-tile-names', default=TEST_TILE_NAMES, help='image tiles')
    parser.add_argument('--city-name', type=str, default=CITY_NAME, help='city name (default austin)')
    parser.add_argument('--random-seed', type=int, default=RANDOM_SEED, help='tf random seed')
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (10)')
    parser.add_argument('--input-size', default=INPUT_SIZE, type=int, help='input size 224')
    parser.add_argument('--ckdir', default=CKDIR, help='ckpt dir (models)')
    parser.add_argument('--num-classes', type=int, default=NUM_CLASS, help='# classes (including background)')
    parser.add_argument('--GPU', type=str, default=GPU, help="GPU used for computation.")

    flags = parser.parse_args()
    flags.input_size = (flags.input_size, flags.input_size)
    return flags


def main(flags):
    # set gpu
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = flags.GPU
    # environment settings
    np.random.seed(flags.random_seed)
    tf.set_random_seed(flags.random_seed)

    for city in ['bellingham', 'bloomington', 'innsbruck', 'sfo', 'tyrol-e']:
        if not os.path.exists('./temp'):
            os.makedirs('./temp')

        flags.city_name = city
        for m_name in MODEL_NAME:
            tf.reset_default_graph()

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
            model = unet.UnetModel({'X':X, 'Y':y}, trainable=mode, model_name=m_name, input_size=flags.input_size)
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
                for (image_name,) in collect_files_test:
                    if flags.city_name in image_name:
                        city_name = re.findall('[a-z\-]*(?=[0-9]+\.)', image_name)[0]
                        tile_id = re.findall('[0-9]+(?=\.tif)', image_name)[0]
                        print('Evaluating {}_{} at patch size: {}'.format(city_name, tile_id, flags.input_size))

                        # load reader
                        iterator_test = image_reader.image_label_iterator(
                            os.path.join(flags.rsr_data_dir, image_name),
                            batch_size=flags.batch_size,
                            tile_dim=meta_test['dim_image'][:2],
                            patch_size=flags.input_size,
                            overlap=0)
                        # run
                        result = model.test('X', sess, iterator_test)
                        raw_pred = patch_extractor.un_patchify(result, meta_test['dim_image'], flags.input_size)

                        file_name = '{}_{}_{}.npy'.format(m_name, city_name, tile_id)
                        np.save(os.path.join('./temp', file_name), raw_pred)
                coord.request_stop()
                coord.join(threads)

            duration = time.time() - start_time
            print('duration {:.2f} minutes'.format(duration/60))

        for tile_num in range(36):
            output = np.zeros((5000, 5000, 2))
            for m_name in MODEL_NAME:
                raw_pred = np.load(os.path.join('./temp', '{}_{}_{}.npy'.format(m_name, flags.city_name, tile_num+1)))
                output += raw_pred

            # combine results
            labels_pred = utils.get_pred_labels(output)
            output_pred = utils.make_output_file(labels_pred, meta_test['colormap'])
            scipy.misc.imsave(os.path.join(FIG_SAVE_DIR, '{}{}.tif'.format(flags.city_name, tile_num+1)),
                              output_pred)

        shutil.rmtree('./temp')


if __name__ == '__main__':
    flags = read_flag()
    main(flags)