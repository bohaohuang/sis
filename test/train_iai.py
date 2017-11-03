import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
sys.path.append(os.path.realpath(r'/home/lab/Documents/bohao/code/sisl'))
from utils import image_reader
from utils import model
from network import unet

TRAIN_DATA_DIR = r'/media/ei-edl01/user/bh163/data/iai/train_patches'
VALID_DATA_DIR = r'/media/ei-edl01/user/bh163/data/iai/valid_patches'
CITY_NAME = 'austin,chicago,kitsap,tyrol-w,vienna'
TRAIN_TILE_NAMES = ','.join(['{}'.format(i) for i in range(6,37)])
VALID_TILE_NAMES = ','.join(['{}'.format(i) for i in range(1,6)])
RANDOM_SEED = 1234
BATCH_SIZE = 10
LEARNING_RATE = 1e-3
INPUT_SIZE = (224, 224)
EPOCHS = 100
CKDIR = r'./models'
NUM_CLASS = 2
N_TRAIN = 8000
GPU = '0'
DECAY_STEP = 60
DECAY_RATE = 0.1
VALID_SIZE = 1000


def read_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data-dir', default=TRAIN_DATA_DIR, help='path to release folder')
    parser.add_argument('--valid-data-dir', default=VALID_DATA_DIR, help='path to release folder')
    parser.add_argument('--train-tile-names', default=TRAIN_TILE_NAMES, help='image tiles')
    parser.add_argument('--valid-tile-names', default=VALID_TILE_NAMES, help='image tiles')
    parser.add_argument('--city-name', type=str, default=CITY_NAME, help='city name (default austin)')
    parser.add_argument('--random-seed', type=int, default=RANDOM_SEED, help='tf random seed')
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (10)')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE, help='learning rate (1e-3)')
    parser.add_argument('--input-size', default=INPUT_SIZE, type=int, help='input size (128, 128)')
    parser.add_argument('--epochs', default=EPOCHS, type=int, help='# epochs (1)')
    parser.add_argument('--ckdir', default=CKDIR, help='ckpt dir (models)')
    parser.add_argument('--num-classes', type=int, default=NUM_CLASS, help='# classes (including background)')
    parser.add_argument('--n-train', type=int, default=N_TRAIN, help='# samples per epoch')
    parser.add_argument("--GPU", type=str, default=GPU, help="GPU used for computation.")
    parser.add_argument("--decay-step", type=float, default=DECAY_STEP, help="Learning rate decay step in number of epochs.")
    parser.add_argument("--decay-rate", type=float, default=DECAY_RATE, help="Learning rate decay rate")
    parser.add_argument('--valid-size', type=int, default=VALID_SIZE, help='#patches to valid')

    flags = parser.parse_args()
    return flags


def image_summary(image, truth, prediction):
    truth_img = image_reader.decode_labels(truth, 10, ds='iai')
    pred_labels = image_reader.get_pred_labels(prediction)
    pred_img = image_reader.decode_labels(pred_labels, 10, ds='iai')
    return np.concatenate([image, truth_img, pred_img], axis=2)


def main(flags):
    # set gpu
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = flags.GPU
    # environment settings
    np.random.seed(flags.random_seed)
    tf.set_random_seed(flags.random_seed)
    # image reader
    city_names = flags.city_name.split(',')
    tile_names = []
    for city_name in city_names:
        tile_names.extend(['{}{}'.format(city_name, tile) for tile in flags.train_tile_names.split(',')])
    image_reader_train = image_reader.ImageReader(flags.train_data_dir, tile_names=tile_names,
                                                  random_rotation=True, random_flip=True, ds='iai')
    train_iterator = image_reader_train.data_iterator_iai(flags.batch_size)

    city_names = flags.city_name.split(',')
    tile_names = []
    for city_name in city_names:
        tile_names.extend(['{}{}'.format(city_name, tile) for tile in flags.valid_tile_names.split(',')])
    image_reader_valid = image_reader.ImageReader(flags.valid_data_dir, tile_names=tile_names,
                                                  patch_num=flags.valid_size, random_rotation=False, random_flip=False,
                                                  ds='iai')
    valid_iterator = image_reader_valid.data_iterator_iai(flags.batch_size * 2)

    # define place holder
    X = tf.placeholder(tf.float32, shape=[None, flags.input_size[0], flags.input_size[1], 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, flags.input_size[0], flags.input_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')

    # initialize model
    model = unet.UnetModel({'X':X, 'Y':y}, trainable=mode, model_name='UnetInria', input_size=flags.input_size)
    model.create_graph('X', flags.num_classes)
    model.make_loss('Y')
    model.make_learning_rate(flags.learning_rate,
                             tf.cast(flags.n_train/flags.batch_size * flags.decay_step, tf.int32), flags.decay_rate)
    model.make_update_ops('X', 'Y')
    model.make_optimizer(model.learning_rate)
    # set ckdir
    model.make_ckdir(flags.ckdir)
    # make summary
    model.make_summary()
    # set up graph and initialize
    config = tf.ConfigProto()

    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)

        if os.path.exists(model.ckdir) and tf.train.get_checkpoint_state(model.ckdir):
            latest_check_point = tf.train.latest_checkpoint(model.ckdir)
            saver.restore(sess, latest_check_point)

        try:
            train_summary_writer = tf.summary.FileWriter(model.ckdir, sess.graph)

            model.train('X', 'Y', flags.epochs, flags.n_train, flags.batch_size, sess, train_summary_writer,
                        train_iterator=train_iterator, valid_iterator=valid_iterator, image_summary=image_summary)
        finally:
            saver.save(sess, '{}/model.ckpt'.format(model.ckdir), global_step=model.global_step)


if __name__ == '__main__':
    flags = read_flag()
    main(flags)