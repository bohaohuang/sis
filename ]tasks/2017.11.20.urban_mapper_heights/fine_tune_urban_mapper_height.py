import os
import time
import argparse
import numpy as np
import tensorflow as tf
import sis_utils
from network import unet
from dataReader import image_reader, patch_extractor
from rsrClassData import rsrClassData

TRAIN_DATA_DIR = 'dcc_urban_mapper_height_train_p'
VALID_DATA_DIR = 'dcc_urban_mapper_height_valid_p'
CITY_NAME = 'JAX,TAM'
RSR_DATA_DIR = r'/media/ei-edl01/data/remote_sensing_data'
PATCH_DIR = r'/home/lab/Documents/bohao/data/urban_mapper'
PRE_TRAINED_MODEL = r'/home/lab/Documents/bohao/code/sis/test/models/UnetInria_Origin_no_aug_resample'
LAYERS_TO_KEEP = '1,2,3,4,5,6,7,8,9'
TRAIN_PATCH_APPENDIX = 'train_augfr_um_p2'
VALID_PATCH_APPENDIX = 'valid_augfr_um_p2'
TRAIN_TILE_NAMES = ','.join(['{}'.format(i) for i in range(20,143)])
VALID_TILE_NAMES = ','.join(['{}'.format(i) for i in range(0,20)])
RANDOM_SEED = 1234
BATCH_SIZE = 5
LEARNING_RATE = 1e-4
INPUT_SIZE = 321
EPOCHS = 15
CKDIR = r'/home/lab/Documents/bohao/code/sis/test/models/UrbanMapper_Height_npy'
MODEL_NAME = 'unet_origin_finetune_um_augfr_9'
HEIGHT_MODE = 'subtract'
DATA_AUG = 'filp,rotate'
NUM_CLASS = 2
N_TRAIN = 8000
GPU = '1'
DECAY_STEP = 10
DECAY_RATE = 0.1


def read_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data-dir', default=TRAIN_DATA_DIR, help='path to release folder')
    parser.add_argument('--valid-data-dir', default=VALID_DATA_DIR, help='path to release folder')
    parser.add_argument('--rsr-data-dir', default=RSR_DATA_DIR, help='path to rsrClassData folder')
    parser.add_argument('--patch-dir', default=PATCH_DIR, help='path to patch directory')
    parser.add_argument('--train-patch-appendix', default=TRAIN_PATCH_APPENDIX, help='train patch appendix')
    parser.add_argument('--valid-patch-appendix', default=VALID_PATCH_APPENDIX, help='valid patch appendix')
    parser.add_argument('--train-tile-names', default=TRAIN_TILE_NAMES, help='image tiles')
    parser.add_argument('--valid-tile-names', default=VALID_TILE_NAMES, help='image tiles')
    parser.add_argument('--city-name', type=str, default=CITY_NAME, help='city name (default austin)')
    parser.add_argument('--random-seed', type=int, default=RANDOM_SEED, help='tf random seed')
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (10)')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE, help='learning rate (1e-3)')
    parser.add_argument('--input-size', default=INPUT_SIZE, type=int, help='input size 224')
    parser.add_argument('--epochs', default=EPOCHS, type=int, help='# epochs (1)')
    parser.add_argument('--ckdir', default=CKDIR, help='ckpt dir (models)')
    parser.add_argument('--num-classes', type=int, default=NUM_CLASS, help='# classes (including background)')
    parser.add_argument('--n-train', type=int, default=N_TRAIN, help='# samples per epoch')
    parser.add_argument('--GPU', type=str, default=GPU, help="GPU used for computation.")
    parser.add_argument('--decay-step', type=float, default=DECAY_STEP, help='Learning rate decay step in number of epochs.')
    parser.add_argument('--decay-rate', type=float, default=DECAY_RATE, help='Learning rate decay rate')
    parser.add_argument('--model-name', type=str, default=MODEL_NAME, help='Model name')
    parser.add_argument('--pre-trained-model', default=PRE_TRAINED_MODEL, help='Path to pretrained model')
    parser.add_argument('--layers-to-keep', default=LAYERS_TO_KEEP, help='layers to keep, range 1 to 9')
    parser.add_argument('--data-aug', type=str, default=DATA_AUG, help='Data augmentation methods')
    parser.add_argument('--height-mode', type=str, default=HEIGHT_MODE, help='How to use heights information')

    flags = parser.parse_args()
    flags.input_size = (flags.input_size, flags.input_size)
    flags.layers_to_keep_num = [int(layer_id) for layer_id in flags.layers_to_keep.split(',')]
    return flags


def main(flags):
    # set gpu
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = flags.GPU
    # environment settings
    np.random.seed(flags.random_seed)
    tf.set_random_seed(flags.random_seed)

    # get weight
    '''tf.reset_default_graph()
    if flags.height_mode == 'subtract':
        kernel = utils.get_unet_first_layer_weight(flags.pre_trained_model, 1)
    elif flags.height_mode == 'all':
        kernel = utils.get_unet_first_layer_weight(flags.pre_trained_model, 2)
    else:
        kernel = utils.get_unet_first_layer_weight(flags.pre_trained_model, 3)
    tf.reset_default_graph()'''

    # data prepare step
    Data = rsrClassData(flags.rsr_data_dir)
    (collect_files_train, meta_train) = Data.getCollectionByName(flags.train_data_dir)
    pe_train = patch_extractor.PatchExtractorUrbanMapper(flags.rsr_data_dir,
                                                               collect_files_train, patch_size=flags.input_size,
                                                               tile_dim=meta_train['dim_image'][:2],
                                                               appendix=flags.train_patch_appendix)
    train_data_dir = pe_train.extract(flags.patch_dir)
    (collect_files_valid, meta_valid) = Data.getCollectionByName(flags.valid_data_dir)
    pe_valid = patch_extractor.PatchExtractorUrbanMapper(flags.rsr_data_dir,
                                                               collect_files_valid, patch_size=flags.input_size,
                                                               tile_dim=meta_train['dim_image'][:2],
                                                               appendix=flags.valid_patch_appendix)
    valid_data_dir = pe_valid.extract(flags.patch_dir)

    # image reader
    coord = tf.train.Coordinator()

    # load reader
    reader_train = image_reader.ImageLabelReaderHeight(train_data_dir, flags.input_size, coord,
                                                       city_list=flags.city_name, tile_list=flags.train_tile_names,
                                                       ds_name='urban_mapper', data_aug=flags.data_aug,
                                                       height_mode=flags.height_mode)
    reader_valid = image_reader.ImageLabelReaderHeight(valid_data_dir, flags.input_size, coord,
                                                       city_list=flags.city_name, tile_list=flags.valid_tile_names,
                                                       ds_name='urban_mapper', data_aug=flags.data_aug,
                                                       height_mode=flags.height_mode)
    reader_train_iter = reader_train.image_height_label_iterator(flags.batch_size)
    reader_valid_iter = reader_valid.image_height_label_iterator(flags.batch_size)

    # define place holder
    if flags.height_mode == 'all':
        X = tf.placeholder(tf.float32, shape=[None, flags.input_size[0], flags.input_size[1], 5], name='X')
    elif flags.height_mode == 'subtract':
        X = tf.placeholder(tf.float32, shape=[None, flags.input_size[0], flags.input_size[1], 4], name='X')
    else:
        X = tf.placeholder(tf.float32, shape=[None, flags.input_size[0], flags.input_size[1], 6], name='X')
    y = tf.placeholder(tf.int32, shape=[None, flags.input_size[0], flags.input_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')

    # initialize model
    model = unet.UnetModel_Height({'X':X, 'Y':y}, trainable=mode, model_name=flags.model_name, input_size=flags.input_size)
    model.create_graph('X', flags.num_classes)
    model.load_weights(flags.pre_trained_model, flags.layers_to_keep_num, kernel)
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

    # run training
    start_time = time.time()
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)

        if os.path.exists(model.ckdir) and tf.train.get_checkpoint_state(model.ckdir):
            latest_check_point = tf.train.latest_checkpoint(model.ckdir)
            saver.restore(sess, latest_check_point)

        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        try:
            train_summary_writer = tf.summary.FileWriter(model.ckdir, sess.graph)
            model.train('X', 'Y', flags.epochs, flags.n_train, flags.batch_size, sess, train_summary_writer,
                        train_iterator=reader_train_iter, valid_iterator=reader_valid_iter, image_summary=sis_utils.image_summary)
        finally:
            coord.request_stop()
            coord.join(threads)
            saver.save(sess, '{}/model.ckpt'.format(model.ckdir), global_step=model.global_step)

    duration = time.time() - start_time
    print('duration {:.2f} hours'.format(duration/60/60))

if __name__ == '__main__':
    flags = read_flag()
    main(flags)