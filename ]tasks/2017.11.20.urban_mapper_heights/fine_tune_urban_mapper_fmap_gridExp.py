import os
import time
import argparse
import numpy as np
import tensorflow as tf
import sis_utils
from network import unet
from dataReader import image_reader, patch_extractor
from rsrClassData import rsrClassData

TRAIN_DATA_DIR = 'dcc_urban_mapper_height_train_p_f'
VALID_DATA_DIR = 'dcc_urban_mapper_height_valid_p_f'
CITY_NAME = 'JAX,TAM'
RSR_DATA_DIR = r'/media/ei-edl01/data/remote_sensing_data'
PATCH_DIR = r'/home/lab/Documents/bohao/data/urban_mapper'
PRE_TRAINED_MODEL = r'/home/lab/Documents/bohao/code/sis/test/models/UnetInria_Origin_fr_resample'
LAYERS_TO_KEEP = '1,2,3,4,5,6,7,8,9'
TRAIN_PATCH_APPENDIX = 'train_augfr_um_npy_p_f'
VALID_PATCH_APPENDIX = 'valid_augfr_um_npy_p_f'
TRAIN_TILE_NAMES = ','.join(['{}'.format(i) for i in range(17,143)])
VALID_TILE_NAMES = ','.join(['{}'.format(i) for i in range(0,17)])
RANDOM_SEED = 1234
BATCH_SIZE = 5
LEARNING_RATE = 1e-4
INPUT_SIZE = 572
EPOCHS = 15
CKDIR = r'/home/lab/Documents/bohao/code/sis/test/models/UrbanMapper_Height_GridExp'
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


def fine_tune_grid_exp(height_mode,
                       layers_to_keep_num,
                       learning_rate,
                       decay_step,
                       decay_rate,
                       epochs,
                       model_name):
    # set gpu
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = flags.GPU
    # environment settings
    np.random.seed(flags.random_seed)
    tf.set_random_seed(flags.random_seed)

    # get weight
    tf.reset_default_graph()
    if height_mode == 'subtract':
        kernel = sis_utils.get_unet_first_layer_weight(flags.pre_trained_model, 4)
    elif height_mode == 'all':
        kernel = sis_utils.get_unet_first_layer_weight(flags.pre_trained_model, 2)
    else:
        kernel = sis_utils.get_unet_first_layer_weight(flags.pre_trained_model, 3)
    tf.reset_default_graph()

    # data prepare step
    Data = rsrClassData(flags.rsr_data_dir)
    (collect_files_train, meta_train) = Data.getCollectionByName(flags.train_data_dir)
    pe_train = patch_extractor.PatchExtractorUrbanMapperHeightFmap(flags.rsr_data_dir,
                                                               collect_files_train, patch_size=flags.input_size,
                                                               tile_dim=meta_train['dim_image'][:2],
                                                               appendix=flags.train_patch_appendix)
    train_data_dir = pe_train.extract(flags.patch_dir)
    (collect_files_valid, meta_valid) = Data.getCollectionByName(flags.valid_data_dir)
    pe_valid = patch_extractor.PatchExtractorUrbanMapperHeightFmap(flags.rsr_data_dir,
                                                               collect_files_valid, patch_size=flags.input_size,
                                                               tile_dim=meta_train['dim_image'][:2],
                                                               appendix=flags.valid_patch_appendix)
    valid_data_dir = pe_valid.extract(flags.patch_dir)

    # image reader
    coord = tf.train.Coordinator()

    # load reader
    reader_train = image_reader.ImageLabelReaderHeightFmap(train_data_dir, flags.input_size, coord,
                                                       city_list=flags.city_name, tile_list=flags.train_tile_names,
                                                       ds_name='urban_mapper', data_aug=flags.data_aug,
                                                       height_mode=flags.height_mode)
    reader_valid = image_reader.ImageLabelReaderHeightFmap(valid_data_dir, flags.input_size, coord,
                                                       city_list=flags.city_name, tile_list=flags.valid_tile_names,
                                                       ds_name='urban_mapper', data_aug=flags.data_aug,
                                                       height_mode=flags.height_mode)
    reader_train_iter = reader_train.image_height_label_fmap_iterator(flags.batch_size)
    reader_valid_iter = reader_valid.image_height_label_fmap_iterator(flags.batch_size)

    # define place holder
    if height_mode == 'all':
        X = tf.placeholder(tf.float32, shape=[None, flags.input_size[0], flags.input_size[1], 5], name='X')
    elif height_mode == 'subtract':
        X = tf.placeholder(tf.float32, shape=[None, flags.input_size[0], flags.input_size[1], 7], name='X')
    else:
        X = tf.placeholder(tf.float32, shape=[None, flags.input_size[0], flags.input_size[1], 6], name='X')
    y = tf.placeholder(tf.int32, shape=[None, flags.input_size[0], flags.input_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')

    # initialize model
    model = unet.UnetModel_Height_Appendix({'X':X, 'Y':y}, trainable=mode, model_name=model_name, input_size=flags.input_size)
    model.create_graph('X', flags.num_classes)
    model.load_weights(flags.pre_trained_model, layers_to_keep_num, kernel)
    model.make_loss('Y')
    model.make_learning_rate(learning_rate,
                             tf.cast(flags.n_train/flags.batch_size * decay_step, tf.int32), decay_rate)
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
            print('Model Loaded {}'.format(model.ckdir))

        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        try:
            train_summary_writer = tf.summary.FileWriter(model.ckdir, sess.graph)
            model.train('X', 'Y', epochs, flags.n_train, flags.batch_size, sess, train_summary_writer,
                        train_iterator=reader_train_iter, valid_iterator=reader_valid_iter, image_summary=sis_utils.image_summary)
        finally:
            coord.request_stop()
            coord.join(threads)
            saver.save(sess, '{}/model.ckpt'.format(model.ckdir), global_step=model.global_step)

    duration = time.time() - start_time
    print('duration {:.2f} hours'.format(duration/60/60))


def evaluate_results(flags, model_name, height_mode):
    result = sis_utils.test_authentic_unet_height_fmap(flags.rsr_data_dir,
                                                       flags.valid_data_dir,
                                                       flags.input_size,
                                                       model_name,
                                                       flags.num_classes,
                                                       flags.ckdir,
                                                       flags.city_name,
                                                       flags.batch_size,
                                                       ds_name='urban_mapper',
                                                       GPU=flags.GPU,
                                                       height_mode=height_mode)
    _, task_dir = sis_utils.get_task_img_folder()
    np.save(os.path.join(task_dir, '{}.npy'.format(model_name)), result)

    return result


if __name__ == '__main__':
    flags = read_flag()
    _, task_dir = sis_utils.get_task_img_folder()

    height_mode = 'subtract'
    epochs = 25
    decay_step = 20
    decay_rate = 0.1
    lr_base = 1e-4
    #for ly2kp in range(7, 8):
    for ly2kp in [7]:
        layers_to_keep_num = [i for i in range(1, ly2kp+1)]
        #for lr in [0.5, 0.25, 0.1, 0.075, 0.05, 0.025, 0.01]:
        for lr in [0.5]:
            learning_rate = lr * lr_base

            model_name = '{}_rescaled_appendix_fmap_EP-{}_DS-{}_DR-{}_LY-{}_LR-{}-{:1.1e}'.\
                format(flags.pre_trained_model.split('/')[-1],
                       epochs,
                       decay_step,
                       decay_rate,
                       ly2kp,
                       lr,
                       lr_base)
            print('Finetuneing model: {}'.format(model_name))

            fine_tune_grid_exp(height_mode,
                               layers_to_keep_num,
                               learning_rate,
                               decay_step,
                               decay_rate,
                               epochs,
                               model_name)

            try:
                print('Evaluating model: {}'.format(model_name))
                result = evaluate_results(flags, model_name, height_mode)
                iou = []
                for k, v in result.items():
                    iou.append(v)
                result_mean = np.mean(iou)
                print('\t Mean IoU on Validation Set: {:.3f}'.format(result_mean))

                with open(os.path.join(task_dir, 'grid_exp_record_1129.txt'), 'a') as record_file:
                    record_file.write('{} {}\n'.format(model_name, result_mean))
            finally:
                print('end')
