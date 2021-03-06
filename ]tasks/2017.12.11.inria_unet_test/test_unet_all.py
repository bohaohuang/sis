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


def test(flags, model_name, save_dir):
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
    if 'appendix' in model_name:
        model = unet.UnetModel_Height_Appendix({'X':X, 'Y':y}, trainable=mode, model_name=model_name, input_size=flags.input_size)
    else:
        model = unet.UnetModel_Origin({'X':X, 'Y':y}, trainable=mode, model_name=model_name, input_size=flags.input_size)
    if 'large' in model_name:
        model.create_graph('X', flags.num_classes, start_filter_num=40)
    else:
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
            latest_check_point = tf.train.latest_checkpoint(model.ckdir)
            saver.restore(sess, latest_check_point)
            print('loaded {}'.format(latest_check_point))

        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        try:
            iou_record = {}
            for (image_name, label_name) in collect_files_test:
                c_names = flags.city_name.split(',')
                for c_name in c_names:
                    if c_name in image_name:
                        city_name = re.findall('[a-z\-]*(?=[0-9]+\.)', image_name)[0]
                        tile_id = re.findall('[0-9]+(?=\.tif)', image_name)[0]

                        # load reader
                        iterator_test = image_reader.image_label_iterator(
                            os.path.join(flags.rsr_data_dir, image_name),
                            batch_size=flags.batch_size,
                            tile_dim=meta_test['dim_image'][:2],
                            patch_size=flags.input_size,
                            overlap=184, padding=92,
                            image_mean=IMG_MEAN)
                        # run
                        result = model.test('X', sess, iterator_test)

                        pred_label_img = sis_utils.get_output_label(result,
                                                                    (meta_test['dim_image'][0]+184, meta_test['dim_image'][1]+184),
                                                                    flags.input_size,
                                                                    meta_test['colormap'], overlap=184,
                                                                    output_image_dim=meta_test['dim_image'],
                                                                    output_patch_size=(flags.input_size[0]-184, flags.input_size[1]-184),
                                                                    make_map=False)
                        # evaluate
                        truth_label_img = scipy.misc.imread(os.path.join(flags.rsr_data_dir, label_name))
                        iou = sis_utils.iou_metric(truth_label_img, pred_label_img * 255)

                        '''ax1 = plt.subplot(121)
                        ax1.imshow(truth_label_img)
                        ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
                        ax2.imshow(pred_label_img)
                        plt.show()'''

                        iou_record[image_name] = iou
                        print('{}_{}: iou={:.2f}'.format(city_name, tile_id, iou*100))
        finally:
            coord.request_stop()
            coord.join(threads)

    duration = time.time() - start_time
    print('duration {:.2f} minutes'.format(duration/60))
    np.save(os.path.join(save_dir, '{}.npy').format(model.model_name), iou_record)

    iou_mean = []
    for _, val in iou_record.items():
        iou_mean.append(val)
    print(np.mean(iou_mean))
    return np.mean(iou_mean)


if __name__ == '__main__':
    flags = read_flag()
    img_dir, task_dir = sis_utils.get_task_img_folder()
    iou_record = []

    model_names = [#'UnetInria_fr_mean_reduced_appendix_EP-100_DS-60.0_LR-0.0001',
                   #'UnetInria_fr_mean_reduced_appendix_large_EP-100_DS-60.0_LR-0.0001',
                   #'UnetInria_fr_mean_reduced_EP-100_DS-40.0_LR-0.001',
                   #'UnetInria_fr_mean_reduced_EP-100_DS-60.0_LR-0.0001',
                   #'UnetInria_fr_mean_reduced_EP-100_DS-60.0_LR-0.0005',
                   'UnetInria_fr_mean_reduced_EP-100_DS-60_LR-0.001',
                   'UnetInria_fr_mean_reduced_large_EP-100_DS-60.0_LR-0.0001'
                  ]

    for model_name in model_names:
        tf.reset_default_graph()
        iou_mean = test(flags, model_name, task_dir)

        '''file_name = 'UnetInria_fr_mean_reduced_EP-100_DS-60.0_LR-0.0001.npy'
        iou_mean = []
        iou = dict(np.load(file_name).tolist())
        for _, val in iou.items():
            iou_mean.append(val)
        print(np.mean(iou_mean))'''

        iou_record.append(iou_mean)
    print(iou_record)
