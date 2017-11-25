import os
import re
import time
import shutil
import argparse
import numpy as np
import tensorflow as tf
import scipy.misc
import matplotlib.pyplot as plt
import utils
from network import unet
from dataReader import image_reader, patch_extractor
from rsrClassData import rsrClassData

TEST_DATA_DIR = 'dcc_inria_valid'
CITY_NAME = 'chicago'
RSR_DATA_DIR = r'/media/ei-edl01/data/remote_sensing_data'
PATCH_DIR = r'/media/ei-edl01/user/bh163/data/iai'
TEST_PATCH_APPENDIX = 'valid_noaug_dcc'
TEST_TILE_NAMES = ','.join(['{}'.format(i) for i in range(1, 6)])
RANDOM_SEED = 1234
BATCH_SIZE = 1
INPUT_SIZE = 572
CKDIR = r'/home/lab/Documents/bohao/code/sis/test/models'
MODEL_NAME = 'UnetInria_Origin_no_aug'
NUM_CLASS = 2
GPU = '1'


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


def test_real(flags, patch_size, test_city):
    # set gpu
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = flags.GPU
    tf.reset_default_graph()
    # environment settings
    np.random.seed(flags.random_seed)
    tf.set_random_seed(flags.random_seed)

    # data prepare step
    Data = rsrClassData(flags.rsr_data_dir)
    (collect_files_test, meta_test) = Data.getCollectionByName(flags.test_data_dir)

    # image reader
    coord = tf.train.Coordinator()

    # define place holder
    X = tf.placeholder(tf.float32, shape=[None, patch_size[0], patch_size[1], 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, patch_size[0], patch_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')

    # initialize model
    model = unet.UnetModel_Origin({'X':X, 'Y':y}, trainable=mode, model_name=flags.model_name, input_size=patch_size)
    model.create_graph('X', flags.num_classes)
    model.make_update_ops('X', 'Y')
    # set ckdir
    model.make_ckdir(flags.ckdir)
    # set up graph and initialize
    config = tf.ConfigProto()

    result_dict = {}

    # run training
    start_time = time.time()
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)

        if os.path.exists(model.ckdir) and tf.train.get_checkpoint_state(model.ckdir):
            latest_check_point = tf.train.latest_checkpoint(model.ckdir)
            saver.restore(sess, latest_check_point)
            print('loaded model {}'.format(latest_check_point))

        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        try:
            for (image_name, label_name) in collect_files_test:
                if test_city in image_name:
                    city_name = re.findall('[a-z\-]*(?=[0-9]+\.)', image_name)[0]
                    tile_id = re.findall('[0-9]+(?=\.tif)', image_name)[0]

                    # load reader
                    iterator_test = image_reader.image_label_iterator(
                        os.path.join(flags.rsr_data_dir, image_name),
                        batch_size=flags.batch_size,
                        tile_dim=meta_test['dim_image'][:2],
                        patch_size=patch_size,
                        overlap=184, padding=92)
                    # run
                    result = model.test('X', sess, iterator_test)

                    pred_label_img = utils.get_output_label(result,
                                                            (meta_test['dim_image'][0]+184, meta_test['dim_image'][1]+184),
                                                            patch_size,
                                                            meta_test['colormap'], overlap=184,
                                                            output_image_dim=meta_test['dim_image'],
                                                            output_patch_size=(patch_size[0]-184, patch_size[1]-184))
                    # evaluate
                    truth_label_img = scipy.misc.imread(os.path.join(flags.rsr_data_dir, label_name))
                    iou = utils.iou_metric(truth_label_img, pred_label_img)

                    print('{}_{}: iou={:.2f}'.format(city_name, tile_id, iou*100))

                    result_dict['{}{}'.format(city_name, tile_id)] = iou
        finally:
            coord.request_stop()
            coord.join(threads)

    duration = time.time() - start_time
    print('duration {:.2f} minutes'.format(duration/60))
    return result_dict


def test_fake(flags, patch_size, test_city):
    result = utils.test_unet(flags.rsr_data_dir,
                             flags.test_data_dir,
                             patch_size,
                             'UNET_PS-224__BS-10__E-100__NT-8000__DS-60__CT-__no_random',
                             flags.num_classes,
                             r'/home/lab/Documents/bohao/code/sis/test/models/GridExp',
                             test_city,
                             flags.batch_size)
    return result


def test_fake_across_city(flags, MODEL_NAME, patch_size):
    # set gpu
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = flags.GPU
    # environment settings
    np.random.seed(flags.random_seed)
    tf.set_random_seed(flags.random_seed)

    result_dict = {}

    for city in ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']:
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
            X = tf.placeholder(tf.float32, shape=[None, patch_size[0], patch_size[1], 3], name='X')
            y = tf.placeholder(tf.int32, shape=[None, patch_size[0], patch_size[1], 1], name='y')
            mode = tf.placeholder(tf.bool, name='mode')

            # initialize model
            model = unet.UnetModel({'X':X, 'Y':y}, trainable=mode, model_name=m_name, input_size=patch_size)
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
                for (image_name, label_name) in collect_files_test:
                    if flags.city_name in image_name:
                        city_name = re.findall('[a-z\-]*(?=[0-9]+\.)', image_name)[0]
                        tile_id = re.findall('[0-9]+(?=\.tif)', image_name)[0]
                        print('Evaluating {}_{} at patch size: {}'.format(city_name, tile_id, patch_size))

                        # load reader
                        iterator_test = image_reader.image_label_iterator(
                            os.path.join(flags.rsr_data_dir, image_name),
                            batch_size=flags.batch_size,
                            tile_dim=meta_test['dim_image'][:2],
                            patch_size=patch_size,
                            overlap=184, padding=92)
                        # run
                        result = model.test('X', sess, iterator_test)
                        raw_pred = patch_extractor.un_patchify(result, meta_test['dim_image'], patch_size)

                        file_name = '{}_{}_{}.npy'.format(m_name, city_name, tile_id)
                        np.save(os.path.join('./temp', file_name), raw_pred)
                coord.request_stop()
                coord.join(threads)

            duration = time.time() - start_time
            print('duration {:.2f} minutes'.format(duration/60))

        for tile_num in range(5):
            output = np.zeros((5000, 5000, 2))
            for m_name in MODEL_NAME:
                raw_pred = np.load(os.path.join('./temp', '{}_{}_{}.npy'.format(m_name, city, tile_num+1)))
                output += raw_pred

            # combine results
            pred_label_img = utils.get_pred_labels(output)
            output_pred = utils.make_output_file(pred_label_img, meta_test['colormap'])
            # evaluate
            truth_label_img = scipy.misc.imread(os.path.join(flags.rsr_data_dir, 'inria', 'truth', '{}{}.tif'.format(city, tile_num+1)))
            iou = utils.iou_metric(truth_label_img, output_pred)
            print('{}_{}: iou={:.2f}'.format(city, tile_id, iou * 100))

            result_dict['{}{}'.format(city, tile_id)] = iou

        shutil.rmtree('./temp')
    return result_dict


def test_real_across_city(flags, MODEL_NAME, patch_size):
    # set gpu
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = flags.GPU
    tf.reset_default_graph()
    # environment settings
    np.random.seed(flags.random_seed)
    tf.set_random_seed(flags.random_seed)

    result_dict = {}

    for city in ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']:
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
            X = tf.placeholder(tf.float32, shape=[None, patch_size[0], patch_size[1], 3], name='X')
            y = tf.placeholder(tf.int32, shape=[None, patch_size[0], patch_size[1], 1], name='y')
            mode = tf.placeholder(tf.bool, name='mode')

            # initialize model
            model = unet.UnetModel_Origin({'X':X, 'Y':y}, trainable=mode, model_name=m_name, input_size=patch_size)
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
                for (image_name, label_name) in collect_files_test:
                    if flags.city_name in image_name:
                        city_name = re.findall('[a-z\-]*(?=[0-9]+\.)', image_name)[0]
                        tile_id = re.findall('[0-9]+(?=\.tif)', image_name)[0]
                        print('Evaluating {}_{} at patch size: {}'.format(city_name, tile_id, patch_size))

                        # load reader
                        iterator_test = image_reader.image_label_iterator(
                            os.path.join(flags.rsr_data_dir, image_name),
                            batch_size=flags.batch_size,
                            tile_dim=meta_test['dim_image'][:2],
                            patch_size=patch_size,
                            overlap=184, padding=92)
                        # run
                        result = model.test('X', sess, iterator_test)
                        raw_pred = patch_extractor.un_patchify_shrink(result,
                                                                        (meta_test['dim_image'][0] + 184,
                                                                        meta_test['dim_image'][1] + 184),
                                                                        tile_dim_output=meta_test['dim_image'],
                                                                        patch_size= patch_size,
                                                                        patch_size_output= (patch_size[0] - 184, patch_size[1] - 184),
                                                                        overlap=184)
                        file_name = '{}_{}_{}.npy'.format(m_name.split('/')[-1], city_name, tile_id)
                        np.save(os.path.join('./temp', file_name), raw_pred)
                coord.request_stop()
                coord.join(threads)

            duration = time.time() - start_time
            print('duration {:.2f} minutes'.format(duration/60))

        for tile_num in range(5):
            output = np.zeros((5000, 5000, 2))
            for m_name in MODEL_NAME:
                raw_pred = np.load(os.path.join('./temp', '{}_{}_{}.npy'.format(m_name.split('/')[-1], city, tile_num+1)))
                output += raw_pred

            # combine results
            pred_label_img = utils.get_pred_labels(output)
            output_pred = utils.make_output_file(pred_label_img, meta_test['colormap'])
            # evaluate
            truth_label_img = scipy.misc.imread(os.path.join(flags.rsr_data_dir, 'inria', 'truth', '{}{}.tif'.format(city, tile_num+1)))
            iou = utils.iou_metric(truth_label_img, output_pred)
            print('{}_{}: iou={:.2f}'.format(city, tile_id, iou * 100))

            result_dict['{}{}'.format(city, tile_id)] = iou

        shutil.rmtree('./temp')
    return result_dict


if __name__ == '__main__':
    flags = read_flag()
    _, task_dir = utils.get_task_img_folder()

    for city in ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']:
        for patch_size in [572, 2636]:
            file_name = '{}_{}_{}.npy'.format('UnetInria_Origin_fr', city, patch_size)
            if not os.path.exists(os.path.join(task_dir, file_name)):
                result_dict = test_real(flags, (patch_size, patch_size), city)
                np.save(os.path.join(task_dir, file_name), result_dict)

    for city in ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']:
        for patch_size in [224, 2800]:
            file_name = '{}_{}_{}.npy'.format(flags.model_name+'_fake', city, patch_size)
            if not os.path.exists(os.path.join(task_dir, file_name)):
                result_dict = test_fake(flags, (patch_size, patch_size), city)
                np.save(os.path.join(task_dir, file_name), result_dict)

    for patch_size in [224, 2800]:
        file_name = '{}_{}_cross.npy'.format(flags.model_name+'_fake', patch_size)
        if not os.path.exists(os.path.join(task_dir, file_name)):
            model_names = ['UNET_austin_no_random', 'UNET_chicago_no_random',
                          'UNET_kitsap_no_random', 'UNET_tyrol-w_no_random',
                          'UNET_vienna_no_random']
            result_dict = test_fake_across_city(flags, model_names, (patch_size, patch_size))
            np.save(os.path.join(task_dir, file_name), result_dict)

    for patch_size in [572, 2636]:
        file_name = '{}_{}_cross.npy'.format(flags.model_name, patch_size)
        if not os.path.exists(os.path.join(task_dir, file_name)):
            model_names = ['UnetInria_Origin_cross_city/UnetInria_austin_Origin_no_aug',
                           'UnetInria_Origin_cross_city/UnetInria_chicago_Origin_no_aug',
                           'UnetInria_Origin_cross_city/UnetInria_kitsap_Origin_no_aug',
                           'UnetInria_Origin_cross_city/UnetInria_tyrol-w_Origin_no_aug',
                           'UnetInria_Origin_cross_city/UnetInria_vienna_Origin_no_aug']
            result_dict = test_real_across_city(flags, model_names, (patch_size, patch_size))
            np.save(os.path.join(task_dir, file_name), result_dict)

    result_mean = np.zeros((2, 4))
    result_std = np.zeros((2, 4))

    # fake, cross region
    iou = []
    for cnt, patch_size in enumerate([224, 2800]):
        for city in ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']:
            file_name = '{}_{}_{}.npy'.format(flags.model_name+'_fake', city, patch_size)
            result = dict(np.load(os.path.join(task_dir, file_name)).tolist())
            for k, v in result.items():
                iou.append(v)
            result_mean[cnt, 0] = np.mean(iou)
            result_std[cnt, 0] = np.std(iou)

    # fake, cross city
    iou = []
    for cnt, patch_size in enumerate([224, 2800]):
        file_name = 'UnetInria_Origin_no_aug_fake_{}_cross.npy'.format(patch_size)
        result = dict(np.load(os.path.join(task_dir, file_name)).tolist())
        for k, v in result.items():
            iou.append(v)
        result_mean[cnt, 1] = np.mean(iou)
        result_std[cnt, 1] = np.std(iou)

    # real, cross region
    iou = []
    for cnt, patch_size in enumerate([572, 2636]):
        for city in ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']:
            file_name = '{}_{}_{}.npy'.format('UnetInria_Origin_fr', city, patch_size)
            result = dict(np.load(os.path.join(task_dir, file_name)).tolist())
            for k, v in result.items():
                iou.append(v)
            result_mean[cnt, 2] = np.mean(iou)
            result_std[cnt, 2] = np.std(iou)

    # real, cross city
    iou = []
    for cnt, patch_size in enumerate([572, 2636]):
        file_name = 'UnetInria_Origin_no_aug_{}_cross.npy'.format(patch_size)
        result = dict(np.load(os.path.join(task_dir, file_name)).tolist())
        for k, v in result.items():
            iou.append(v)
        result_mean[cnt, 3] = np.mean(iou)
        result_std[cnt, 3] = np.std(iou)

    _, N = result_mean.shape
    ind = np.arange(N)
    width = 0.35
    ax = plt.subplot(111)
    rect1 = ax.bar(ind, result_mean[0, :], width, color='g')
    rect2 = ax.bar(ind + width, result_mean[1, :], width, color='y')
    plt.xticks(ind+width/2, ('Common\nX Region', 'Common\nX City', 'Autentic\nX Region', 'Autentic\nX City'))
    plt.ylabel('mean IoU')
    plt.ylim(ymin=0.6, ymax=0.8)
    plt.title('U-Net Model Comparison')
    ax.legend((rect1[0], rect2[0]), ('PatchSize: 224/572', 'PatchSize: 2800/2636'))
    utils.barplot_autolabel(ax, rect1, margin=0.01)
    utils.barplot_autolabel(ax, rect2, margin=0.01)
    plt.show()

    print(result_mean)
    print(result_std)
