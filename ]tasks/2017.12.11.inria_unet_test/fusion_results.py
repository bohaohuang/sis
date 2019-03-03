import os
import re
import time
import argparse
import numpy as np
import tensorflow as tf
import scipy.misc
import sklearn.metrics
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


def test_and_save(flags, model_name, save_dir):
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
    elif 'Res' in model_name:
        model = unet.ResUnetModel_Crop({'X': X, 'Y': y}, trainable=mode, model_name=model_name,
                                      input_size=flags.input_size)
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

    # make fold if not exists
    save_path = os.path.join(save_dir, 'temp_save', model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        return save_path

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
            for (image_name, label_name) in collect_files_test:
                c_names = flags.city_name.split(',')
                for c_name in c_names:
                    if c_name in image_name:
                        city_name = re.findall('[a-z\-]*(?=[0-9]+\.)', image_name)[0]
                        tile_id = re.findall('[0-9]+(?=\.tif)', image_name)[0]

                        print('Scoring {}_{} using {}...'.format(city_name, tile_id, model_name))

                        # load reader
                        iterator_test = image_reader.image_label_iterator(
                            os.path.join(flags.rsr_data_dir, image_name),
                            batch_size=flags.batch_size,
                            tile_dim=meta_test['dim_image'][:2],
                            patch_size=flags.input_size,
                            overlap=184, padding=92,
                            image_mean=IMG_MEAN)
                        # run
                        result = model.test('X', sess, iterator_test, soft_pred=True)

                        pred_label_img = sis_utils.get_output_label(result,
                                                                    (meta_test['dim_image'][0]+184, meta_test['dim_image'][1]+184),
                                                                    flags.input_size,
                                                                    meta_test['colormap'], overlap=184,
                                                                    output_image_dim=meta_test['dim_image'],
                                                                    output_patch_size=(flags.input_size[0]-184, flags.input_size[1]-184),
                                                                    make_map=False, soft_pred=True)
                        file_name = os.path.join(save_path, '{}_{}.npy'.format(city_name, tile_id))
                        np.save(file_name, pred_label_img)
        finally:
            coord.request_stop()
            coord.join(threads)

    duration = time.time() - start_time
    print('duration {:.2f} minutes'.format(duration/60))
    return save_path


def get_fusion_ious(model_dirs):
    model_name = [a.split('/')[-1] for a in model_dirs]
    iou_record = {}
    print('Evaluating using {}...'.format('+'.join(model_name)))

    # data prepare step
    Data = rsrClassData(flags.rsr_data_dir)
    (collect_files_test, meta_test) = Data.getCollectionByName(flags.test_data_dir)
    for (image_name, label_name) in collect_files_test:
        c_names = flags.city_name.split(',')
        for c_name in c_names:
            if c_name in image_name:
                city_name = re.findall('[a-z\-]*(?=[0-9]+\.)', image_name)[0]
                tile_id = re.findall('[0-9]+(?=\.tif)', image_name)[0]

                preds = np.zeros((5000, 5000, 2))
                for dir in model_dirs:
                    preds += np.load(os.path.join(dir, '{}_{}.npy'.format(city_name, tile_id)))
                pred_labels = sis_utils.get_pred_labels(preds)

                # evaluate
                truth_label_img = scipy.misc.imread(os.path.join(flags.rsr_data_dir, label_name))
                iou = sis_utils.iou_metric(truth_label_img, pred_labels * 255)
                iou_record[image_name] = iou
                print('{}_{}: iou={:.2f}'.format(city_name, tile_id, iou * 100))

    if len(model_name) <= 3:
        np.save(os.path.join(task_dir, '{}.npy').format('_'.join(model_name)), iou_record)
    else:
        np.save(os.path.join(task_dir, 'fusion_{}_{}.npy').format(len(model_name),
                                                                  '_'.join([a[:4] for a in model_name])), iou_record)

    iou_mean = []
    for _, val in iou_record.items():
        iou_mean.append(val)
    print(np.mean(iou_mean))
    return np.mean(iou_mean)


def plot_comparison(model_names):
    ious = np.zeros((len(model_names)+1, 25))
    city_names = []
    ylims = np.array([70, 65, 20, 70, 70])

    for cnt, m_name in enumerate(model_names):
        file_name = os.path.join(task_dir, '{}.npy'.format(m_name))
        iou = dict(np.load(file_name).tolist())
        for cnt2, (k, v) in enumerate(iou.items()):
            ious[cnt][cnt2] = v*100

    file_name = os.path.join(task_dir, '{}.npy'.format('_'.join(model_names)))
    iou = dict(np.load(file_name).tolist())
    for cnt2, (k, v) in enumerate(iou.items()):
        city_names.append(k.split('/')[-1].split('.')[0])
        ious[-1][cnt2] = v*100

    # plot the figures
    N = 5
    w = 0.2
    fig = plt.figure(figsize=(14, 12))
    ax = []

    ind = np.arange(N)
    for i in range(5):
        ax.append(plt.subplot(511+i))
        rect = []
        for cnt in range(len(model_names)):
            rect.append(ax[i].bar(ind+w*cnt, ious[cnt][(i*5):((i+1)*5)], w, label=model_names[cnt]))
        rect.append(ax[i].bar(ind+w*len(model_names), ious[-1][(i*5):((i+1)*5)], w, label='Fusion'))
        plt.xticks(ind, city_names[(i*5):((i+1)*5)])
        plt.xlabel('Tile Name')
        plt.ylabel('IoU')
        plt.ylim(ymin=ylims[i])
        for r in rect:
            sis_utils.barplot_autolabel(ax[i], r)
    ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.1), fancybox=True)

    if len(model_names) <= 3:
        file_name = 'fusion_cmp_{}.png'.format('+'.join(model_names))
    else:
        file_name = 'fusion_cmp_{}.png'.format('+'.join([a[:4] for a in model_names]))
    plt.savefig(os.path.join(img_dir, file_name))
    plt.show()


def get_diff_ordered(model_name_0, model_name_1):
    import operator
    file_name_0 = os.path.join(task_dir, '{}.npy'.format(model_name_0))
    file_name_1 = os.path.join(task_dir, '{}.npy'.format(model_name_1))
    iou_0 = dict(np.load(file_name_0).tolist())
    iou_1 = dict(np.load(file_name_1).tolist())
    iou_diff = {}
    for city in iou_0.keys():
        iou_diff[city] = iou_0[city] - iou_1[city]
    sorted_iou_record = sorted(iou_diff.items(), key=operator.itemgetter(1), reverse=True)

    for item in sorted_iou_record:
        print(item)


if __name__ == '__main__':
    flags = read_flag()
    img_dir, task_dir = sis_utils.get_task_img_folder()
    iou_record = []

    model_names = ['UnetInria_fr_mean_reduced_appendix_large_EP-100_DS-60.0_LR-0.0001',
                   'UnetInria_fr_mean_reduced_large_EP-100_DS-60.0_LR-0.0001',
                   'UnetInria_fr_mean_reduced_appendix_EP-100_DS-60.0_LR-0.0001',
                   'UnetInria_fr_mean_reduced_EP-100_DS-60.0_LR-0.0001',
                   'UnetInria_fr_mean_reduced_EP-100_DS-60.0_LR-0.0005',
                   'UnetInria_fr_mean_reduced_EP-100_DS-60_LR-0.001',
                   'ResUnetCrop Inria_fr_resample_mean_reduced_EP-100_DS-60_LR-0.0001'
                  ]

    model_preds = []
    for model_name in model_names:
        tf.reset_default_graph()
        save_path = test_and_save(flags, model_name, task_dir)
        model_preds.append(save_path)


    #get_fusion_ious(model_preds[:2])
    #get_fusion_ious([model_preds[0], model_preds[3]])
    #get_fusion_ious(model_preds[:3])
    #get_fusion_ious([model_preds[0], model_preds[1], model_preds[2], model_preds[-1]])
    #get_fusion_ious(model_preds[:4])
    #get_fusion_ious(model_preds[:6])
    #get_fusion_ious(model_preds)

    plot_comparison([model_names[0], model_names[1], model_names[2], model_names[-1]])
