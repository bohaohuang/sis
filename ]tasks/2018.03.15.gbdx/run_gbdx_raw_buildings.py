import os
import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import sis_utils
import util_functions
import uabDataReader
import uabUtilreader
from bohaoCustom import uabMakeNetwork_UNet

# settings
gpu = 0
batch_size = 5
input_size = [572, 572]
tile_size = [2541, 2541]
util_functions.tf_warn_level(3)
img_dir, task_dir = sis_utils.get_task_img_folder()

for tile_cnt in range(50):

    tf.reset_default_graph()

    model_dir = r'/hdd6/Models/UNET_rand_gird/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'
    gt_dir = r'/media/ei-edl01/data/uab_datasets/sp/DATA_BUILDING_AND_PANEL'
    # compute img mean
    data_path = r'/media/ei-edl01/user/as667/cropped_brighten'
    imgs = glob(os.path.join(data_path, '*.tif'))[:50]

    '''img_mean = [[], [], []]
    for img_file in tqdm(imgs):
        img_rbg = imageio.imread(img_file)
        img_mean[0].extend(img_rbg[:, :, 0].flatten())
        img_mean[1].extend(img_rbg[:, :, 1].flatten())
        img_mean[2].extend(img_rbg[:, :, 2].flatten())
    img_mean[0] = np.mean(img_mean[0])
    img_mean[1] = np.mean(img_mean[1])
    img_mean[2] = np.mean(img_mean[2])
    print(img_mean)'''
    img_mean = [65.44005765, 77.18910836, 59.67908051]

    # make the model
    # define place holder
    X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')
    model = uabMakeNetwork_UNet.UnetModelCrop({'X':X, 'Y':y},
                                              trainable=mode,
                                              input_size=input_size,
                                              batch_size=5, start_filter_num=32)
    # create graph
    model.create_graph('X', class_num=2)

    # load data
    large_tile = [os.path.basename(imgs[tile_cnt])]
    reader = uabDataReader.ImageLabelReader(gtInds=[0],
                                            dataInds=[0],
                                            nChannels=3,
                                            parentDir=data_path,
                                            chipFiles=[large_tile],
                                            chip_size=input_size,
                                            tile_size=tile_size,
                                            batchSize=batch_size,
                                            block_mean=img_mean,
                                            overlap=model.get_overlap(),
                                            padding=np.array((model.get_overlap()/2, model.get_overlap()/2)),
                                            isTrain=False)
    test_reader = reader.readManager

    # run algo
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        model.load(model_dir, sess)
        result1 = model.test('X', sess, test_reader)
    image_pred = uabUtilreader.un_patchify_shrink(result1,
                                                  [tile_size[0] + model.get_overlap(), tile_size[1] + model.get_overlap()],
                                                  tile_size, input_size,
                                                  [input_size[0] - model.get_overlap(), input_size[1] - model.get_overlap()],
                                                  overlap=model.get_overlap())
    pred = util_functions.get_pred_labels(image_pred) * 255

    # view result
    lt = imageio.imread(os.path.join(data_path, large_tile[0]))
    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(121)
    plt.axis('off')
    plt.imshow(lt)
    ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
    plt.imshow(pred)
    plt.axis('off')
    plt.suptitle(os.path.basename(imgs[tile_cnt]))
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'gbdx_raw_brighten',
                             '{}_result_cm_building.png'.format(os.path.basename(imgs[tile_cnt])[:-5])))
    plt.close(fig)
    # plt.show()
