import os
import csv
import imageio
import numpy as np
import tensorflow as tf
import uabPreprocClasses
import uab_collectionFunctions
import uab_DataHandlerFunctions
import bohaoCustom.uabPreprocClasses as bPreproc
from bohaoCustom import uabMakeNetwork_UNet


def name_list_convert(name_list):
    list_2_return = []
    for name_str in name_list:
        list_2_return.append([i.strip() for i in name_str.split(' ')])
    return list_2_return


def image_reader(name_list, channels, parent_dir, img_mean, divide_num=5, sample_num=200):
    # divide into 5 cities
    city_len = int(len(name_list)/divide_num)
    select_pool = []
    for i in range(divide_num):
        select_pool.append(name_list[i*city_len:(i+1)*city_len])

    # generate samples
    idx = np.random.permutation(city_len)
    for s in range(sample_num):
        for i in range(divide_num):
            img = []
            for c in channels:
                img.append(imageio.imread(os.path.join(parent_dir, select_pool[i][idx[s]][c])))
            img = np.dstack(img) - img_mean
            yield np.expand_dims(img, axis=0)


if __name__ == '__main__':
    # settings
    input_size = [572, 572]
    num_classes = 2
    fname = 'fileList.txt'
    gpu = 1
    model_dir = r'/hdd6/Models/UNET_rand_gird/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'

    # make network
    # define place holder
    X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')
    model = uabMakeNetwork_UNet.UnetModelCrop({'X': X, 'Y': y},
                                              trainable=mode,
                                              input_size=input_size)
    model.create_graph('X', class_num=num_classes)

    # get extracted patch directory
    blCol = uab_collectionFunctions.uabCollection('inria')
    opDetObj = bPreproc.uabOperTileDivide(255)  # inria GT has value 0 and 255, we map it back to 0 and 1
    # [3] is the channel id of GT
    rescObj = uabPreprocClasses.uabPreprocMultChanOp([], 'GT_Divide.tif', 'Map GT to (0, 1)', [3], opDetObj)
    rescObj.run(blCol)
    img_mean = blCol.getChannelMeans([0, 1, 2])  # get mean of rgb info

    # extract patches
    extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4],  # extract all 4 channels
                                                    cSize=input_size,  # patch size as 572*572
                                                    numPixOverlap=int(model.get_overlap() / 2),  # overlap as 92
                                                    extSave=['jpg', 'jpg', 'jpg', 'png'],
                                                    # save rgb files as jpg and gt as png
                                                    isTrain=True,
                                                    gtInd=3,
                                                    pad=model.get_overlap())  # pad around the tiles
    patchDir = extrObj.run(blCol)

    # get patches files
    patch_name_list = os.path.join(patchDir, fname)
    with open(patch_name_list, 'r') as f:
        name_list = f.readlines()

    name_list = name_list_convert(name_list)

    # get reader
    reader = image_reader(name_list, [0, 1, 2], patchDir, img_mean)

    # run the model
    # run algo
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        model.load(model_dir, sess)
        file_name = os.path.join(r'/hdd6/temp', 'encoded_unet.csv')
        with open(file_name, 'w+') as f:
            for X_batch in reader:
                encoding = sess.run(model.encoding, feed_dict={model.inputs['X']: X_batch,
                                                               model.trainable: False})
                encoding = encoding.reshape((-1,)).tolist()
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(['{:3.4e}'.format(x) for x in encoding])
