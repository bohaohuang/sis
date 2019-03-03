import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import sis_utils
import ersa_utils
import uabDataReader
import uabCrossValMaker
import uab_collectionFunctions
from nn import nn_utils
from bohaoCustom import uabMakeNetwork_UNet


class bayes_update:
    def __init__(self):
        self.m = 0
        self.mean = 0
        self.var = 0

    def update(self, d):
        n = d.shape[0]
        mu_n = np.mean(d)
        sig_n = np.var(d)
        factor_m = self.m / (self.m + n)
        factor_n = 1 - factor_m

        mean_update = factor_m * self.mean + factor_n * mu_n
        self.var = factor_m * (self.var + self.mean ** 2) + factor_n * (sig_n + mu_n ** 2) - mean_update ** 2
        self.mean = mean_update

        self.m += n

        return np.array([self.mean, self.var])


class UnetModelCrop(uabMakeNetwork_UNet.UnetModelCrop):
    def conv_conv_pool(self, input_, n_filters, training, name, kernal_size=(3, 3),
                       conv_stride=(1, 1), pool=True, pool_size=(2, 2), pool_stride=(2, 2),
                       activation=tf.nn.relu, padding='same', bn=True, dropout=None):
        net = input_
        activations = []

        with tf.variable_scope('layer{}'.format(name)):
            for i, F in enumerate(n_filters):
                net = tf.layers.conv2d(net, F, kernal_size, activation=None, strides=conv_stride,
                                       padding=padding, name='conv_{}'.format(i + 1))
                #activations.append(net)
                if bn:
                    net = tf.layers.batch_normalization(net, training=training, name='bn_{}'.format(i+1))
                activations.append(net)
                net = activation(net, name='relu_{}'.format(name, i + 1))
                if dropout is not None:
                    net = tf.layers.dropout(net, rate=self.dropout_rate, training=training,
                                            name='drop_{}'.format(name, i + 1))

            if pool is False:
                return net, activations

            pool = tf.layers.max_pooling2d(net, pool_size, strides=pool_stride, name='pool_{}'.format(name))
            return net, pool, activations

    def create_graph(self, x_name, class_num, start_filter_num=32):
        self.class_num = class_num
        self.activations = []
        sfn = self.sfn

        # downsample
        conv1, pool1, acts = self.conv_conv_pool(self.inputs[x_name], [sfn, sfn], self.trainable, name='conv1',
                                                 padding='valid', dropout=self.dropout_rate)
        self.activations.extend(acts)
        conv2, pool2, acts = self.conv_conv_pool(pool1, [sfn*2, sfn*2], self.trainable, name='conv2',
                                                 padding='valid', dropout=self.dropout_rate)
        self.activations.extend(acts)
        conv3, pool3, acts = self.conv_conv_pool(pool2, [sfn*4, sfn*4], self.trainable, name='conv3',
                                                 padding='valid', dropout=self.dropout_rate)
        self.activations.extend(acts)
        conv4, pool4, acts = self.conv_conv_pool(pool3, [sfn*8, sfn*8], self.trainable, name='conv4',
                                                 padding='valid', dropout=self.dropout_rate)
        self.activations.extend(acts)
        self.encoding, acts = self.conv_conv_pool(pool4, [sfn*16, sfn*16], self.trainable, name='conv5', pool=False,
                                                  padding='valid', dropout=self.dropout_rate)
        self.activations.extend(acts)

        # upsample
        up6 = self.crop_upsample_concat(self.encoding, conv4, 8, name='6')
        conv6, acts = self.conv_conv_pool(up6, [sfn*8, sfn*8], self.trainable, name='up6', pool=False,
                                          padding='valid', dropout=self.dropout_rate)
        self.activations.extend(acts)
        up7 = self.crop_upsample_concat(conv6, conv3, 32, name='7')
        conv7, acts = self.conv_conv_pool(up7, [sfn*4, sfn*4], self.trainable, name='up7', pool=False,
                                          padding='valid', dropout=self.dropout_rate)
        self.activations.extend(acts)
        up8 = self.crop_upsample_concat(conv7, conv2, 80, name='8')
        conv8, acts = self.conv_conv_pool(up8, [sfn*2, sfn*2], self.trainable, name='up8', pool=False,
                                          padding='valid', dropout=self.dropout_rate)
        self.activations.extend(acts)
        up9 = self.crop_upsample_concat(conv8, conv1, 176, name='9')
        conv9, acts = self.conv_conv_pool(up9, [sfn, sfn], self.trainable, name='up9', pool=False,
                                          padding='valid', dropout=self.dropout_rate)
        self.activations.extend(acts)

        self.pred = tf.layers.conv2d(conv9, class_num, (1, 1), name='final', activation=None, padding='same')
        self.activations.extend([self.pred])
        self.output = tf.nn.softmax(self.pred)

    def save_activations(self, rgb_list, gt_list, rgb_dir, img_mean, gpu, pretrained_model_dir, path_to_save,
                         input_size, tile_size, batch_size, load_epoch_num):
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        activation_dict = dict()
        for file_name, file_name_truth in zip(rgb_list, gt_list):
            tile_name = file_name_truth.split('_')[0]
            print('Evaluating {} ... '.format(tile_name))
            # prepare the reader
            reader = uabDataReader.ImageLabelReader(gtInds=[0],
                                                    dataInds=[0],
                                                    nChannels=3,
                                                    parentDir=rgb_dir,
                                                    chipFiles=[file_name],
                                                    chip_size=input_size,
                                                    tile_size=tile_size,
                                                    batchSize=batch_size,
                                                    block_mean=img_mean,
                                                    overlap=self.get_overlap(),
                                                    padding=np.array((self.get_overlap()/2, self.get_overlap()/2)),
                                                    isTrain=False)
            rManager = reader.readManager
            total_len = np.ceil((tile_size[0] + self.get_overlap()) / (input_size[0] - self.get_overlap())) * \
                        np.ceil((tile_size[1] + self.get_overlap()) / (input_size[1] - self.get_overlap()))
            if self.config is None:
                self.config = tf.ConfigProto(allow_soft_placement=True)
            with tf.Session(config=self.config) as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                self.load(pretrained_model_dir, sess, epoch=load_epoch_num, best_model=False)
                for X_batch in tqdm(rManager, total=total_len):
                    for layer_id in range(len(self.activations)):
                        layer_val = sess.run(self.activations[layer_id], feed_dict={self.inputs['X']: X_batch,
                                                                                    self.trainable: False})
                        for chan_id in range(layer_val.shape[-1]):
                            f_i_t = layer_val[:, :, :, chan_id].flatten()
                            act_name = 'f_{}_{}'.format(layer_id, chan_id)
                            if act_name not in activation_dict:
                                activation_dict[act_name] = bayes_update()
                            activation_dict[act_name].update(f_i_t)

        save_name = os.path.join(path_to_save, 'activation_list.pkl')
        ersa_utils.save_file(save_name, activation_dict)


if __name__ == '__main__':
    # settings
    gpu = 1
    batch_size = 1
    input_size = [572, 572]
    city_name = 'DC'
    nn_utils.tf_warn_level(3)

    if city_name == 'Atlanta':
        tile_size = [2000, 3000]
    elif city_name == 'Austin':
        tile_size = [2000, 3000]
    elif city_name == 'DC':
        tile_size = [2500, 2500]

    img_dir, task_dir = sis_utils.get_task_img_folder()

    path_to_save = os.path.join(task_dir, 'dtda_new', city_name, 'valid')
    ersa_utils.make_dir_if_not_exist(path_to_save)

    model_dir = r'/hdd6/Models/UNET_rand_gird/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_' \
                r'EP100_LR0.0001_DS60_DR0.1_SFN32'

    tf.reset_default_graph()

    blCol = uab_collectionFunctions.uabCollection(city_name)
    blCol.readMetadata()
    file_list, parent_dir = blCol.getAllTileByDirAndExt([0, 1, 2])
    file_list_truth, parent_dir_truth = blCol.getAllTileByDirAndExt(3)
    img_mean = blCol.getChannelMeans([0, 1, 2])

    # make the model
    # define place holder
    X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')
    model = UnetModelCrop({'X': X, 'Y': y}, trainable=mode, input_size=input_size, batch_size=5)
    # create graph
    model.create_graph('X', class_num=2)

    # evaluate on tiles
    model.save_activations(file_list, file_list_truth, parent_dir, img_mean, gpu, model_dir,
                           path_to_save, input_size, tile_size, batch_size, load_epoch_num=95)

    #########################################################################################################

    path_to_save = os.path.join(task_dir, 'dtda_new', city_name, 'train')
    ersa_utils.make_dir_if_not_exist(path_to_save)

    model_dir = r'/hdd6/Models/UNET_rand_gird/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_' \
                r'EP100_LR0.0001_DS60_DR0.1_SFN32'

    tf.reset_default_graph()
    # use first 5 tiles for validation
    tile_size = [5000, 5000]
    blCol = uab_collectionFunctions.uabCollection('inria')
    blCol.readMetadata()
    file_list, parent_dir = blCol.getAllTileByDirAndExt([0, 1, 2])
    file_list_truth, parent_dir_truth = blCol.getAllTileByDirAndExt(4)
    idx, file_list = uabCrossValMaker.uabUtilGetFolds(None, file_list, 'force_tile')
    idx_truth, file_list_truth = uabCrossValMaker.uabUtilGetFolds(None, file_list_truth, 'force_tile')
    file_list_valid = uabCrossValMaker.make_file_list_by_key(
        idx, file_list, [i for i in range(0, 6)],
        filter_list=['bellingham', 'bloomington', 'sfo', 'tyrol-e', 'innsbruck'])
    file_list_valid_truth = uabCrossValMaker.make_file_list_by_key(
        idx_truth, file_list_truth, [i for i in range(0, 6)],
        filter_list=['bellingham', 'bloomington', 'sfo', 'tyrol-e', 'innsbruck'])
    img_mean = blCol.getChannelMeans([0, 1, 2])

    # make the model
    # define place holder
    X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')
    model = UnetModelCrop({'X': X, 'Y': y}, trainable=mode, input_size=input_size, batch_size=5)
    # create graph
    model.create_graph('X', class_num=2)

    # evaluate on tiles
    model.save_activations(file_list_valid, file_list_valid_truth, parent_dir, img_mean, gpu, model_dir,
                           path_to_save, input_size, tile_size, batch_size, load_epoch_num=95)
