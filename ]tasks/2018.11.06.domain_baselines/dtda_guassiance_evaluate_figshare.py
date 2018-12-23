import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import utils
import ersa_utils
import uabDataReader
import uabCrossValMaker
import uab_collectionFunctions
from nn import nn_utils
from bohaoCustom import uabMakeNetwork_UNet


class UnetModelCrop(uabMakeNetwork_UNet.UnetModelCrop):
    @staticmethod
    def get_shifts(shift_dict, layer_ids):
        shift_list = []
        for l in layer_ids:
            for act_name, shift_val in shift_dict.items():
                if 'f_{}_'.format(l) in act_name:
                    shift_list.append(shift_val)
        return shift_list

    def conv_conv_pool(self, input_, n_filters, shift, training, name, kernal_size=(3, 3),
                       conv_stride=(1, 1), pool=True, pool_size=(2, 2), pool_stride=(2, 2),
                       activation=tf.nn.relu, padding='same', bn=True, dropout=None):
        net = input_
        activations = []
        layer_cnt = 0

        with tf.variable_scope('layer{}'.format(name)):
            for i, F in enumerate(n_filters):
                net = tf.layers.conv2d(net, F, kernal_size, activation=None, strides=conv_stride,
                                       padding=padding, name='conv_{}'.format(i + 1))
                if bn:
                    net = tf.layers.batch_normalization(net, training=training, name='bn_{}'.format(i+1))
                output_list = []
                for n_cnt in range(net.shape[-1]):
                    output_list.append(
                        #shift[layer_cnt][0] * (net[:, :, :, n_cnt] - shift[layer_cnt][1]) + shift[layer_cnt][2])
                        1 * (net[:, :, :, n_cnt] - 0) + 0)
                    layer_cnt += 1
                net = tf.stack(output_list, axis=-1)
                activations.append(net)
                net = activation(net, name='relu_{}'.format(name, i + 1))
                if dropout is not None:
                    net = tf.layers.dropout(net, rate=self.dropout_rate, training=training,
                                            name='drop_{}'.format(name, i + 1))

            if pool is False:
                return net, activations

            pool = tf.layers.max_pooling2d(net, pool_size, strides=pool_stride, name='pool_{}'.format(name))
            return net, pool, activations

    def create_graph(self, x_name, shift_dict, class_num, start_filter_num=32):
        self.class_num = class_num
        self.activations = []
        sfn = self.sfn

        # downsample
        shift = self.get_shifts(shift_dict, [0, 1])
        conv1, pool1, acts = self.conv_conv_pool(self.inputs[x_name], [sfn, sfn], shift, self.trainable, name='conv1',
                                                 padding='valid', dropout=self.dropout_rate)
        self.activations.extend(acts)
        shift = self.get_shifts(shift_dict, [2, 3])
        conv2, pool2, acts = self.conv_conv_pool(pool1, [sfn*2, sfn*2], shift, self.trainable, name='conv2',
                                                 padding='valid', dropout=self.dropout_rate)
        self.activations.extend(acts)
        shift = self.get_shifts(shift_dict, [4, 5])
        conv3, pool3, acts = self.conv_conv_pool(pool2, [sfn*4, sfn*4], shift, self.trainable, name='conv3',
                                                 padding='valid', dropout=self.dropout_rate)
        self.activations.extend(acts)
        shift = self.get_shifts(shift_dict, [6, 7])
        conv4, pool4, acts = self.conv_conv_pool(pool3, [sfn*8, sfn*8], shift, self.trainable, name='conv4',
                                                 padding='valid', dropout=self.dropout_rate)
        self.activations.extend(acts)
        shift = self.get_shifts(shift_dict, [8, 9])
        self.encoding, acts = self.conv_conv_pool(pool4, [sfn*16, sfn*16], shift, self.trainable, name='conv5', pool=False,
                                                  padding='valid', dropout=self.dropout_rate)
        self.activations.extend(acts)

        # upsample
        up6 = self.crop_upsample_concat(self.encoding, conv4, 8, name='6')
        shift = self.get_shifts(shift_dict, [10, 11])
        conv6, acts = self.conv_conv_pool(up6, [sfn*8, sfn*8], shift, self.trainable, name='up6', pool=False,
                                          padding='valid', dropout=self.dropout_rate)
        self.activations.extend(acts)
        up7 = self.crop_upsample_concat(conv6, conv3, 32, name='7')
        shift = self.get_shifts(shift_dict, [12, 13])
        conv7, acts = self.conv_conv_pool(up7, [sfn*4, sfn*4], shift, self.trainable, name='up7', pool=False,
                                          padding='valid', dropout=self.dropout_rate)
        self.activations.extend(acts)
        shift = self.get_shifts(shift_dict, [14, 15])
        up8 = self.crop_upsample_concat(conv7, conv2, 80, name='8')
        conv8, acts = self.conv_conv_pool(up8, [sfn*2, sfn*2], shift, self.trainable, name='up8', pool=False,
                                          padding='valid', dropout=self.dropout_rate)
        self.activations.extend(acts)
        shift = self.get_shifts(shift_dict, [16, 17])
        up9 = self.crop_upsample_concat(conv8, conv1, 176, name='9')
        conv9, acts = self.conv_conv_pool(up9, [sfn, sfn], shift, self.trainable, name='up9', pool=False,
                                          padding='valid', dropout=self.dropout_rate)
        self.activations.extend(acts)

        shift = self.get_shifts(shift_dict, [18])
        self.pred = tf.layers.conv2d(conv9, class_num, (1, 1), name='final', activation=None, padding='same')
        output_list = []
        for n_cnt in range(self.pred.shape[-1]):
            output_list.append(shift[n_cnt][0] * (self.pred[:, :, :, n_cnt] - shift[n_cnt][1]) + shift[n_cnt][2])
        self.pred = tf.stack(output_list, axis=-1)
        self.output = tf.nn.softmax(self.pred)


if __name__ == '__main__':
    # settings
    gpu = 1
    batch_size = 1
    input_size = [572, 572]
    city_name = 'DC'
    nn_utils.tf_warn_level(3)

    if city_name == 'Atlanta':
        tile_size = [2000, 3000]
    elif city_name == 'DC':
        tile_size = [2500, 2500]
    else:
        tile_size = [2000, 3000]

    img_dir, task_dir = utils.get_task_img_folder()

    path_to_save = os.path.join(task_dir, 'dtda_new', city_name, 'shift_dict.pkl')
    shift_dict = ersa_utils.load_file(path_to_save)

    for k, v in shift_dict.items():
        print(k, v)

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
    Z = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='Z')
    y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')
    model = UnetModelCrop({'X': X, 'Y': y}, trainable=mode, input_size=input_size, batch_size=5)
    #model = uabMakeNetwork_UNet.UnetModelDTDA({'X': X, 'Z': Z, 'Y': y}, trainable=mode, input_size=input_size, batch_size=5)
    # create graph
    model.create_graph('X', shift_dict, class_num=2)
    #model.create_graph('X', 'Z', class_num=2)

    # evaluate on tiles
    #model.load_source_weights(model_dir, shift_dict, gpu=gpu)
    model.evaluate(file_list, file_list_truth, parent_dir, parent_dir_truth,
                   input_size, tile_size, batch_size, img_mean, model_dir, gpu,
                   save_result_parent_dir='domain_baseline_new', ds_name=city_name, best_model=False,
                   load_epoch_num=95, show_figure=True)
