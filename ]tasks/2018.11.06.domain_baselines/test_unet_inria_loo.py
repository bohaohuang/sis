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
                        shift[layer_cnt][0] * (net[:, :, :, n_cnt] - shift[layer_cnt][1]) + shift[layer_cnt][2])
                        #1 * (net[:, :, :, n_cnt] - 0) + 0)
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
    tile_size = [5000, 5000]
    city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
    nn_utils.tf_warn_level(3)

    img_dir, task_dir = utils.get_task_img_folder()

    for city_id in [0]:
        path_to_save = os.path.join(task_dir, 'dtda', city_list[city_id], 'shift_dict2.pkl')
        shift_dict = ersa_utils.load_file(path_to_save)

        model_dir = r'/hdd6//Models/Inria_Domain_LOO/UnetCrop_inria_aug_leave_{}_0_PS(572, 572)_BS5_' \
                    r'EP100_LR0.0001_DS60_DR0.1_SFN32'.format(city_id)

        tf.reset_default_graph()

        blCol = uab_collectionFunctions.uabCollection('inria')
        blCol.readMetadata()
        file_list, parent_dir = blCol.getAllTileByDirAndExt([0, 1, 2])
        file_list_truth, parent_dir_truth = blCol.getAllTileByDirAndExt(4)
        idx, file_list = uabCrossValMaker.uabUtilGetFolds(None, file_list, 'force_tile')
        idx_truth, file_list_truth = uabCrossValMaker.uabUtilGetFolds(None, file_list_truth, 'force_tile')
        # use first 5 tiles for validation
        exclude_cities = [city_list[a] for a in range(5) if a != city_id]
        file_list_valid = uabCrossValMaker.make_file_list_by_key(
            idx, file_list, [i for i in range(0, 6)],
            filter_list=['bellingham', 'bloomington', 'sfo', 'tyrol-e', 'innsbruck'] + exclude_cities)
        file_list_valid_truth = uabCrossValMaker.make_file_list_by_key(
            idx_truth, file_list_truth, [i for i in range(0, 6)],
            filter_list=['bellingham', 'bloomington', 'sfo', 'tyrol-e', 'innsbruck']+ exclude_cities)
        img_mean = blCol.getChannelMeans([0, 1, 2])

        # make the model
        # define place holder
        X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
        y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
        mode = tf.placeholder(tf.bool, name='mode')
        model = UnetModelCrop({'X': X, 'Y': y}, trainable=mode, input_size=input_size, batch_size=5)
        # create graph
        model.create_graph('X', shift_dict, class_num=2)

        # evaluate on tiles
        model.evaluate(file_list_valid, file_list_valid_truth, parent_dir, parent_dir_truth,
                       input_size, tile_size, batch_size, img_mean, model_dir, gpu,
                       save_result_parent_dir='domain_baseline2', ds_name='inria', best_model=False,
                       load_epoch_num=95, show_figure=False)
