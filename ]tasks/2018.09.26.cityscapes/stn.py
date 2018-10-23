import os
import tensorflow as tf
from nn import basicNetwork
from nn import nn_utils


class STN(basicNetwork.SegmentationNetwork):
    """
    Implements the style transfer network from
    https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf
    """
    def __init__(self, class_num, input_size, dropout_rate=None, name='stn', suffix='', learn_rate=1e-4,
                 decay_step=60, decay_rate=0.1, epochs=100, batch_size=5, start_filter_num=32, pad=40,
                 stn_func=None, dis_func=None):
        """
        Initialize the object
        :param class_num: class number in labels, determine the # of output units
        :param input_size: input patch size
        :param dropout_rate: dropout rate in each layer, if it is None, no dropout will be used
        :param name: name of this network
        :param suffix: used to create a unique name of the network
        :param learn_rate: start learning rate
        :param decay_step: #steps before the learning rate decay
        :param decay_rate: learning rate will be decayed to lr*decay_rate
        :param epochs: #epochs to train
        :param batch_size: batch size
        :param start_filter_num: #filters at the first layer
        """
        self.sfn = start_filter_num
        self.pad = pad
        if stn_func is None:
            self.stn_func = self.create_stn
        else:
            self.stn_func = stn_func
        if dis_func is None:
            self.dis_func = self.create_dis
        else:
            self.dis_func = dis_func
        self.refine = None
        self.true_logit = None
        self.fake_logit = None
        self.d_loss = None
        self.g_loss = None
        super().__init__(class_num, input_size, dropout_rate, name, suffix, learn_rate, decay_step,
                         decay_rate, epochs, batch_size)

    def create_stn(self, feature):
        with tf.variable_scope('stn'):
            orig = feature
            padding = tf.constant([[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]])
            pred = tf.pad(feature, padding, 'REFLECT', name='reflect_pad')
            pred = nn_utils.conv_conv_pool(pred, [self.sfn], self.mode, 'conv_block_1', (9, 9), (1, 1), pool=False,
                                           activation=tf.nn.relu)
            pred = nn_utils.conv_conv_pool(pred, [self.sfn * 2], self.mode, 'conv_block_2', (3, 3), (2, 2), pool=False,
                                           activation=tf.nn.relu)
            pred = nn_utils.conv_conv_pool(pred, [self.sfn * 4], self.mode, 'conv_block_3', (3, 3), (2, 2), pool=False,
                                           activation=tf.nn.relu)
            for i in range(5):
                pred = self.res_block(pred, self.sfn * 4, str(i))

            pred = self.trans_2d_block(pred, self.sfn * 2, '1')
            pred = self.trans_2d_block(pred, self.sfn, '2')

            refine = nn_utils.conv_conv_pool(tf.concat([pred, orig], axis=-1), [self.class_num], self.mode, '3', (9, 9), pool=False,
                                             activation=tf.nn.sigmoid)
        return refine

    def create_dis(self, refine, reuse):
        with tf.variable_scope('discr', reuse=reuse):
            # downsample
            conv1 = nn_utils.conv_conv_pool(refine, [self.sfn], self.mode, name='conv1', kernel_size=(5, 5),
                                            conv_stride=(2, 2), padding='valid', dropout=self.dropout_rate, pool=False)
            conv2 = nn_utils.conv_conv_pool(conv1, [self.sfn], self.mode, name='conv2', kernel_size=(5, 5),
                                            conv_stride=(2, 2), padding='valid', dropout=self.dropout_rate, pool=False)
            conv3 = nn_utils.conv_conv_pool(conv2, [self.sfn * 2], self.mode, name='conv3', kernel_size=(5, 5),
                                            conv_stride=(2, 2), padding='valid', dropout=self.dropout_rate, pool=False)
            conv4 = nn_utils.conv_conv_pool(conv3, [self.sfn * 2], self.mode, name='conv4', kernel_size=(3, 3),
                                            conv_stride=(2, 2), padding='valid', dropout=self.dropout_rate, pool=False)
            conv5 = nn_utils.conv_conv_pool(conv4, [self.sfn * 4], self.mode, name='conv5', kernel_size=(3, 3),
                                            conv_stride=(2, 2), padding='valid', dropout=self.dropout_rate, pool=False)
            conv6 = nn_utils.conv_conv_pool(conv5, [self.sfn * 4], self.mode, name='conv6', kernel_size=(3, 3),
                                            conv_stride=(2, 2), padding='valid', dropout=self.dropout_rate, pool=False)
            flat = tf.reshape(conv6, shape=[self.bs, 6 * 14 * self.sfn * 4])
            return nn_utils.fc_fc(flat, [100, 1], self.mode, name='fc_final', activation=None, dropout=False)

    def create_graph(self, feature, **kwargs):
        """
        Create graph for the U-Net
        :param feature: input image
        :param start_filter_num: #filters at the start layer, #filters in U-Net grows exponentially
        :return:
        """
        self.refine = self.stn_func(feature)

        self.true_logit = self.create_dis(kwargs['feature_valid'], reuse=False)
        self.fake_logit = self.create_dis(self.refine, reuse=True)

    @staticmethod
    def res_block(input_, n_filter, name):
        res = input_
        conv = input_
        with tf.variable_scope('res{}'.format(name)):
            for i in range(2):
                conv = tf.layers.conv2d(conv, n_filter, (3, 3), name='res{}_conv_{}'.format(name, i))
                conv = tf.layers.batch_normalization(conv, name='res{}_batchnorm_{}'.format(name, i))
                if i < 1:
                    conv = tf.nn.relu(conv, name='res{}_relu_{}'.format(name, i))
            res = res[:, 2:-2, 2:-2, :]
            conv = tf.add(res, conv, name='res{}_add'.format(name))
        return conv

    @staticmethod
    def trans_2d_block(input_, n_filter, name):
        with tf.variable_scope('trans_{}'.format(name)):
            input_ = tf.layers.conv2d_transpose(input_, n_filter, (3, 3), strides=(2, 2), padding='SAME',
                                                name='trans_{}_conv'.format(name))
            input_ = tf.layers.batch_normalization(input_, name='trans_{}_batchnorm'.format(name))
            input_ = tf.nn.relu(input_, name='trans_{}_relu'.format(name))
        return input_

    def make_loss(self, label, loss_type='xent', **kwargs):
        """
        Make loss to optimize for the network
        U-Net's output is smaller than input, thus ground truth need to be cropped
        :param label: input labels, can be generated by tf.data.Dataset
        :param loss_type:
            xent: cross entropy loss
        :return:
        """
        with tf.variable_scope('loss'):
            pred_flat = tf.reshape(self.refine, [-1, self.class_num])
            _, w, h, _ = label.get_shape().as_list()
            y = tf.image.resize_image_with_crop_or_pad(label, w - self.get_overlap(), h - self.get_overlap())
            y_flat = tf.reshape(tf.squeeze(y, axis=[3]), [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(y_flat, self.class_num - 1)), 1)
            gt = tf.gather(y_flat, indices)
            prediction = tf.gather(pred_flat, indices)

            pred = tf.argmax(prediction, axis=-1, output_type=tf.int32)

            self.loss_iou = self.create_resetable_metric(tf.metrics.mean_iou, var_name='loss_iou',
                                                         scope=tf.get_variable_scope().name,
                                                         labels=gt, predictions=pred, num_classes=self.class_num,
                                                         name='loss_iou')

            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt))

            self.loss_xent = self.create_resetable_metric(tf.metrics.mean, var_name='loss_xent',
                                                          scope=tf.get_variable_scope().name,
                                                          values=self.loss, name='loss_xent')

        with tf.variable_scope('adv_loss'):
            d_loss_fake_0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logit,
                                                                                   labels=tf.zeros([self.bs, 1])))
            d_loss_fake_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logit,
                                                                                   labels=tf.ones([self.bs, 1])))
            d_loss_real_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.true_logit,
                                                                                   labels=tf.ones([self.bs, 1])))
            self.g_loss = d_loss_fake_1
            self.d_loss = d_loss_fake_0 + d_loss_real_1
