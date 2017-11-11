import tensorflow as tf
from network import network


class UnetModel(network.Network):
    def __init__(self, inputs, trainable, model_name, input_size, dropout_rate=0.2):
        network.Network.__init__(self, inputs, trainable, model_name, dropout_rate)
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 3, 3], name='validation_images')
        self.update_ops = None

    def create_graph(self, x_name, class_num):
        # TODO add a parameter here: start_filter_num=32
        self.class_num = class_num

        # downsample
        conv1, pool1 = self.conv_conv_pool(self.inputs[x_name], [32, 32], self.trainable, name='conv1')
        conv2, pool2 = self.conv_conv_pool(pool1, [64, 64], self.trainable, name='conv2')
        conv3, pool3 = self.conv_conv_pool(pool2, [128, 128], self.trainable, name='conv3')
        conv4, pool4 = self.conv_conv_pool(pool3, [256, 256], self.trainable, name='conv4')
        conv5 = self.conv_conv_pool(pool4, [512, 512], self.trainable, name='conv5', pool=False)

        # upsample
        up6 = self.upsample_concat(conv5, conv4, name='6')
        conv6 = self.conv_conv_pool(up6, [256, 256], self.trainable, name='up6', pool=False)
        up7 = self.upsample_concat(conv6, conv3, name='7')
        conv7 = self.conv_conv_pool(up7, [128, 128], self.trainable, name='up7', pool=False)
        up8 = self.upsample_concat(conv7, conv2, name='8')
        conv8 = self.conv_conv_pool(up8, [64, 64], self.trainable, name='up8', pool=False)
        up9 = self.upsample_concat(conv8, conv1, name='9')
        conv9 = self.conv_conv_pool(up9, [32, 32], self.trainable, name='up9', pool=False)

        self.pred = tf.layers.conv2d(conv9, class_num, (1, 1), name='final', activation=None, padding='same')

    def make_learning_rate(self, lr, decay_steps, decay_rate):
        self.learning_rate = tf.train.exponential_decay(lr, self.global_step, decay_steps,
                                                        decay_rate, staircase=True)

    def make_loss(self, y_name):
        with tf.variable_scope('loss'):
            pred_flat = tf.reshape(tf.nn.softmax(self.pred), [-1, self.class_num])
            y_flat = tf.reshape(tf.squeeze(self.inputs[y_name], axis=[3]), [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(y_flat, self.class_num - 1)), 1)
            gt = tf.gather(y_flat, indices)
            prediction = tf.gather(pred_flat, indices)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt))

    def make_update_ops(self, x_name, y_name):
        tf.add_to_collection('inputs', self.inputs[x_name])
        tf.add_to_collection('inputs', self.inputs[y_name])
        tf.add_to_collection('outputs', self.pred)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def make_optimizer(self, lr):
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss, global_step=self.global_step)

    def make_summary(self):
        tf.summary.histogram('Predicted Prob', tf.argmax(tf.nn.softmax(self.pred), 1))
        tf.summary.scalar('Cross Entropy', self.loss)
        tf.summary.scalar('learning rate', self.learning_rate)
        self.summary = tf.summary.merge_all()

    def train(self, x_name, y_name, epoch_num, n_train, batch_size, sess, summary_writer,
              train_iterator=None, train_reader=None, valid_iterator=None, valid_reader=None,
              image_summary=None, verb_step=100):
        # define summary operations
        valid_cross_entropy_summary_op = tf.summary.scalar('xent_validation', self.valid_cross_entropy)
        valid_image_summary_op = tf.summary.image('Validation_images_summary', self.valid_images,
                                                  max_outputs=10)
        for epoch in range(epoch_num):
            for step in range(0, n_train, batch_size):
                if train_iterator is not None:
                    X_batch, y_batch = next(train_iterator)
                else:
                    X_batch, y_batch = sess.run(train_reader)
                _, self.global_step_value = sess.run([self.optimizer, self.global_step],
                                                     feed_dict={self.inputs[x_name]:X_batch,
                                                                self.inputs[y_name]:y_batch,
                                                                self.trainable: True})
                if self.global_step_value % verb_step == 0:
                    pred_train, step_cross_entropy, step_summary = sess.run([self.pred, self.loss, self.summary],
                                                                            feed_dict={self.inputs[x_name]: X_batch,
                                                                                       self.inputs[y_name]: y_batch,
                                                                                       self.trainable: False})
                    summary_writer.add_summary(step_summary, self.global_step_value)
                    print('Epoch {:d} step {:d}\tcross entropy = {:.3f}'.
                          format(epoch, self.global_step_value, step_cross_entropy))
            # validation
            if valid_iterator is not None:
                X_batch_val, y_batch_val = next(valid_iterator)
            else:
                X_batch_val, y_batch_val = sess.run(valid_reader)
            pred_valid, cross_entropy_valid = sess.run([self.pred, self.loss],
                                                       feed_dict={self.inputs[x_name]: X_batch_val,
                                                                  self.inputs[y_name]: y_batch_val,
                                                                  self.trainable: False})
            print('Validation cross entropy: {:.3f}'.format(cross_entropy_valid))
            valid_cross_entropy_summary = sess.run(valid_cross_entropy_summary_op,
                                                   feed_dict={self.valid_cross_entropy: cross_entropy_valid})
            summary_writer.add_summary(valid_cross_entropy_summary, self.global_step_value)

            if image_summary is not None:
                valid_image_summary = sess.run(valid_image_summary_op,
                                               feed_dict={self.valid_images:
                                                              image_summary(X_batch_val, y_batch_val, pred_valid)})
                summary_writer.add_summary(valid_image_summary, self.global_step_value)
