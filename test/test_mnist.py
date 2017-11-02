import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from network import network


class MnistModel(network.Network):
    def create_graph(self, x_name, class_num):
        net = self.fc_fc(self.inputs[x_name], [500, 100], self.trainable, name='fc1')
        net = self.fc_fc(net, [class_num],
                         self.trainable, name='fc2', dropout=False, activation=None)

        self.pred =  net

    def create_graph_conv(self, x_name, class_num):
        conv1, pool1 = self.conv_conv_pool(tf.reshape(self.inputs[x_name], [-1, 28, 28, 1]),
                                           [8, 8], self.trainable, name='conv1', bn=False)
        conv2, pool2 = self.conv_conv_pool(pool1, [16, 16], self.trainable, name='conv2', bn=False)
        net = self.fc_fc(tf.reshape(pool2, [-1, 16*7*7]), [500, 100],
                         self.trainable, name='fc1', dropout=False)
        net = self.fc_fc(net, [class_num],
                         self.trainable, name='fc2', dropout=False, activation=None)

        self.pred =  net

    def make_loss(self, y_name):
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.inputs[y_name],
                                                                               logits=self.pred))

    def make_optimizer(self, lr):
        self.optimizer = tf.train.GradientDescentOptimizer(lr).minimize(self.loss, global_step=self.global_step)

    def make_accuracy(self, y_name):
        with tf.variable_scope('evaluation'):
            correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.inputs[y_name], 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy

    def make_summary(self):
        tf.summary.histogram('Predicted Prob', tf.argmax(self.pred, 1))
        tf.summary.scalar('Cross Entropy', self.loss)
        self.summary = tf.summary.merge_all()

    def validate(self, sess, x_name, y_name, summary_writer):
        validate_acc = 0
        for v in range(10):
            batch = mnist.validation.next_batch(100)
            acc, step_summary = sess.run([self.make_accuracy('Y'), self.summary],
                                         feed_dict={self.inputs[x_name]:batch[0],
                                                    self.inputs[y_name]:batch[1]})
            validate_acc += (1/10) * acc
            summary_writer.add_summary(step_summary, self.global_step_value)
        return validate_acc

    def train(self, sess, x_name, y_name, step_num, valid_step, batch_size, summary_writer):
        for i in range(step_num):
            # Valid
            if i % valid_step == 0:
                validate_acc = self.validate(sess, x_name, y_name, summary_writer)
                print('step {}, validation accuracy {}'.format(i, validate_acc))

            # Train
            batch = mnist.train.next_batch(batch_size)
            _, self.global_step_value = sess.run([self.optimizer, self.global_step],
                                                 feed_dict={self.inputs[x_name]: batch[0],
                                                            self.inputs[y_name]: batch[1]})


# make ckdir
ckdir = r'./models/'

# Import data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Model Inputs
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

model = MnistModel({'X':X, 'Y':Y}, trainable=True, model_name='MnistConv')
model.create_graph_conv('X', 10)
model.make_loss('Y')
model.make_optimizer(lr=0.1)
model.make_ckdir(ckdir)
model.make_summary()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_summary_writer = tf.summary.FileWriter(model.ckdir, sess.graph)

    model.train(sess, 'X', 'Y', 4000, 250, 50, train_summary_writer)
