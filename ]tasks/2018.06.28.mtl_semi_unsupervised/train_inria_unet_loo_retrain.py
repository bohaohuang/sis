import os
import time
import argparse
import numpy as np
import tensorflow as tf
import uabRepoPaths
import uabUtilreader
import util_functions
import uabCrossValMaker
import bohaoCustom.uabPreprocClasses as bPreproc
import uabPreprocClasses
import uab_collectionFunctions
import uab_DataHandlerFunctions
from bohaoCustom import uabDataReader
from bohaoCustom import uabMakeNetwork
from bohaoCustom import uabMakeNetwork_UNet

RUN_ID = 0
BATCH_SIZE = 5
LEARNING_RATE = 1e-4
INPUT_SIZE = 572
TILE_SIZE = 5000
EPOCHS = 100
NUM_CLASS = 2
N_TRAIN = 8000
N_VALID = 1000
GPU = 0
DECAY_STEP = 60
DECAY_RATE = 0.1
MODEL_NAME = 'inria_loo_mtl_retrain_{}_{}'
SFN = 32
LEAVE_CITY = 0


class ImageLabelReaderRotation(uabDataReader.ImageLabelReader):
    def __init__(self, gtInds, dataInds, parentDir, chipFiles, chip_size, batchSize=1,
                 nChannels=1, padding=np.array((0, 0)), block_mean=None, dataAug=''):
        super(ImageLabelReaderRotation, self).__init__(gtInds, dataInds, parentDir,
                                                       chipFiles,
                                                       chip_size,
                                                       batchSize, nChannels, padding,
                                                       block_mean, dataAug)

    def readFromDiskIteratorTrain(self, image_dir, chipFiles, batch_size, patch_size,
                                  padding=(0, 0), dataAug=''):
        assert batch_size == 1
        # this is a iterator for training
        nDims = len(chipFiles[0])
        while True:
            image_batch = np.zeros((4, patch_size[0], patch_size[1], nDims))
            # select number to sample
            idx_batch = np.random.permutation(len(chipFiles))

            for cnt, randInd in enumerate(idx_batch):
                row = chipFiles[randInd]

                blockList = []
                nDims = 0
                for file in row:
                    img = util_functions.uabUtilAllTypeLoad(os.path.join(image_dir, file))
                    if len(img.shape) == 2:
                        img = np.expand_dims(img, axis=2)
                    nDims += img.shape[2]
                    blockList.append(img)
                block = np.dstack(blockList).astype(np.float32)

                if self.block_mean is not None:
                    block -= self.block_mean

                if dataAug != '':
                    augDat = uabUtilreader.doDataAug(block, nDims, dataAug, is_np=True,
                                                     img_mean=self.block_mean)
                else:
                    augDat = block

                if (np.array(padding) > 0).any():
                    augDat = uabUtilreader.pad_block(augDat, padding)

                image_batch[0, :, :, :] = augDat
                for i in range(1, 4):
                    image_batch[i, :, :, :] = np.rot90(augDat, k=i, axes=(0, 1))

                if (cnt + 1) % batch_size == 0:
                    yield image_batch[:, :, :, 1:], image_batch[:, :, :, :1]


class UnetModelPredictRot(uabMakeNetwork_UNet.UnetModelPredict):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=None,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32):
        uabMakeNetwork.Network.__init__(self, inputs, trainable, dropout_rate,
                                        learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'UnetPredictRot'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_iou = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 3, 3], name='validation_images')
        self.update_ops = None
        self.config = None

    def create_graph(self, x_name, class_num, start_filter_num=32):
        self.class_num = class_num
        sfn = self.sfn

        # downsample
        conv1, pool1 = self.conv_conv_pool(self.inputs[x_name], [sfn, sfn], self.trainable, name='conv1',
                                           padding='valid', dropout=self.dropout_rate)
        conv2, pool2 = self.conv_conv_pool(pool1, [sfn*2, sfn*2], self.trainable, name='conv2',
                                           padding='valid', dropout=self.dropout_rate)
        conv3, pool3 = self.conv_conv_pool(pool2, [sfn*4, sfn*4], self.trainable, name='conv3',
                                           padding='valid', dropout=self.dropout_rate)
        conv4, pool4 = self.conv_conv_pool(pool3, [sfn*8, sfn*8], self.trainable, name='conv4',
                                           padding='valid', dropout=self.dropout_rate)
        self.encoding = self.conv_conv_pool(pool4, [sfn*16, sfn*16], self.trainable, name='conv5', pool=False,
                                    padding='valid', dropout=self.dropout_rate)

        # upsample
        up6 = self.crop_upsample_concat(self.encoding, conv4, 8, name='6')
        conv6 = self.conv_conv_pool(up6, [sfn*8, sfn*8], self.trainable, name='up6', pool=False,
                                    padding='valid', dropout=self.dropout_rate)
        up7 = self.crop_upsample_concat(conv6, conv3, 32, name='7')
        conv7 = self.conv_conv_pool(up7, [sfn*4, sfn*4], self.trainable, name='up7', pool=False,
                                    padding='valid', dropout=self.dropout_rate)
        up8 = self.crop_upsample_concat(conv7, conv2, 80, name='8')
        conv8 = self.conv_conv_pool(up8, [sfn*2, sfn*2], self.trainable, name='up8', pool=False,
                                    padding='valid', dropout=self.dropout_rate)
        up9 = self.crop_upsample_concat(conv8, conv1, 176, name='9')
        conv9 = self.conv_conv_pool(up9, [sfn, sfn], self.trainable, name='up9', pool=False,
                                    padding='valid', dropout=self.dropout_rate)

        self.pred = tf.layers.conv2d(conv9, class_num, (1, 1), name='final', activation=None, padding='same')
        self.output = tf.nn.softmax(self.pred)

        with tf.variable_scope('rotation'):
            rot_slice = tf.to_int32(tf.random_uniform([1], 1, 4))[0]
            self.building_loss = 1e-3 * tf.reduce_mean(
                tf.norm(conv9[0, :, :, :] - tf.image.rot90(conv9[rot_slice, :, :, :], k=4-rot_slice), axis=2))

            '''self.building_loss = 0
            images = [conv9[0, :, :, :]]
            for i in range(1, 4):
                images.append(tf.image.rot90(conv9[i, :, :, :], k=4-i))
            images = tf.stack(images)
            image_mean = tf.reduce_mean(images, axis=0)
            for i in range(4):
                self.building_loss += tf.reduce_mean(tf.norm(images[i, :, :, :] - image_mean, axis=2))
            #self.building_loss = self.building_loss * 1e-3 - (tf.exp(tf.reduce_sum(image_mean)))
            self.building_loss = - (tf.exp(tf.reduce_sum(image_mean)))'''

    def make_optimizer(self, train_var_filter):
        with tf.control_dependencies(self.update_ops):
            if train_var_filter is None:
                 seg_optm = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                                global_step=self.global_step)
                 clf_optm = tf.train.AdamOptimizer(self.learning_rate * 0.05).minimize(self.building_loss,
                                                                                       global_step=None)
                 self.optimizer = [seg_optm, clf_optm]

    def make_loss(self, y_name, loss_type='xent', **kwargs):
        with tf.variable_scope('loss'):
            pred_flat = tf.reshape(self.pred, [-1, self.class_num])
            _, w, h, _ = self.inputs[y_name].get_shape().as_list()
            y = tf.image.resize_image_with_crop_or_pad(self.inputs[y_name], w-self.get_overlap(), h-self.get_overlap())
            y_flat = tf.reshape(tf.squeeze(y, axis=[3]), [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(y_flat, self.class_num - 1)), 1)
            gt = tf.gather(y_flat, indices)
            prediction = tf.gather(pred_flat, indices)

            pred = tf.argmax(prediction, axis=-1, output_type=tf.int32)
            intersect = tf.cast(tf.reduce_sum(gt * pred), tf.float32)
            union = tf.cast(tf.reduce_sum(gt), tf.float32) + tf.cast(tf.reduce_sum(pred), tf.float32) \
                    - tf.cast(tf.reduce_sum(gt * pred), tf.float32)
            self.loss_iou = tf.convert_to_tensor([intersect, union])
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt))

    def make_update_ops(self, x_name, y_name):
        tf.add_to_collection('inputs', self.inputs[x_name])
        tf.add_to_collection('inputs', self.inputs[y_name])
        tf.add_to_collection('outputs', self.pred)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def make_summary(self, hist=False):
        if hist:
            tf.summary.histogram('Predicted Prob', tf.argmax(tf.nn.softmax(self.pred), 1))
        tf.summary.scalar('Cross Entropy', self.loss)
        tf.summary.scalar('Classify Loss', self.building_loss)
        tf.summary.scalar('learning rate', self.learning_rate)
        self.summary = tf.summary.merge_all()

    def train_config(self, x_name, y_name, y_name_2, n_train, n_valid, patch_size, ckdir, loss_type='xent',
                     train_var_filter=None, hist=False, par_dir=None, **kwargs):
        self.make_loss(y_name, loss_type, **kwargs)
        self.make_learning_rate(n_train)
        self.make_update_ops(x_name, y_name)
        self.make_optimizer(train_var_filter)
        self.make_ckdir(ckdir, patch_size, par_dir)
        self.make_summary(hist)
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.n_train = n_train
        self.n_valid = n_valid

    def train(self, x_name, y_name, y_name_2, n_train, sess, summary_writer, n_valid=1000,
              train_reader=None, train_reader_building=None, valid_reader=None,
              image_summary=None, verb_step=100, save_epoch=5,
              img_mean=np.array((0, 0, 0), dtype=np.float32),
              continue_dir=None, valid_iou=False):
        # define summary operations
        valid_cross_entropy_summary_op = tf.summary.scalar('xent_validation', self.valid_cross_entropy)
        valid_iou_summary_op = tf.summary.scalar('iou_validation', self.valid_iou)
        valid_image_summary_op = tf.summary.image('Validation_images_summary', self.valid_images,
                                                  max_outputs=10)

        if continue_dir is not None and os.path.exists(continue_dir):
            self.load(continue_dir, sess)
            gs = sess.run(self.global_step)
            start_epoch = int(np.ceil(gs/n_train*self.bs))
            start_step = gs - int(start_epoch*n_train/self.bs)
        else:
            start_epoch = 0
            start_step = 0

        cross_entropy_valid_min = np.inf
        iou_valid_max = 0
        for epoch in range(start_epoch, self.epochs):
            start_time = time.time()
            for step in range(start_step, n_train, self.bs):
                X_batch, y_batch = train_reader.readerAction(sess)
                _, self.global_step_value = sess.run([self.optimizer[0], self.global_step],
                                                     feed_dict={self.inputs[x_name]:X_batch,
                                                                self.inputs[y_name]:y_batch,
                                                                self.trainable: True})
                X_batch_rot, _ = train_reader_building.readerAction(sess)
                _, self.global_step_value = sess.run([self.optimizer[1], self.global_step],
                                                     feed_dict={self.inputs[x_name]: X_batch_rot,
                                                                self.trainable: True})
                if self.global_step_value % verb_step == 0:
                    step_loss_building, step_cross_entropy, step_summary = \
                        sess.run([self.building_loss, self.loss, self.summary],
                                 feed_dict={self.inputs[x_name]: X_batch, self.inputs[y_name]: y_batch,
                                            self.trainable: False})
                    summary_writer.add_summary(step_summary, self.global_step_value)
                    print('Epoch {:d} step {:d}\tcross entropy = {:.3f}, rotation loss = {:.3f}'.
                          format(epoch, self.global_step_value, step_cross_entropy, step_loss_building))
            # validation
            cross_entropy_valid_mean = []
            iou_valid_mean = np.zeros(2)
            X_batch_val, y_batch_val, pred_valid = None, None, None
            for step in range(0, n_valid, self.bs):
                X_batch_val, y_batch_val = valid_reader.readerAction(sess)
                pred_valid, cross_entropy_valid, iou_valid = sess.run([self.pred, self.loss, self.loss_iou],
                                                                      feed_dict={self.inputs[x_name]: X_batch_val,
                                                                                 self.inputs[y_name]: y_batch_val,
                                                                                 self.trainable: False})
                cross_entropy_valid_mean.append(cross_entropy_valid)
                iou_valid_mean += iou_valid
            cross_entropy_valid_mean = np.mean(cross_entropy_valid_mean)
            iou_valid_mean = iou_valid_mean[0] / iou_valid_mean[1]
            duration = time.time() - start_time
            if valid_iou:
                print('Validation IoU: {:.3f}, duration: {:.3f}'.format(iou_valid_mean, duration))
            else:
                print('Validation cross entropy: {:.3f}, duration: {:.3f}'.format(cross_entropy_valid_mean,
                                                                                  duration))
            valid_cross_entropy_summary = sess.run(valid_cross_entropy_summary_op,
                                                   feed_dict={self.valid_cross_entropy: cross_entropy_valid_mean})
            valid_iou_summary = sess.run(valid_iou_summary_op,
                                         feed_dict={self.valid_iou: iou_valid_mean})
            summary_writer.add_summary(valid_cross_entropy_summary, self.global_step_value)
            summary_writer.add_summary(valid_iou_summary, self.global_step_value)
            if valid_iou:
                if iou_valid_mean > iou_valid_max:
                    iou_valid_max = iou_valid_mean
                    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                    saver.save(sess, '{}/best_model.ckpt'.format(self.ckdir))

            else:
                if cross_entropy_valid_mean < cross_entropy_valid_min:
                    cross_entropy_valid_min = cross_entropy_valid_mean
                    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                    saver.save(sess, '{}/best_model.ckpt'.format(self.ckdir))

            if image_summary is not None:
                valid_image_summary = sess.run(valid_image_summary_op,
                                               feed_dict={self.valid_images:
                                                              image_summary(X_batch_val[:,:,:,:3], y_batch_val, pred_valid,
                                                                            img_mean)})
                summary_writer.add_summary(valid_image_summary, self.global_step_value)

            if epoch % save_epoch == 0:
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                saver.save(sess, '{}/model_{}.ckpt'.format(self.ckdir, epoch), global_step=self.global_step)


def read_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (10)')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE, help='learning rate (1e-3)')
    parser.add_argument('--input-size', default=INPUT_SIZE, type=int, help='input size 224')
    parser.add_argument('--tile-size', default=TILE_SIZE, type=int, help='tile size 5000')
    parser.add_argument('--epochs', default=EPOCHS, type=int, help='# epochs (1)')
    parser.add_argument('--num-classes', type=int, default=NUM_CLASS, help='# classes (including background)')
    parser.add_argument('--n-train', type=int, default=N_TRAIN, help='# samples per epoch')
    parser.add_argument('--n-valid', type=int, default=N_VALID, help='# patches to valid')
    parser.add_argument('--GPU', type=str, default=GPU, help="GPU used for computation.")
    parser.add_argument('--decay-step', type=float, default=DECAY_STEP, help='Learning rate decay step in number of epochs.')
    parser.add_argument('--decay-rate', type=float, default=DECAY_RATE, help='Learning rate decay rate')
    parser.add_argument('--model-name', type=str, default=MODEL_NAME, help='Model name')
    parser.add_argument('--run-id', type=str, default=RUN_ID, help='id of this run')
    parser.add_argument('--sfn', type=int, default=SFN, help='filter number of the first layer')
    parser.add_argument('--leave-city', type=int, default=LEAVE_CITY, help='city id to leave-out in training')

    flags = parser.parse_args()
    flags.input_size = (flags.input_size, flags.input_size)
    flags.tile_size = (flags.tile_size, flags.tile_size)
    flags.model_name = flags.model_name.format(flags.leave_city, flags.run_id)
    return flags


def main(flags):
    # make network
    # define place holder
    X = tf.placeholder(tf.float32, shape=[None, flags.input_size[0], flags.input_size[1], 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, flags.input_size[0], flags.input_size[1], 1], name='y')
    y2 = tf.placeholder(tf.float32, shape=[None, 1], name='y2')
    mode = tf.placeholder(tf.bool, name='mode')
    model = UnetModelPredictRot({'X':X, 'Y':y, 'Y2':y2},
                                trainable=mode,
                                model_name=flags.model_name,
                                input_size=flags.input_size,
                                batch_size=flags.batch_size,
                                learn_rate=flags.learning_rate,
                                decay_step=flags.decay_step,
                                decay_rate=flags.decay_rate,
                                epochs=flags.epochs,
                                start_filter_num=flags.sfn)
    model.create_graph('X', class_num=flags.num_classes)

    # create collection
    # the original file is in /ei-edl01/data/uab_datasets/inria
    blCol = uab_collectionFunctions.uabCollection('inria')
    opDetObj = bPreproc.uabOperTileDivide(255)          # inria GT has value 0 and 255, we map it back to 0 and 1
    # [3] is the channel id of GT
    rescObj = uabPreprocClasses.uabPreprocMultChanOp([], 'GT_Divide.tif', 'Map GT to (0, 1)', [3], opDetObj)
    rescObj.run(blCol)
    img_mean = blCol.getChannelMeans([0, 1, 2])         # get mean of rgb info

    # extract patches
    extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4],
                                                    cSize=flags.input_size,
                                                    numPixOverlap=int(model.get_overlap()),
                                                    extSave=['jpg', 'jpg', 'jpg', 'png'],
                                                    isTrain=True,
                                                    gtInd=3,
                                                    pad=model.get_overlap()/2)
    patchDir = extrObj.run(blCol)

    # make data reader
    # use uabCrossValMaker to get fileLists for training and validation
    idx_city, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'city')
    idx_tile, _ = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'force_tile')
    idx = [j * 10 + i for i, j in zip(idx_city, idx_tile)]
    # use first city for validation
    filter_train = []
    filter_valid = []
    for i in range(5):
        for j in range(1, 37):
            if i != flags.leave_city and j > 5:
                filter_train.append(j * 10 + i)
            elif i == flags.leave_city and j <= 5:
                filter_valid.append(j * 10 + i)
    # use first city for validation
    file_list_train = uabCrossValMaker.make_file_list_by_key(idx, file_list, filter_train)
    file_list_valid = uabCrossValMaker.make_file_list_by_key(idx, file_list, filter_valid)

    dataReader_train = uabDataReader.ImageLabelReader([3], [0, 1, 2], patchDir, file_list_train, flags.input_size,
                                                      flags.batch_size, dataAug='flip,rotate',
                                                      block_mean=np.append([0], img_mean), batch_code=0)
    dataReader_train_rot = ImageLabelReaderRotation([3], [0, 1, 2], patchDir, file_list_valid, flags.input_size,
                                                    1, dataAug='flip,rotate', block_mean=np.append([0], img_mean))
    # no augmentation needed for validation
    dataReader_valid = uabDataReader.ImageLabelReader([3], [0, 1, 2], patchDir, file_list_valid, flags.input_size,
                                                      flags.batch_size, dataAug=' ',
                                                      block_mean=np.append([0], img_mean), batch_code=0)

    # train
    start_time = time.time()

    model.train_config('X', 'Y', 'Y2', flags.n_train, flags.n_valid, flags.input_size, uabRepoPaths.modelPath,
                       loss_type='xent', par_dir='Inria_Domain_LOO')
    model.run(train_reader=dataReader_train,
              train_reader_building=dataReader_train_rot,
              valid_reader=dataReader_valid,
              pretrained_model_dir=None,        # train from scratch, no need to load pre-trained model
              isTrain=True,
              img_mean=img_mean,
              verb_step=100,                    # print a message every 100 step(sample)
              save_epoch=5,                     # save the model every 5 epochs
              gpu=GPU,
              tile_size=flags.tile_size,
              patch_size=flags.input_size)

    duration = time.time() - start_time
    print('duration {:.2f} hours'.format(duration/60/60))


if __name__ == '__main__':
    flags = read_flag()
    main(flags)
