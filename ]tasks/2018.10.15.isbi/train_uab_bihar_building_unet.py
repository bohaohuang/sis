import os
import time
import argparse
import numpy as np
import tensorflow as tf
import sis_utils
import ersa_utils
import uabRepoPaths
import uabDataReader
import uabCrossValMaker
import uab_collectionFunctions
from reader import reader_utils
from bohaoCustom import uabMakeNetwork_UNet

RUN_ID = 1
BATCH_SIZE = 5
LEARNING_RATE = 1e-7
INPUT_SIZE = 300
EPOCHS = 30
NUM_CLASS = 2
N_TRAIN = 200
N_VALID = 5
GPU = 0
DECAY_STEP = 30
DECAY_RATE = 0.8
START_LAYER = 10
MODEL_NAME = 'bihar_building_{}'
DS_NAME = 'bihar_building'
SFN = 32


class myUnetModelCrop(uabMakeNetwork_UNet.UnetModelCrop):
    def train(self, x_name, y_name, n_train, sess, summary_writer, n_valid=1000,
              train_reader=None, valid_reader=None,
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
            cross_entropy_valid_mean = []
            iou_valid_mean = np.zeros(2)
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
                                                              image_summary(X_batch_val[:,92:-92,92:-92,:3], y_batch_val, pred_valid,
                                                                            img_mean)})
                summary_writer.add_summary(valid_image_summary, self.global_step_value)

            if epoch % save_epoch == 0:
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                saver.save(sess, '{}/model_{}.ckpt'.format(self.ckdir, epoch), global_step=self.global_step)


def resize_patches(files, par_dir, patch_size, save_dir):
    def load_and_resize(par, f, patch_size, preserve, pad=92):
        patch_size = np.array(patch_size) - np.array([pad*2, pad*2])
        data = ersa_utils.load_file(os.path.join(par, f))
        data = ersa_utils.pad_image(data, pad)
        return reader_utils.resize_image(data, patch_size, preserve)

    f_name_list = []
    for file in files:
        r_patch = load_and_resize(par_dir[0], file[0], patch_size, True).astype(np.uint8)
        g_patch = load_and_resize(par_dir[1], file[1], patch_size, True).astype(np.uint8)
        b_patch = load_and_resize(par_dir[2], file[2], patch_size, True).astype(np.uint8)
        gt_patch = load_and_resize(par_dir[3], file[3], patch_size, True).astype(np.uint8)

        ersa_utils.save_file(os.path.join(save_dir, '{}jpg'.format(file[0][:-3])), r_patch)
        ersa_utils.save_file(os.path.join(save_dir, '{}jpg'.format(file[1][:-3])), g_patch)
        ersa_utils.save_file(os.path.join(save_dir, '{}jpg'.format(file[2][:-3])), b_patch)
        ersa_utils.save_file(os.path.join(save_dir, '{}png'.format(file[3][:-3])), gt_patch)

        f_line = '{}jpg {}jpg {}jpg {}png\n'.format(file[0][:-3], file[1][:-3], file[2][:-3], file[3][:-3])
        f_name_list.append(f_line)
    file_name = os.path.join(save_dir, 'fileList.txt')
    ersa_utils.save_file(file_name, f_name_list)


def read_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (10)')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE, help='learning rate (1e-3)')
    parser.add_argument('--input-size', default=INPUT_SIZE, type=int, help='input size 224')
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
    parser.add_argument('--ds-name', type=str, default=DS_NAME, help='name of the dataset')
    parser.add_argument('--start-layer', type=int, default=START_LAYER, help='start layer to unfreeze')

    flags = parser.parse_args()
    flags.input_size = (flags.input_size, flags.input_size)
    flags.model_name = flags.model_name.format(flags.run_id)
    return flags


def main(flags):
    np.random.seed(int(flags.run_id))
    tf.set_random_seed(int(flags.run_id))

    # make network
    # define place holder
    X = tf.placeholder(tf.float32, shape=[None, flags.input_size[0], flags.input_size[1], 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, flags.input_size[0], flags.input_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')
    model = myUnetModelCrop({'X': X, 'Y': y}, trainable=mode, model_name=flags.model_name, input_size=flags.input_size,
                            batch_size=flags.batch_size, learn_rate=flags.learning_rate, decay_step=flags.decay_step,
                            decay_rate=flags.decay_rate, epochs=flags.epochs, start_filter_num=flags.sfn)
    model.create_graph('X', class_num=flags.num_classes)

    # create collection
    # the original file is in /ei-edl01/data/uab_datasets/inria
    blCol = uab_collectionFunctions.uabCollection(flags.ds_name)
    blCol.readMetadata()
    img_mean = blCol.getChannelMeans([0, 1, 2])  # get mean of rgb info
    print(img_mean)

    img_dir, task_dir = sis_utils.get_task_img_folder()
    save_dir = os.path.join(task_dir, 'bihar_patches')
    ersa_utils.make_dir_if_not_exist(save_dir)
    files, par_dir = blCol.getAllTileByDirAndExt([0, 1, 2, 3])
    resize_patches(files, par_dir, flags.input_size, save_dir)

    patchDir = save_dir

    # make data reader
    # use first 5 tiles for validation
    idx, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'tile')
    # use first city for validation
    #assert len(file_list) == flags.n_train + flags.n_valid
    file_list_train = [a for a in file_list[:45]]
    file_list_valid = [a for a in file_list[-5:]]

    with tf.name_scope('image_loader'):
        # GT has no mean to subtract, append a 0 for block mean
        dataReader_train = uabDataReader.ImageLabelReader([3], [0, 1, 2], patchDir, file_list_train, flags.input_size,
                                                          None,
                                                          flags.batch_size, dataAug='flip,rotate',
                                                          block_mean=np.append([0], img_mean))
        # no augmentation needed for validation
        dataReader_valid = uabDataReader.ImageLabelReader([3], [0, 1, 2], patchDir, file_list_valid, flags.input_size,
                                                          None,
                                                          flags.batch_size, dataAug=' ', block_mean=np.append([0], img_mean))

    # train
    start_time = time.time()

    model.train_config('X', 'Y', flags.n_train, flags.n_valid, flags.input_size, uabRepoPaths.modelPath,
                       loss_type='xent', par_dir='{}'.format(flags.ds_name))
    model.run(train_reader=dataReader_train,
              valid_reader=dataReader_valid,
              pretrained_model_dir=None,   # train from scratch, no need to load pre-trained model
              isTrain=True,
              img_mean=img_mean,
              verb_step=100,                        # print a message every 100 step(sample)
              save_epoch=200,                         # save the model every 5 epochs
              gpu=GPU,
              patch_size=flags.input_size)

    duration = time.time() - start_time
    print('duration {:.2f} hours'.format(duration/60/60))


if __name__ == '__main__':
    flags = read_flag()
    main(flags)
