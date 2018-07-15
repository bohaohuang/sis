import os
import time
import imageio
import argparse
import numpy as np
import tensorflow as tf
from glob import glob
import uabRepoPaths
import uabCrossValMaker
import bohaoCustom.uabPreprocClasses as bPreproc
import uabPreprocClasses
import uabUtilreader
import util_functions
import uab_collectionFunctions
import uab_DataHandlerFunctions
from bohaoCustom import uabDataReader
from bohaoCustom import uabMakeNetwork_UNet

RUN_ID = 0
BATCH_SIZE = 5
LEARNING_RATE = 1e-5
INPUT_SIZE = 572
TILE_SIZE = 5000
EPOCHS = 40
NUM_CLASS = 2
N_TRAIN = 8000
N_VALID = 1000
GPU = 1
DECAY_STEP = 60
DECAY_RATE = 0.1
MODEL_NAME = 'inria_loo_mtl_iter_{}_{}'
SFN = 32
LEAVE_CITY = 1
PRED_FILE_DIR = r'/media/ei-edl01/user/bh163/tasks/2018.06.28.mtl_semi_unsupervised'
FINETUNE_DIR = r'/hdd6/Models/Inria_Domain_LOO/UnetCrop_inria_aug_leave_{}_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'


def make_gt(pred_dir, save_dir, prefix, model_name='unet', threshold=0.1):
    city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
    for city_num in range(5):
        pred_dir = os.path.join(pred_dir, 'inria_all', 'pred')
        if model_name == 'unet':
            patch_size = (572, 572)
            overlap = 92
            pad = 92
        else:
            patch_size = (321, 321)
            overlap = 0
            pad = 0
        # extract patches
        tile_files = sorted(glob(os.path.join(pred_dir, '{}*.png'.format(city_list[city_num]))))
        pred_building_binary = []
        for file in tile_files:
            gt = imageio.imread(file)
            gt = np.expand_dims(gt, axis=2)
            gt = uabUtilreader.pad_block(gt, np.array([pad, pad]))
            for patch in uabUtilreader.patchify(gt, (5184, 5184), patch_size, overlap):
                if model_name == 'deeplab':
                    pred_raw = np.sum(patch) / (patch_size[0] * patch_size[1])
                else:
                    pred_raw = np.sum(util_functions.crop_center(patch, 388, 388)) / (patch_size[0] * patch_size[1])
                if pred_raw > threshold:
                    pred_building_binary.append(1)
                else:
                    pred_building_binary.append(0)
        np.save(save_dir, pred_building_binary)


class UnetModelCrop_Iter(uabMakeNetwork_UNet.UnetModelPredict):
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
                X_batch, _, building_truth = train_reader_building.readerAction(sess)
                _, self.global_step_value = sess.run([self.optimizer[1], self.global_step],
                                                     feed_dict={self.inputs[x_name]: X_batch,
                                                                self.inputs[y_name_2]: building_truth,
                                                                self.trainable: True})
                if self.global_step_value % verb_step == 0:
                    pred_train, step_cross_entropy, step_summary = sess.run([self.pred, self.loss, self.summary],
                                                                            feed_dict={self.inputs[x_name]: X_batch,
                                                                                       self.inputs[y_name]: y_batch,
                                                                                       self.inputs[y_name_2]: building_truth,
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
                                                              image_summary(X_batch_val[:,:,:,:3], y_batch_val, pred_valid,
                                                                            img_mean)})
                summary_writer.add_summary(valid_image_summary, self.global_step_value)

            if epoch % save_epoch == 0:
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                saver.save(sess, '{}/model_{}.ckpt'.format(self.ckdir, epoch), global_step=self.global_step)

                # remake gts
                blCol = uab_collectionFunctions.uabCollection('inria')
                blCol.readMetadata()
                file_list, parent_dir = blCol.getAllTileByDirAndExt([0, 1, 2])
                file_list_truth, parent_dir_truth = blCol.getAllTileByDirAndExt(4)
                idx, file_list = uabCrossValMaker.uabUtilGetFolds(None, file_list, 'force_tile')
                idx_truth, file_list_truth = uabCrossValMaker.uabUtilGetFolds(None, file_list_truth, 'force_tile')

                # use first 5 tiles for validation
                city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
                file_list_valid = uabCrossValMaker.make_file_list_by_key(
                    idx, file_list, [i for i in range(0, 6)],
                    filter_list=['bellingham', 'bloomington', 'sfo', 'tyrol-e', 'innsbruck'] +
                                [a for a in city_list if a != city_list[flags.leave_city]])
                file_list_valid_truth = uabCrossValMaker.make_file_list_by_key(
                    idx_truth, file_list_truth, [i for i in range(0, 6)],
                    filter_list=['bellingham', 'bloomington', 'sfo', 'tyrol-e', 'innsbruck'] +
                                [a for a in city_list if a != city_list[flags.leave_city]])
                img_mean = blCol.getChannelMeans([0, 1, 2])

                self.evaluate(file_list_valid, file_list_valid_truth, parent_dir, parent_dir_truth, (572, 572),
                              (5000, 5000), 50, img_mean, self.ckdir, flags.GPU,
                              save_result_parent_dir='domain_selection', ds_name='inria', best_model=False)
                result_dir = os.path.join(uabRepoPaths.evalPath, self.model_name)
                make_gt(result_dir, flags.pred_file_dir, 'iter')

    def run(self, train_reader=None, train_reader_building=None, valid_reader=None, test_reader=None,
            pretrained_model_dir=None, layers2load=None, isTrain=False, img_mean=np.array((0, 0, 0), dtype=np.float32),
            verb_step=100, save_epoch=5, gpu=None, tile_size=(5000, 5000), patch_size=(572, 572), truth_val=1,
            continue_dir=None, load_epoch_num=None, valid_iou=False, best_model=True):
        if gpu is not None:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        if isTrain:
            coord = tf.train.Coordinator()
            with tf.Session(config=self.config) as sess:
                # init model
                init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
                sess.run(init)
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                # load model
                if pretrained_model_dir is not None:
                    if layers2load is not None:
                        self.load_weights(pretrained_model_dir, layers2load)
                    else:
                        self.load(pretrained_model_dir, sess, saver, epoch=load_epoch_num)
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)
                try:
                    train_summary_writer = tf.summary.FileWriter(self.ckdir, sess.graph)
                    self.train('X', 'Y', 'Y2', self.n_train, sess, train_summary_writer,
                               n_valid=self.n_valid, train_reader=train_reader, valid_reader=valid_reader,
                               train_reader_building=train_reader_building,
                               image_summary=util_functions.image_summary, img_mean=img_mean,
                               verb_step=verb_step, save_epoch=save_epoch, continue_dir=continue_dir,
                               valid_iou=valid_iou)
                finally:
                    coord.request_stop()
                    coord.join(threads)
                    saver.save(sess, '{}/model.ckpt'.format(self.ckdir), global_step=self.global_step)
        else:
            if self.config is None:
                self.config = tf.ConfigProto(allow_soft_placement=True)
            pad = self.get_overlap()
            with tf.Session(config=self.config) as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                result = self.test('X', sess, test_reader)
            image_pred = uabUtilreader.un_patchify_shrink(result,
                                                          [tile_size[0] + pad, tile_size[1] + pad],
                                                          tile_size,
                                                          patch_size,
                                                          [patch_size[0] - pad, patch_size[1] - pad],
                                                          overlap=pad)
            return util_functions.get_pred_labels(image_pred) * truth_val


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
    parser.add_argument('--pred-file-dir', type=str, default=PRED_FILE_DIR, help='building/non-building prediction dir')
    parser.add_argument('--finetune-dir', type=str, default=FINETUNE_DIR, help='pretrained model dir')

    flags = parser.parse_args()
    flags.input_size = (flags.input_size, flags.input_size)
    flags.tile_size = (flags.tile_size, flags.tile_size)
    flags.model_name = flags.model_name.format(flags.leave_city, flags.run_id)
    flags.finetune_dir = flags.finetune_dir.format(flags.leave_city)
    return flags


def main(flags):
    flags.pred_file_dir = os.path.join(flags.pred_file_dir,
                                       'iter_pred_building_binary_{}.npy'.format(flags.leave_city))

    # make network
    # define place holder
    X = tf.placeholder(tf.float32, shape=[None, flags.input_size[0], flags.input_size[1], 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, flags.input_size[0], flags.input_size[1], 1], name='y')
    y2 = tf.placeholder(tf.float32, shape=[None, 1], name='y2')
    mode = tf.placeholder(tf.bool, name='mode')
    model = UnetModelCrop_Iter({'X':X, 'Y':y, 'Y2':y2},
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
    model.load_weights(flags.finetune_dir, '1,2,3,4,5,6,7,8,9')

    # create collection
    # the original file is in /ei-edl01/data/uab_datasets/inria
    blCol = uab_collectionFunctions.uabCollection('inria')
    opDetObj = bPreproc.uabOperTileDivide(255)          # inria GT has value 0 and 255, we map it back to 0 and 1
    # [3] is the channel id of GT
    rescObj = uabPreprocClasses.uabPreprocMultChanOp([], 'GT_Divide.tif', 'Map GT to (0, 1)', [3], opDetObj)
    rescObj.run(blCol)
    img_mean = blCol.getChannelMeans([0, 1, 2])         # get mean of rgb info

    # extract patches
    extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4], # extract all 4 channels
                                                    cSize=flags.input_size, # patch size as 572*572
                                                    numPixOverlap=int(model.get_overlap()/2),  # overlap as 92
                                                    extSave=['jpg', 'jpg', 'jpg', 'png'], # save rgb files as jpg and gt as png
                                                    isTrain=True,
                                                    gtInd=3,
                                                    pad=model.get_overlap()) # pad around the tiles
    patchDir = extrObj.run(blCol)

    # make data reader
    # use uabCrossValMaker to get fileLists for training and validation
    idx, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'city')
    # use first city for validation
    file_list_train = uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(5) if i != flags.leave_city])
    file_list_valid = uabCrossValMaker.make_file_list_by_key(idx, file_list, [flags.leave_city])

    dataReader_train = uabDataReader.ImageLabelReader([3], [0, 1, 2], patchDir, file_list_train, flags.input_size,
                                                      flags.batch_size, dataAug='flip,rotate',
                                                      block_mean=np.append([0], img_mean), batch_code=0)
    dataReader_train_building = uabDataReader.ImageLabelReaderBuildingCustom(
        [3], [0, 1, 2], patchDir, file_list_valid, flags.input_size, flags.batch_size, dataAug='flip,rotate',
        percent_file=flags.pred_file_dir, block_mean=np.append([0], img_mean), patch_prob=0.1, binary=True)
    # no augmentation needed for validation
    dataReader_valid = uabDataReader.ImageLabelReader([3], [0, 1, 2], patchDir, file_list_valid, flags.input_size,
                                                      flags.batch_size, dataAug=' ',
                                                      block_mean=np.append([0], img_mean), batch_code=0)

    # train
    start_time = time.time()

    model.train_config('X', 'Y', 'Y2', flags.n_train, flags.n_valid, flags.input_size, uabRepoPaths.modelPath,
                       loss_type='xent', par_dir='Inria_Domain_LOO')
    model.run(train_reader=dataReader_train,
              train_reader_building=dataReader_train_building,
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
