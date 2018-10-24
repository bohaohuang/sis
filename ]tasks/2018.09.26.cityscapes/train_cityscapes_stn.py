import os
import time
import argparse
import scipy.misc
import skimage.transform
import numpy as np
import tensorflow as tf
from glob import glob
import stn
import ersaPath
import ersa_utils
from nn import hook, nn_utils
from reader import reader_utils
import cityscapes_reader, cityscapes_labels


# define parameters
BATCH_SIZE = 4
DS_NAME = 'cityscapes'
LEARNING_RATE = (1e-6, 1e-6, 1e-6)
TILE_SIZE = (512, 1024)
EPOCHS = 40
NUM_CLASS = 19
PAR_DIR = DS_NAME+'/stn'
SUFFIX = 'rand_scale'
N_TRAIN = 2996
N_VALID = 500
VAL_MULT = 5
GPU = 0
DECAY_STEP = (10, 10, 10)
DECAY_RATE = (0.1, 0.1, 0.1)
VERB_STEP = 120
SAVE_EPOCH = 5
DATA_DIR = r'/hdd/cityscapes'
RGB_TYPE = 'leftImg8bit'
GT_TYPE = 'gtFine'
RGB_EXT = RGB_TYPE
GT_EXT = 'labelTrainIds'
FORCE_RUN = False
CITYSCAPES_TRAIN_ID_TO_EVAL_ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                  23, 24, 25, 26, 27, 28, 31, 32, 33]


class DataReaderSegmentationTrainValid(object):
    def __init__(self, input_size, file_list_train, file_list_valid, batch_size=5, chan_mean=None, aug_func=None,
                 random=True, has_gt=True, gt_dim=0, include_gt=True, valid_mult=1, global_func=None):
        self.input_size = input_size
        self.file_list_train = file_list_train
        self.file_list_valid = file_list_valid
        self.batch_size = batch_size
        self.chan_mean = chan_mean
        if aug_func is None:
            aug_func = []
        if type(aug_func) is not list:
            aug_func = [aug_func]
        if global_func is None:
            global_func = []
        if type(global_func) is not list:
            global_func = [global_func]
        self.aug_func = aug_func
        self.global_func = global_func
        self.random = random
        self.has_gt = has_gt
        self.gt_dim = gt_dim
        self.include_gt = include_gt

        # read one set of files to get #channels
        self.channel_num = 0
        for f in self.file_list_train[0]:
            self.channel_num += ersa_utils.get_img_channel_num(f)
        if self.chan_mean is None:
            self.chan_mean = np.zeros(self.channel_num - self.gt_dim)

        self.valid_mult = valid_mult

    def data_reader_helper(self, files, is_train):
        data_block = []
        for f in files:
            data_block.append(ersa_utils.load_file(f))
        data_block = np.dstack(data_block)
        for aug_func in self.global_func:
            data_block = aug_func(data_block)
        if is_train:
            for aug_func in self.aug_func:
                data_block = aug_func(data_block)
        if self.has_gt:
            ftr_block = data_block[:, :, :-2*self.gt_dim]
            ftr_block = ftr_block - self.chan_mean
            lbl_block = data_block[:, :, -2*self.gt_dim:-self.gt_dim]
            prd_block = data_block[:, :, -self.gt_dim:]
            for cnt, id in enumerate(CITYSCAPES_TRAIN_ID_TO_EVAL_ID):
                np.place(prd_block, prd_block == id, cnt)
            if self.include_gt:
                return ftr_block, lbl_block, prd_block
            else:
                return ftr_block
        else:
            data_block = data_block - self.chan_mean
            return data_block

    def data_reader(self, file_list, is_train, random=False):
        """
        Read feature and label, or feature alone
        :param file_list: list of files to read data
        :param is_train: the dataset is used for training or not
        :param random: include randomness in reading of not
        :return: data read from file list, one line each time
        """
        if random:
            file_list = np.random.permutation(file_list)
        for files in file_list:
            yield self.data_reader_helper(files, is_train)

    def get_dataset(self):
        """
        Create a tf.Dataset from the generator defined
        :return: a tf.Dataset object
        """
        def generator_train(): return self.data_reader(self.file_list_train, True, self.random)

        def generator_valid(): return self.data_reader(self.file_list_valid, False, self.random)

        if self.has_gt and self.include_gt:
            dataset_train = tf.data.Dataset.from_generator(generator_train, (tf.float32, tf.int32, tf.float32,),
                                                           ((self.input_size[0], self.input_size[1],
                                                             self.channel_num - 2 * self.gt_dim),
                                                            (self.input_size[0], self.input_size[1], self.gt_dim),
                                                            (self.input_size[0], self.input_size[1], self.gt_dim),))
            dataset_valid = tf.data.Dataset.from_generator(generator_valid, (tf.float32, tf.int32, tf.float32),
                                                           ((self.input_size[0], self.input_size[1],
                                                             self.channel_num - 2 * self.gt_dim),
                                                            (self.input_size[0], self.input_size[1], self.gt_dim),
                                                            (self.input_size[0], self.input_size[1], self.gt_dim),))
        elif self.has_gt and not self.include_gt:
            dataset_train = tf.data.Dataset.from_generator(generator_train, (tf.float32, ),
                                                           ((self.input_size[0], self.input_size[1],
                                                             self.channel_num - 2 * self.gt_dim), ))
            dataset_valid = tf.data.Dataset.from_generator(generator_valid, (tf.float32, ),
                                                           ((self.input_size[0], self.input_size[1],
                                                             self.channel_num - 2 * self.gt_dim), ))
        else:
            dataset_train = tf.data.Dataset.from_generator(
                generator_train, (tf.float32, ), ((self.input_size[0], self.input_size[1], self.channel_num), ))
            dataset_valid = tf.data.Dataset.from_generator(
                generator_valid, (tf.float32, ), ((self.input_size[0], self.input_size[1], self.channel_num), ))
        return dataset_train, dataset_valid

    def read_op(self):
        """
        Get tf iterator as well as init operation for the dataset
        :return: reader operation and init operation
        """
        dataset_train, dataset_valid = self.get_dataset()
        dataset_train = dataset_train.repeat()
        dataset_train = dataset_train.batch(self.batch_size)

        dataset_valid = dataset_valid.batch(self.batch_size * self.valid_mult)

        iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
        reader_op = iterator.get_next()
        train_init_op = iterator.make_initializer(dataset_train)
        valid_init_op = iterator.make_initializer(dataset_valid)

        return train_init_op, valid_init_op, reader_op


def read_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (10)')
    parser.add_argument('--ds-name', default=DS_NAME, type=str, help='dataset name')
    parser.add_argument('--learning-rate', type=tuple, default=LEARNING_RATE, help='learning rate (1e-3)')
    parser.add_argument('--tile-size', default=TILE_SIZE, type=tuple, help='tile size 5000')
    parser.add_argument('--epochs', default=EPOCHS, type=int, help='# epochs (1)')
    parser.add_argument('--num-classes', type=int, default=NUM_CLASS, help='# classes (including background)')
    parser.add_argument('--model-par-dir', type=str, default=PAR_DIR, help='parent directory name to save the model')
    parser.add_argument('--model-suffix', type=str, default=SUFFIX, help='suffix in the model directory')
    parser.add_argument('--n-train', type=int, default=N_TRAIN, help='# samples per epoch')
    parser.add_argument('--n-valid', type=int, default=N_VALID, help='# patches to valid')
    parser.add_argument('--val-mult', type=int, default=VAL_MULT, help='validation_bs=val_mult*train_bs')
    parser.add_argument('--GPU', type=str, default=GPU, help="GPU used for computation.")
    parser.add_argument('--decay-step', type=tuple, default=DECAY_STEP, help='Learning rate decay step in number of epochs.')
    parser.add_argument('--decay-rate', type=tuple, default=DECAY_RATE, help='Learning rate decay rate')
    parser.add_argument('--verb-step', type=int, default=VERB_STEP, help='#steps between two verbose prints')
    parser.add_argument('--save-epoch', type=int, default=SAVE_EPOCH, help='#epochs between two model save')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR, help='root directory of cityscapes')
    parser.add_argument('--rgb-type', type=str, default=RGB_TYPE, help='rgb type in cityscapes')
    parser.add_argument('--gt-type', type=str, default=GT_TYPE, help='gt type in cityscapes')
    parser.add_argument('--rgb-ext', type=str, default=RGB_EXT, help='rgb extension in their file names')
    parser.add_argument('--gt-ext', type=str, default=GT_EXT, help='gt extensions in their file names')
    parser.add_argument('--force-run', type=bool, default=FORCE_RUN, help='force run collection maker or not')

    flags = parser.parse_args()
    return flags


def get_image_list(data_dir, pred_dir, fold='train'):
    file_list = []
    pred_files = sorted(glob(os.path.join(pred_dir, '*.png')))
    for f in pred_files:
        city_name = os.path.basename(f).split('\'')[1].split('_')[0]
        image_id = '_'.join(os.path.basename(f).split('\'')[1].split('_')[1:])
        rgb_file = os.path.join(data_dir, 'leftImg8bit', fold, city_name, '{}_{}_leftImg8bit.png'.format(city_name, image_id))
        gt_file = os.path.join(data_dir, 'gtFine', fold, city_name, '{}_{}_gtFine_labelTrainIds.png'.format(city_name, image_id))
        file_list.append([rgb_file, gt_file, f])
    return file_list


def resize_image(img, size):
    if img.shape[2] == 4:
        # resize rgb and gt separately
        resize_img = np.zeros((size[0], size[1], 4))
        resize_img[:, :, :3] = scipy.misc.imresize(img[:, :, :3], size)
        resize_img[:, :, -1] = scipy.misc.imresize(img[:, :, -1], size, 'nearest')
        return resize_img
    elif img.shape[2] == 5:
        resize_img = np.zeros((size[0], size[1], 5))
        resize_img[:, :, :3] = scipy.misc.imresize(img[:, :, :3], size)
        resize_img[:, :, -2] = skimage.transform.resize(img[:, :, -2], size, order=0, preserve_range=True, mode='reflect')
        resize_img[:, :, -1] = skimage.transform.resize(img[:, :, -1], size, order=0, preserve_range=True, mode='reflect')
        return resize_img
    else:
        return scipy.misc.imresize(img, size)


def main(flags):
    nn_utils.set_gpu(GPU)

    # define network
    model = stn.STN(flags.num_classes, flags.tile_size, suffix=flags.model_suffix, learn_rate=flags.learning_rate,
                    decay_step=flags.decay_step, decay_rate=flags.decay_rate, epochs=flags.epochs,
                    batch_size=flags.batch_size)

    cm_train = cityscapes_reader.CollectionMakerCityscapes(flags.data_dir, flags.rgb_type, flags.gt_type, 'train',
                                                           flags.rgb_ext, flags.gt_ext, ['png', 'png'],
                                                           clc_name='{}_train'.format(flags.ds_name),
                                                           force_run=flags.force_run)

    cm_valid = cityscapes_reader.CollectionMakerCityscapes(flags.data_dir, flags.rgb_type, flags.gt_type, 'val',
                                                           flags.rgb_ext, flags.gt_ext, ['png', 'png'],
                                                           clc_name='{}_valid'.format(flags.ds_name),
                                                           force_run=flags.force_run)
    cm_train.print_meta_data()
    cm_valid.print_meta_data()

    train_pred_dir = r'/home/lab/Documents/bohao/data/deeplab_model/vis_train/raw_segmentation_results'
    valid_pred_dir = r'/home/lab/Documents/bohao/data/deeplab_model/vis/raw_segmentation_results'
    file_list_train = get_image_list(flags.data_dir, train_pred_dir, 'train')
    file_list_valid = get_image_list(flags.data_dir, valid_pred_dir, 'val')

    resize_func = lambda img: resize_image(img, flags.tile_size)
    train_init_op, valid_init_op, reader_op = DataReaderSegmentationTrainValid(
            flags.tile_size, file_list_train, file_list_valid,
            flags.batch_size, cm_train.meta_data['chan_mean'], aug_func=[reader_utils.image_flipping_hori],
            random=True, has_gt=True, gt_dim=1, include_gt=True, valid_mult=flags.val_mult, global_func=resize_func)\
        .read_op()
    feature, label, pred = reader_op
    train_init_op_valid, _, reader_op = DataReaderSegmentationTrainValid(
        flags.tile_size, file_list_valid, file_list_train,
        flags.batch_size, cm_valid.meta_data['chan_mean'], aug_func=[reader_utils.image_flipping_hori],
        random=True, has_gt=True, gt_dim=1, include_gt=True, valid_mult=flags.val_mult, global_func=resize_func) \
        .read_op()
    _, _, pred_valid = reader_op

    model.create_graph(pred, feature_valid=pred_valid, rgb=feature)

    model.compile(pred, label, flags.n_train, flags.n_valid, flags.tile_size, ersaPath.PATH['model'],
                  par_dir=flags.model_par_dir, val_mult=flags.val_mult, loss_type='xent')
    train_hook = hook.ValueSummaryHook(flags.verb_step, [model.loss, model.g_loss, model.d_loss, model.lr_op[0],
                                                         model.lr_op[1], model.lr_op[2]],
                                       value_names=['seg_loss', 'g_loss', 'd_loss', 'lr_seg', 'lr_g', 'lr_d'],
                                       print_val=[0, 1, 2])
    model_save_hook = hook.ModelSaveHook(model.get_epoch_step()*flags.save_epoch, model.ckdir)
    valid_loss_hook = hook.ValueSummaryHookIters(model.get_epoch_step(), [model.loss_xent, model.loss_iou],
                                                 value_names=['valid_loss', 'valid_mIoU'], log_time=True,
                                                 run_time=model.n_valid)
    image_hook = hook.ImageValidSummaryHook(model.input_size, model.get_epoch_step(), feature, label, model.refine,
                                            cityscapes_labels.image_summary, img_mean=cm_train.meta_data['chan_mean'])
    start_time = time.time()
    model.train(train_hooks=[train_hook, model_save_hook], valid_hooks=[valid_loss_hook, image_hook],
                train_init=[train_init_op, train_init_op_valid], valid_init=valid_init_op)
    print('Duration: {:.3f}'.format((time.time() - start_time)/3600))


if __name__ == '__main__':
    flags = read_flag()
    main(flags)
