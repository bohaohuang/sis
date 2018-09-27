import time
import argparse
import scipy.misc
import numpy as np
import ersaPath
from nn import deeplab, hook, nn_utils
from reader import dataReaderSegmentation, reader_utils
import cityscapes_reader, cityscapes_labels


# define parameters
BATCH_SIZE = 1
DS_NAME = 'cityscapes'
LEARNING_RATE = 1e-5
TILE_SIZE = (512, 1024)
EPOCHS = 30
NUM_CLASS = 19
PAR_DIR = DS_NAME
SUFFIX = 'test'
N_TRAIN = 2995
N_VALID = 500
VAL_MULT = 5
GPU = 0
DECAY_STEP = 15
DECAY_RATE = 0.1
VERB_STEP = 100
SAVE_EPOCH = 5
DATA_DIR = r'/hdd/cityscapes'
RGB_TYPE = 'leftImg8bit'
GT_TYPE = 'gtFine'
RGB_EXT = RGB_TYPE
GT_EXT = 'labelTrainIds'
FORCE_RUN = False


def read_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (10)')
    parser.add_argument('--ds-name', default=DS_NAME, type=str, help='dataset name')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE, help='learning rate (1e-3)')
    parser.add_argument('--tile-size', default=TILE_SIZE, type=tuple, help='tile size 5000')
    parser.add_argument('--epochs', default=EPOCHS, type=int, help='# epochs (1)')
    parser.add_argument('--num-classes', type=int, default=NUM_CLASS, help='# classes (including background)')
    parser.add_argument('--model-par-dir', type=str, default=PAR_DIR, help='parent directory name to save the model')
    parser.add_argument('--model-suffix', type=str, default=SUFFIX, help='suffix in the model directory')
    parser.add_argument('--n-train', type=int, default=N_TRAIN, help='# samples per epoch')
    parser.add_argument('--n-valid', type=int, default=N_VALID, help='# patches to valid')
    parser.add_argument('--val-mult', type=int, default=VAL_MULT, help='validation_bs=val_mult*train_bs')
    parser.add_argument('--GPU', type=str, default=GPU, help="GPU used for computation.")
    parser.add_argument('--decay-step', type=float, default=DECAY_STEP, help='Learning rate decay step in number of epochs.')
    parser.add_argument('--decay-rate', type=float, default=DECAY_RATE, help='Learning rate decay rate')
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


def resize_image(img, size):
    if img.shape[2] == 4:
        # resize rgb and gt separately
        resize_img = np.zeros((size[0], size[1], 4))
        resize_img[:, :, :3] = scipy.misc.imresize(img[:, :, :3], size)
        resize_img[:, :, -1] = scipy.misc.imresize(img[:, :, -1], size, 'nearest')
        return resize_img
    else:
        return scipy.misc.imresize(img, size)


def main(flags):
    nn_utils.set_gpu(GPU)

    # define network
    model = deeplab.DeepLab(flags.num_classes, flags.tile_size, suffix=flags.model_suffix, learn_rate=flags.learning_rate,
                            decay_step=flags.decay_step, decay_rate=flags.decay_rate,
                            epochs=flags.epochs, batch_size=flags.batch_size)

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

    resize_func = lambda img: resize_image(img, flags.tile_size)
    train_init_op, valid_init_op, reader_op = dataReaderSegmentation.DataReaderSegmentationTrainValid(
            flags.tile_size, cm_train.meta_data['file_list'], cm_valid.meta_data['file_list'],
            flags.batch_size, cm_train.meta_data['chan_mean'], aug_func=[reader_utils.image_flipping_hori],
            random=True, has_gt=True, gt_dim=1, include_gt=True, valid_mult=flags.val_mult, global_func=resize_func)\
        .read_op()
    feature, label = reader_op

    model.create_graph(feature)
    model.compile(feature, label, flags.n_train, flags.n_valid, flags.tile_size, ersaPath.PATH['model'],
                  par_dir=flags.model_par_dir, val_mult=flags.val_mult, loss_type='xent')
    train_hook = hook.ValueSummaryHook(flags.verb_step, [model.loss, model.lr_op],
                                       value_names=['train_loss', 'learning_rate'], print_val=[0])
    model_save_hook = hook.ModelSaveHook(model.get_epoch_step()*flags.save_epoch, model.ckdir)
    valid_loss_hook = hook.ValueSummaryHook(model.get_epoch_step(), [model.loss],
                                            value_names=['valid_loss'], log_time=True,
                                            run_time=model.n_valid)
    image_hook = hook.ImageValidSummaryHook(model.get_epoch_step(), model.valid_images, feature, label, model.pred,
                                            cityscapes_labels.image_summary, img_mean=cm_train.meta_data['chan_mean'])
    start_time = time.time()
    model.train(train_hooks=[train_hook, model_save_hook], valid_hooks=[valid_loss_hook, image_hook],
               train_init=train_init_op, valid_init=valid_init_op)
    print('Duration: {:.3f}'.format((time.time() - start_time)/3600))


if __name__ == '__main__':
    flags = read_flag()
    main(flags)
