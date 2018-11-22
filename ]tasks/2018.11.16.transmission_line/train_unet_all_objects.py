import os
import time
import argparse
import numpy as np
import tensorflow as tf
from functools import partial
import ersaPath
import ersa_utils
from nn import unet, hook, nn_utils
from preprocess import patchExtractor
from collection import collectionMaker, collectionEditor
from reader import dataReaderSegmentation, reader_utils

from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000

# settings
NUM_CLASS = 2
PATCH_SIZE = (572, 572)
DS_NAME = 'infrastructure'
PAR_DIR = 'infrastructure'
SUFFIX = '5objs_weight{}'
RUN_ID = 0
DECAY_STEP = 60
DECAY_RATE = 0.1
EPOCHS = 100
BATCH_SIZE = 5
VAL_MULT = 5
GPU = 0
N_TRAIN = 8000
N_VALID = 1000
VERB_STEP = 50
SAVE_EPOCH = 5
LEARN_RATE = 1e-4
POS_WEIGHT = 0.999
DATA_DIR = r'/media/ei-edl01/data/uab_datasets/infrastructure/data/Original_Tiles'


def read_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (10)')
    parser.add_argument('--ds-name', default=DS_NAME, type=str, help='dataset name')
    parser.add_argument('--patch-size', default=PATCH_SIZE, type=tuple, help='patch size 572')
    parser.add_argument('--epochs', default=EPOCHS, type=int, help='# epochs (1)')
    parser.add_argument('--num-classes', type=int, default=NUM_CLASS, help='# classes (including background)')
    parser.add_argument('--par-dir', type=str, default=PAR_DIR, help='parent directory name to save the model')
    parser.add_argument('--suffix', type=str, default=SUFFIX, help='suffix of the model name')
    parser.add_argument('--run-id', type=int, default=RUN_ID, help='a number indicates the run and control randomness')
    parser.add_argument('--n-train', type=int, default=N_TRAIN, help='# samples per epoch')
    parser.add_argument('--n-valid', type=int, default=N_VALID, help='# patches to valid')
    parser.add_argument('--val-mult', type=int, default=VAL_MULT, help='validation_bs=val_mult*train_bs')
    parser.add_argument('--GPU', type=str, default=GPU, help="GPU used for computation.")
    parser.add_argument('--decay-step', type=float, default=DECAY_STEP, help='Learning rate decay step in number of epochs.')
    parser.add_argument('--decay-rate', type=float, default=DECAY_RATE, help='Learning rate decay rate')
    parser.add_argument('--verb-step', type=int, default=VERB_STEP, help='#steps between two verbose prints')
    parser.add_argument('--save-epoch', type=int, default=SAVE_EPOCH, help='#epochs between two model save')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR, help='root directory of cityscapes')
    parser.add_argument('--learn-rate', type=float, default=LEARN_RATE, help='learning rate')
    parser.add_argument('--pos-weight', type=float, default=POS_WEIGHT, help='weight for H1 class')

    flags = parser.parse_args()
    flags.suffix = flags.suffix.format(flags.pos_weight)
    return flags


def main(flags):
    nn_utils.set_gpu(flags.GPU)
    np.random.seed(flags.run_id)
    tf.set_random_seed(flags.run_id)

    # define network
    model = unet.UNet(flags.num_classes, flags.patch_size, suffix=flags.suffix, learn_rate=flags.learn_rate,
                      decay_step=flags.decay_step, decay_rate=flags.decay_rate, epochs=flags.epochs,
                      batch_size=flags.batch_size)
    overlap = model.get_overlap()

    cm = collectionMaker.read_collection(raw_data_path=flags.data_dir,
                                         field_name='Tucson,Colwich,Clyde,Wilmington',
                                         field_id=','.join(str(i) for i in range(1, 16)),
                                         rgb_ext='RGB',
                                         gt_ext='GT',
                                         file_ext='tif,png',
                                         force_run=False,
                                         clc_name=flags.ds_name)
    gt_d255 = collectionEditor.SingleChanSwitch(cm.clc_dir, {2:0, 3:1, 4:0, 5:0, 6:0, 7:0}, ['GT', 'GT_switch'],
                                                'tower_only').run(force_run=False, file_ext='png', d_type=np.uint8, )
    cm.replace_channel(gt_d255.files, True, ['GT', 'GT_switch'])
    cm.print_meta_data()

    file_list_train = cm.load_files(field_name='Tucson,Colwich,Clyde,Wilmington',
                                    field_id=','.join(str(i) for i in range(4, 16)),
                                    field_ext='RGB,GT_switch')
    file_list_valid = cm.load_files(field_name='Tucson,Colwich,Clyde,Wilmington', field_id='1,2,3',
                                    field_ext='RGB,GT_switch')

    patch_list_train = patchExtractor.PatchExtractor(flags.patch_size,
                                                     ds_name=flags.ds_name + '_tower_only',
                                                     overlap=overlap, pad=overlap // 2). \
        run(file_list=file_list_train, file_exts=['jpg', 'png'], force_run=False).get_filelist()
    patch_list_valid = patchExtractor.PatchExtractor(flags.patch_size,
                                                     ds_name=flags.ds_name + '_tower_only',
                                                     overlap=overlap, pad=overlap // 2). \
        run(file_list=file_list_valid, file_exts=['jpg', 'png'], force_run=False).get_filelist()
    chan_mean = cm.meta_data['chan_mean']

    train_init_op, valid_init_op, reader_op = \
        dataReaderSegmentation.DataReaderSegmentationTrainValid(
            flags.patch_size, patch_list_train, patch_list_valid, batch_size=flags.batch_size,
            chan_mean=chan_mean,
            aug_func=[reader_utils.image_flipping, reader_utils.image_rotating],
            random=True, has_gt=True, gt_dim=1, include_gt=True, valid_mult=flags.val_mult).read_op()
    feature, label = reader_op

    model.create_graph(feature)
    model.compile(feature, label, flags.n_train, flags.n_valid, flags.patch_size, ersaPath.PATH['model'],
                  par_dir=flags.par_dir, loss_type='xent', pos_weight=flags.pos_weight)
    train_hook = hook.ValueSummaryHook(flags.verb_step, [model.loss, model.lr_op],
                                       value_names=['train_loss', 'learning_rate'],
                                       print_val=[0])
    model_save_hook = hook.ModelSaveHook(model.get_epoch_step() * flags.save_epoch, model.ckdir)
    valid_loss_hook = hook.ValueSummaryHookIters(model.get_epoch_step(), [model.loss_iou, model.loss_xent],
                                                 value_names=['IoU', 'valid_loss'], log_time=True,
                                                 run_time=model.n_valid)
    image_hook = hook.ImageValidSummaryHook(model.input_size, model.get_epoch_step(), feature, label,
                                            model.pred,
                                            partial(nn_utils.image_summary, label_num=flags.num_classes),
                                            img_mean=chan_mean)
    start_time = time.time()
    model.train(train_hooks=[train_hook, model_save_hook], valid_hooks=[valid_loss_hook, image_hook],
                train_init=train_init_op, valid_init=valid_init_op)
    print('Duration: {:.3f}'.format((time.time() - start_time) / 3600))


if __name__ == '__main__':
    flags = read_flag()
    main(flags)
