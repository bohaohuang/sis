import time
import argparse
import numpy as np
import tensorflow as tf
import ersaPath
from nn import unet, hook, nn_utils
from preprocess import patchExtractor
from reader import dataReaderSegmentation, reader_utils
from collection import collectionMaker

# settings
NUM_CLASS = 2
PATCH_SIZE = (572, 572)
TILE_SIZE = (5000, 5000)
DS_NAME = 'aemo_hist'
PAR_DIR = 'aemo/new4'
FROM_SCRATCH = False
DECAY_STEP = 30
DECAY_RATE = 0.1
EPOCHS = 80
BATCH_SIZE = 5
VAL_MULT = 5
START_LAYER = 6
GPU = 0
N_TRAIN = 785
N_VALID = 395
VERB_STEP = 50
SAVE_EPOCH = 20
MODEL_DIR = r'/work/bh163/misc/unet_reweight'
DATA_DIR = r'/work/bh163/data/aemo_hist'


def read_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (10)')
    parser.add_argument('--ds-name', default=DS_NAME, type=str, help='dataset name')
    parser.add_argument('--tile-size', default=TILE_SIZE, type=tuple, help='tile size 5000')
    parser.add_argument('--patch-size', default=PATCH_SIZE, type=tuple, help='patch size 572')
    parser.add_argument('--epochs', default=EPOCHS, type=int, help='# epochs (1)')
    parser.add_argument('--num-classes', type=int, default=NUM_CLASS, help='# classes (including background)')
    parser.add_argument('--par-dir', type=str, default=PAR_DIR, help='parent directory name to save the model')
    parser.add_argument('--n-train', type=int, default=N_TRAIN, help='# samples per epoch')
    parser.add_argument('--n-valid', type=int, default=N_VALID, help='# patches to valid')
    parser.add_argument('--val-mult', type=int, default=VAL_MULT, help='validation_bs=val_mult*train_bs')
    parser.add_argument('--GPU', type=str, default=GPU, help="GPU used for computation.")
    parser.add_argument('--decay-step', type=float, default=DECAY_STEP, help='Learning rate decay step in number of epochs.')
    parser.add_argument('--decay-rate', type=float, default=DECAY_RATE, help='Learning rate decay rate')
    parser.add_argument('--verb-step', type=int, default=VERB_STEP, help='#steps between two verbose prints')
    parser.add_argument('--save-epoch', type=int, default=SAVE_EPOCH, help='#epochs between two model save')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR, help='root directory of cityscapes')
    parser.add_argument('--from-scratch', type=bool, default=FROM_SCRATCH, help='from scratch or not')
    parser.add_argument('--start-layer', type=int, default=START_LAYER, help='start layer to unfreeze')
    parser.add_argument('--model-dir', type=str, default=MODEL_DIR, help='pretrained model directory')

    flags = parser.parse_args()
    return flags


def main(flags):
    nn_utils.set_gpu(flags.GPU)
    if flags.start_layer >= 10:
        suffix_base = 'aemo'
    else:
        suffix_base = 'aemo_up{}'.format(flags.start_layer)
    if flags.from_scratch:
        suffix_base += '_scratch'
    for lr in [1e-3, 1e-4]:
        for run_id in range(4):
            suffix = '{}_{}'.format(suffix_base, run_id)
            tf.reset_default_graph()

            np.random.seed(run_id)
            tf.set_random_seed(run_id)

            # define network
            model = unet.UNet(flags.num_classes, flags.patch_size, suffix=suffix, learn_rate=lr,
                              decay_step=flags.decay_step, decay_rate=flags.decay_rate,
                              epochs=flags.epochs, batch_size=flags.batch_size)
            overlap = model.get_overlap()

            cm = collectionMaker.read_collection(raw_data_path=flags.data_dir,
                                                 field_name='aus10,aus30,aus50',
                                                 field_id='',
                                                 rgb_ext='.*rgb',
                                                 gt_ext='.*gt',
                                                 file_ext='tif',
                                                 force_run=False,
                                                 clc_name=flags.ds_name)
            cm.print_meta_data()

            file_list_train = cm.load_files(field_name='aus10,aus30', field_id='', field_ext='.*rgb,.*gt')
            file_list_valid = cm.load_files(field_name='aus50', field_id='', field_ext='.*rgb,.*gt')

            patch_list_train = patchExtractor.PatchExtractor(flags.patch_size, flags.tile_size,
                                                             flags.ds_name + '_train_hist',
                                                             overlap, overlap // 2). \
                run(file_list=file_list_train, file_exts=['jpg', 'png'], force_run=False).get_filelist()
            patch_list_valid = patchExtractor.PatchExtractor(flags.patch_size, flags.tile_size,
                                                             flags.ds_name + '_valid_hist',
                                                             overlap, overlap // 2). \
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
            if flags.start_layer >= 10:
                model.compile(feature, label, flags.n_train, flags.n_valid, flags.patch_size, ersaPath.PATH['model'],
                              par_dir=flags.par_dir, loss_type='xent')
            else:
                model.compile(feature, label, flags.n_train, flags.n_valid, flags.patch_size, ersaPath.PATH['model'],
                              par_dir=flags.par_dir, loss_type='xent',
                              train_var_filter=['layerup{}'.format(i) for i in range(flags.start_layer, 10)])
            train_hook = hook.ValueSummaryHook(flags.verb_step, [model.loss, model.lr_op],
                                               value_names=['train_loss', 'learning_rate'],
                                               print_val=[0])
            model_save_hook = hook.ModelSaveHook(model.get_epoch_step() * flags.save_epoch, model.ckdir)
            valid_loss_hook = hook.ValueSummaryHookIters(model.get_epoch_step(), [model.loss_xent, model.loss_iou],
                                                         value_names=['valid_loss', 'IoU'], log_time=True,
                                                         run_time=model.n_valid)
            image_hook = hook.ImageValidSummaryHook(model.input_size, model.get_epoch_step(), feature, label,
                                                    model.pred,
                                                    nn_utils.image_summary, img_mean=chan_mean)
            start_time = time.time()
            if not flags.from_scratch:
                model.load(flags.model_dir)
            model.train(train_hooks=[train_hook, model_save_hook], valid_hooks=[valid_loss_hook, image_hook],
                        train_init=train_init_op, valid_init=valid_init_op)
            print('Duration: {:.3f}'.format((time.time() - start_time) / 3600))


if __name__ == '__main__':
    flags = read_flag()
    main(flags)
