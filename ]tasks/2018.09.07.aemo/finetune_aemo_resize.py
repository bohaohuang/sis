import os
import time
import argparse
import numpy as np
import tensorflow as tf
import ersaPath
import ersa_utils
from nn import unet, hook, nn_utils
from collection import collectionMaker
from reader import dataReaderSegmentation, reader_utils

# settings
NUM_CLASS = 2
PATCH_SIZE = (572, 572)
TILE_SIZE = (5000, 5000)
FROM_SCRATCH = False
PAR_DIR = 'aemo_resize_new_loss'
DECAY_STEP = 30
DECAY_RATE = 0.1
EPOCHS = 80
BATCH_SIZE = 5
VAL_MULT = 5
START_LAYER = '10'
GPU = 1
N_TRAIN = 785
N_VALID = 395
VERB_STEP = 50
SAVE_EPOCH = 20
LEARN_RATE = '1e-3'
MODEL_DIR = r'/hdd6/Models/spca/UnetCropWeighted_GridChipPretrained6Weighted4_PS(572, 572)_BS5_' \
            r'EP100_LR0.0001_DS50_DR0.1_SFN32'
DATA_DIR = r'/hdd/ersa/patch_extractor/aemo_resize'


def read_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (10)')
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
    parser.add_argument('--start-layer', type=str, default=START_LAYER, help='start layer to unfreeze')
    parser.add_argument('--model-dir', type=str, default=MODEL_DIR, help='pretrained model directory')
    parser.add_argument('--learn-rate', type=str, default=LEARN_RATE, help='learning rate')

    flags = parser.parse_args()
    flags.par_dir = 'aemo/' + flags.par_dir
    flags.learn_rate = ersa_utils.str2list(flags.learn_rate, d_type=float)
    flags.start_layer = ersa_utils.str2list(flags.start_layer, d_type=int)
    return flags


def main(flags):
    nn_utils.set_gpu(flags.GPU)
    for start_layer in flags.start_layer:
        if start_layer >= 10:
            suffix_base = 'aemo'
        else:
            suffix_base = 'aemo_up{}'.format(start_layer)
        if flags.from_scratch:
            suffix_base += '_scratch'
        for lr in flags.learn_rate:
            for run_id in range(1):
                suffix = '{}_{}'.format(suffix_base, run_id)
                tf.reset_default_graph()

                np.random.seed(run_id)
                tf.set_random_seed(run_id)

                # define network
                model = unet.UNet(flags.num_classes, flags.patch_size, suffix=suffix, learn_rate=lr,
                                  decay_step=flags.decay_step, decay_rate=flags.decay_rate,
                                  epochs=flags.epochs, batch_size=flags.batch_size)

                file_list = os.path.join(flags.data_dir, 'file_list.txt')
                lines = ersa_utils.load_file(file_list)

                patch_list_train = []
                patch_list_valid = []
                train_tile_names = ['aus10', 'aus30']
                valid_tile_names = ['aus50']

                for line in lines:
                    tile_name = os.path.basename(line.split(' ')[0]).split('_')[0].strip()
                    if tile_name in train_tile_names:
                        patch_list_train.append(line.strip().split(' '))
                    elif tile_name in valid_tile_names:
                        patch_list_valid.append(line.strip().split(' '))
                    else:
                        raise ValueError

                cm = collectionMaker.read_collection('aemo_align')
                chan_mean = cm.meta_data['chan_mean']

                train_init_op, valid_init_op, reader_op = \
                    dataReaderSegmentation.DataReaderSegmentationTrainValid(
                        flags.patch_size, patch_list_train, patch_list_valid, batch_size=flags.batch_size,
                        chan_mean=chan_mean,
                        aug_func=[reader_utils.image_flipping, reader_utils.image_rotating],
                        random=True, has_gt=True, gt_dim=1, include_gt=True, valid_mult=flags.val_mult).read_op()
                feature, label = reader_op

                model.create_graph(feature)
                if start_layer >= 10:
                    model.compile(feature, label, flags.n_train, flags.n_valid, flags.patch_size, ersaPath.PATH['model'],
                                  par_dir=flags.par_dir, loss_type='xent')
                else:
                    model.compile(feature, label, flags.n_train, flags.n_valid, flags.patch_size, ersaPath.PATH['model'],
                                  par_dir=flags.par_dir, loss_type='xent',
                                  train_var_filter=['layerup{}'.format(i) for i in range(start_layer, 10)])
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
