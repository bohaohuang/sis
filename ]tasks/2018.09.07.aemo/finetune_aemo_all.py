import time
import argparse
import numpy as np
import tensorflow as tf
import uabDataReader
import uabRepoPaths
import uabCrossValMaker
import uab_collectionFunctions
import uab_DataHandlerFunctions
from bohaoCustom import uabMakeNetwork_UNet

RUN_ID = 0
BATCH_SIZE = 5
LEARNING_RATE = 1e-3
INPUT_SIZE = 572
TILE_SIZE = 5000
EPOCHS = 80
NUM_CLASS = 2
N_TRAIN = 785
N_VALID = 395
GPU = 1
DECAY_STEP = 30
DECAY_RATE = 0.1
START_LAYER = 10
MODEL_NAME = 'aemo_{}_all'
DS_NAME = 'aemo_comb'
SFN = 32
MODEL_DIR = r'/hdd6/Models/spca/UnetCropWeighted_GridChipPretrained6Weighted4_PS(572, 572)_BS5_' \
            r'EP100_LR0.0001_DS50_DR0.1_SFN32'


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
    parser.add_argument('--model-dir', type=str, default=MODEL_DIR, help='pretrained model directory')
    parser.add_argument('--ds-name', type=str, default=DS_NAME, help='name of the dataset')
    parser.add_argument('--start-layer', type=int, default=START_LAYER, help='start layer to unfreeze')

    flags = parser.parse_args()
    flags.input_size = (flags.input_size, flags.input_size)
    flags.tile_size = (flags.tile_size, flags.tile_size)
    flags.model_name = flags.model_name.format(flags.run_id)
    return flags


def main(flags):
    np.random.seed(int(flags.run_id))
    tf.set_random_seed(int(flags.run_id))

    if flags.start_layer >= 10:
        pass
    else:
        flags.model_name += '_up{}'.format(flags.start_layer)

    # make network
    # define place holder
    X = tf.placeholder(tf.float32, shape=[None, flags.input_size[0], flags.input_size[1], 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, flags.input_size[0], flags.input_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')
    model = uabMakeNetwork_UNet.UnetModelCrop({'X': X, 'Y': y},
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
    blCol = uab_collectionFunctions.uabCollection(flags.ds_name)
    blCol.readMetadata()
    img_mean = blCol.getChannelMeans([1, 2, 3])  # get mean of rgb info

    # extract patches
    extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 3],
                                                    cSize=flags.input_size,
                                                    numPixOverlap=int(model.get_overlap()),
                                                    extSave=['png', 'jpg', 'jpg', 'jpg'],
                                                    isTrain=True,
                                                    gtInd=0,
                                                    pad=int(model.get_overlap()//2))
    patchDir = extrObj.run(blCol)

    # make data reader
    # use first 5 tiles for validation
    idx, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'tile')
    file_list = uabCrossValMaker.make_file_list_by_key(idx, file_list, [0, 1, 2, 3, 4, 5])
    file_list = np.random.permutation(file_list)
    file_list_train = file_list[:-100]
    file_list_valid = file_list[-100:]

    with tf.name_scope('image_loader'):
        # GT has no mean to subtract, append a 0 for block mean
        dataReader_train = uabDataReader.ImageLabelReader([0], [1, 2, 3], patchDir, file_list_train, flags.input_size,
                                                          flags.tile_size,
                                                          flags.batch_size, dataAug='flip,rotate',
                                                          block_mean=np.append([0], img_mean))
        # no augmentation needed for validation
        dataReader_valid = uabDataReader.ImageLabelReader([0], [1, 2, 3], patchDir, file_list_valid, flags.input_size,
                                                          flags.tile_size,
                                                          flags.batch_size, dataAug=' ', block_mean=np.append([0], img_mean))

    # train
    start_time = time.time()

    if flags.start_layer >= 10:
        model.train_config('X', 'Y', flags.n_train, flags.n_valid, flags.input_size, uabRepoPaths.modelPath,
                           loss_type='xent', par_dir='aemo/{}'.format(flags.ds_name))
    else:
        model.train_config('X', 'Y', flags.n_train, flags.n_valid, flags.input_size, uabRepoPaths.modelPath,
                           loss_type='xent', par_dir='aemo/{}'.format(flags.ds_name),
                           train_var_filter=['layerup{}'.format(i) for i in range(flags.start_layer, 10)])
    model.run(train_reader=dataReader_train,
              valid_reader=dataReader_valid,
              pretrained_model_dir=flags.model_dir,   # train from scratch, no need to load pre-trained model
              isTrain=True,
              img_mean=img_mean,
              verb_step=100,                        # print a message every 100 step(sample)
              save_epoch=5,                         # save the model every 5 epochs
              gpu=GPU,
              tile_size=flags.tile_size,
              patch_size=flags.input_size)

    duration = time.time() - start_time
    print('duration {:.2f} hours'.format(duration/60/60))


if __name__ == '__main__':
    flags = read_flag()
    main(flags)
