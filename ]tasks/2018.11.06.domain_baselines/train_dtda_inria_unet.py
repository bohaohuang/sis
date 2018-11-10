import os
import time
import argparse
import numpy as np
import tensorflow as tf
import ersa_utils
import uabDataReader
import uabRepoPaths
import uabCrossValMaker
import bohaoCustom.uabPreprocClasses as bPreproc
import uabPreprocClasses
import uab_collectionFunctions
import uab_DataHandlerFunctions
from bohaoCustom import uabMakeNetwork_UNet

RUN_ID = 0
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
INPUT_SIZE = 572
TILE_SIZE = 5000
EPOCHS = 100
NUM_CLASS = 2
N_TRAIN = 8000
N_VALID = 1000
GPU = 0
DECAY_STEP = 60
DECAY_RATE = 0.1
MODEL_NAME = 'inria_aug_leave_{}_{}'
SFN = 32
LEAVE_CITY = 0
LAM = 0.1
MODEL_DIR = r'/hdd6//Models/Inria_Domain_LOO/UnetCrop_inria_aug_leave_{}_0_PS(572, 572)_BS5_' \
            r'EP100_LR0.0001_DS60_DR0.1_SFN32'
CITY_LIST = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
WEIGHT_DIR = r'/media/ei-edl01/user/bh163/tasks/2018.11.06.domain_baselines/dtda/{}'


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
    parser.add_argument('--lam', type=float, default=LAM, help='lambda in the cost function')
    parser.add_argument('--model-dir', type=str, default=MODEL_DIR, help='pretrained model dir')
    parser.add_argument('--weight-dir', type=str, default=WEIGHT_DIR, help='path to save weights')

    flags = parser.parse_args()
    flags.input_size = (flags.input_size, flags.input_size)
    flags.tile_size = (flags.tile_size, flags.tile_size)
    flags.model_dir = flags.model_dir.format(flags.leave_city)
    flags.weight_dir = flags.weight_dir.format(CITY_LIST[flags.leave_city])
    flags.model_name = flags.model_name.format(flags.leave_city, flags.run_id)
    return flags


def get_pretrained_weights(flags):
    save_name = os.path.join(flags.weight_dir, 'weight.pkl')

    if not os.path.exists(save_name):
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
        train_vars = [v for v in tf.trainable_variables()]

        weight_dict = dict()

        with tf.Session() as sess:
            model.load(flags.model_dir, sess, epoch=95)
            for v in train_vars:
                theta = sess.run(v)
                weight_dict[v.name] = theta
        ersa_utils.save_file(save_name, weight_dict)
    else:
        weight_dict = ersa_utils.load_file(save_name)

    tf.reset_default_graph()
    return weight_dict


def main(flags, weight_dict):
    # make network
    # define place holder
    X = tf.placeholder(tf.float32, shape=[None, flags.input_size[0], flags.input_size[1], 3], name='X')
    Z = tf.placeholder(tf.float32, shape=[None, flags.input_size[0], flags.input_size[1], 3], name='Z')
    y = tf.placeholder(tf.int32, shape=[None, flags.input_size[0], flags.input_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')
    model = uabMakeNetwork_UNet.UnetModelDTDA({'X': X, 'Z': Z, 'Y': y},
                                              trainable=mode,
                                              model_name=flags.model_name,
                                              input_size=flags.input_size,
                                              batch_size=flags.batch_size,
                                              learn_rate=flags.learning_rate,
                                              decay_step=flags.decay_step,
                                              decay_rate=flags.decay_rate,
                                              epochs=flags.epochs,
                                              start_filter_num=flags.sfn)
    model.create_graph('X', 'Z', class_num=flags.num_classes)

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
    idx_city, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'city')
    idx_tile, _ = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'force_tile')
    idx = [j * 10 + i for i, j in zip(idx_city, idx_tile)]
    # use first city for validation
    filter_source = []
    filter_target = []
    filter_valid = []
    for i in range(5):
        for j in range(1, 37):
            if i != flags.leave_city and j > 5:
                filter_source.append(j * 10 + i)
            elif i == flags.leave_city and j > 5:
                filter_target.append(j * 10 + i)
            elif i == flags.leave_city and j <= 5:
                filter_valid.append(j * 10 + i)
    # use first city for validation
    file_list_source = uabCrossValMaker.make_file_list_by_key(idx, file_list, filter_source)
    file_list_target = uabCrossValMaker.make_file_list_by_key(idx, file_list, filter_target)
    file_list_valid = uabCrossValMaker.make_file_list_by_key(idx, file_list, filter_valid)

    with tf.name_scope('image_loader'):
        # GT has no mean to subtract, append a 0 for block mean
        dataReader_source = uabDataReader.ImageLabelReader([3], [0, 1, 2], patchDir, file_list_source, flags.input_size,
                                                          flags.tile_size,
                                                          flags.batch_size, dataAug='flip,rotate',
                                                          block_mean=np.append([0], img_mean))
        # no augmentation needed for validation
        dataReader_target = uabDataReader.ImageLabelReader([3], [0, 1, 2], patchDir, file_list_target, flags.input_size,
                                                          flags.tile_size,
                                                          flags.batch_size, dataAug='flip,rotate',
                                                          block_mean=np.append([0], img_mean))

        dataReader_valid = uabDataReader.ImageLabelReader([3], [0, 1, 2], patchDir, file_list_valid, flags.input_size,
                                                           flags.tile_size,
                                                           flags.batch_size, dataAug='flip,rotate',
                                                           block_mean=np.append([0], img_mean))

    # train
    start_time = time.time()

    model.train_config('X', 'Y', 'Z', flags.n_train, flags.n_valid, flags.input_size, uabRepoPaths.modelPath,
                       loss_type='xent', par_dir='domain_baseline', lam=flags.lam)
    model.load_source_weights(weight_dict)
    model.run(train_reader_source=dataReader_source,
              train_reader_target=dataReader_target,
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
    weight_dict = get_pretrained_weights(flags)
    main(flags, weight_dict)
