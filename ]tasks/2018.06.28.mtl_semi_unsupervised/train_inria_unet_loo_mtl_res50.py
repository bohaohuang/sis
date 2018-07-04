import os
import time
import argparse
import numpy as np
import tensorflow as tf
import uabRepoPaths
import uabCrossValMaker
import bohaoCustom.uabPreprocClasses as bPreproc
import uabPreprocClasses
import uab_collectionFunctions
import uab_DataHandlerFunctions
from bohaoCustom import uabDataReader
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
MODEL_NAME = 'inria_loo_mtl_res50_{}_{}'
SFN = 32
LEAVE_CITY = 0
PRED_FILE_DIR = r'/media/ei-edl01/user/bh163/tasks/2018.06.28.mtl_semi_unsupervised'


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

    flags = parser.parse_args()
    flags.input_size = (flags.input_size, flags.input_size)
    flags.tile_size = (flags.tile_size, flags.tile_size)
    flags.model_name = flags.model_name.format(flags.leave_city, flags.run_id)
    return flags


def main(flags):
    flags.pred_file_dir = os.path.join(flags.pred_file_dir,
                                       'res50_pred_building_binary_{}.npy'.format(flags.leave_city))

    # make network
    # define place holder
    X = tf.placeholder(tf.float32, shape=[None, flags.input_size[0], flags.input_size[1], 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, flags.input_size[0], flags.input_size[1], 1], name='y')
    y2 = tf.placeholder(tf.float32, shape=[None, 1], name='y2')
    mode = tf.placeholder(tf.bool, name='mode')
    model = uabMakeNetwork_UNet.UnetModelPredict({'X':X, 'Y':y, 'Y2':y2},
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
