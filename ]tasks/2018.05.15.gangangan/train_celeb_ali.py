import time
import argparse
import numpy as np
import tensorflow as tf
import uabDataReader
import uabRepoPaths
from bohaoCustom import uabMakeNetwork_ALI


class ImageLabelReader_celeb(uabDataReader.ImageLabelReader):
    def __init__(self, data_dir, batch_size, patch_size, is_train=True):
        self.isQueue = 0
        self.readManager = self.readFromDiskIteratorTrain(data_dir, batch_size, patch_size, is_train)

    def readFromDiskIteratorTrain(self, data_dir, batch_size, patch_size, is_train):
        import os
        import imageio
        import scipy.misc

        def read_img(data_dir, img_id):
            img = imageio.imread(os.path.join(data_dir, '{:06}.jpg'.format(img_id + 1)))
            return img

        n_sample = 202599

        idx = np.random.permutation(n_sample)
        if is_train:
            idx = idx[:200000]
        else:
            idx = idx[200000:]
        while True:
            image_batch = np.zeros((batch_size, patch_size[0], patch_size[1], 3)).astype(np.float32)
            for cnt, img_id in enumerate(idx):
                block = read_img(data_dir, img_id)
                image_batch[cnt % batch_size, :, :, :] = scipy.misc.imresize(block, patch_size)
                if ((cnt + 1) % batch_size == 0):
                    yield image_batch, None


RUN_ID = 0
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
INPUT_SIZE = 64
TILE_SIZE = 5000
EPOCHS = 25
NUM_CLASS = 3
N_TRAIN = 200000
N_VALID = 2599
GPU = 1
DECAY_STEP = 25
DECAY_RATE = 0.1
MODEL_NAME = 'celeb_z{}_lrm{}_tanh'
SFN = 64
Z_DIM = 100
LR_MULT = 1
DATA_DIR = r'/home/lab/Documents/bohao/data/data/celebA'


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
    parser.add_argument('--z-dim', type=int, default=Z_DIM, help='dimension of latent variable')
    parser.add_argument('--lr-mult', type=int, default=LR_MULT, help='Multifactor of G and D LR')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR, help='data dir to celeb dataset')

    flags = parser.parse_args()
    flags.input_size = (flags.input_size, flags.input_size)
    flags.tile_size = (flags.tile_size, flags.tile_size)
    flags.model_name = flags.model_name.format(flags.z_dim, flags.lr_mult)
    return flags


def main(flags):
    # make network
    # define place holder
    X = tf.placeholder(tf.float32, shape=[None, flags.input_size[0], flags.input_size[1], 3], name='X')
    z = tf.placeholder(tf.float32, shape=[None, 1, 1, flags.z_dim], name='z')
    mode = tf.placeholder(tf.bool, name='mode')
    model = uabMakeNetwork_ALI.ALI({'X': X, 'Z': z},
                                   trainable=mode,
                                   model_name=flags.model_name,
                                   input_size=flags.input_size,
                                   batch_size=flags.batch_size,
                                   learn_rate=flags.learning_rate,
                                   decay_step=flags.decay_step,
                                   decay_rate=flags.decay_rate,
                                   epochs=flags.epochs,
                                   start_filter_num=flags.sfn,
                                   z_dim=flags.z_dim,
                                   lr_mult=flags.lr_mult,
                                   raw_marginal=None)
    model.create_graph('X', class_num=flags.num_classes, reduce_dim=False)

    # prepare data
    dataReader_train = ImageLabelReader_celeb(flags.data_dir, flags.batch_size,
                                              (flags.input_size[0], flags.input_size[1]), True)
    dataReader_valid = ImageLabelReader_celeb(flags.data_dir, flags.batch_size,
                                              (flags.input_size[0], flags.input_size[1]), False)
    img_mean = np.zeros(3)

    # train
    start_time = time.time()

    model.train_config('X', 'Z', flags.n_train, flags.n_valid, flags.input_size, uabRepoPaths.modelPath,
                       par_dir='ALI_CELEB')
    model.run(train_reader=dataReader_train,
              valid_reader=dataReader_valid,
              pretrained_model_dir=None,
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
