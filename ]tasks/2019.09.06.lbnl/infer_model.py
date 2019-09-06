import os
import imageio
import numpy as np
import tensorflow as tf
import uabUtilreader
import util_functions
import uabDataReader
from glob import glob
from natsort import natsorted
from bohaoCustom import uabMakeNetwork_UNet

gpu = 1
batch_size = 5
input_size = [572, 572]
tile_size = [2500, 2500]
util_functions.tf_warn_level(3)
XFOLD = 2
PRED_DIR = r''
model_dir = r'/hdd6/Models/LBNL/UnetModelCrop_LBNL_{}_0.0001_0.0001_5_AEMO_SD_1_PS(572, 572)_BS4_' \
            r'EP150_LR0.0001_DS100_DR0.1_SFN32'.format(XFOLD)
EPOCH_NUM = 145


class UnetModelCrop(uabMakeNetwork_UNet.UnetModelCrop):
    def run(self, train_reader=None, valid_reader=None, test_reader=None, pretrained_model_dir=None, layers2load=None,
            isTrain=False, img_mean=np.array((0, 0, 0), dtype=np.float32), verb_step=100, save_epoch=5, gpu=None,
            tile_size=(5000, 5000), patch_size=(572, 572), truth_val=1, continue_dir=None, load_epoch_num=None,
            valid_iou=False, best_model=True):
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
                    self.train('X', 'Y', self.n_train, sess, train_summary_writer,
                               n_valid=self.n_valid, train_reader=train_reader, valid_reader=valid_reader,
                               image_summary=util_functions.image_summary, img_mean=img_mean,
                               verb_step=verb_step, save_epoch=save_epoch, continue_dir=continue_dir, valid_iou=valid_iou)
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
                self.load(pretrained_model_dir, sess, epoch=load_epoch_num, best_model=best_model)
                self.model_name = pretrained_model_dir.split('/')[-1]
                result = self.test('X', sess, test_reader)
            image_pred = uabUtilreader.un_patchify_shrink(result,
                                                          [tile_size[0] + pad, tile_size[1] + pad],
                                                          tile_size,
                                                          patch_size,
                                                          [patch_size[0] - pad, patch_size[1] - pad],
                                                          overlap=pad)
            return util_functions.get_pred_labels(image_pred) * truth_val


# settings
img_mean = np.array([54.29754073358309, 43.75329986786511, 36.900178842746556])

# make the model
# define place holder
X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = UnetModelCrop({'X': X, 'Y': y}, trainable=mode, input_size=input_size,
                      batch_size=batch_size, start_filter_num=32)
# create graph
model.create_graph('X', class_num=2)

for YEAR in [15, 16, 17]:
    data_dir = r'/hdd/lbnl/SDall{}/data/Original_Tiles'.format(YEAR)
    file_list = natsorted(glob(os.path.join(data_dir, '*.tif')))
    for img_file in file_list:
        file_name = os.path.splitext(os.path.basename(img_file))[0]
        print('Evaluating {} at year 20{}...'.format(file_name, YEAR))

        # prepare the reader
        reader = uabDataReader.ImageLabelReader(gtInds=[0],
                                                dataInds=[0],
                                                nChannels=3,
                                                parentDir=data_dir,
                                                chipFiles=[[os.path.basename(img_file)]],
                                                chip_size=input_size,
                                                tile_size=tile_size,
                                                batchSize=batch_size,
                                                block_mean=img_mean,
                                                overlap=model.get_overlap(),
                                                padding=np.array((model.get_overlap() / 2, model.get_overlap() / 2)),
                                                isTrain=False)
        rManager = reader.readManager

        # run the model
        pred = model.run(pretrained_model_dir=model_dir,
                         test_reader=rManager,
                         tile_size=tile_size,
                         patch_size=input_size,
                         gpu=gpu, load_epoch_num=EPOCH_NUM, best_model=False)
        save_dir = os.path.join(PRED_DIR, str(YEAR))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        imageio.imsave(os.path.join(save_dir, '{}.png'.format(file_name)), pred)
