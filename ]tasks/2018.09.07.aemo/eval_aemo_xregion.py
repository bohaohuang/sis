import os
import numpy as np
import tensorflow as tf
import uabUtilreader
import utils
import ersa_utils
import util_functions
import uabCrossValMaker
import uab_collectionFunctions
from bohaoCustom import uabMakeNetwork_UNet

gpu = 1
batch_size = 5
input_size = [572, 572]
tile_size = [5000, 5000]
util_functions.tf_warn_level(3)
model_dir = r'/hdd6/Models/aemo/aemo/UnetCrop_aemo_ft_0_xregion_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32'
ds_name = 'aemo'
img_dir, task_dir = utils.get_task_img_folder()
SAVE_DIR = os.path.join(task_dir, 'confmap_uab_{}'.format(os.path.basename(model_dir)))
ersa_utils.make_dir_if_not_exist(SAVE_DIR)
TILE_CNT = 0
TILE_NAME = ['aus10_4453x10891_gt_d255', 'aus30_13678x10766_gt_d255', 'aus50_4808x3179_gt_d255']


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
            global TILE_CNT
            ersa_utils.save_file(os.path.join(SAVE_DIR, '{}.npy'.format(TILE_NAME[TILE_CNT])), image_pred[:, :, 1])
            TILE_CNT += 1
            return util_functions.get_pred_labels(image_pred) * truth_val


# settings
blCol = uab_collectionFunctions.uabCollection(ds_name)
blCol.readMetadata()
file_list, parent_dir = blCol.getAllTileByDirAndExt([1, 2, 3])
file_list_truth, parent_dir_truth = blCol.getAllTileByDirAndExt(0)
idx, file_list = uabCrossValMaker.uabUtilGetFolds(None, file_list, 'tile')
idx_truth, file_list_truth = uabCrossValMaker.uabUtilGetFolds(None, file_list_truth, 'tile')
# use first 5 tiles for validation
file_list_valid = uabCrossValMaker.make_file_list_by_key(idx, file_list, [1, 3, 5])
file_list_valid_truth = uabCrossValMaker.make_file_list_by_key(idx_truth, file_list_truth, [1, 3, 5])
img_mean = blCol.getChannelMeans([1, 2, 3])

# make the model
# define place holder
X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = UnetModelCrop({'X':X, 'Y':y}, trainable=mode, input_size=input_size,
                      batch_size=batch_size, start_filter_num=32)
# create graph
model.create_graph('X', class_num=2)

# evaluate on tiles
model.evaluate(file_list_valid, file_list_valid_truth, parent_dir, parent_dir_truth,
               input_size, tile_size, batch_size, img_mean, model_dir, gpu,
               save_result_parent_dir='aemo/uab', ds_name=ds_name, best_model=False, load_epoch_num=30)
