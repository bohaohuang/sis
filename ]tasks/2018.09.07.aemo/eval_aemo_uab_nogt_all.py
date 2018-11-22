import os
import numpy as np
import tensorflow as tf
from glob import glob
import uabUtilreader
import utils
import ersa_utils
import util_functions
import uabDataReader
import uabCrossValMaker
import uab_collectionFunctions
from visualize import visualize_utils
from bohaoCustom import uabMakeNetwork_UNet

gpu = 0
batch_size = 5
input_size = [572, 572]
tile_size = [5000, 5000]
util_functions.tf_warn_level(3)
model_dir = r'/hdd6/Models/aemo/aemo_comb/UnetCrop_aemo_0_all_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32'
ds_name = 'aemo_comb'
img_dir, task_dir = utils.get_task_img_folder()


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
            return util_functions.get_pred_labels(image_pred) * truth_val, image_pred[:, :, 1]


def get_file_list(root_dir):
    for i in range(1, 6):
        if i != 2:
            for p, s, f in os.walk(root_dir.format(i)):
                file_list = [os.path.basename(x) for x in f]
                yield file_list, p
        else:
            for j in range(1, 3):
                for p, s, f in os.walk(os.path.join(root_dir.format(i), str(j))):
                    file_list = [os.path.join(x) for x in f]
                    yield file_list, p

def get_blank_regions(img):
    img_tmp = img.astype(np.float32)
    img_tmp = np.sum(img_tmp, axis=2)
    blank_mask = (img_tmp < 0.1).astype(np.int)
    return blank_mask


# settings
def eval_tiles():
    blCol = uab_collectionFunctions.uabCollection(ds_name)
    blCol.readMetadata()
    file_list, parent_dir = blCol.getAllTileByDirAndExt([1, 2, 3])
    file_list_truth, parent_dir_truth = blCol.getAllTileByDirAndExt(0)
    idx, file_list = uabCrossValMaker.uabUtilGetFolds(None, file_list, 'tile')
    idx_truth, file_list_truth = uabCrossValMaker.uabUtilGetFolds(None, file_list_truth, 'tile')
    # use first 5 tiles for validation
    file_list_valid = uabCrossValMaker.make_file_list_by_key(idx, file_list, [0, 1, 2, 3, 4, 5])
    file_list_valid_truth = uabCrossValMaker.make_file_list_by_key(idx_truth, file_list_truth, [0, 1, 2, 3, 4, 5])
    img_mean = blCol.getChannelMeans([1, 2, 3])

    # make the model
    # define place holder
    X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')
    model = UnetModelCrop({'X': X, 'Y': y}, trainable=mode, input_size=input_size,
                          batch_size=batch_size, start_filter_num=32)
    # create graph
    model.create_graph('X', class_num=2)

    # evaluate on each sub folder
    root_dir = r'/home/lab/Documents/bohao/data/aemo_all/align/0584270470{}0_01'
    for fl, p_dir in get_file_list(root_dir):
        for f in fl:
            print('Evaluating {} in {}'.format(f, p_dir))

            pred_save_dir = os.path.join(task_dir, 'aemo_all', '/'.join(p_dir.split('/')[7:]))
            ersa_utils.make_dir_if_not_exist(pred_save_dir)
            # prepare the reader
            reader = uabDataReader.ImageLabelReader(gtInds=[0],
                                                    dataInds=[0],
                                                    nChannels=3,
                                                    parentDir=p_dir,
                                                    chipFiles=[[f]],
                                                    chip_size=input_size,
                                                    tile_size=tile_size,
                                                    batchSize=batch_size,
                                                    block_mean=img_mean,
                                                    overlap=model.get_overlap(),
                                                    padding=np.array((model.get_overlap() / 2, model.get_overlap() / 2)),
                                                    isTrain=False)
            rManager = reader.readManager

            # run the model
            pred, conf_map = model.run(pretrained_model_dir=model_dir,
                                       test_reader=rManager,
                                       tile_size=tile_size,
                                       patch_size=input_size,
                                       gpu=gpu, load_epoch_num=75, best_model=False)
            pred_name = 'pred_{}.png'.format(f.split('.')[0])
            conf_name = 'conf_{}.npy'.format(f.split('.')[0])

            ersa_utils.save_file(os.path.join(pred_save_dir, pred_name), pred.astype(np.uint8))
            ersa_utils.save_file(os.path.join(pred_save_dir, conf_name), conf_map)


def check_stats(check_fig=False):
    root_dir = r'/home/lab/Documents/bohao/data/aemo_all/align/0584270470{}0_01'
    data_dir = r'/media/ei-edl01/data/aemo/TILES/'
    for fl, p_dir in get_file_list(root_dir):
        rgb_file_dir = os.path.join(data_dir, '/'.join(p_dir.split('/')[8:]))
        pred_save_dir = os.path.join(task_dir, 'aemo_all', '/'.join(p_dir.split('/')[7:]))
        pred_files = sorted(glob(os.path.join(pred_save_dir, '*.png')))
        conf_files = sorted(glob(os.path.join(pred_save_dir, '*.npy')))
        for file_pred, file_conf in zip(pred_files, conf_files):
            print('Processing file {}'.format(file_pred))
            pred = ersa_utils.load_file(file_pred)
            conf = ersa_utils.load_file(file_conf)
            rgb = ersa_utils.load_file(os.path.join(rgb_file_dir, os.path.basename(file_pred)[5:-3]) + 'tif')
            bm = 1 - get_blank_regions(rgb)
            pred = bm * pred
            conf = bm * conf
            if check_fig:
                visualize_utils.compare_three_figure(rgb, pred, conf)
            ersa_utils.save_file(file_pred, pred)
            ersa_utils.save_file(file_conf, conf)


if __name__ == '__main__':
    #eval_tiles()
    check_stats()
