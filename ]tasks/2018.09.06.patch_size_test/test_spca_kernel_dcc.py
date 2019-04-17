import os
import time
import imageio
import numpy as np
import scipy.signal
import tensorflow as tf
import uabCrossValMaker
import uab_collectionFunctions
import uabDataReader
import uabRepoPaths
import uabUtilreader
import util_functions
from bohaoCustom import uabMakeNetwork_UNet, uabMakeNetwork_DeepLabV2


cached_2d_windows = dict()
def _window_2D(window_size, window_in, power=2):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    # Memoization
    global cached_2d_windows
    key = "{}_{}".format(window_size, power)
    if key in cached_2d_windows:
        wind = cached_2d_windows[key]
    else:
        wind = _spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, 3), 3)
        wind = wind * wind.transpose(1, 0, 2)
        cached_2d_windows[key] = wind
    region = (window_size - window_in) // 2
    return wind[region:region+window_in, region:region+window_in, :]


def _spline_window(window_size, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """
    intersection = int(window_size/4)
    wind_outer = (abs(2*(scipy.signal.triang(window_size))) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(scipy.signal.triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)

    return wind


def _rotate_mirror_undo(im_mirrs):
    """
    merges a list of 8 np arrays (images) of shape (x, y, nb_channels) generated
    from the `_rotate_mirror_do` function. Each images might have changed and
    merging them implies to rotated them back in order and average things out.

    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    """
    origs = []
    origs.append(np.array(im_mirrs[0]))
    origs.append(np.rot90(np.array(im_mirrs[1]), axes=(0, 1), k=3))
    origs.append(np.rot90(np.array(im_mirrs[2]), axes=(0, 1), k=2))
    origs.append(np.rot90(np.array(im_mirrs[3]), axes=(0, 1), k=1))
    origs.append(np.array(im_mirrs[4])[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[5]), axes=(0, 1), k=3)[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[6]), axes=(0, 1), k=2)[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[7]), axes=(0, 1), k=1)[:, ::-1])
    return np.mean(origs, axis=0)


class ImageLabelReader(uabDataReader.ImageLabelReader):
    def __init__(self, gtInds, dataInds, parentDir, chipFiles, chip_size, tile_size, batchSize, nChannels=1,
                 overlap=0, padding=np.array((0, 0)), block_mean=None, dataAug='', random=True, isTrain=True, aug=0):
        self.aug = aug
        super(ImageLabelReader, self).__init__(gtInds, dataInds, parentDir, chipFiles, chip_size, tile_size, batchSize,
                                               nChannels, overlap, padding, block_mean, dataAug, random, isTrain)

    def _aug(self, img):
        if self.aug == 0:
            return img
        elif 1 <= self.aug < 4:
            return np.rot90(np.array(img), axes=(0, 1), k=self.aug)
        elif self.aug == 4:
            return np.array(img)[:, ::-1]
        elif 5 <= self.aug < 8:
            return np.rot90(np.array(img)[:, ::-1], axes=(0, 1), k=self.aug-4)

    def readFromDiskIteratorTest(self, image_dir, chipFiles, batch_size, tile_dim, patch_size, overlap=0,
                                 padding=(0, 0)):
        # this is a iterator for test
        for row in chipFiles:
            blockList = []
            nDims = 0
            for cnt, file in enumerate(row):
                if type(image_dir) is list:
                    img = util_functions.uabUtilAllTypeLoad(os.path.join(image_dir[cnt], file))
                else:
                    img = util_functions.uabUtilAllTypeLoad(os.path.join(image_dir, file))
                if len(img.shape) == 2:
                    img = np.expand_dims(img, axis=2)
                nDims += img.shape[2]
                blockList.append(img)
            block = np.dstack(blockList).astype(np.float32)

            if self.block_mean is not None:
                block -= self.block_mean

            if (np.array(padding) > 0).any():
                block = uabUtilreader.pad_block(block, padding)
                tile_dim = tile_dim + padding * 2

            block = self._aug(block)

            ind = 0
            image_batch = np.zeros((batch_size, patch_size[0], patch_size[1], nDims))
            for patch in uabUtilreader.patchify(block, tile_dim, patch_size, overlap=overlap):
                # print(str(ind) +': '+ str(patch.shape))
                image_batch[ind, :, :, :] = patch
                ind += 1
                if ind == batch_size:
                    ind = 0
                    yield image_batch
            # yield the last chunk
            if ind > 0:
                yield image_batch[:ind, :, :, :]


class UnetModelCrop(uabMakeNetwork_UNet.UnetModelCrop):
    def evaluate(self, rgb_list, gt_list, rgb_dir, gt_dir, input_size, tile_size, batch_size, img_mean,
                 model_dir, gpu=None, save_result=True, save_result_parent_dir=None, show_figure=False,
                 verb=True, ds_name='default', load_epoch_num=None, best_model=True):
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None

        if show_figure:
            import matplotlib.pyplot as plt

        if save_result:
            self.model_name = model_dir.split('/')[-1]
            if save_result_parent_dir is None:
                score_save_dir = os.path.join(uabRepoPaths.evalPath, self.model_name, ds_name)
            else:
                score_save_dir = os.path.join(uabRepoPaths.evalPath, save_result_parent_dir,
                                              self.model_name, ds_name)
            if not os.path.exists(score_save_dir):
                os.makedirs(score_save_dir)
            with open(os.path.join(score_save_dir, 'result.txt'), 'w'):
                pass

        iou_record = []
        iou_return = {}
        for file_name, file_name_truth in zip(rgb_list, gt_list):
            tile_name = file_name_truth.split('_')[0]
            if verb:
                print('Evaluating {} ... '.format(tile_name))
            start_time = time.time()
            truth_label_img = imageio.imread(os.path.join(gt_dir, file_name_truth))
            if tile_size is None:
                tile_size_temp = truth_label_img.shape[:2]
            else:
                tile_size_temp = tile_size

            # prepare the reader
            pred_all = []
            for aug_cnt in range(8):
                reader = ImageLabelReader(gtInds=[0], dataInds=[0], nChannels=3, parentDir=rgb_dir,
                                          chipFiles=[file_name], chip_size=input_size, tile_size=tile_size_temp,
                                          batchSize=batch_size, block_mean=img_mean, overlap=self.get_overlap(),
                                          padding=np.array((self.get_overlap()/2, self.get_overlap()/2)),
                                          isTrain=False, aug=aug_cnt)
                rManager = reader.readManager

                # run the model
                pred = self.run(pretrained_model_dir=model_dir,
                                test_reader=rManager,
                                tile_size=tile_size_temp,
                                patch_size=input_size,
                                gpu=gpu, load_epoch_num=load_epoch_num, best_model=best_model)
                pred_all.append(pred)

            # average results
            pred_all = _rotate_mirror_undo(pred_all)
            pred = util_functions.get_pred_labels(pred_all) * 1

            iou = util_functions.iou_metric(truth_label_img, pred, divide_flag=True)
            iou_record.append(iou)
            iou_return[tile_name] = iou

            duration = time.time() - start_time
            if verb:
                print('{} mean IoU={:.3f}, duration: {:.3f}'.format(tile_name, iou[0]/iou[1], duration))

            # save results
            if save_result:
                pred_save_dir = os.path.join(score_save_dir, 'pred')
                if not os.path.exists(pred_save_dir):
                    os.makedirs(pred_save_dir)
                imageio.imsave(os.path.join(pred_save_dir, tile_name+'.png'), pred.astype(np.uint8))
                with open(os.path.join(score_save_dir, 'result.txt'), 'a+') as file:
                    file.write('{} {}\n'.format(tile_name, iou))

            if show_figure:
                plt.figure(figsize=(12, 6))
                ax1 = plt.subplot(121)
                ax1.imshow(truth_label_img)
                plt.title('Truth')
                ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
                ax2.imshow(pred)
                plt.title('pred')
                plt.suptitle('{} Results on {} IoU={:3f}'.format(self.model_name, file_name_truth.split('_')[0], iou[0]/iou[1]))
                plt.tight_layout()
                plt.show()

        iou_record = np.array(iou_record)
        mean_iou = np.sum(iou_record[:, 0]) / np.sum(iou_record[:, 1])
        print('Overall mean IoU={:.3f}'.format(mean_iou))
        if save_result:
            if save_result_parent_dir is None:
                score_save_dir = os.path.join(uabRepoPaths.evalPath, self.model_name, ds_name)
            else:
                score_save_dir = os.path.join(uabRepoPaths.evalPath, save_result_parent_dir, self.model_name,
                                              ds_name)
            with open(os.path.join(score_save_dir, 'result.txt'), 'a+') as file:
                file.write('{}'.format(mean_iou))

        return iou_return

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
            kernel = _window_2D(patch_size[0], patch_size[0]-self.get_overlap())

            if self.config is None:
                self.config = tf.ConfigProto(allow_soft_placement=True)
            pad = self.get_overlap()
            with tf.Session(config=self.config) as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                self.load(pretrained_model_dir, sess, epoch=load_epoch_num, best_model=best_model)
                self.model_name = pretrained_model_dir.split('/')[-1]
                result = self.test('X', sess, test_reader)
            n_patch = result.shape[0]
            for patch_cnt in range(n_patch):
                result[patch_cnt, :, :, :] *= kernel
            image_pred = uabUtilreader.un_patchify_shrink(result,
                                                          [tile_size[0] + pad, tile_size[1] + pad],
                                                          tile_size,
                                                          patch_size,
                                                          [patch_size[0] - pad, patch_size[1] - pad],
                                                          overlap=pad)
            return image_pred


class DeeplabV3(uabMakeNetwork_DeepLabV2.DeeplabV3):
    def evaluate(self, rgb_list, gt_list, rgb_dir, gt_dir, input_size, tile_size, batch_size, img_mean,
                 model_dir, gpu=None, save_result=True, save_result_parent_dir=None, show_figure=False,
                 verb=True, ds_name='default', load_epoch_num=None, best_model=True):
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None

        if show_figure:
            import matplotlib.pyplot as plt

        if save_result:
            self.model_name = model_dir.split('/')[-1]
            if save_result_parent_dir is None:
                score_save_dir = os.path.join(uabRepoPaths.evalPath, self.model_name, ds_name)
            else:
                score_save_dir = os.path.join(uabRepoPaths.evalPath, save_result_parent_dir,
                                              self.model_name, ds_name)
            if not os.path.exists(score_save_dir):
                os.makedirs(score_save_dir)
            with open(os.path.join(score_save_dir, 'result.txt'), 'w'):
                pass

        iou_record = []
        iou_return = {}
        for file_name, file_name_truth in zip(rgb_list, gt_list):
            tile_name = file_name_truth.split('_')[0]
            if verb:
                print('Evaluating {} ... '.format(tile_name))
            start_time = time.time()
            truth_label_img = imageio.imread(os.path.join(gt_dir, file_name_truth))
            if tile_size is None:
                tile_size_temp = truth_label_img.shape[:2]
            else:
                tile_size_temp = tile_size

            # prepare the reader
            pred_all = []
            for aug_cnt in range(8):
                reader = ImageLabelReader(gtInds=[0], dataInds=[0], nChannels=3, parentDir=rgb_dir,
                                          chipFiles=[file_name], chip_size=input_size, tile_size=tile_size_temp,
                                          batchSize=batch_size, block_mean=img_mean, overlap=self.get_overlap(),
                                          padding=np.array((self.get_overlap()/2, self.get_overlap()/2)),
                                          isTrain=False, aug=aug_cnt)
                rManager = reader.readManager

                # run the model
                pred = self.run(pretrained_model_dir=model_dir,
                                test_reader=rManager,
                                tile_size=tile_size_temp,
                                patch_size=input_size,
                                gpu=gpu, load_epoch_num=load_epoch_num, best_model=best_model)
                pred_all.append(pred)

            # average results
            pred_all = _rotate_mirror_undo(pred_all)
            pred = util_functions.get_pred_labels(pred_all) * 1

            iou = util_functions.iou_metric(truth_label_img, pred, divide_flag=True)
            iou_record.append(iou)
            iou_return[tile_name] = iou

            duration = time.time() - start_time
            if verb:
                print('{} mean IoU={:.3f}, duration: {:.3f}'.format(tile_name, iou[0]/iou[1], duration))

            # save results
            if save_result:
                pred_save_dir = os.path.join(score_save_dir, 'pred')
                if not os.path.exists(pred_save_dir):
                    os.makedirs(pred_save_dir)
                imageio.imsave(os.path.join(pred_save_dir, tile_name+'.png'), pred.astype(np.uint8))
                with open(os.path.join(score_save_dir, 'result.txt'), 'a+') as file:
                    file.write('{} {}\n'.format(tile_name, iou))

            if show_figure:
                plt.figure(figsize=(12, 6))
                ax1 = plt.subplot(121)
                ax1.imshow(truth_label_img)
                plt.title('Truth')
                ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
                ax2.imshow(pred)
                plt.title('pred')
                plt.suptitle('{} Results on {} IoU={:3f}'.format(self.model_name, file_name_truth.split('_')[0], iou[0]/iou[1]))
                plt.tight_layout()
                plt.show()

        iou_record = np.array(iou_record)
        mean_iou = np.sum(iou_record[:, 0]) / np.sum(iou_record[:, 1])
        print('Overall mean IoU={:.3f}'.format(mean_iou))
        if save_result:
            if save_result_parent_dir is None:
                score_save_dir = os.path.join(uabRepoPaths.evalPath, self.model_name, ds_name)
            else:
                score_save_dir = os.path.join(uabRepoPaths.evalPath, save_result_parent_dir, self.model_name,
                                              ds_name)
            with open(os.path.join(score_save_dir, 'result.txt'), 'a+') as file:
                file.write('{}'.format(mean_iou))

        return iou_return

    def run(self, train_reader=None, valid_reader=None, test_reader=None, pretrained_model_dir=None, layers2load=None,
            isTrain=False, img_mean=np.array((0, 0, 0), dtype=np.float32), verb_step=100, save_epoch=5, gpu=None,
            tile_size=(5000, 5000), patch_size=(572, 572), truth_val=1, continue_dir=None, load_epoch_num=None,
            fineTune=False, valid_iou=False, best_model=True):
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
                        if not fineTune:
                            restore_var = [v for v in tf.global_variables() if 'resnet_v1' in v.name and
                                       not 'Adam' in v.name]
                            loader = tf.train.Saver(var_list=restore_var)
                            self.load(pretrained_model_dir, sess, loader, epoch=load_epoch_num)
                        else:
                            self.load(pretrained_model_dir, sess, saver, epoch=load_epoch_num)
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)
                try:
                    train_summary_writer = tf.summary.FileWriter(self.ckdir, sess.graph)
                    self.train('X', 'Y', self.n_train, sess, train_summary_writer,
                               n_valid=self.n_valid, train_reader=train_reader, valid_reader=valid_reader,
                               image_summary=util_functions.image_summary, img_mean=img_mean,
                               verb_step=verb_step, save_epoch=save_epoch, continue_dir=continue_dir,
                               valid_iou=valid_iou)
                finally:
                    coord.request_stop()
                    coord.join(threads)
                    saver.save(sess, '{}/model.ckpt'.format(self.ckdir), global_step=self.global_step)
        else:
            kernel = _window_2D(patch_size[0] + 182, patch_size[0])

            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                self.load(pretrained_model_dir, sess, epoch=load_epoch_num, best_model=best_model)
                self.model_name = pretrained_model_dir.split('/')[-1]
                result = self.test('X', sess, test_reader)
            n_patch = result.shape[0]
            for patch_cnt in range(n_patch):
                result[patch_cnt, :, :, :] *= kernel
            image_pred = uabUtilreader.un_patchify(result, tile_size, patch_size)
            return image_pred


# settings
gpu = 0
batch_size = 1
tile_size = [5000, 5000]
util_functions.tf_warn_level(3)
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
model_type = 'deeplab'

if model_type == 'unet':
    patch_size = [572, 828, 1084, 1340, 1596, 1852, 2092, 2332, 2636]
    model_dir = r'/hdd6/Models/UNET_rand_gird/UnetCrop_spca_aug_grid_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'
else:
    patch_size = [832, 1088, 1344, 1600, 1856, 2096, 2640]
    model_dir = r'/hdd6/Models/DeepLab_rand_grid/DeeplabV3_spca_aug_grid_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32'

for ps in patch_size:
    input_size = [ps, ps]
    tf.reset_default_graph()
    blCol = uab_collectionFunctions.uabCollection('spca')
    blCol.readMetadata()
    file_list, parent_dir = blCol.getAllTileByDirAndExt([1, 2, 3])
    file_list_truth, parent_dir_truth = blCol.getAllTileByDirAndExt(0)
    idx, file_list = uabCrossValMaker.uabUtilGetFolds(None, file_list, 'force_tile')
    idx_truth, file_list_truth = uabCrossValMaker.uabUtilGetFolds(None, file_list_truth, 'force_tile')
    # use first 5 tiles for validation
    file_list_valid = uabCrossValMaker.make_file_list_by_key(
        idx, file_list, [i for i in range(250, 500)])
    file_list_valid_truth = uabCrossValMaker.make_file_list_by_key(
        idx_truth, file_list_truth, [i for i in range(250, 500)])
    img_mean = blCol.getChannelMeans([1, 2, 3])

    # make the model
    # define place holder
    X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')
    if model_type == 'unet':
        model = UnetModelCrop({'X': X, 'Y': y}, trainable=mode, input_size=input_size,
                              batch_size=batch_size, start_filter_num=32)
    else:
        model = DeeplabV3({'X': X, 'Y': y}, trainable=mode, input_size=input_size,
                          batch_size=batch_size, start_filter_num=32)
    # create graph
    model.create_graph('X', class_num=2)

    # evaluate on tiles
    model.evaluate(file_list_valid, file_list_valid_truth, parent_dir, parent_dir_truth,
                   input_size, tile_size, batch_size, img_mean, model_dir, gpu,
                   save_result_parent_dir='size', ds_name='spca_{}_kernel'.format(ps), best_model=True)
