import os
import time
import imageio
import skimage.transform
import numpy as np
import tensorflow as tf
from skimage import measure
from PIL import Image, ImageDraw
import uabUtilreader
import sis_utils
import ersa_utils
import util_functions
import uabRepoPaths
import uabDataReader
import uabCrossValMaker
import uab_collectionFunctions
from reader import reader_utils
from bohaoCustom import uabMakeNetwork_UNet


def get_lines(img, patch_size):
    img_binary = (img > 0).astype(np.uint8)
    lbl = measure.label(img_binary)
    props = measure.regionprops(lbl)
    vert_list = []
    # get vertices
    for rp in props:
        vert_list.append(rp.centroid)

    # add lines
    im = Image.new('L', patch_size.tolist())
    for i in range(len(vert_list)):
        for j in range(i+1, len(vert_list)):
            ImageDraw.Draw(im).line((vert_list[i][1], vert_list[i][0],
                                     vert_list[j][1], vert_list[j][0]), fill=1, width=15)
    im_lines = (np.array(im, dtype=float) - img.astype(float) > 0).astype(np.uint8)
    im = (img + im_lines).astype(np.uint8)
    assert set(np.unique(im).tolist()).issubset({0, 1, 2})
    return im


class myImageLabelReader(uabDataReader.ImageLabelReader):
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

            if not np.all([np.array(tile_dim) == block.shape[:2]]):
                block = skimage.transform.resize(block, tile_dim, order=0, preserve_range=True, mode='reflect')

            if self.block_mean is not None:
                block -= self.block_mean

            if (np.array(padding) > 0).any():
                block = uabUtilreader.pad_block(block, padding)
                tile_dim = tile_dim + padding * 2

            ind = 0
            image_batch = np.zeros((batch_size, patch_size[0], patch_size[1], nDims))

            for patch in uabUtilreader.patchify(block, tile_dim, patch_size, overlap=overlap):
                patch_gt = (patch[:, :, 0] > 0).astype(np.uint8)
                patch[:, :, 0] = get_lines(patch_gt, np.array(patch_size))
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
            tile_size = ersa_utils.load_file(os.path.join(rgb_dir[0], file_name[0])).shape[:2]
            tile_size = np.array(tile_size) // 2

            tile_name = file_name_truth.split('_')[0]
            if verb:
                print('Evaluating {} ... '.format(tile_name))
            start_time = time.time()

            # prepare the reader
            reader = myImageLabelReader(gtInds=[0], dataInds=[0], nChannels=3, parentDir=rgb_dir,
                                        chipFiles=[file_name], chip_size=input_size, tile_size=tile_size,
                                        batchSize=batch_size, block_mean=img_mean, overlap=self.get_overlap(),
                                        padding=np.array((self.get_overlap()/2, self.get_overlap()/2)),
                                        isTrain=False)
            rManager = reader.readManager

            # run the model
            pred = self.run(pretrained_model_dir=model_dir,
                            test_reader=rManager,
                            tile_size=tile_size,
                            patch_size=input_size,
                            gpu=gpu, load_epoch_num=load_epoch_num, best_model=best_model, tile_name=tile_name)

            truth_label_img = imageio.imread(os.path.join(gt_dir, file_name_truth))
            truth_label_img = reader_utils.resize_image(truth_label_img, tile_size, preserve_range=True)
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
                plt.figure(figsize=(12, 4))
                ax1 = plt.subplot(121)
                ax1.imshow(truth_label_img)
                plt.title('Truth')
                ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
                ax2.imshow(pred)
                plt.title('pred')
                plt.suptitle('{} Results on {} IoU={:3f}'.format(self.model_name, file_name_truth.split('_')[0], iou[0]/iou[1]))
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
            valid_iou=False, best_model=True, tile_name=None):
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
            ersa_utils.save_file(os.path.join(SAVE_DIR, '{}.png'.format(tile_name)),
                                 (image_pred[:, :, 1] * 255).astype(np.uint8))
            return util_functions.get_pred_labels(image_pred) * truth_val


# settings
gpu = 1
batch_size = 5
input_size = [572, 572]
tile_size = [5000, 5000]
util_functions.tf_warn_level(3)
ds_name = 'lines_v3'
img_dir, task_dir = sis_utils.get_task_img_folder()

blCol = uab_collectionFunctions.uabCollection(ds_name)
blCol.readMetadata()
file_list, parent_dir = blCol.getAllTileByDirAndExt([0, 1, 2])
file_list_truth, parent_dir_truth = blCol.getAllTileByDirAndExt(3)
idx, file_list = uabCrossValMaker.uabUtilGetFolds(None, file_list, 'force_tile')
idx_truth, file_list_truth = uabCrossValMaker.uabUtilGetFolds(None, file_list_truth, 'force_tile')
# use first 5 tiles for validation
file_list_valid = uabCrossValMaker.make_file_list_by_key(idx, file_list, [1, 2, 3])
file_list_valid_truth = uabCrossValMaker.make_file_list_by_key(idx_truth, file_list_truth, [1, 2, 3])
img_mean = blCol.getChannelMeans([1, 2, 3])
img_mean = np.concatenate([np.array([0]), img_mean])
city_id = 0

# make the model
# define place holder
for weight in [1, 5]:
    model_dir = r'/hdd6/Models/lines/UnetCrop_lines_city{}_pw{}_0_PS(572, 572)_BS5_' \
                r'EP100_LR0.0001_DS60_DR0.1_SFN32'.format(city_id, weight)
    SAVE_DIR = os.path.join(task_dir, 'confmap_uab_{}'.format(os.path.basename(model_dir)))
    ersa_utils.make_dir_if_not_exist(SAVE_DIR)
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 4], name='X')
    y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')
    model = UnetModelCrop({'X': X, 'Y': y}, trainable=mode, input_size=input_size,
                          batch_size=batch_size, start_filter_num=32)
    # create graph
    model.create_graph('X', class_num=2)

    # evaluate on tiles
    model.evaluate(file_list_valid, file_list_valid_truth, parent_dir, parent_dir_truth,
                   input_size, tile_size, batch_size, img_mean, model_dir, gpu,
                   save_result_parent_dir='lines', ds_name=ds_name, best_model=False,
                   load_epoch_num=85)
