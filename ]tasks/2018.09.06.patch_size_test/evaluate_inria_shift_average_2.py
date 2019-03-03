import os
import time
import imageio
import numpy as np
import tensorflow as tf
import sis_utils
import uabCrossValMaker
import uabDataReader
import uabUtilreader
import uab_collectionFunctions
import util_functions
from bohaoCustom import uabMakeNetwork_UNet


class myImageLabelReader(uabDataReader.ImageLabelReader):
    def __init__(self, gtInds, dataInds, parentDir, chipFiles, chip_size, tile_size, batchSize, nChannels=1,
                 overlap=0, padding=np.array((0, 0)), block_mean=None, dataAug='', random=True, isTrain=True,
                 shift=0):
        self.shift = shift
        super().__init__(gtInds, dataInds, parentDir, chipFiles, chip_size, tile_size, batchSize, nChannels,
                         overlap, padding, block_mean, dataAug, random, isTrain)

    @staticmethod
    def shift_block(block, shift):
        block = np.roll(block, -shift, axis=1)
        return block

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
            block = block[:, :4576, :]
            tile_dim = np.array((5000, 4576))
            block = self.shift_block(block, self.shift)
            if self.block_mean is not None:
                block -= self.block_mean

            if (np.array(padding) > 0).any():
                block = uabUtilreader.pad_block(block, padding)
                tile_dim = tile_dim + padding * 2

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


# settings
gpu = 0
batch_size = 49
input_size = [572, 572]
tile_size = [5000, 5000]
shift_max = 32
shift_list = []
for i in range(shift_max // 2):
    shift_list.append(i)
    shift_list.append(i+16)
for slide_step in shift_list:
    util_functions.tf_warn_level(3)
    city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
    model_dir = r'/hdd6/Models/Inria_decay/UnetCrop_inria_decay_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60.0_DR0.1_SFN32'
    img_dir, task_dir = sis_utils.get_task_img_folder()

    tf.reset_default_graph()
    blCol = uab_collectionFunctions.uabCollection('inria')
    blCol.readMetadata()
    file_list, parent_dir = blCol.getAllTileByDirAndExt([0, 1, 2])
    file_list_truth, parent_dir_truth = blCol.getAllTileByDirAndExt(4)
    idx, file_list = uabCrossValMaker.uabUtilGetFolds(None, file_list, 'force_tile')
    idx_truth, file_list_truth = uabCrossValMaker.uabUtilGetFolds(None, file_list_truth, 'force_tile')
    # use first 5 tiles for validation
    file_list_valid = uabCrossValMaker.make_file_list_by_key(
        idx, file_list, [i for i in range(0, 6)],
        filter_list=['bellingham', 'bloomington', 'sfo', 'tyrol-e', 'innsbruck'])
    file_list_valid_truth = uabCrossValMaker.make_file_list_by_key(
        idx_truth, file_list_truth, [i for i in range(0, 6)],
        filter_list=['bellingham', 'bloomington', 'sfo', 'tyrol-e', 'innsbruck'])
    img_mean = blCol.getChannelMeans([0, 1, 2])

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # make the model
    # define place holder
    X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')
    model = uabMakeNetwork_UNet.UnetModelCrop({'X': X, 'Y': y},
                                              trainable=mode,
                                              input_size=input_size,
                                              batch_size=batch_size,
                                              start_filter_num=32)
    # create graph
    model.create_graph('X', class_num=2)
    score_save_dir = os.path.join(task_dir, 'unet_patch_test_8', 'slide_step_{}'.format(slide_step))

    iou_record = []
    if not os.path.exists(score_save_dir):
        os.makedirs(score_save_dir)
    with open(os.path.join(score_save_dir, 'result.txt'), 'w'):
        pass
    for file_name, file_name_truth in zip(file_list_valid, file_list_valid_truth):
        tile_name = file_name_truth.split('_')[0]
        print('Evaluating {} ... '.format(tile_name))
        start_time = time.time()

        pred_overall = np.zeros((tile_size[0], tile_size[1], 2))

        reader = myImageLabelReader(gtInds=[0],
                                    dataInds=[0],
                                    nChannels=3,
                                    parentDir=parent_dir,
                                    chipFiles=[file_name],
                                    chip_size=input_size,
                                    tile_size=tile_size,
                                    batchSize=batch_size,
                                    block_mean=img_mean,
                                    overlap=model.get_overlap(),
                                    padding=np.array((model.get_overlap() // 2, model.get_overlap() // 2)),
                                    isTrain=False,
                                    shift=slide_step)
        rManager = reader.readManager

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            model.load(model_dir, sess, epoch=None, best_model=True)
            model.model_name = model_dir.split('/')[-1]
            result = model.test('X', sess, rManager)
        pad = model.get_overlap()
        image_pred = uabUtilreader.un_patchify_shrink(result,
                                                      [tile_size[0] + pad, 4576 + pad],
                                                      [5000, 4576],
                                                      input_size,
                                                      [input_size[0] - pad, input_size[1] - pad],
                                                      overlap=pad)
        pred_overall = util_functions.get_pred_labels(image_pred) * 1
        pred_overall = np.roll(pred_overall, shift=slide_step, axis=1)
        pred_overall = pred_overall[:, 1000:-1000]
        #pred_overall = pred_overall[:, shift_max-slide_step:-slide_step-1]
        truth_label_img = imageio.imread(os.path.join(parent_dir_truth, file_name_truth))
        #truth_label_img = np.roll(truth_label_img, -slide_step, axis=1)
        truth_label_img = truth_label_img[:, :4576]
        truth_label_img = truth_label_img[:, 1000:-1000]
        iou = util_functions.iou_metric(truth_label_img, pred_overall, divide_flag=True)
        duration = time.time() - start_time
        print('{} mean IoU={:.3f}, duration: {:.3f}'.format(tile_name, iou[0] / iou[1], duration))
        iou_record.append(iou)

        pred_save_dir = os.path.join(score_save_dir, 'pred')
        if not os.path.exists(pred_save_dir):
            os.makedirs(pred_save_dir)
        imageio.imsave(os.path.join(pred_save_dir, tile_name + '.png'), pred_overall.astype(np.uint8))
        with open(os.path.join(score_save_dir, 'result.txt'), 'a+') as file:
            file.write('{} {}\n'.format(tile_name, iou))
    iou_record = np.array(iou_record)
    mean_iou = np.sum(iou_record[:, 0]) / np.sum(iou_record[:, 1])
    print('Overall mean IoU={:.3f}'.format(mean_iou))
    with open(os.path.join(score_save_dir, 'result.txt'), 'a+') as file:
        file.write('{}'.format(mean_iou))
