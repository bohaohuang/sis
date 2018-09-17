import os
import imageio
import numpy as np
import tensorflow as tf
import utils
import uabCrossValMaker
import uabDataReader
import uab_collectionFunctions
import util_functions
import uabUtilreader
from uabUtilreader import crop_image
from bohaoCustom import uabMakeNetwork_UNet


def patchify(block, tile_dim, patch_size, overlap=0, insert_dim=False):
    slide_step = patch_size[0] - overlap
    max_h = (tile_dim[0] - patch_size[0]).astype(np.int32)
    max_w = (tile_dim[1] - patch_size[1]).astype(np.int32)
    h_step = (tile_size[0] + 184) % (input_size[0] - (input_size[0] - slide_step))
    w_step = (tile_size[1] + 184) % (input_size[1] - (input_size[1] - slide_step))
    patch_grid_h = np.arange(0, max_h, slide_step).astype(np.int32)
    patch_grid_w = np.arange(0, max_w, slide_step).astype(np.int32)
    if h_step != 0:
        patch_grid_h = np.concatenate(patch_grid_h, max_h)
    if w_step != 0:
        patch_grid_w = np.concatenate(patch_grid_w, max_w)
    for corner_h in patch_grid_h:
        for corner_w in patch_grid_w:
            if insert_dim:
                yield np.expand_dims(crop_image(block, patch_size, (corner_h, corner_w)), axis=0)
            else:
                yield crop_image(block, patch_size, (corner_h, corner_w))


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
            if self.block_mean is not None:
                block -= self.block_mean

            if (np.array(padding) > 0).any():
                block = uabUtilreader.pad_block(block, padding)
                tile_dim = tile_dim + padding * 2

            ind = 0
            image_batch = np.zeros((batch_size, patch_size[0], patch_size[1], nDims))
            for patch in patchify(block, tile_dim, patch_size, overlap=overlap):
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
gpu = 1
batch_size = 1
input_size = [572, 572]
tile_size = [5000, 5000]
block_size = 4656
for slide_step in [388]:
    util_functions.tf_warn_level(3)
    city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
    model_dir = r'/hdd6/Models/Inria_decay/UnetCrop_inria_decay_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60.0_DR0.1_SFN32'
    img_dir, task_dir = utils.get_task_img_folder()

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

    # make coords
    output_size = [388, 388]
    coords = []
    max_h = tile_size[0] + 184 - input_size[0]
    max_w = tile_size[1] + 184 - input_size[1]
    h_step = (tile_size[0] + 184) % (input_size[0] - (input_size[0]-slide_step))
    w_step = (tile_size[1] + 184) % (input_size[1] - (input_size[1]-slide_step))
    patch_grid_h = np.arange(0, max_h, slide_step).astype(np.int32)
    patch_grid_w = np.arange(0, max_w, slide_step).astype(np.int32)
    if h_step != 0:
        patch_grid_h = np.concatenate(patch_grid_h, max_h)
    if w_step != 0:
        patch_grid_w = np.concatenate(patch_grid_w, max_w)
    for corner_h in patch_grid_h:
        for corner_w in patch_grid_w:
            coords.append([corner_h, corner_w])

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
    score_save_dir = os.path.join(task_dir, 'unet_patch_test', 'slide_step_{}'.format(slide_step))

    iou_record = []
    if not os.path.exists(score_save_dir):
        os.makedirs(score_save_dir)
    with open(os.path.join(score_save_dir, 'result.txt'), 'w'):
        pass
    for file_name, file_name_truth in zip(file_list_valid, file_list_valid_truth):
        tile_name = file_name_truth.split('_')[0]
        print('Evaluating {} ... '.format(tile_name))

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
                                    overlap=input_size[0]-slide_step,
                                    padding=np.array((model.get_overlap() / 2, model.get_overlap() / 2)),
                                    isTrain=False)
        rManager = reader.readManager

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            model.load(model_dir, sess, epoch=None, best_model=True)
            model.model_name = model_dir.split('/')[-1]
            for cnt, X_batch in enumerate(rManager):
                pred = sess.run(model.output, feed_dict={model.inputs['X']: X_batch,
                                                         model.trainable: False})
                pred_overall[coords[cnt][0]:coords[cnt][0]+output_size[0],
                             coords[cnt][1]:coords[cnt][1]+output_size[1], :] += pred[0, :, :, :]
        pred_overall = np.argmax(pred_overall, axis=2)
        truth_label_img = imageio.imread(os.path.join(parent_dir_truth, file_name_truth))
        iou = util_functions.iou_metric(truth_label_img, pred_overall, divide_flag=True)
        print('{} mean IoU={:.3f}'.format(tile_name, iou[0] / iou[1]))
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
