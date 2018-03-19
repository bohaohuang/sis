import os
import csv
import numpy as np
import tensorflow as tf
import utils
import uabCrossValMaker
import uabDataReader
import uab_collectionFunctions
from bohaoCustom import uabMakeNetwork_DeepLabV2

# settings
gpu = 1
batch_size = 1
input_size = [321, 321]
tile_size = [5000, 5000]
test_city = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
img_dir, task_dir = utils.get_task_img_folder()

tf.reset_default_graph()

model_dir = r'/hdd6/Models/DeepLab_rand_grid/' \
            r'DeeplabV3_res101_inria_aug_grid_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32'
blCol = uab_collectionFunctions.uabCollection('inria')
blCol.readMetadata()
file_list, parent_dir = blCol.getAllTileByDirAndExt([0, 1, 2])
file_list_truth, parent_dir_truth = blCol.getAllTileByDirAndExt(4)
idx, file_list = uabCrossValMaker.uabUtilGetFolds(None, file_list, 'force_tile')
idx_truth, file_list_truth = uabCrossValMaker.uabUtilGetFolds(None, file_list_truth, 'force_tile')

for tc in test_city:
    # use first 5 tiles for validation
    file_list_valid = uabCrossValMaker.make_file_list_by_key(
        idx, file_list, [i for i in range(0, 6)],
        filter_list=['bellingham', 'bloomington', 'sfo', 'tyrol-e', 'innsbruck'].extend([c for c in test_city if c != tc]))
    file_list_valid_truth = uabCrossValMaker.make_file_list_by_key(
        idx_truth, file_list_truth, [i for i in range(0, 6)],
        filter_list=['bellingham', 'bloomington', 'sfo', 'tyrol-e', 'innsbruck'].extend([c for c in test_city if c != tc]))
    img_mean = blCol.getChannelMeans([0, 1, 2])

    # make the model
    # define place holder
    X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')
    model = uabMakeNetwork_DeepLabV2.DeeplabV3({'X':X, 'Y':y},
                                               trainable=mode,
                                               input_size=input_size,
                                               batch_size=5)
    # create graph
    model.create_graph('X', class_num=2)

    # load data
    reader = uabDataReader.ImageLabelReader(gtInds=[0],
                                            dataInds=[0],
                                            nChannels=3,
                                            parentDir=parent_dir,
                                            chipFiles=file_list,
                                            chip_size=input_size,
                                            tile_size=tile_size,
                                            batchSize=batch_size,
                                            block_mean=img_mean,
                                            overlap=model.get_overlap(),
                                            padding=np.array((model.get_overlap() / 2, model.get_overlap() / 2)),
                                            isTrain=False)
    test_reader = reader.readManager

    # run algo
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        model.load(model_dir, sess)
        file_name = os.path.join(r'/hdd6/temp', '{}.csv'.format(tc))
        with open(file_name, 'w+') as f:
            for cnt, X_batch in enumerate(test_reader):
                pred = sess.run(model.encoding, feed_dict={model.inputs['X']: X_batch,
                                                           model.trainable: False})
                result = pred.reshape((-1,)).tolist()
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(['{:3.4e}'.format(x) for x in result])
