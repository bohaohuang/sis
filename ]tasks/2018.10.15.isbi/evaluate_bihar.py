import os
import time
import imageio
import numpy as np
import tensorflow as tf
import uabDataReader
import util_functions
import uab_collectionFunctions
from bohaoCustom import uabMakeNetwork_DeepLabV2


def make_preds(model, model_dir, file_list):
    for file_name in file_list:
        tile_name = os.path.basename(file_name)[:-4]
        print('Evaluating {} ... '.format(tile_name))
        start_time = time.time()
        tile_size = imageio.imread(file_name).shape[:2]

        print(img_mean)

        # prepare the reader
        reader = uabDataReader.ImageLabelReader(gtInds=[0],
                                                dataInds=[0],
                                                nChannels=3,
                                                parentDir=os.path.dirname(file_name),
                                                chipFiles=[[os.path.basename(file_name)]],
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
                         gpu=gpu, load_epoch_num=None, best_model=False)

        print(np.unique(pred))
        #from visualize import visualize_utils
        #visualize_utils.compare_figures([imageio.imread(file_name), pred], (1, 2), fig_size=(12, 5))

        duration = time.time() - start_time
        print('duration: {:.3f}'.format(duration))


gpu = 1
batch_size = 1
input_size = [2000, 2000]
util_functions.tf_warn_level(3)
ds_name = 'bihar_building'

# settings
blCol = uab_collectionFunctions.uabCollection(ds_name)
blCol.readMetadata()
img_mean = blCol.getChannelMeans([0, 1, 2])
print(img_mean)

# make the model
# define place holder
model_dir = r'/hdd6/Models/bihar_building/DeeplabV3_bihar_building_1_PS(300, 300)_BS5_EP30_LR0.0001_DS20_DR0.8_SFN32'
X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = uabMakeNetwork_DeepLabV2.DeeplabV3({'X': X, 'Y': y}, trainable=mode, input_size=input_size,
                                           batch_size=batch_size, start_filter_num=32)
# create graph
model.create_graph('X', class_num=2)

# walk through folders
from glob import glob
files = sorted(glob(os.path.join(r'/media/ei-edl01/data/uab_datasets/bihar/patch_for_building_detection/c', '*.tif')))

make_preds(model, model_dir, files)

'''model.config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=model.config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    model.load(model_dir, sess, epoch=None, best_model=False)

    input_ = np.expand_dims(fig - img_mean, axis=0)

    pred = sess.run(model.output, feed_dict={model.inputs['X']: input_,
                                             model.trainable: False})

    pred = np.argmax(pred[0, :, :, :], axis=-1)

    from visualize import visualize_utils
    visualize_utils.compare_figures([fig, pred], (1, 2), fig_size=(12, 5))'''
