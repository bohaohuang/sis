import numpy as np
from nn import unet, nn_utils
from collection import collectionMaker, collectionEditor

# settings
class_num = 2
patch_size = (572, 572)
tile_size = (5000, 5000)
suffix = 'aemo'
lr = 1e-5
ds = 20
dr = 0.1
epochs = 10
bs = 5
gpu = 0

# define network
unet = unet.UNet(class_num, patch_size, suffix=suffix, learn_rate=lr, decay_step=ds, decay_rate=dr,
                 epochs=epochs, batch_size=bs)
overlap = unet.get_overlap()

cm = collectionMaker.read_collection(raw_data_path=r'/home/lab/Documents/bohao/data/aemo',
                                     field_name='aus',
                                     field_id='',
                                     rgb_ext='.*rgb',
                                     gt_ext='.*gt',
                                     file_ext='tif',
                                     force_run=False,
                                     clc_name='aemo')
gt_d255 = collectionEditor.SingleChanMult(cm.clc_dir, 1/255, ['.*gt', 'gt_d255']).\
    run(force_run=False, file_ext='tif', d_type=np.uint8,)
cm.replace_channel(gt_d255.files, True, ['gt', 'gt_d255'])
cm.print_meta_data()

file_list_valid = cm.load_files(field_id='', field_ext='.*rgb,.*gt_d255')
chan_mean = cm.meta_data['chan_mean'][:3]


import os
import tensorflow as tf
import uabDataReader
import uabUtilreader
from bohaoCustom import uabMakeNetwork_UNet
tf.reset_default_graph()
# make the model
# define place holder
X = tf.placeholder(tf.float32, shape=[None, patch_size[0], patch_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, patch_size[0], patch_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = uabMakeNetwork_UNet.UnetModelCrop({'X': X, 'Y': y},
                                          trainable=mode,
                                          input_size=patch_size,
                                          batch_size=5, start_filter_num=32)
# create graph
model.create_graph('X', class_num=2)

# sp detector
reader = uabDataReader.ImageLabelReader(gtInds=[0],
                                        dataInds=[0],
                                        nChannels=3,
                                        parentDir=r'/home/lab/Documents/bohao/data/aemo',
                                        chipFiles=[[os.path.basename(file_list_valid[0][0])]],
                                        chip_size=patch_size,
                                        tile_size=tile_size,
                                        batchSize=bs,
                                        block_mean=chan_mean,
                                        overlap=model.get_overlap(),
                                        padding=np.array((model.get_overlap() / 2, model.get_overlap() / 2)),
                                        isTrain=False)
test_reader = reader.readManager
# run algo
model_dir = r'/hdd6/Models/UNET_rand_gird/UnetCrop_spca_aug_grid_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    model.load(model_dir, sess)
    result = model.test('X', sess, test_reader)
image_pred = uabUtilreader.un_patchify_shrink(result,
                                              [tile_size[0] + model.get_overlap(),
                                               tile_size[1] + model.get_overlap()],
                                              tile_size, patch_size,
                                              [patch_size[0] - model.get_overlap(),
                                               patch_size[1] - model.get_overlap()],
                                              overlap=model.get_overlap())
import matplotlib.pyplot as plt
plt.imshow(image_pred[:, :, 1])
plt.show()
