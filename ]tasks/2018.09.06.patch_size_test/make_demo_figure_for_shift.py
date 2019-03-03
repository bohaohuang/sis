import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import ersa_utils, sis_utils
import uab_collectionFunctions
from nn import nn_utils
from bohaoCustom import uabMakeNetwork_UNet

model_dir = r'/hdd6/Models/Inria_decay/UnetCrop_inria_decay_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60.0_DR0.1_SFN32'
rgb_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles/austin1_RGB.tif'

rgb = ersa_utils.load_file(rgb_dir)
rgb = rgb[1200:2200, 900:2300, :]
height = 2200 - 1200
width = 2300 - 900
rgb_temp = np.copy(rgb)
nn_utils.set_gpu(-1)
img_dir, task_dir = sis_utils.get_task_img_folder()

blCol = uab_collectionFunctions.uabCollection('inria')
img_mean = blCol.getChannelMeans([0, 1, 2])

X = tf.placeholder(tf.float32, shape=[None, 572, 572, 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, 572, 572, 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = uabMakeNetwork_UNet.UnetModelCrop({'X': X, 'Y': y},
                                          trainable=mode,
                                          input_size=(572, 572),
                                          batch_size=1,
                                          start_filter_num=32)
# create graph
model.create_graph('X', class_num=2)

sess = tf.Session()
init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
sess.run(init)
model.load(model_dir, sess, epoch=95)

shift = 0
plt.figure(figsize=(8, 6))
for w in range(shift, width-572, 388):
    for h in range(0, height-572, 388):
        plt.axhline(h + 92, 0, width, color='r', linestyle='--')
        plt.axhline(h + 92 + 388, 0, width, color='r', linestyle='--')
        plt.axvline(w + 92, 0, height, color='r', linestyle='--')
        plt.axvline(w + 92 + 388, 0, height, color='r', linestyle='--')

        X_batch = np.expand_dims(rgb[h:h+572, w:w+572, :]-img_mean, axis=0)
        pred = sess.run(model.output, feed_dict={model.inputs['X']: X_batch,
                                                 model.trainable: False})
        pred = np.argmax(np.squeeze(pred, axis=0), axis=-1)
        mask_locs = np.where(pred == 1)
        locs_num = len(mask_locs[0])
        X_batch = rgb[h:h+572, w:w+572, :]
        X_batch = X_batch[92:-92, 92:-92, :]
        print(X_batch.shape, pred.shape)
        X_batch[mask_locs[0], mask_locs[1], np.zeros(locs_num, dtype=int)] = 255
        rgb_temp[h+92:h+92+388, w+92:w+92+388, :] = X_batch

plt.imshow(rgb_temp)
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'shift_demo_shift_{}.png'.format(shift)))
plt.show()
sess.close()
