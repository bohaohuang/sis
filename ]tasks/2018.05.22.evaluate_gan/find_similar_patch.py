import os
import imageio
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import utils
import uabPreprocClasses
import uab_collectionFunctions
import uab_DataHandlerFunctions
import bohaoCustom.uabPreprocClasses as bPreproc
from bohaoCustom import uabMakeNetwork_UNetEncoder


def crop_center(img, cropx, cropy):
    y,x,_ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx,:]


def random_crop(img, crop_x, crop_y):
    x, y, _ = img.shape
    x_range = int(x - crop_x)
    y_range = int(y - crop_y)
    start_x = np.random.randint(0, x_range)
    start_y = np.random.randint(0, y_range)
    return img[start_x:start_x+crop_x, start_y:start_y+crop_y, :], start_x, start_y


# settings
inria_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles'
img_dir, task_dir = utils.get_task_img_folder()
model_name ='vae'
city_name = 'austin'
city_id = 1
img_name = '{}{}_RGB.tif'.format(city_name, city_id)
img = imageio.imread(os.path.join(inria_dir, img_name))

tile_dim = [5000, 5000]
patch_size = [321, 321]
vae_size = [256, 256]

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# load encoded features in valid set
feature_file = os.path.join(task_dir, '{}_inria.csv'.format(model_name))
feature = pd.read_csv(feature_file, sep=',', header=None).values
patch_file = os.path.join(task_dir, '{}_inria.txt'.format(model_name))
with open(patch_file, 'r') as f:
    patch_names = f.readlines()

# define model
X = tf.placeholder(tf.float32, shape=[None, vae_size[0], vae_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, vae_size[0], vae_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = uabMakeNetwork_UNetEncoder.VGGVAE({'X':X, 'Y':y},
                                          trainable=mode,
                                          input_size=vae_size,
                                          batch_size=1,
                                          start_filter_num=32,
                                          latent_num=500)
# create graph
model.create_graph('X', class_num=3)
model_dir = r'/hdd6/Models/VGGVAE/VGGVAE_inria_z500_0_PS(256, 256)_BS5_EP400_LR1e-05_DS200.0_DR0.5_SFN32'

# load gmm model
model_file_name = os.path.join(task_dir, 'gmm_models_{}_{}.npy'.format(model_name, 80))
gmm_models = np.load(model_file_name)

# load patch dir
blCol = uab_collectionFunctions.uabCollection('inria')
opDetObj = bPreproc.uabOperTileDivide(255)          # inria GT has value 0 and 255, we map it back to 0 and 1
# [3] is the channel id of GT
rescObj = uabPreprocClasses.uabPreprocMultChanOp([], 'GT_Divide.tif', 'Map GT to (0, 1)', [3], opDetObj)
rescObj.run(blCol)
img_mean = blCol.getChannelMeans([0, 1, 2])         # get mean of rgb info

# extract patches
extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4],
                                                cSize=(321, 321),
                                                numPixOverlap=0,
                                                extSave=['jpg', 'jpg', 'jpg', 'png'],
                                                isTrain=True,
                                                gtInd=3,
                                                pad=0)
patchDir = extrObj.run(blCol)

# evaluate
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    model.load(model_dir, sess, best_model=False)

    while True:
        patch, start_x, start_y = random_crop(img, vae_size[0], vae_size[1])
        img2encode = np.expand_dims(patch, axis=0)
        pred = sess.run([model.z_mean, model.z_sigma], feed_dict={model.inputs['X']: img2encode,
                                                                  model.trainable: False})
        encoded = np.zeros((1, model.latent_num * 2))
        encoded[0, :model.latent_num] = pred[0][0, :]
        encoded[0, model.latent_num:] = pred[1][0, :]

        # find closest patch
        plt.figure(figsize=(15, 6))
        grid = plt.GridSpec(2, 5, wspace=0.4, hspace=0.3)
        plt.subplot(grid[:2, :2])
        plt.imshow(patch)
        plt.title('{}_{} ({}, {})'.format(city_name, city_id, start_x, start_y))
        plt.axis('off')

        dist = np.linalg.norm(encoded - feature, ord=2, axis=1)
        top6_idx = np.argsort(dist)[:6]
        for plt_cnt, idx in enumerate(top6_idx):
            similar_img = np.zeros((321, 321, 3), dtype=np.uint8)
            for c_cnt in range(3):
                similar_img[:, :, c_cnt] = imageio.imread(os.path.join(patchDir, '{}_RGB{}.jpg'.
                                                                       format(patch_names[idx][:-1], c_cnt)))
            plt.subplot(grid[plt_cnt//3, plt_cnt%3+2])
            plt.imshow(similar_img)

            plt.title('{}: {:.3f}'.format(patch_names[idx][:-1].split('_')[0], dist[idx]))
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, 'similar_patch_{}_{}_({},{}).png'.
                                 format(city_name, city_id, start_x, start_y)))
        plt.show()
