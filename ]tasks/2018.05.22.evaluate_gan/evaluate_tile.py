import os
import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import utils
import uabUtilreader
from bohaoCustom import uabMakeNetwork_UNetEncoder


def crop_center(img, cropx, cropy):
    y,x,_ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx,:]


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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

llh_mask = [[] for i in range(5)]
patchify_reader = uabUtilreader.patchify(img, tile_dim, patch_size, 0)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    model.load(model_dir, sess, best_model=False)

    for patch in patchify_reader:
        img2encode = np.expand_dims(crop_center(patch, vae_size[0], vae_size[1]), axis=0)

        pred = sess.run([model.z_mean, model.z_sigma], feed_dict={model.inputs['X']: img2encode,
                                                                  model.trainable: False})
        encoded = np.zeros((1, model.latent_num * 2))
        encoded[0, :model.latent_num] = pred[0][0, :]
        encoded[0, model.latent_num:] = pred[1][0, :]

        # compute llh
        llh = np.zeros(5)
        for i in range(5):
            llh[i] = gmm_models[i].score_samples(encoded)[0]
            llh_mask[i].append(llh[i] * np.ones((patch_size[0], patch_size[1], 1)))

# get mask
plt.figure(figsize=(15, 5))
for i in range(5):
    conf_map = uabUtilreader.un_patchify(np.stack(llh_mask[i], axis=0), tile_dim, patch_size, 0)
    plt.subplot(151 + i)
    plt.imshow(conf_map[:, :, 0])
    plt.axis('off')
plt.tight_layout()
#plt.savefig(os.path.join(img_dir, '{}_{}_{}_gmm_all.png'.format(city_name, city_id, model_name)))
plt.show()
