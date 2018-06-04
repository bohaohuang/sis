"""
Do Inception test with ALI models
Use pretrained ResNet50 model
"""
gpu = 1
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
import imageio
import numpy as np
import tensorflow as tf
import utils
from bohaoCustom import uabMakeNetwork_ALI


def make_thumbnail(img_batch):
    img_num, height, width, _ = img_batch.shape
    n_row = int(np.floor(np.sqrt(img_num)))
    img = np.zeros((1, n_row*height, n_row*width, 3))
    for i in range(n_row):
        for j in range(n_row):
            img[0, (i*height):((i+1)*height), (j*width):((j+1)*width)] = img_batch[n_row*i + j]
    return img

# settings
batch_size = 100
input_size = [64, 64]
input_size_fit = (224, 224)
img_dir, task_dir = utils.get_task_img_folder()
img_temp_dir = os.path.join(img_dir, 'temp')
if not os.path.exists(img_temp_dir):
    os.makedirs(img_temp_dir)

# make the model
# define place holder
with open(os.path.join(task_dir, 'kl_record.txt'), 'w+') as f:
    for lr in ['5e-06']:
        for z_dim in [800]:
            latent_num = z_dim

            for DS in [400.0]:
                # generate images
                tf.reset_default_graph()
                X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
                z = tf.placeholder(tf.float32, shape=[None, 1, 1, latent_num], name='z')
                mode = tf.placeholder(tf.bool, name='mode')
                model = uabMakeNetwork_ALI.ALI({'X': X, 'Z': z},
                                               trainable=mode,
                                               input_size=input_size,
                                               batch_size=batch_size,
                                               start_filter_num=64,
                                               z_dim=latent_num,
                                               raw_marginal=None)
                # create graph
                model.create_graph('X', class_num=3)
                model_dir = r'/hdd6/Models/ALI_Inria/ALI_inria_z{}_lrm1_rawm_PS(64, 64)_BS128_EP400_LR{}_DS{}_DR0.1_SFN64' \
                    .format(z_dim, lr, DS)

                with tf.Session() as sess:
                    init = tf.global_variables_initializer()
                    sess.run(init)
                    model.load(model_dir, sess, best_model=False)
                    Z_batch_sample = np.random.normal(size=(model.bs, 1, 1, model.z_dim)).astype(np.float32)
                    valid_img_gen = sess.run(model.G_x, feed_dict={model.inputs['Z']: Z_batch_sample,
                                                                   model.train_g: False, model.train_d: False})
                thumbnail = make_thumbnail(valid_img_gen)
                imageio.imsave(os.path.join(img_dir, 'lr{}_z{}_ds{}_thumb.png'.format(lr, z_dim, DS)),
                               thumbnail[0, :, :, :])
