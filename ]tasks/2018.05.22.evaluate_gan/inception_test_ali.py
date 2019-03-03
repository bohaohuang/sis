"""
Do Inception test with ALI models
Use pretrained ResNet50 model
"""
gpu = 1
import os
import time
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
import keras
import imageio
import scipy.misc
import numpy as np
import tensorflow as tf
from scipy.stats import entropy
import sis_utils
from bohaoCustom import uabMakeNetwork_ALI


def patch_iterator(parent_dir, file_list, patch_size, img_mean):
    for files in file_list:
        img = np.zeros((patch_size[0], patch_size[1], 3), dtype=np.uint8)
        for cnt, f in enumerate(files[:3]):
            img[:, :, cnt] = scipy.misc.imresize(imageio.imread(os.path.join(parent_dir, f)), patch_size)
        yield np.expand_dims(img - img_mean, axis=0)


def batch_resize(batch_img, resize_shape, mult=255):
    n, _, _, _ = batch_img.shape
    resize_img = np.zeros((n, resize_shape[0], resize_shape[1], 3), dtype=np.uint8)
    for i in range(n):
        resize_img[i, :, :, :] = scipy.misc.imresize(batch_img[i, :, :, :] * mult, resize_shape)
    return resize_img


# settings
batch_size = 500
input_size = [64, 64]
input_size_fit = (224, 224)
img_dir, task_dir = sis_utils.get_task_img_folder()
TEST_SAMPLE = 50000
BASE_SAMPLE = 5000
img_temp_dir = os.path.join(img_dir, 'temp')
if not os.path.exists(img_temp_dir):
    os.makedirs(img_temp_dir)

# make the model
# define place holder
with open(os.path.join(task_dir, 'kl_record.txt'), 'w+') as f:
    for lr in ['1e-05', '5e-06']:
        for z_dim in [500, 800, 1000]:
            latent_num = z_dim

            for DS in [400.0, 200.0]:
                start_time = time.time()

                p_yx = np.zeros((TEST_SAMPLE, 1000))
                p_y = BASE_SAMPLE/TEST_SAMPLE * np.ones(1000)

                for cnt, eval_iter in enumerate(range(0, TEST_SAMPLE, BASE_SAMPLE)):
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

                        for cnt_2, _ in enumerate(range(0, BASE_SAMPLE, model.bs)):
                            Z_batch_sample = np.random.normal(size=(model.bs, 1, 1, model.z_dim)).astype(np.float32)
                            valid_img_gen = sess.run(model.G_x, feed_dict={model.inputs['Z']: Z_batch_sample,
                                                                           model.train_g: False, model.train_d: False})
                            resize_img = batch_resize(valid_img_gen, input_size_fit)
                            np.save(os.path.join(img_temp_dir, 'file_{}.npy'.format(cnt_2)),
                                    batch_resize(valid_img_gen, input_size_fit))

                    # test images
                    keras.backend.clear_session()
                    tf.reset_default_graph()
                    res50 = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet')
                    for cnt_2, _ in enumerate(range(0, BASE_SAMPLE, model.bs)):
                        fc1000 = res50.predict(np.load(os.path.join(img_temp_dir, 'file_{}.npy'.format(cnt_2))))
                        p_yx[cnt*BASE_SAMPLE + cnt_2*model.bs: cnt*BASE_SAMPLE + (cnt_2+1)*model.bs, :] = fc1000
                        y = np.argmax(fc1000, axis=1)
                        y_hist, _ = np.histogram(y, bins=1000)
                        p_y += y_hist

                # calculate kl divergence
                kl = 0
                for i in range(TEST_SAMPLE):
                    kl += entropy(p_yx[i, :], p_y)
                kl = kl/TEST_SAMPLE

                print('lr:{} z:{} ds:{}, kl={:.3f}, duration={:.2f}'.format(lr, z_dim, DS, kl,
                                                                            time.time() - start_time))
                f.write('lr{} z{} ds{} {}\n'.format(lr, z_dim, DS, kl))
