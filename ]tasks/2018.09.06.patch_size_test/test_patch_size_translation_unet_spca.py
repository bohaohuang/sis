import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import utils
import ersa_utils
import uab_collectionFunctions
from collection import collectionMaker as cm
from bohaoCustom import uabMakeNetwork_UNet


def overlap_reader(img, gt, y, x, patch_size, stride=1, block_size=1, output_size=388):
    ref_val = img[y-92-block_size:y-92, x-92-block_size:x-92, :]

    pad_y = -np.min([0, y - patch_size])
    pad_x = -np.min([0, x - patch_size])
    img = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), 'reflect')
    gt = np.pad(gt, ((pad_y, pad_y), (pad_x, pad_x)), 'reflect')
    new_y = y + pad_y
    new_x = x + pad_x

    for cnt_y, y_slide in enumerate(range(new_y, new_y+output_size-block_size, stride)):
        for cnt_x, x_slide in enumerate(range(new_x, new_x+output_size-block_size, stride)):
            patch_img = img[y_slide-patch_size:y_slide, x_slide-patch_size:x_slide, :]
            patch_gt = gt[y_slide-patch_size:y_slide, x_slide-patch_size:x_slide]
            assert np.all([patch_img[patch_size-92-block_size-cnt_y:patch_size-92-cnt_y,
                           patch_size-92-block_size-cnt_x:patch_size-92-cnt_x, :] == ref_val])

            yield patch_img, patch_gt, cnt_y, cnt_x


if __name__ == '__main__':
    force_run = False
    pretrained_model_dir = r'/hdd6/Models/UNET_rand_gird/UnetCrop_spca_aug_grid_0_PS(572, 572)_BS5_' \
                           r'EP100_LR0.0001_DS60_DR0.1_SFN32'
    blCol = uab_collectionFunctions.uabCollection('spca')
    img_mean = blCol.getChannelMeans([1, 2, 3])

    for field_name in ['Fresno', 'Modesto', 'Stockton']:
        for field_id in range(250, 500):
            y = 800
            x = 800
            patch_size = 572
            stride = 1
            block_size = 300
            output_size = 388
            tf.reset_default_graph()
            record_matrix = []

            img_dir, task_dir = utils.get_task_img_folder()
            save_file_name = os.path.join(task_dir, 'corr_{}{}_ps{}_bs{}_spca.npy'.
                                          format(field_name, field_id, patch_size, block_size))

            # load data
            if not os.path.exists(save_file_name) or force_run:
                rgb_file = os.path.join(r'/media/ei-edl01/data/uab_datasets/spca/data/Original_Tiles',
                                        '{}{}_RGB.jpg'.format(field_name, field_id))
                gt_file = os.path.join(r'/media/ei-edl01/data/uab_datasets/spca/data/Original_Tiles',
                                       '{}{}_GT.png'.format(field_name, field_id))
                try:
                    rgb = ersa_utils.load_file(rgb_file)
                    gt = ersa_utils.load_file(gt_file)
                except OSError:
                    continue
                reader = overlap_reader(rgb, gt, y, x, patch_size, stride, block_size, output_size)

                # make the model
                # define place holder
                X = tf.placeholder(tf.float32, shape=[None, patch_size, patch_size, 3], name='X')
                y = tf.placeholder(tf.int32, shape=[None, patch_size, patch_size, 1], name='y')
                mode = tf.placeholder(tf.bool, name='mode')
                model = uabMakeNetwork_UNet.UnetModelCrop({'X': X, 'Y': y},
                                                          trainable=mode,
                                                          input_size=(patch_size, patch_size),
                                                          batch_size=5, start_filter_num=32)
                # create graph
                model.create_graph('X', class_num=2)

                with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                    init = tf.global_variables_initializer()
                    sess.run(init)
                    model.load(pretrained_model_dir, sess, epoch=95, best_model=False)

                    for patch_img, patch_gt, cnt_y, cnt_x in tqdm(reader, total=7744):
                        x_batch = np.expand_dims(patch_img-img_mean, axis=0)
                        pred = sess.run(model.output, feed_dict={model.inputs['X']: x_batch,
                                                                 model.trainable: False})
                        record_matrix.append(
                            pred[0, output_size-block_size-cnt_y:output_size-cnt_y,
                            output_size-block_size-cnt_x:output_size-cnt_x, 1].flatten())
                record_matrix = np.array(record_matrix)
                np.save(save_file_name, record_matrix)
            else:
                record_matrix = np.load(save_file_name)

            print(record_matrix.shape)
