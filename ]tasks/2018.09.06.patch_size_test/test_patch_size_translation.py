import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import sis_utils
import ersa_utils
import uab_collectionFunctions
from collection import collectionMaker as cm
from bohaoCustom import uabMakeNetwork_DeepLabV2


def overlap_reader(img, gt, y, x, patch_size, stride=1, block_size=1):
    ref_val = img[y:y+block_size, x:x+block_size, :]

    pad_y = -np.min([0, y - patch_size])
    pad_x = -np.min([0, x - patch_size])
    img = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), 'reflect')
    gt = np.pad(gt, ((pad_y, pad_y), (pad_x, pad_x)), 'reflect')
    new_y = y + pad_y
    new_x = x + pad_x

    for cnt_y, y_slide in enumerate(range(new_y-patch_size+block_size, new_y, stride)):
        for cnt_x, x_slide in enumerate(range(new_x-patch_size+block_size, new_x, stride)):
            patch_img = img[y_slide:y_slide+patch_size, x_slide:x_slide+patch_size, :]
            patch_gt = gt[y_slide:y_slide+patch_size, x_slide:x_slide+patch_size]
            target_y = patch_size - cnt_y
            target_x = patch_size - cnt_x
            assert np.all([patch_img[target_y-block_size:target_y, target_x-block_size:target_x, :] == ref_val])

            yield patch_img, patch_gt, cnt_y, cnt_x, target_y, target_x


if __name__ == '__main__':
    force_run = False
    pretrained_model_dir = r'/hdd6/Models/Inria_decay/DeeplabV3_inria_decay_0_PS(321, 321)_BS5_' \
                           r'EP100_LR1e-05_DS40.0_DR0.1_SFN32'
    for field_name in ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']:
        for field_id in ['1', '2', '3', '4', '5']:
            y = 500
            x = 500
            patch_size = 321
            stride = 1
            block_size = 100
            tf.reset_default_graph()
            record_matrix = []

            img_dir, task_dir = sis_utils.get_task_img_folder()
            save_file_name = os.path.join(task_dir, 'corr_{}{}_ps{}_bs{}.npy'.
                                          format(field_name, field_id, patch_size, block_size))

            blCol = uab_collectionFunctions.uabCollection('inria')
            img_mean = blCol.getChannelMeans([0, 1, 2])

            # load data
            if not os.path.exists(save_file_name) or force_run:
                clc = cm.read_collection(clc_name='Inria')
                clc.print_meta_data()
                files = clc.load_files(field_name=field_name, field_id=field_id, field_ext='RGB,gt_d255')
                rgb = ersa_utils.load_file(files[0][0])
                gt = ersa_utils.load_file(files[0][1])
                reader = overlap_reader(rgb, gt, y, x, patch_size, stride, block_size)

                # make the model
                # define place holder
                X = tf.placeholder(tf.float32, shape=[None, patch_size, patch_size, 3], name='X')
                y = tf.placeholder(tf.int32, shape=[None, patch_size, patch_size, 1], name='y')
                mode = tf.placeholder(tf.bool, name='mode')
                model = uabMakeNetwork_DeepLabV2.DeeplabV3({'X': X, 'Y': y},
                                                           trainable=mode,
                                                           input_size=(patch_size, patch_size),
                                                           batch_size=5, start_filter_num=32)
                # create graph
                model.create_graph('X', class_num=2)

                with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                    init = tf.global_variables_initializer()
                    sess.run(init)
                    model.load(pretrained_model_dir, sess, epoch=95, best_model=False)

                    for patch_img, patch_gt, cnt_y, cnt_x, target_y, target_x in tqdm(reader, total=48841):
                        x_batch = np.expand_dims(patch_img-img_mean, axis=0)
                        pred = sess.run(model.output, feed_dict={model.inputs['X']: x_batch,
                                                                 model.trainable: False})
                        record_matrix.append(pred[0, target_y-block_size:target_y, target_x-block_size:target_x, 1].flatten())
                record_matrix = np.array(record_matrix)
                np.save(save_file_name, record_matrix)
            else:
                record_matrix = np.load(save_file_name)

            print(record_matrix.shape)
