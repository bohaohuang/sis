import tensorflow as tf
from nn import unet, nn_utils
from collection import collectionMaker
from bohaoCustom import uabMakeNetwork_UNet

class_num = 2
patch_size = (572, 572)
tile_size = (5000, 5000)
batch_size = 1
gpu = 0
model_dir = r'/hdd6/Models/Inria_decay/UnetCrop_inria_decay_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60.0_DR0.1_SFN32'

cm = collectionMaker.read_collection('inria')
#cm.print_meta_data()

file_list_valid = cm.load_files(field_id=','.join(str(i) for i in range(5)), field_ext='RGB,gt_d255')
chan_mean = cm.meta_data['chan_mean']

nn_utils.tf_warn_level(3)
model = unet.UNet(class_num, patch_size)

feature = tf.placeholder(dtype=tf.float32, shape=[None, patch_size[0], patch_size[1], 3])
model.create_graph(feature)

model_names_ersa = [i.name for i in tf.trainable_variables()]
for n in model_names_ersa:
    print(n)

tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=[None, patch_size[0], patch_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, patch_size[0], patch_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = uabMakeNetwork_UNet.UnetModelCrop({'X': X, 'Y': y},
                                          trainable=mode, input_size=patch_size,
                                          start_filter_num=32)
model.create_graph('X', class_num=2)

model_names_uab = [i.name for i in tf.trainable_variables()]
for n in model_names_uab:
    print(n)

for e, u in zip(model_names_ersa[1:], model_names_uab):
    assert e == u
    print(e, u)

'''model.evaluate(file_list_valid, patch_size, tile_size, batch_size, chan_mean, model_dir, gpu,
               save_result_parent_dir='spca', sfn=32, force_run=True, score_results=True, split_char='.',
               load_epoch_num=95)'''
