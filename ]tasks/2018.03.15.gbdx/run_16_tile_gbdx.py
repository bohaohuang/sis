import os
import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
import utils
import util_functions
import uabDataReader
import uabUtilreader
import uab_collectionFunctions
from bohaoCustom import uabMakeNetwork_UNet
from util_functions import iou_metric

# settings
gpu = 0
batch_size = 1
input_size = [572, 572]
tile_size = [2541, 2541]
util_functions.tf_warn_level(3)
img_dir, task_dir = utils.get_task_img_folder()
tile_cnt = 12

tf.reset_default_graph()

model_dir = r'/media/ei-edl01/data/uab_datasets/sp/]shared_models'
gt_dir = r'/media/ei-edl01/data/uab_datasets/sp/DATA_BUILDING_AND_PANEL'
# compute img mean
data_path = r'/media/ei-edl01/user/as667/ctims'
img_mean = [[], [], []]
imgs = glob(os.path.join(data_path, '*.jpg'))
tile_ids = sorted([os.path.basename(a).split('.')[0][:-4] for a in imgs])

'''for img_file in imgs:
    img_rbg = imageio.imread(img_file)
    tile_sizes.append(img_rbg.shape[:2])
    img_mean[0].extend(img_rbg[:, :, 0].flatten())
    img_mean[1].extend(img_rbg[:, :, 1].flatten())
    img_mean[2].extend(img_rbg[:, :, 2].flatten())
img_mean[0] = np.mean(img_mean[0])
img_mean[1] = np.mean(img_mean[1])
img_mean[2] = np.mean(img_mean[2])
print(img_mean)
print(tile_sizes)'''
img_mean = [58.893722732055885, 67.51008454137144, 68.20242335032503]

# make the model
# define place holder
X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = uabMakeNetwork_UNet.UnetModelCrop({'X':X, 'Y':y},
                                          trainable=mode,
                                          input_size=input_size,
                                          batch_size=5, start_filter_num=32)
# create graph
model.create_graph('X', class_num=2)

# load data
large_tile = ['{}_RGB.jpg'.format(tile_ids[tile_cnt])]
reader = uabDataReader.ImageLabelReader(gtInds=[0],
                                        dataInds=[0],
                                        nChannels=3,
                                        parentDir=data_path,
                                        chipFiles=[large_tile],
                                        chip_size=input_size,
                                        tile_size=tile_size,
                                        batchSize=batch_size,
                                        block_mean=img_mean,
                                        overlap=model.get_overlap(),
                                        padding=np.array((model.get_overlap()/2, model.get_overlap()/2)),
                                        isTrain=False)
test_reader = reader.readManager

# run algo
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    model.load(model_dir, sess)
    result = model.test('X', sess, test_reader)
image_pred = uabUtilreader.un_patchify_shrink(result,
                                              [tile_size[0] + model.get_overlap(), tile_size[1] + model.get_overlap()],
                                              tile_size, input_size,
                                              [input_size[0] - model.get_overlap(), input_size[1] - model.get_overlap()],
                                              overlap=model.get_overlap())
pred = util_functions.get_pred_labels(image_pred) * 255
gt = imageio.imread(os.path.join(gt_dir, '{}_GT.png'.format(tile_ids[tile_cnt].replace('_', '-'))))

cfm_norm = image_pred[:, :, 1]/np.sum(image_pred, axis=2)
cfm_norm = cfm_norm.astype(np.uint8)
#imageio.imsave(os.path.join(data_path, '{}_pred.png'.format(tile_sizes[tile_cnt])), pred)

iou = iou_metric(gt, pred/255)

# view result
lt = imageio.imread(os.path.join(data_path, large_tile[0]))
plt.figure(figsize=(15, 6))
ax1 = plt.subplot(131)
plt.axis('off')
plt.imshow(lt)
ax2 = plt.subplot(132, sharex=ax1, sharey=ax1)
plt.imshow(pred)
plt.title('Pred IoU={:.3f}'.format(iou))
plt.axis('off')
ax3 = plt.subplot(133, sharex=ax1, sharey=ax1)
plt.imshow(gt)
plt.title('GT')
plt.axis('off')
plt.suptitle(tile_ids[tile_cnt])
plt.tight_layout()
plt.savefig(os.path.join(img_dir, '{}_result_cm.png'.format(tile_ids[tile_cnt])))
plt.show()
