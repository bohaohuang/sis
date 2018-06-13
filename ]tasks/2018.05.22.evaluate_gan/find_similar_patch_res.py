import os
import keras
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
model_name ='res50'
city_name = 'tyrol-w'
city_id = 1
img_name = '{}{}_RGB.tif'.format(city_name, city_id)
img = imageio.imread(os.path.join(inria_dir, img_name))

tile_dim = [5000, 5000]
patch_size = [321, 321]
res50_size = [224, 224]

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# load encoded features in valid set
feature_file = os.path.join(task_dir, '{}_inria_2048.csv'.format(model_name))
feature = pd.read_csv(feature_file, sep=',', header=None).values
patch_file = os.path.join(task_dir, '{}_inria_2048.txt'.format(model_name))
with open(patch_file, 'r') as f:
    patch_names = f.readlines()

# load gmm model
model_file_name = os.path.join(task_dir, 'gmm_models_2048_{}_{}.npy'.format(model_name, 150))
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
res50 = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet')
fc2048 = keras.models.Model(inputs=res50.input, outputs=res50.get_layer('flatten_1').output)

# evaluate
while True:
    patch, start_x, start_y = random_crop(img, res50_size[0], res50_size[1])
    img2encode = np.expand_dims(patch, axis=0)
    encoded = fc2048.predict(img2encode).reshape((-1,))

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
    plt.savefig(os.path.join(img_dir, 'similar_patch_{}_{}_({},{})_2048.png'.
                             format(city_name, city_id, start_x, start_y)))
    plt.show()
