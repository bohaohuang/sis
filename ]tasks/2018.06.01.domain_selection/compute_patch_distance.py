"""
This file has the following functions:
1. Extract feature vector from a tile in larger grid
2. Compute distance between two tiles in extracted feature vectors
"""

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import keras
import imageio
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import utils
import uabCrossValMaker
import uab_collectionFunctions
import uab_DataHandlerFunctions


def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]


def get_city_name_id(file_dir):
    file_name = os.path.basename(file_dir)
    name_id = file_name.split('_')[0]
    name = ''.join([i for i in name_id if not i.isdigit()])
    idx = int(''.join([i for i in name_id if i.isdigit()]))
    return name, idx


def seperate_rgb_files(file_list):
    city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
    train_set = [[] for i in range(len(city_list))]
    valid_set = [[] for i in range(len(city_list))]

    for file_name in sorted(file_list):
        name, idx = get_city_name_id(file_name)
        if name in city_list:
            if idx < 6:
                train_set[city_list.index(name)].append(file_name)
            else:
                valid_set[city_list.index(name)].append(file_name)
    return train_set, valid_set


def sparse_patchify(file_name, patch_size, patch_num):
    """
    only works for square images
    """
    img = imageio.imread(file_name)
    w, h, c = img.shape
    patches = np.zeros((patch_num, patch_size, patch_size, c), dtype=np.uint8)
    assert w == h
    patch_n_single = int(np.floor(np.sqrt(patch_num)))
    step = w//patch_n_single
    offset = (step - patch_size) // 2
    cnt = 0
    for x in range(0, w-patch_size, step):
        for y in range(0, h-patch_size, step):
            patches[cnt, :, :, :] = img[x+offset:x+offset+patch_size, y+offset:y+offset+patch_size]
            cnt += 1
    return patches


def read_patch_file(patch_list, patch_size, patch_dir):
    patches = np.zeros((1, patch_size, patch_size, 3), dtype=np.uint8)
    for cnt, file in enumerate(patch_list[:3]):
        patches[0, :, :, cnt] = crop_center(imageio.imread(os.path.join(patch_dir, file)), patch_size, patch_size)
    return patches


def compute_distance_patch(train_set, patches, patch_size, patch_num, patch_dir):
    _, task_dir = utils.get_task_img_folder()
    res50 = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet')
    fc2048 = keras.models.Model(inputs=res50.input, outputs=res50.get_layer('flatten_1').output)
    train_vectors = []
    valid_vectors = []
    train_vector_name = os.path.join(task_dir, 'train_l{}_ps{}_pn{}_2048.npy'.format(len(train_set), patch_size, patch_num))
    valid_vector_name = os.path.join(task_dir, 'patch_vector_2048.npy'.format(len(patches), patch_size, patch_num))

    if not os.path.exists(train_vector_name):
        print('Extract vectors from set 1...')
        for file in tqdm(train_set):
            patches_val = sparse_patchify(file, patch_size, patch_num)
            train_vectors.append(fc2048.predict(patches_val))
        train_vectors = np.concatenate(train_vectors)
        np.save(train_vector_name, train_vectors)
    else:
        print('Load vectors from set 1...')
        train_vectors = np.load(train_vector_name)

    if not os.path.exists(valid_vector_name):
        print('Extract vectors from set 2...')
        for file in tqdm(patches):
            patches_val = read_patch_file(file, patch_size, patch_dir)
            valid_vectors.append(fc2048.predict(patches_val))
        valid_vectors = np.concatenate(valid_vectors)
        np.save(valid_vector_name, valid_vectors)
    else:
        print('Load vectors from set 2...')
        valid_vectors = np.load(valid_vector_name)

    return train_vectors, valid_vectors


def get_patch_prob(train_vec, patch_vec):
    n_sample = patch_vec.shape[0]
    dist = np.zeros(n_sample)
    for cnt, vec in enumerate(tqdm(patch_vec)):
        dists = np.sum(np.square(train_vec - vec), axis=1)
        dist[cnt] = np.mean(dists)
    dist_neg = -dist
    dist_neg = (dist_neg - np.min(dist_neg)) / (np.max(dist_neg) - np.min(dist_neg))
    dist_neg = dist_neg / np.sum(dist_neg)
    '''plt.plot(np.arange(n_sample), np.sort(dist_neg))
    plt.title('{} {}'.format(np.min(dist), np.max(dist)))
    plt.show()
    dist_neg = -np.array(np.exp(dist))/np.max(np.abs(np.exp(dist)))'''
    # return np.exp(dist_neg) / np.sum(np.exp(dist_neg), axis=0)
    return dist_neg


# get train valid images set
img_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles'
rgb_files = glob(os.path.join(img_dir, '*_RGB.tif'))
train_set, valid_set = seperate_rgb_files(rgb_files)
img_dir, task_dir = utils.get_task_img_folder()

# compute distance patch-wise
patch_size = 224
patch_num = 25
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
city_num = 2
blCol = uab_collectionFunctions.uabCollection('inria')
img_mean = blCol.getChannelMeans([0, 1, 2])
extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4],
                                                cSize=(321, 321),
                                                numPixOverlap=0,
                                                extSave=['jpg', 'jpg', 'jpg', 'png'],
                                                isTrain=True,
                                                gtInd=3,
                                                pad=0)
patchDir = extrObj.run(blCol)
chipFiles = os.path.join(patchDir, 'fileList.txt')
idx, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'force_tile')
# use first 5 tiles for validation
file_list_train = uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(6, 37)])

train_vectors, valid_vectors = \
    compute_distance_patch(train_set[city_num], file_list_train, patch_size, patch_num, patchDir)
prob = get_patch_prob(train_vectors, valid_vectors)
plt.plot(np.arange(prob.shape[0]), np.sort(prob))
plt.title('{} {}'.format(np.min(prob), np.max(prob)))
plt.show()
np.save(os.path.join(task_dir, 'patch_prob_{}_2048.npy'.format(city_list[city_num])), prob)

top_num = 1000
top_idx = np.argsort(prob)[::-1][:top_num]
city_dict = {'aus': 0, 'chi': 1, 'kit': 2, 'tyr': 3, 'vie': 4}
city_cnt = np.zeros(5)
for i in range(top_num):
    city_name = file_list_train[top_idx[i]][0][:3]
    city_cnt[city_dict[city_name]] += 1
plt.bar(np.arange(5), city_cnt)

plt.figure()
plt.hist(prob, bins=100)
plt.show()
