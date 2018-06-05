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
from glob import glob
from tqdm import tqdm


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


def compute_distance(set_1, set_2, patch_size, patch_num):
    res50 = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet')
    train_vectors = []
    valid_vectors = []
    print('Extract vectors from set 1...')
    for file in tqdm(set_1):
        patches = sparse_patchify(file, patch_size, patch_num)
        train_vectors.append(res50.predict(patches))
    train_vectors = np.concatenate(train_vectors)

    print('Extract vectors from set 2...')
    for file in tqdm(set_2):
        patches = sparse_patchify(file, patch_size, patch_num)
        valid_vectors.append(res50.predict(patches))
    valid_vectors = np.concatenate(valid_vectors)

    print('Compute Distance...')
    dist = 0
    for i in train_vectors:
        dist += np.sum(np.sum(np.square(valid_vectors - i), axis=1))
    return dist/train_vectors.shape[0]


# get train valid images set
'''img_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles'
rgb_files = glob(os.path.join(img_dir, '*_RGB.tif'))
train_set, valid_set = seperate_rgb_files(rgb_files)

# compute distance
patch_size = 224
dist = []
for i in range(5):
    dist.append(compute_distance(train_set[0], valid_set[i], patch_size, 100))
    print(i, dist[-1])'''

dist = np.array([591.5087363891602,
                 849.8030891113282,
                 602.1756367797851,
                 569.3498864746093,
                 567.6342014770507])
dist = dist / 500

dist_neg = -np.array(dist)
prob = np.exp(dist_neg) / np.sum(np.exp(dist_neg), axis=0)
print(prob)
