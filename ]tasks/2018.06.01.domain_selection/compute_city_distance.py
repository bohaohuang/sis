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
import sis_utils


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


def compute_distance_tile(set_1, set_2, patch_size, patch_num):
    _, task_dir = sis_utils.get_task_img_folder()
    res50 = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet')
    train_vectors = []
    valid_vectors = []
    train_vector_name = os.path.join(task_dir, 'train_l{}_ps{}_pn{}.npy'.format(len(set_1), patch_size, patch_num))
    valid_vector_name = os.path.join(task_dir, 'valid_l{}_ps{}_pn{}.npy'.format(len(set_2), patch_size, patch_num))

    if not os.path.exists(train_vector_name):
        print('Extract vectors from set 1...')
        for file in tqdm(set_1):
            patches = sparse_patchify(file, patch_size, patch_num)
            train_vectors.append(res50.predict(patches))
        train_vectors = np.concatenate(train_vectors)
        np.save(train_vector_name, train_vectors)
    else:
        print('Load vectors from set 1...')
        train_vectors = np.load(train_vector_name)

    if not os.path.exists(valid_vector_name):
        print('Extract vectors from set 2...')
        for file in tqdm(set_2):
            patches = sparse_patchify(file, patch_size, patch_num)
            valid_vectors.append(res50.predict(patches))
        valid_vectors = np.concatenate(valid_vectors)
        np.save(valid_vector_name, valid_vectors)
    else:
        print('Load vectors from set 2...')
        valid_vectors = np.load(valid_vector_name)

    dist_list = []
    tile_id_list = []
    dist = np.zeros(train_vectors.shape[0])
    for cnt, val_vec in enumerate(valid_vectors):
        dist += np.sum(np.square(train_vectors - val_vec), axis=1)
        if cnt % patch_num == 0:
            dist_list.append(np.min(dist))
            tile_id_list.append(np.argmin(dist))
            dist = np.zeros(train_vectors.shape[0])

    return dist_list, tile_id_list


# get train valid images set
img_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles'
rgb_files = glob(os.path.join(img_dir, '*_RGB.tif'))
train_set, valid_set = seperate_rgb_files(rgb_files)

# compute distance city-wise
'''patch_size = 224
dist = []
for i in range(5):
    dist.append(compute_distance(train_set[0], valid_set[i], patch_size, 100))
    print(i, dist[-1])
dist_neg = -np.array(dist)
prob = np.exp(dist_neg) / np.sum(np.exp(dist_neg), axis=0)
print(prob)'''

# compute distance tile-wise
val = []
patch_size = 224
patch_num = 25
for i in range(5):
    val += valid_set[i]
dist_list, tile_id_list = compute_distance_tile(train_set[0], val, patch_size, patch_num)
sort_idx = np.argsort(dist_list)[::-1]

'''import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.subplot(211)
plt.hist(dist_list, bins=100)
for i in range(0, 155, 31):
    plt.axvline(dist_list[sort_idx[i]], color='r', linestyle='--')
plt.subplot(212)
plt.hist(tile_id_list, bins=5*patch_num)
for i in range(0, 5 * patch_num, patch_num):
    plt.axvline(i, color='r', linestyle='--')
plt.show()'''

file_group = [[] for i in range(5)]
file_group_save = [[] for i in range(5)]
for cnt in range(len(val)):
    file_group[cnt // 31].append(val[sort_idx[cnt]])
    file_name = os.path.basename(val[sort_idx[cnt]])
    name_id = file_name.split('_')[0]
    file_group_save[cnt // 31].append(name_id)
_, task_dir = sis_utils.get_task_img_folder()
np.save(os.path.join(task_dir, 'file_group_austin.npy'), file_group_save)

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))
city_dict = {'austin': 0, 'chicago': 1, 'kitsap': 2, 'tyrol-w': 3, 'vienna': 4}
for i in range(5):
    print(len(file_group[i]))
    plt.subplot(151 + i)
    city_cnt = np.zeros(5)

    for j in file_group[i]:
        file_name = os.path.basename(j)
        name_id = file_name.split('_')[0]
        name = ''.join([i for i in name_id if not i.isdigit()])
        city_cnt[city_dict[name]] += 1

    plt.bar(np.arange(5), city_cnt)

plt.show()
