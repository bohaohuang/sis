import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import uabCrossValMaker
import uab_collectionFunctions
import uab_DataHandlerFunctions
from bohaoCustom import uabDataReader


def get_patch_by_name(patch_dir, p_name, patch_size):
    img = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    for i in range(3):
        img_name = p_name[i] # '{}_RGB{}.jpg'.format(p_name, i)
        img[:, :, i] = imageio.imread(os.path.join(patch_dir, img_name))
    return img


patch_prob = np.load('/media/ei-edl01/user/bh163/tasks/2018.06.01.domain_selection/patch_prob_austin_2048.npy')
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']

# create collection
# the original file is in /ei-edl01/data/uab_datasets/inria
blCol = uab_collectionFunctions.uabCollection('inria')
img_mean = blCol.getChannelMeans([0, 1, 2])

# extract patches
extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4],  # extract all 4 channels
                                                cSize=(321, 321),  # patch size as 572*572
                                                numPixOverlap=0,  # overlap as 92
                                                extSave=['jpg', 'jpg', 'jpg', 'png'],
                                                # save rgb files as jpg and gt as png
                                                isTrain=True,
                                                gtInd=3,
                                                pad=0)  # pad around the tiles
patchDir = extrObj.run(blCol)

# make data reader
chipFiles = os.path.join(patchDir, 'fileList.txt')
# use uabCrossValMaker to get fileLists for training and validation
idx, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'force_tile')
# use first 5 tiles for validation
file_list_train = uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(6, 37)])
file_list_valid = uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(0, 6)])

patch_id_dict = {}
for cnt, item in enumerate(file_list_train):
    p_name = '_'.join(item[0].split('_')[:2])
    patch_id_dict[p_name] = cnt

dataReader_train = uabDataReader.ImageLabelReaderPatchSampleControl(
    [3], [0, 1, 2], patchDir, file_list_train, (321, 321), 100, patch_prob, patch_name=True,
    block_mean=np.append([0], img_mean))

patch_cnt = np.zeros(len(file_list_train), dtype=np.uint64)
c_cnt = np.zeros(5)
city_dict = {'aus': 0, 'chi': 1, 'kit': 2, 'tyr': 3, 'vie': 4}
for reader_cnt in tqdm(range(100000)):
    idx_batch = np.random.choice(len(file_list_train), 100, p=patch_prob)
    for i in idx_batch:
        row = file_list_train[i]
        p_name = '_'.join(row[0].split('_')[:2])
        c_cnt[city_dict[p_name[:3]]] += 1
        patch_cnt[patch_id_dict[p_name]] += 1
plt.bar(np.arange(5), c_cnt)
plt.xticks(np.arange(5), city_list)
plt.show()

top_num = 28
top_idx = np.argsort(patch_cnt)[::-1][:top_num]
c_cnt = np.zeros(5)
plt.figure(figsize=(18, 10))
for i in range(top_num):
    plt.subplot(4, 7, 1+i)
    plt.imshow(get_patch_by_name(patchDir, file_list_train[top_idx[i]], 321))
    plt.axis('off')
    p_name = '_'.join(file_list_train[top_idx[i]][0].split('_')[:2])
    p_num = patch_cnt[top_idx[i]]
    c_cnt[city_dict[p_name[:3]]] += 1
    plt.title('{}:{}'.format(p_name, p_num))
plt.tight_layout()
print(c_cnt)

plt.figure(figsize=(8, 5))
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
city_key = [city_dict[i[0][:3]] for i in file_list_train]
sort_city_key = np.array([city_key[i] for i in np.argsort(patch_cnt)[::-1]])
for i in range(5):
    plt.bar(np.arange(len(file_list_train))[sort_city_key == i],
            np.sort(patch_cnt)[::-1][sort_city_key == i], label=city_list[i])
plt.ylabel('cnt')
plt.legend()
plt.tight_layout()

plt.show()
