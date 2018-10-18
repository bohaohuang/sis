import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import utils
import ersa_utils
import processBlock
from preprocess import histMatching
from collection import collectionMaker, collectionEditor

np.random.seed(1004)
img_dir, task_dir = utils.get_task_img_folder()

# get spca stats
save_file = os.path.join(task_dir, 'spca_rgb_stats.npy')
def get_spcastats():
    spca_dir = r'/media/ei-edl01/data/uab_datasets/spca/data/Original_Tiles'
    spca_files = glob(os.path.join(spca_dir, '*_RGB.jpg'))
    idx = np.random.permutation(len(spca_files))
    spca = np.zeros((3, 255))
    for i in tqdm(idx[:100]):
        rgb = ersa_utils.load_file(spca_files[i])
        for c in range(3):
            cnt, _ = np.histogram(rgb[:, :, c], bins=np.arange(256))
            spca[c, :] += cnt
    spca = spca / 100
    return spca
spca = processBlock.ValueComputeProcess('spca_rgb_stats', task_dir, save_file, get_spcastats).run().val

n_col = 5
fig = plt.figure(figsize=(14, 8))
c_list = ['r', 'g', 'b']
for c in range(3):
    plt.subplot(3, n_col, 1 + n_col * c)
    plt.bar(np.arange(255), spca[c, :], color=c_list[c])
    plt.ylim([0, 40e4])
    if c == 0:
        plt.title('California')


# get before pad stats
suffix = 'aemo'
cm = collectionMaker.read_collection(raw_data_path=r'/home/lab/Documents/bohao/data/{}'.format(suffix),
                                     field_name='aus10,aus30,aus50',
                                     field_id='',
                                     rgb_ext='.*rgb',
                                     gt_ext='.*gt',
                                     file_ext='tif',
                                     force_run=False,
                                     clc_name=suffix)
gt_d255 = collectionEditor.SingleChanMult(cm.clc_dir, 1/255, ['.*gt', 'gt_d255']).\
    run(force_run=False, file_ext='tif', d_type=np.uint8,)
cm.replace_channel(gt_d255.files, True, ['gt', 'gt_d255'])
# hist matching
ref_file = r'/media/ei-edl01/data/uab_datasets/spca/data/Original_Tiles/Fresno1_RGB.jpg'
ga = histMatching.HistMatching(ref_file, color_space='RGB', ds_name=suffix)
file_list = [f[0] for f in cm.meta_data['rgb_files']]
hist_match = ga.run(force_run=False, file_list=file_list)
cm.add_channel(hist_match.get_files(), '.*rgb_hist')
cm.print_meta_data()
aemo_files = cm.load_files(field_name='aus10,aus30,aus50', field_id='', field_ext='.*rgb')
rgb_files = aemo_files[:6]
rgb_hist_files = aemo_files[-6:]

aemo = np.zeros((3, 255))
for rgb_file in rgb_files:
    rgb = ersa_utils.load_file(rgb_file[0])
    for c in range(3):
        rgb_cnt, _ = np.histogram(rgb[:, :, c], bins=np.arange(256))
        aemo[c, :] += rgb_cnt
aemo = aemo / len(rgb_files)
for c in range(3):
    plt.subplot(3, n_col, 2 + n_col * c)
    plt.bar(np.arange(255), aemo[c, :], color=c_list[c])
    plt.ylim([0, 40e4])
    if c == 0:
        plt.title('AEMO')

aemo_hist = np.zeros((3, 255))
for rgb_file in rgb_hist_files:
    rgb = ersa_utils.load_file(rgb_file[0])
    for c in range(3):
        rgb_cnt, _ = np.histogram(rgb[:, :, c], bins=np.arange(256))
        aemo_hist[c, :] += rgb_cnt
aemo_hist = aemo_hist / len(rgb_hist_files)
for c in range(3):
    plt.subplot(3, n_col, 3 + n_col * c)
    plt.bar(np.arange(255), aemo_hist[c, :], color=c_list[c])
    plt.ylim([0, 40e4])
    if c == 0:
        plt.title('AEMO Hist')


# get before pad stats
suffix = 'aemo_pad'
cm = collectionMaker.read_collection(raw_data_path=r'/home/lab/Documents/bohao/data/{}'.format(suffix),
                                     field_name='aus10,aus30,aus50',
                                     field_id='',
                                     rgb_ext='.*rgb',
                                     gt_ext='.*gt',
                                     file_ext='tif',
                                     force_run=False,
                                     clc_name=suffix)
gt_d255 = collectionEditor.SingleChanMult(cm.clc_dir, 1/255, ['.*gt', 'gt_d255']).\
    run(force_run=False, file_ext='tif', d_type=np.uint8,)
cm.replace_channel(gt_d255.files, True, ['gt', 'gt_d255'])
# hist matching
ref_file = r'/media/ei-edl01/data/uab_datasets/spca/data/Original_Tiles/Fresno1_RGB.jpg'
ga = histMatching.HistMatching(ref_file, color_space='RGB', ds_name=suffix)
file_list = [f[0] for f in cm.meta_data['rgb_files']]
hist_match = ga.run(force_run=False, file_list=file_list)
cm.add_channel(hist_match.get_files(), '.*rgb_hist')
cm.print_meta_data()
aemo_files = cm.load_files(field_name='aus10,aus30,aus50', field_id='', field_ext='.*rgb')
rgb_files = aemo_files[:6]
rgb_hist_files = aemo_files[-6:]

aemo = np.zeros((3, 255))
for rgb_file in rgb_files:
    rgb = ersa_utils.load_file(rgb_file[0])
    for c in range(3):
        rgb_cnt, _ = np.histogram(rgb[:, :, c], bins=np.arange(256))
        aemo[c, :] += rgb_cnt
aemo = aemo / len(rgb_files)
for c in range(3):
    plt.subplot(3, n_col, 4 + n_col * c)
    plt.bar(np.arange(255), aemo[c, :], color=c_list[c])
    plt.ylim([0, 40e4])
    if c == 0:
        plt.title('AEMO Pad')

aemo_hist = np.zeros((3, 255))
for rgb_file in rgb_hist_files:
    rgb = ersa_utils.load_file(rgb_file[0])
    for c in range(3):
        rgb_cnt, _ = np.histogram(rgb[:, :, c], bins=np.arange(256))
        aemo_hist[c, :] += rgb_cnt
aemo_hist = aemo_hist / len(rgb_hist_files)
for c in range(3):
    plt.subplot(3, n_col, 5 + n_col * c)
    plt.bar(np.arange(255), aemo_hist[c, :], color=c_list[c])
    plt.ylim([0, 40e4])
    if c == 0:
        plt.title('AEMO Pad Hist')

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'aemo_stats_rgb_pad_hist_agg_cmp.png'))
plt.show()
