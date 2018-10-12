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


suffix = 'aemo'
np.random.seed(1004)
img_dir, task_dir = utils.get_task_img_folder()

cm = collectionMaker.read_collection(raw_data_path=r'/home/lab/Documents/bohao/data/aemo',
                                     field_name='aus10,aus30,aus50',
                                     field_id='',
                                     rgb_ext='.*rgb',
                                     gt_ext='.*gt',
                                     file_ext='tif',
                                     force_run=False,
                                     clc_name='aemo')
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

'''for rgb_file, rgb_hist_file in zip(rgb_files, rgb_hist_files):
    fig = plt.figure(figsize=(12, 8))
    c_list = ['r', 'g', 'b']
    for c in range(3):
        plt.subplot(331 + 3 * c)
        plt.bar(np.arange(255), spca[c, :], color=c_list[c])
        plt.ylim([0, 40e4])
        if c == 0:
            plt.title('California')

    rgb = ersa_utils.load_file(rgb_file[0])
    rgb_hist = ersa_utils.load_file(rgb_hist_file[0])

    for c in range(3):
        rgb_cnt, _ = np.histogram(rgb[:, :, c], bins=np.arange(256))
        rgb_hist_cnt, _ = np.histogram(rgb_hist[:, :, c], bins=np.arange(256))

        plt.subplot(331 + 1 + 3*c)
        plt.bar(np.arange(255), rgb_cnt, color=c_list[c])
        plt.ylim([0, 40e4])
        if c == 0:
            plt.title('AEMO')
        plt.subplot(331 + 2 + 3*c)
        plt.bar(np.arange(255), rgb_hist_cnt, color=c_list[c])
        plt.ylim([0, 40e4])
        if c == 0:
            plt.title('AEMO Hist Adjust')

    plt.tight_layout()
    plt.show()'''


n_col = 1 + len(rgb_files)
fig = plt.figure(figsize=(18, 8))
c_list = ['r', 'g', 'b']
for c in range(3):
    plt.subplot(3, n_col, 1 + n_col * c)
    plt.bar(np.arange(255), spca[c, :], color=c_list[c])
    plt.ylim([0, 40e4])
    if c == 0:
        plt.title('California')

for plt_cnt, rgb_file in enumerate(rgb_hist_files):
    rgb = ersa_utils.load_file(rgb_file[0])
    for c in range(3):
        rgb_cnt, _ = np.histogram(rgb[:, :, c], bins=np.arange(256))

        plt.subplot(3, n_col, 2 + plt_cnt + n_col*c)
        plt.bar(np.arange(255), rgb_cnt, color=c_list[c])
        plt.ylim([0, 40e4])
        if c == 0:
            plt.title('AEMO HAdj Fig #{}'.format(plt_cnt))

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'aemo_stats_rgb_hist.png'))
plt.show()
