import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import utils
import ersa_utils
import processBlock

np.random.seed(1004)
img_dir, task_dir = utils.get_task_img_folder()

spca_dir = r'/media/ei-edl01/data/uab_datasets/spca/data/Original_Tiles'
aemo_dir = r'/home/lab/Documents/bohao/data/aemo/aemo_pad'
aemohist_dir = r'/hdd/ersa/preprocess/aemo_pad/hist_matching'

spca_files = glob(os.path.join(spca_dir, '*_RGB.jpg'))
idx = np.random.permutation(len(spca_files))
spca_files = [spca_files[i] for i in idx]
aemo_files = glob(os.path.join(aemo_dir, '*rgb.tif'))
aemo_hist_files = glob(os.path.join(aemohist_dir, '*histRGB.tif'))

save_file = os.path.join(task_dir, 'spca_panel_stats.npy')
def get_spcastats():
    print('Extacting panel pixels in spca...')
    spca_stats = np.zeros((3, 255))
    for rgb_file in tqdm(spca_files):
        gt_file = rgb_file[:-7] + 'GT.png'
        rgb = ersa_utils.load_file(rgb_file)
        gt = ersa_utils.load_file(gt_file)

        for c in range(3):
            cnt, _ = np.histogram(rgb[:, :, c] * gt, bins=np.arange(256))
            if np.sum(gt) > 0:
                spca_stats[c, :] += cnt / np.sum(gt)
    spca_stats = spca_stats / len(spca_files)
    return spca_stats
spca = processBlock.ValueComputeProcess('spca_panel_stats', task_dir, save_file, get_spcastats).run().val

save_file = os.path.join(task_dir, 'aemo_panel_stats.npy')
def get_spcastats():
    print('Extracting panel pixels in aemo...')
    aemo_stats = np.zeros((3, 255))
    for rgb_file in tqdm(aemo_files):
        gt_file = rgb_file[:-7] + 'gt_d255.tif'
        rgb = ersa_utils.load_file(rgb_file)
        gt = ersa_utils.load_file(gt_file)

        for c in range(3):
            cnt, _ = np.histogram(rgb[:, :, c] * gt, bins=np.arange(256))
            aemo_stats[c, :] += cnt / np.sum(gt)
        aemo_stats = aemo_stats / len(aemo_files)
    return aemo_stats
aemo = processBlock.ValueComputeProcess('aemo_panel_stats', task_dir, save_file, get_spcastats).run().val

save_file = os.path.join(task_dir, 'aemohist_panel_stats.npy')
def get_spcastats():
    print('Extracting panel pixels in aemo...')
    aemo_stats = np.zeros((3, 255))
    for rgb_file in tqdm(aemo_hist_files):
        gt_file = os.path.join(aemo_dir, os.path.basename(rgb_file[:-15]) + 'gt_d255.tif')
        rgb = ersa_utils.load_file(rgb_file)
        gt = ersa_utils.load_file(gt_file)

        for c in range(3):
            cnt, _ = np.histogram(rgb[:, :, c] * gt, bins=np.arange(256))
            aemo_stats[c, :] += cnt / np.sum(gt)
        aemo_stats = aemo_stats / len(aemo_files)
    return aemo_stats
aemohist = processBlock.ValueComputeProcess('aemohist_panel_stats', task_dir, save_file, get_spcastats).run().val

plt.figure(figsize=(8, 6))
color_list = ['r', 'g', 'b']
spca[:, 0] = 0
aemo[:, 0] = 0
aemohist[:, 0] = 0
ax1 = plt.subplot(311)
for i in range(3):
    plt.plot(spca[i, :], color_list[i])
plt.title('RGB Distribution of Panel Pixels')
plt.ylabel('cnt/#panel pixels')
ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
for i in range(3):
    plt.plot(aemo[i, :], color_list[i])
plt.ylabel('cnt/#panel pixels')
ax3 = plt.subplot(313, sharex=ax1, sharey=ax1)
for i in range(3):
    plt.plot(aemohist[i, :], color_list[i])
plt.xlabel('Intensity')
plt.ylabel('cnt/#panel pixels')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'panel_rgb_stats.png'))
plt.show()
