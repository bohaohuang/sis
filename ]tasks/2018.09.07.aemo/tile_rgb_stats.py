import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import sis_utils
import ersa_utils
import processBlock

np.random.seed(1004)
img_dir, task_dir = sis_utils.get_task_img_folder()

spca_dir = r'/media/ei-edl01/data/uab_datasets/spca/data/Original_Tiles'
aemo_dir = r'/home/lab/Documents/bohao/data/aemo/aemo_pad'
aemohist_dir = r'/hdd/ersa/preprocess/aemo_pad/hist_matching'
force_run = True

spca_files = glob(os.path.join(spca_dir, '*_RGB.jpg'))
idx = np.random.permutation(len(spca_files))
spca_files = [spca_files[i] for i in idx]
aemo_files = glob(os.path.join(aemo_dir, '*rgb.tif'))
#aemo_hist_files = glob(os.path.join(aemohist_dir, '*histRGB.tif'))
aemo_hist_files = glob(os.path.join(r'/home/lab/Documents/bohao/data/aemo/aemo_hist2', '*rgb.tif'))

save_file = os.path.join(task_dir, 'spca_tile_stats.npy')
def get_spcastats():
    print('Extacting panel pixels in spca...')
    spca_stats = np.zeros((3, 255))
    for rgb_file in tqdm(spca_files):
        rgb = ersa_utils.load_file(rgb_file)

        for c in range(3):
            cnt, _ = np.histogram(rgb[:, :, c], bins=np.arange(256))
            spca_stats[c, :] += cnt
    spca_stats = spca_stats / len(spca_files)
    return spca_stats
spca = processBlock.ValueComputeProcess('spca_tile_stats', task_dir, save_file, get_spcastats).run(force_run).val

save_file = os.path.join(task_dir, 'aemo_tile_stats.npy')
def get_spcastats():
    print('Extracting panel pixels in aemo...')
    aemo_stats = np.zeros((3, 255))
    for rgb_file in tqdm(aemo_files):
        rgb = ersa_utils.load_file(rgb_file)

        for c in range(3):
            cnt, _ = np.histogram(rgb[:, :, c], bins=np.arange(256))
            aemo_stats[c, :] += cnt
        aemo_stats = aemo_stats / len(aemo_files)
    return aemo_stats
aemo = processBlock.ValueComputeProcess('aemo_tile_stats', task_dir, save_file, get_spcastats).run(force_run).val

save_file = os.path.join(task_dir, 'aemohist3_tile_stats.npy')
def get_spcastats():
    print('Extracting panel pixels in aemo...')
    aemo_stats = np.zeros((3, 255))
    for rgb_file in tqdm(aemo_hist_files):
        rgb = ersa_utils.load_file(rgb_file)

        for c in range(3):
            cnt, _ = np.histogram(rgb[:, :, c], bins=np.arange(256))
            aemo_stats[c, :] += cnt
        aemo_stats = aemo_stats / len(aemo_hist_files)
    return aemo_stats
aemohist = processBlock.ValueComputeProcess('aemohist3_tile_stats', task_dir, save_file, get_spcastats).run(force_run).val

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
plt.ylabel('cnt')
plt.tight_layout()
#plt.savefig(os.path.join(img_dir, 'panel_rgb_stats3.png'))
plt.show()
