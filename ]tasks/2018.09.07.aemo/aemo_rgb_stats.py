import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import sis_utils
import ersa_utils

data_dir = r'/home/lab/Documents/bohao/data/aemo/aemo_pad'
tile_ids = ['10', '30', '50']
color_code = ['r', 'g', 'b']
img_dir, task_dir = sis_utils.get_task_img_folder()


plt.figure(figsize=(12, 8))
for tile_cnt, tile_id in enumerate(tile_ids):
    rgb_hist = np.zeros((3, 255))
    gt_hist = np.zeros((3, 255))
    rgb_files = glob(os.path.join(data_dir, 'aus{}*rgb.tif'.format(tile_id)))
    gt_files = glob(os.path.join(data_dir, 'aus{}*d255.tif'.format(tile_id)))
    for file, gt_file in zip(rgb_files, gt_files):
        rgb = ersa_utils.load_file(file)
        gt = ersa_utils.load_file(gt_file)

        rgb_mask = np.sum(rgb.astype(np.float32), axis=2)

        for c in range(3):
            rgb_slice = rgb[:, :, c]
            cnt, _ = np.histogram(rgb_slice[np.where(rgb_mask < 255 * 3)].flatten(), bins=np.arange(256))
            rgb_hist[c, :] += cnt / (5000 * 5000)
            cnt, _ = np.histogram(rgb_slice[np.where(gt == 1)].flatten(), bins=np.arange(256))
            gt_hist[c, :] += cnt / (np.sum(gt))
            gt_hist[c, -1] = 0

    plt.subplot(321 + tile_cnt * 2)
    for c in range(3):
        plt.plot(np.arange(255), ersa_utils.savitzky_golay(rgb_hist[c, :], 11, 2), color=color_code[c], linewidth=2)
    plt.title('Tile-wise RGB Hist ({})'.format(tile_id))
    plt.ylim([-0.005, 0.03])
    plt.ylabel('Probability')

    if tile_cnt == 2:
        plt.xlabel('Intensity')

    plt.subplot(321 + tile_cnt * 2 + 1)
    for c in range(3):
        plt.plot(np.arange(255), ersa_utils.savitzky_golay(gt_hist[c, :], 11, 2), color=color_code[c], linewidth=2)
    plt.title('Panel-wise RGB Hist ({})'.format(tile_id))
    plt.ylim([-0.005, 0.03])
    # plt.ylabel('Counts')

    if tile_cnt == 2:
        plt.xlabel('Intensity')

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'aemo_tile_panel_rgb_stats_noartifacts.png'))
plt.show()
