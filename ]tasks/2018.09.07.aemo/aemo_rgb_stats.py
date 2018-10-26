import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import utils
import ersa_utils

data_dir = r'/home/lab/Documents/bohao/data/aemo/aemo_pad'
tile_ids = ['10', '30', '50']
color_code = ['r', 'g', 'b']
img_dir, task_dir = utils.get_task_img_folder()

plt.figure(figsize=(6, 8))
for tile_cnt, tile_id in enumerate(tile_ids):
    rgb_hist = np.zeros((3, 255))
    rgb_files = glob(os.path.join(data_dir, 'aus{}*rgb.tif'.format(tile_id)))
    for file in rgb_files:
        rgb = ersa_utils.load_file(file)
        for c in range(3):
            cnt, _ = np.histogram(rgb[:, :, c].flatten(), bins=np.arange(256))
            rgb_hist[c, :] += cnt

    plt.subplot(311 + tile_cnt)
    for c in range(3):
        plt.plot(np.arange(255), ersa_utils.savitzky_golay(rgb_hist[c, :], 11, 2), color=color_code[c], linewidth=2)
    plt.title(tile_id)
    plt.ylabel('Counts')

    if tile_cnt == 2:
        plt.xlabel('Intensity')

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'aemo_tile_rgb_stats.png'))
plt.show()
