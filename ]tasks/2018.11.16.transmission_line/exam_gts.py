import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from glob import glob
from natsort import natsorted
import ersa_utils

data_dir = r'/home/lab/Documents/bohao/data/transmission_line'
save_dir = os.path.join(data_dir, 'info')

rgb_files = natsorted([a for a in glob(os.path.join(data_dir, 'raw', '*.tif'))
                           if 'multiclass' not in a])
csv_files = natsorted(glob(os.path.join(data_dir, 'raw', '*.csv')))
patch_size = 500
color_dict = {'TT': 'b', 'DT': 'r'}

for rgb_file, csv_file in zip(rgb_files, csv_files):
    npy_file = os.path.basename(rgb_file)[:-3] + 'npy'

    if 'Wilmington' in rgb_file:
        # load data
        coords = ersa_utils.load_file(os.path.join(save_dir, npy_file))
        rgb = ersa_utils.load_file(rgb_file)

        # plot patch
        for item in coords:
            for cell in item:
                if len(cell['label']) > 0:
                    patch = rgb[cell['h']:cell['h']+patch_size, cell['w']:cell['w']+patch_size, :]
                    fig, ax = plt.subplots(1)
                    for l, b in zip(cell['label'], cell['box']):
                        ax.imshow(patch)
                        rect = patches.Rectangle((b[1], b[0]), b[3]-b[1], b[2]-b[0], linewidth=2, edgecolor=color_dict[l],
                                                 facecolor='none')
                        ax.add_patch(rect)
                    plt.axis('off')
                    plt.tight_layout()
                    plt.show()
