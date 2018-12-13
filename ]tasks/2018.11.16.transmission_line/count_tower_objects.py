import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from natsort import natsorted
from skimage import measure
import utils
import ersa_utils


def count_objects(binary_map):
    _, cnts = measure.label(binary_map, return_num=True)
    return cnts


data_dir = r'/home/lab/Documents/bohao/data/transmission_line/raw'
city_list = ['Tucson', 'Colwich', 'Clyde', 'Wilmington']
img_dir, task_dir = utils.get_task_img_folder()
tower_vals = [3, 1]
tower_names = ['Transmission_Tower', 'Distribution_Tower']
total_counts = np.zeros((len(tower_names), len(city_list)))

tower_dict = {}
for cn in city_list:
    tower_type = {}
    for tt in tower_names:
        tower_type[tt] = [0]
    tower_dict[cn] = tower_type

for tn in tower_names:
    ersa_utils.make_dir_if_not_exist(os.path.join(img_dir, tn))

for city_id in range(len(city_list)):
    gt_files = natsorted(glob(os.path.join(data_dir, '*{}*_multiclass.tif'.format(city_list[city_id]))))
    rgb_files = ['_'.join(a.split('_')[:-1])+'.tif' for a in gt_files]
    for rgb_file_name, gt_file_name in zip(rgb_files, gt_files):
        rgb = ersa_utils.load_file(rgb_file_name)
        gt = ersa_utils.load_file(gt_file_name)
        prefix = os.path.basename(rgb_file_name)[7:-4]

        print('{:<20}: '.format(prefix), end='')
        for t_cnt, (tower_val, tower_name) in enumerate(zip(tower_vals, tower_names)):
            binary_map = (gt == tower_val).astype(np.int)
            tower_cnts = count_objects(binary_map)
            print('{:<15} {:<4}\t'.format(tower_name, tower_cnts), end='')
            total_counts[t_cnt][city_id] += tower_cnts
        print()

plt.figure(figsize=(8, 6))
X = np.arange(len(city_list))
width = 0.35
ax1 = plt.subplot(211)
plt.bar(X, total_counts[0, :])
plt.ylabel('Cnts')
plt.title(tower_names[0])
ax2 = plt.subplot(212, sharex=ax1)
plt.bar(X, total_counts[1, :])
plt.xticks(X, city_list)
plt.xlabel('City Name')
plt.ylabel('Cnts')
plt.title(tower_names[1])
plt.tight_layout()
print(total_counts)
plt.savefig(os.path.join(img_dir, 'tower_num_cnts.png'))
plt.show()
