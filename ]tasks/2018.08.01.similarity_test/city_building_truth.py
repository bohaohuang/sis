import os
import imageio
import numpy as np
from tqdm import tqdm


def make_building_truth(ps, task_dir, cnn_name, patchDir, patch_names, force_run=False):
    truth_file_building = os.path.join(task_dir, 'truth_inria_building_{}.npy'.format(cnn_name))
    if not os.path.exists(truth_file_building) or force_run:
        print('Making ground truth building...')
        truth_building = np.zeros(len(patch_names))
        for cnt, file in enumerate(tqdm(patch_names)):
            gt_name = os.path.join(patchDir, '{}_GT_Divide.png'.format(file[:-1]))
            gt = imageio.imread(gt_name)
            portion = np.sum(gt) / (ps * ps)
            if portion > 0.2:
                truth_building[cnt] = 1
        np.save(truth_file_building, truth_building)
    else:
        truth_building = np.load(truth_file_building)
    return truth_building


def make_city_truth(task_dir, cnn_name, patch_names, force_run=False):
    truth_file_city = os.path.join(task_dir, 'truth_inria_city_2048_{}.npy'.format(cnn_name))
    if not os.path.exists(truth_file_city) or force_run:
        print('Making ground truth city...')
        truth_city = np.zeros(len(patch_names))
        city_dict = {'austin': 0, 'chicago': 1, 'kitsap': 2, 'tyrol-w': 3, 'vienna': 4}
        for cnt, file in enumerate(tqdm(patch_names)):
            city_name = ''.join([i for i in file.split('_')[0] if not i.isdigit()])
            truth_city[cnt] = city_dict[city_name]
        np.save(truth_file_city, truth_city)
    else:
        truth_city = np.load(truth_file_city)
    return truth_city
