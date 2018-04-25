import os
import csv
import scipy.spatial
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
import utils
import uabCrossValMaker

# settings
force_run = False
random_seed = 0
img_dir, task_dir = utils.get_task_img_folder()
file_name = os.path.join(task_dir, 'res50_fc1000_inria.csv')
input_size = 321
patchDir = r'/hdd/uab_datasets/Results/PatchExtr/inria/chipExtrReg_cSz321x321_pad0'

dist_file_name = os.path.join(task_dir, 'deeplab_inria_patch_dist.npy')
if not os.path.exists(dist_file_name) or force_run:
    features = []
    with open(file_name, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            features.append(row)
    features = np.array(features).astype(np.float32)
    dist_mat = scipy.spatial.distance.pdist(features, 'euclidean')
    np.save(dist_file_name, dist_mat)
else:
    dist_mat = np.load(dist_file_name)

print(dist_mat.shape)
