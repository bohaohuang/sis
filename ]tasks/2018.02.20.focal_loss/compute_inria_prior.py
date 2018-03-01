import os
import imageio
import numpy as np
from glob import glob
from tqdm import tqdm

gt_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles'
gt_files = glob(os.path.join(gt_dir, '*_GT.tif'))

gt = 0
total = len(gt_files) * 5000 ** 2
for file in tqdm(gt_files):
    gt_img = imageio.imread(file)
    gt += np.sum(gt_img/255)
print(gt/total)
