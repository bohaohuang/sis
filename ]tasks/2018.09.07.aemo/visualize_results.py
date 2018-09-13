import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from skimage import measure
import ersa_utils

# get test data
gammas = [2.5, 1, 2.5]
sample_id = 3
data_dir = r'/media/ei-edl01/data/aemo/samples/0584007740{}0_01'.format(sample_id)
files = sorted(glob(os.path.join(data_dir, 'TILES', '*.tif')))

# adjust gamma
gamma_save_dir = os.path.join(data_dir, 'gamma_adjust')
orig_files = sorted(glob(os.path.join(gamma_save_dir, '*.tif')))
pred_files = sorted(glob(os.path.join(data_dir, 'bh_pred_ct', '*histRGB.tif')))

b_cnt = 0
for rgb, pred in zip(orig_files, pred_files):
    # r = ersa_utils.load_file(rgb)
    p = ersa_utils.load_file(pred)

    '''plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(121)
    plt.imshow(r)
    plt.axis('off')
    ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
    plt.imshow(p)
    plt.axis('off')
    plt.tight_layout()
    plt.show()'''

    lbl = measure.label(p)
    building_num = len(np.unique(lbl))
    print(building_num)
    b_cnt += building_num
print(b_cnt)
