import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from skimage import measure
import ersa_utils


def visualize_fn_object(rgb, gt, pred):
    h, w, _ = rgb.shape
    lbl = measure.label(gt)
    building_idx = np.unique(lbl)

    for cnt, idx in enumerate(tqdm(building_idx)):
        on_target = np.sum(pred[np.where(lbl == idx)])
        building_size = np.sum(gt[np.where(lbl == idx)])
        if building_size != 0:
            yield on_target / building_size


ds = 40
pred_dir = r'/hdd/Results/aemo/unet_aemo_PS(572, 572)_BS5_EP60_LR0.001_DS{}_DR0.1/default/pred'.format(ds)
orig_dir = r'/home/lab/Documents/bohao/data/aemo'

pred_files = sorted(glob(os.path.join(pred_dir, '*.png')))
rgb_files = sorted(glob(os.path.join(orig_dir, '*rgb.tif')))[-2:]
gt_files = sorted(glob(os.path.join(orig_dir, '*gt.tif')))[-2:]

rate_list = []
for (pred, rgb, gt) in zip(pred_files, rgb_files, gt_files):
    p = ersa_utils.load_file(pred)
    r = ersa_utils.load_file(rgb)
    g = (ersa_utils.load_file(gt) / 255).astype(np.uint8)

    for rate in visualize_fn_object(r, g, p):
        rate_list.append(rate)

plt.figure(figsize=(8, 6))
plt.hist(rate_list, bins=100)
plt.xlabel('Detection Rate')
plt.ylabel('Cnt')
plt.title('Object-wise Detection Rate (Overall={:.2f})'.format(np.mean(rate_list)*100))
plt.show()
