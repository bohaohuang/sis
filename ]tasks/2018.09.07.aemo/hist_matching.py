import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import ersa_utils
import rio_hist.match
from preprocess import histMatching


def hist_match(img_s, img_t, nbr_bins=255):
    im_res = img_s.copy()
    for d in range(img_s.shape[2]):
        im_hist_s, bins = np.histogram(img_s[:, :, d].flatten(), nbr_bins, normed=True)
        im_hist_t, bins = np.histogram(img_t[:, :, d].flatten(), nbr_bins, normed=True)

        cdfsrc = im_hist_s.cumsum()  # cumulative distribution function
        cdfsrc = (255 * cdfsrc / cdfsrc[-1]).astype(np.uint8)  # normalize

        cdftint = im_hist_t.cumsum()  # cumulative distribution function
        cdftint = (255 * cdftint / cdftint[-1]).astype(np.uint8)  # normalize

        im2 = np.interp(img_s[:, :, d].flatten(), bins[:-1], cdfsrc)

        im3 = np.interp(im2, cdftint, bins[:-1])

        im_res[:, :, d] = im3.reshape((img_s.shape[0], img_s.shape[1]))

        return im_res


# get test data
for sample_id in [1, 2, 3]:
    data_dir = r'/media/ei-edl01/data/aemo/samples/0584007740{}0_01'.format(sample_id)
    files = sorted(glob(os.path.join(data_dir, 'TILES', '*.tif')))

    # ref_dir = r'/media/ei-edl01/data/aemo/samples/058400774020_01'
    # ref_file = sorted(glob(os.path.join(ref_dir, 'TILES', '*.tif')))[5]
    ref_file = r'/media/ei-edl01/data/uab_datasets/spca/data/Original_Tiles/Fresno1_RGB.jpg'

    im_s = ersa_utils.load_file(files[5])
    im_t = ersa_utils.load_file(ref_file)

    # hist matching
    hist_save_dir = os.path.join(data_dir, 'hist_match_ct')
    ersa_utils.make_dir_if_not_exist(hist_save_dir)
    ga = histMatching.HistMatching(ref_file, path=hist_save_dir, color_space='Lab')
    ga.run(force_run=True, file_list=files)
