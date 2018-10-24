import os
import numpy as np
from glob import glob
from tqdm import tqdm
import utils
import ersa_utils
from collection import collectionMaker


def cust_hist_match(dist_s, dist_t, img_s):
    bins = np.arange(dist_s.shape[1]+1)
    im_res = np.zeros_like(img_s)
    for d in range(dist_s.shape[0]):
        im_hist_s = dist_s[d, :] / np.sum(dist_s[d, :])
        im_hist_t = dist_t[d, :] / np.sum(dist_t[d, :])

        cdfsrc = im_hist_s.cumsum()
        cdfsrc = (255 * cdfsrc / cdfsrc[-1]).astype(np.uint8)
        cdftint = im_hist_t.cumsum()
        cdftint = (255 * cdftint / cdftint[-1]).astype(np.uint8)

        im2 = np.interp(img_s[:, :, d].flatten(), bins[:-1], cdfsrc)
        im3 = np.interp(im2, cdftint, bins[:-1])
        im_res[:, :, d] = im3.reshape((img_s.shape[0], img_s.shape[1]))
    return im_res


if __name__ == '__main__':
    img_dir, task_dir = utils.get_task_img_folder()

    # get aemo stats
    cm = collectionMaker.read_collection('aemo_pad')
    cm.print_meta_data()

    aemo_files = cm.load_files(field_name='aus10,aus30,aus50', field_id='', field_ext='.*rgb')
    rgb_files = aemo_files[:6]

    aemo = np.zeros((3, 255))
    for rgb_file in rgb_files:
        rgb = ersa_utils.load_file(rgb_file[0])
        for c in range(3):
            rgb_cnt, _ = np.histogram(rgb[:, :, c], bins=np.arange(256))
            aemo[c, :] += rgb_cnt
    aemo = aemo / len(rgb_files)
    aemo[:, :2] = 0
    aemo[:, -1] = aemo[:, -2]

    for i in range(3):
        aemo[i, :] = ersa_utils.savitzky_golay(aemo[i, :], 11, 2)

    spca_stat_file = os.path.join(task_dir, 'spca_rgb_stats.npy')
    spca = ersa_utils.load_file(spca_stat_file)

    aemo_stat_file = os.path.join(task_dir, 'aemo_panel_stats.npy')
    aemo = ersa_utils.load_file(aemo_stat_file)

    spca[:, 0] = 0
    aemo[:, 0] = 0
    aemo[:, -1] = aemo[:, -2]

    # get test data
    data_dir = r'/home/lab/Documents/bohao/data/aemo/aemo_pad'
    files = sorted(glob(os.path.join(data_dir, '*rgb.tif')))
    save_dir = r'/home/lab/Documents/bohao/data/aemo/aemo_hist2'

    for file in tqdm(files):
        im_s = ersa_utils.load_file(file)

        im_res = cust_hist_match(aemo, spca, im_s)
        ersa_utils.save_file(os.path.join(save_dir, os.path.basename(file)), im_res)
