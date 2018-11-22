"""
Makeup all tiles available in AEMO and process them, this includes:
1. Make up the files
2. Histogram matching them
"""

import os
import numpy as np
from glob import glob
from functools import partial
from natsort import natsorted
import ersa_utils
import processBlock


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


def get_blank_regions(img):
    img_tmp = img.astype(np.float32)
    img_tmp = np.sum(img_tmp, axis=2)
    blank_mask = (img_tmp < 0.1).astype(np.int)
    return blank_mask


def process_different_situation(blank_region, orig_img, code):
    h, w = blank_region.shape
    makeup = np.zeros((h, w, 3))
    end_point = 0
    if code == 0:
        for i in range(w):
            if blank_region[h-1, i] != 0:
                end_point = i
        strip = orig_img[:, end_point:end_point*2, :]
        makeup[:, :end_point, :] = strip[:, ::-1, :]
    elif code == 1:
        for i in range(h-1, -1, -1):
            if blank_region[i, 0] != 0:
                end_point = i
        strip = orig_img[h-end_point*2:h-end_point, :, :]
        makeup[h-end_point:h, :, :] = strip[::-1, :, :]
    elif code == 2:
        for i in range(h):
            if blank_region[i, 0] != 0:
                end_point = i
        strip = orig_img[end_point:2*end_point, :, :]
        makeup[:end_point, :, :] = strip[::-1, :, :]
    elif code == 3:
        for i in range(w):
            if blank_region[0, i] != 0:
                end_point = i
        strip = orig_img[:, end_point:end_point*2, :]
        makeup[:, :end_point, :] = strip[:, ::-1, :]
    elif code == 4:
        for i in range(w-1, -1, -1):
            if blank_region[0, i] != 0:
                end_point = i
        end_point = w - end_point
        strip = orig_img[:, w-end_point*2:w-end_point, :]
        makeup[:, w-end_point:, :] = strip[:, ::-1, :]
    elif code == 5:
        for i in range(h):
            if blank_region[i, w-1] != 0:
                end_point = i
        strip = orig_img[end_point:end_point*2, :, :]
        makeup[:end_point, :, :] = strip[::-1, :, :]
    elif code == 6:
        for i in range(h-1, -1, -1):
            if blank_region[i, w-1] != 0:
                end_point = i
        end_point = h - end_point
        strip = orig_img[h-end_point*2:h-end_point, :, :]
        makeup[h-end_point:, :, :] = strip[::-1, :, :]
    elif code == 7:
        for i in range(w-1, -1, -1):
            if blank_region[h-1, i] != 0:
                end_point = i
        strip = orig_img[:, w - end_point * 2:w - end_point, :]
        makeup[:, w - end_point:, :] = strip[:, ::-1, :]
    else:
        return orig_img.astype(np.uint8)
    for i in range(3):
        makeup[:, :, i] = makeup[:, :, i] * blank_region
    return (orig_img + makeup).astype(np.uint8)


def makeup_aemo_img(img, code):
    for c in code:
        bm = get_blank_regions(img)
        img = process_different_situation(bm, img, int(c))

    return img


def get_files(root_dir):
    file_list = []
    for i in range(1, 6):
        if i != 2:
            for p, s, f in os.walk(root_dir.format(i)):
                sort_f = natsorted(f)
                for name in sort_f:
                    if 'tif' in name:
                        file_list.append(os.path.join(p, name))
        else:
            for p, s, f in os.walk(os.path.join(root_dir.format(i), '1')):
                sort_f = natsorted(f)
                for name in sort_f:
                    if 'tif' in name:
                        file_list.append(os.path.join(p, name))
            for p, s, f in os.walk(os.path.join(root_dir.format(i), '2')):
                sort_f = natsorted(f)
                for name in sort_f:
                    if 'tif' in name:
                        file_list.append(os.path.join(p, name))
    return file_list


def process_files(save_dir, file_list, code_list):
    for f, c in zip(file_list, code_list):
        print('processing: {} with code {}'.format(f,c))
        sub_dir = os.path.join(save_dir, '/'.join(f.split('/')[5:-1]))
        ersa_utils.make_dir_if_not_exist(sub_dir)
        save_name = os.path.join(sub_dir, os.path.basename(f))

        rgb = ersa_utils.load_file(f)
        rgb_new = makeup_aemo_img(rgb, c)
        
        ersa_utils.save_file(save_name, rgb_new)


def get_aemo_stats(data_dir, suffix='*.tif'):
    rgb_files = glob(os.path.join(data_dir, suffix))
    dist = np.zeros((3, 255))
    for rgb_file in rgb_files:
        rgb = ersa_utils.load_file(rgb_file)
        for c in range(3):
            rgb_cnt, _ = np.histogram(rgb[:, :, c], bins=np.arange(256))
            dist[c, :] += rgb_cnt
    dist[:, :2] = 0
    dist[:, -1] = dist[:, -2]
    return dist


def match_files(save_dir, root_dir, source_dist):
    for i in range(1, 6):
        if i != 2:
            for p, s, f in os.walk(root_dir.format(i)):
                save_sub_dir = os.path.join(save_dir, '/'.join(p.split('/')[8:]))
                ersa_utils.make_dir_if_not_exist(save_sub_dir)
                target_dist = get_aemo_stats(p)
                align_files(p, save_sub_dir, source_dist, target_dist)
        else:
            for p, s, f in os.walk(os.path.join(root_dir.format(i), '1')):
                save_sub_dir = os.path.join(save_dir, '/'.join(p.split('/')[8:]))
                ersa_utils.make_dir_if_not_exist(save_sub_dir)
                target_dist = get_aemo_stats(p)
                align_files(p, save_sub_dir, source_dist, target_dist)
            for p, s, f in os.walk(os.path.join(root_dir.format(i), '2')):
                save_sub_dir = os.path.join(save_dir, '/'.join(p.split('/')[8:]))
                ersa_utils.make_dir_if_not_exist(save_sub_dir)
                target_dist = get_aemo_stats(p)
                align_files(p, save_sub_dir, source_dist, target_dist)


def align_files(data_dir, save_dir, source_dist, target_dist):
    rgb_files = glob(os.path.join(data_dir, '*.tif'))
    for file in rgb_files:
        print('aligning {}'.format(file))
        im_s = ersa_utils.load_file(file)

        im_res = cust_hist_match(target_dist, source_dist, im_s)
        ersa_utils.save_file(os.path.join(save_dir, os.path.basename(file)), im_res)


if __name__ == '__main__':
    # 1. makeup the files
    code_list = ['02', '0', '0', '60', '2', '8', '8', '6', '2', '8', '8', '6', '24', '4', '4', '46',  # 10
                 '02', '0', '60', '24', '4', '46',  # 20/1
                 '02', '0', '60', '2', '8', '6', '2', '8', '6', '24', '4', '46',  # 20/2
                 '02', '0', '0', '60', '2', '8', '8', '6', '2', '8', '8', '6', '24', '4', '4', '46',  # 30
                 '02', '0', '0', '60', '2', '8', '8', '6', '24', '4', '4', '46',  # 40
                 '02', '0', '060', '2', '8', '6', '2', '8', '6', '24', '4', '46',  # 50
                 ]
    data_dir = r'/media/ei-edl01/data/aemo/TILES/0584270470{}0_01'
    file_list = get_files(data_dir)

    target_dir = r'/home/lab/Documents/bohao/data/aemo_all'

    pb = processBlock.BasicProcess('tile_makeup', target_dir, func=partial(process_files, save_dir=target_dir,
                                                                           file_list=file_list, code_list=code_list)).\
        run(force_run=False)

    align_dir = r'/home/lab/Documents/bohao/data/aemo_all/align'
    dist_target = get_aemo_stats(r'/home/lab/Documents/bohao/data/aemo/aemo_align', '*rgb.tif')
    match_files(align_dir, target_dir + '/TILES/0584270470{}0_01', dist_target)
