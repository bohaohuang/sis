import os
import csv
import scipy.spatial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import load_model
import sis_utils
from rst_utils import misc_utils, feature_extractor
from evaluate_tower_performance import get_center_point, read_polygon_csv_data
from post_processing_utils import visualize_results, break_lines, load_data, get_edge_info, connect_lines, prune_lines


def get_region_bounds(region, tile_dim):
    for i in range(2):
        region[i] = max((0, region[i]))
        if region[i] + region[i+2] > tile_dim[i]:
            region[i] = tile_dim[i] - region[i+2]
    return region


def get_features(raw_rgb, centers, model):
    feature = []
    for c in centers:
        c = [int(a) for a in c]
        region = [c[0] - patch_size[0] // 2, c[1] - patch_size[1] // 2, *patch_size]
        region = get_region_bounds(region, raw_rgb.shape[:2])
        img_patch = raw_rgb[region[0]:region[0] + region[2], region[1]:region[1] + region[3], :]
        feature.append(model.get_feature(img_patch))
    return np.array(feature)


def get_potential_connection_pair(centers, radius):
    kdt = scipy.spatial.KDTree(np.array(centers))
    linked_pairs = list(kdt.query_pairs(radius))
    return linked_pairs


if __name__ == '__main__':
    # directories
    img_dir, task_dir = sis_utils.get_task_img_folder()
    dirs = {
        'task': task_dir,
        'image': img_dir,
        'raw': r'/home/lab/Documents/bohao/data/transmission_line/raw',
        'conf': r'/media/ei-edl01/user/bh163/tasks/2018.11.16.transmission_line/'
                r'confmap_uab_UnetCrop_lines_pw30_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32',
        'line': r'/media/ei-edl01/data/uab_datasets/lines/data/Original_Tiles'
    }
    data_dir = r'/home/lab/Documents/bohao/data/transmission_line'
    raw_dir = os.path.join(data_dir, 'raw')

    # settings
    city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']
    tile_range = [12, 15, 8, 12]
    patch_size = (224, 224)
    misc_utils.set_gpu(0)
    radius = 2500
    width = 7
    th = 5
    model_name = 'faster_rcnn'
    merge_range = 100

    for city_id in range(4):
        for tile_id in range(1, 4):
            K.clear_session()
            res50 = feature_extractor.Res50()
            print('Processing {}: tile {}'.format(city_list[city_id], tile_id))

            # load data
            preds, raw_rgb, conf_img, line_gt, tower_gt, tower_pred, tower_conf = \
                load_data(dirs, model_name, city_id, tile_id, merge_range=merge_range)
            tile_dim = raw_rgb.shape

            tower_pred = tower_gt
            features = get_features(raw_rgb, tower_pred, res50)

            # connect lines
            '''tower_pairs, tower_dists, line_confs = \
                get_edge_info(tower_pred, conf_img, radius=radius, width=width,
                              tile_min=(0, 0), tile_max=raw_rgb.shape)
            raw_pairs = connect_lines(tower_pairs, line_confs, th, cut_n=2)
            raw_pairs, _ = prune_lines(raw_pairs, tower_pred)'''
            raw_pairs = get_potential_connection_pair(tower_pred, radius)
            connected_pairs = []

            # get line prediction
            K.clear_session()
            model_save_path = os.path.join(task_dir, 'model', 'model.hdf5')
            model = load_model(model_save_path)
            for cp in raw_pairs:
                start_feature = features[cp[0], :]
                end_feature = features[cp[1], :]
                start_pos = np.array(tower_pred[cp[0]]) / np.array(tile_dim[:2])
                end_pos = np.array(tower_pred[cp[1]]) / np.array(tile_dim[:2])

                line = [*(start_feature.tolist()), *(start_pos.tolist()),
                        *(end_feature.tolist()), *(end_pos.tolist())]
                line = np.expand_dims(np.array(line), axis=0)

                pred = model.predict(line)
                if pred[0][1] > 0.5:
                    connected_pairs.append(cp)

            #break_lines(connected_pairs, tower_pred)
            visualize_results(img_dir, city_id, tile_id, raw_rgb, line_gt, tower_pred, tower_gt, connected_pairs,
                              None, None, save_fig=True, post_str='_mlp', close_file=True)
