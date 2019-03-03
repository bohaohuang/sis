import os
import csv
import scipy.spatial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sis_utils
from rst_utils import misc_utils, feature_extractor, processBlock
from evaluate_tower_performance import get_center_point, read_polygon_csv_data


def find_close_points(p1, p2, points, th=10):
    def dist2line(p1, p2, p3):
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
        return np.linalg.norm(np.cross(p2 - p1, p3 - p1))/np.linalg.norm(p2 - p1)

    def is_between(p1, p2, p3):
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
        if 1 - scipy.spatial.distance.cosine((p1 - p3), (p2 - p3)) < 0:
            return True
        else:
            return False

    point_ids = []
    for cnt, p3 in enumerate(points):
        if dist2line(p1, p2, p3[::-1]) < th and is_between(p1, p2, p3[::-1]):
            point_ids.append(cnt)

    return point_ids


def read_line_csv_data(csv_file, centers):
    label_order = ['SS', 'OT', 'DT', 'TT', 'OL', 'DL', 'TL']
    df = pd.read_csv(csv_file)
    df['temp_label'] = pd.Categorical(df['Label'], categories=label_order, ordered=True)
    df.sort_values('temp_label', inplace=True, kind='mergesort')

    for name, group in df.groupby('Object', sort=False):
        label = group['Label'].values[0]
        if group['Type'].values[0] == 'Line' and label in ['TL', 'DL']:
            for j in range(group.shape[0] - 1):
                r0, c0 = int(group['X'].values[j]), int(group['Y'].values[j])
                r1, c1 = int(group['X'].values[j + 1]), int(group['Y'].values[j + 1])
                p1 = (r0, c0)
                p2 = (r1, c1)
                online_points = find_close_points(p1, p2, centers)
                yield (r0, c0), (r1, c1), online_points


def get_region_bounds(region, tile_dim):
    for i in range(2):
        region[i] = max((0, region[i]))
        if region[i] + region[i+2] > tile_dim[i]:
            region[i] = tile_dim[i] - region[i+2]
    return region


def add_point_if_not_nearby(p1, p2, points, th=20):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p1_flag = True
    p2_flag = True
    for p3 in points:
        p3 = np.array(p3[::-1])
        if np.linalg.norm(p3 - p1) < th:
            p1_flag = False
        if np.linalg.norm(p3 - p2) < th:
            p2_flag = False
    if p1_flag:
        points.append(p1.tolist()[::-1])
    if p2_flag:
        points.append(p2.tolist()[::-1])
    return points


def write_features(raw_dir, city_list, city_id, tile_id, model):
    raw_rgb = misc_utils.load_file(os.path.join(raw_dir, 'USA_{}_{}.tif'.format(city_list[city_id], tile_id)))
    tile_dim = raw_rgb.shape

    # get tower truth
    gt_list = []
    csv_file_name = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
    for label, bbox in read_polygon_csv_data(csv_file_name):
        y, x = get_center_point(*bbox)
        gt_list.append([y, x])

    # get features
    feature = []
    for start, stop, online_points in read_line_csv_data(csv_file_name, gt_list):
        centers = add_point_if_not_nearby(start, stop, [gt_list[a] for a in online_points])

        for c in centers:
            c = [int(a) for a in c]
            region = [c[0] - patch_size[0] // 2, c[1] - patch_size[1] // 2, *patch_size]
            region = get_region_bounds(region, tile_dim)
            img_patch = raw_rgb[region[0]:region[0] + region[2], region[1]:region[1] + region[3], :]
            feature.append(model.get_feature(img_patch))
    return np.array(feature)


if __name__ == '__main__':
    # directories
    img_dir, task_dir = sis_utils.get_task_img_folder()
    data_dir = r'/home/lab/Documents/bohao/data/transmission_line'
    raw_dir = os.path.join(data_dir, 'raw')

    # settings
    city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']
    tile_range = [12, 15, 8, 12]
    patch_size = (224, 224)
    misc_utils.set_gpu(0)
    res50 = feature_extractor.Res50()

    ftr_file_name_train = os.path.join(task_dir, 'mlp_tower_pair_ftr_train.npy')
    lbl_file_name_train = os.path.join(task_dir, 'mlp_tower_pair_lbl_train.npy')
    all_ftr_train = []
    all_lbl_train = []

    ftr_file_name_valid = os.path.join(task_dir, 'mlp_tower_pair_ftr_valid.npy')
    lbl_file_name_valid = os.path.join(task_dir, 'mlp_tower_pair_lbl_valid.npy')
    all_ftr_valid = []
    all_lbl_valid = []

    for city_id in range(4):
        for tile_id in range(1, tile_range[city_id]):
            print('Processing {}: tile {}'.format(city_list[city_id], tile_id))

            pb = processBlock.ValueComputeProcess('{}{}_fe'.format(city_id, tile_id), task_dir,
                                                  os.path.join(task_dir, '{}_{}_feature.npy'.format(city_id, tile_id)),
                                                  func=write_features)
            feature = pb.run(force_run=False, raw_dir=raw_dir, city_list=city_list, city_id=city_id,
                             tile_id=tile_id, model=res50).val
            raw_rgb = misc_utils.load_file(os.path.join(raw_dir, 'USA_{}_{}.tif'.format(city_list[city_id], tile_id)))
            tile_dim = raw_rgb.shape

            # get tower truth
            gt_list = []
            csv_file_name = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
            for label, bbox in read_polygon_csv_data(csv_file_name):
                y, x = get_center_point(*bbox)
                gt_list.append([y, x])

            current_cnt = 0
            connected_pair = []
            all_centers = []
            for start, stop, online_points in read_line_csv_data(csv_file_name, gt_list):
                centers = add_point_if_not_nearby(start, stop, [gt_list[a] for a in online_points])
                all_centers.extend(centers)

                # write h1 samples
                for i in range(len(centers) - 1):
                    start_feature = feature[current_cnt+i, :]
                    end_feature = feature[current_cnt+i+1, :]
                    start_pos = np.array(centers[i]) / np.array(tile_dim[:2])
                    end_pos = np.array(centers[i+1]) / np.array(tile_dim[:2])

                    flag = np.random.randint(0, 3)
                    # TODO write order based on flag
                    if flag == 0:
                        line = [*(start_feature.tolist()), *(start_pos.tolist()),
                                *(end_feature.tolist()), *(end_pos.tolist())]
                        if tile_id < 4:
                            all_ftr_valid.append(line)
                            all_lbl_valid.append(1)
                        else:
                            all_ftr_train.append(line)
                            all_lbl_train.append(1)
                        connected_pair.append((i, i + 1))
                    elif flag == 1:
                        line = [*(end_feature.tolist()), *(end_pos.tolist()),
                                *(start_feature.tolist()), *(start_pos.tolist()),]
                        if tile_id < 4:
                            all_ftr_valid.append(line)
                            all_lbl_valid.append(1)
                        else:
                            all_ftr_train.append(line)
                            all_lbl_train.append(1)
                        connected_pair.append((i, i + 1))
                    else:
                        line = [*(start_feature.tolist()), *(start_pos.tolist()),
                                *(end_feature.tolist()), *(end_pos.tolist())]
                        if tile_id < 4:
                            all_ftr_valid.append(line)
                            all_lbl_valid.append(1)
                        else:
                            all_ftr_train.append(line)
                            all_lbl_train.append(1)
                        connected_pair.append((i, i + 1))

                        line = [*(end_feature.tolist()), *(end_pos.tolist()),
                                *(start_feature.tolist()), *(start_pos.tolist()),]
                        if tile_id < 4:
                            all_ftr_valid.append(line)
                            all_lbl_valid.append(1)
                        else:
                            all_ftr_train.append(line)
                            all_lbl_train.append(1)
                        connected_pair.append((i, i + 1))
                current_cnt += len(centers)

            # write h0 samples
            h0_cnt = 0
            while h0_cnt < current_cnt * 2:
                pair = np.random.choice(current_cnt, 2)
                pair = tuple(np.sort(pair).tolist())
                if pair not in connected_pair:
                    start_feature = feature[pair[0], :]
                    end_feature = feature[pair[1], :]
                    start_pos = np.array(all_centers[pair[0]]) / np.array(tile_dim[:2])
                    end_pos = np.array(all_centers[pair[1]]) / np.array(tile_dim[:2])

                    line = [*(start_feature.tolist()), *(start_pos.tolist()),
                            *(end_feature.tolist()), *(end_pos.tolist())]
                    if tile_id < 4:
                        all_ftr_valid.append(line)
                        all_lbl_valid.append(0)
                    else:
                        all_ftr_train.append(line)
                        all_lbl_train.append(0)
                    connected_pair.append(pair)
                    h0_cnt += 1

            h0_cnt = 0
            while h0_cnt < current_cnt * 2:
                tp = np.random.choice(current_cnt, 1)[0]
                fp = [np.random.randint(0, tile_dim[0]-patch_size[0]), np.random.randint(0, tile_dim[1]-patch_size[1])]

                start_feature = feature[tp, :]
                img_temp = np.expand_dims(raw_rgb[fp[0]:fp[0]+patch_size[0], fp[1]:fp[1]+patch_size[1], :], axis=0)
                end_feature = res50.get_feature(img_temp)

                start_pos = np.array(all_centers[tp]) / np.array(tile_dim[:2])
                end_pos = (np.array(fp) + np.array(patch_size) // 2) / np.array(tile_dim[:2])

                line = [*(start_feature.tolist()), *(start_pos.tolist()),
                        *(end_feature.tolist()), *(end_pos.tolist())]
                if tile_id < 4:
                    all_ftr_valid.append(line)
                    all_lbl_valid.append(0)
                else:
                    all_ftr_train.append(line)
                    all_lbl_train.append(0)

                h0_cnt += 1

            h0_cnt = 0
            while h0_cnt < current_cnt * 2:
                fp_1 = [np.random.randint(0, tile_dim[0] - patch_size[0]),
                        np.random.randint(0, tile_dim[1] - patch_size[1])]
                fp_2 = [np.random.randint(0, tile_dim[0] - patch_size[0]),
                        np.random.randint(0, tile_dim[1] - patch_size[1])]

                img_temp = np.expand_dims(raw_rgb[fp_1[0]:fp_1[0] + patch_size[0],
                                          fp_1[1]:fp_1[1]+patch_size[1], :], axis=0)
                start_feature = res50.get_feature(img_temp)
                img_temp = np.expand_dims(raw_rgb[fp_2[0]:fp_2[0] + patch_size[0],
                                          fp_2[1]:fp_2[1]+patch_size[1], :], axis=0)
                end_feature = res50.get_feature(img_temp)

                start_pos = (np.array(fp_1) + np.array(patch_size) // 2) / np.array(tile_dim[:2])
                end_pos = (np.array(fp_2) + np.array(patch_size) // 2) / np.array(tile_dim[:2])

                line = [*(start_feature.tolist()), *(start_pos.tolist()),
                        *(end_feature.tolist()), *(end_pos.tolist())]
                if tile_id < 4:
                    all_ftr_valid.append(line)
                    all_lbl_valid.append(0)
                else:
                    all_ftr_train.append(line)
                    all_lbl_train.append(0)

                h0_cnt += 1

    all_ftr_train = np.array(all_ftr_train)
    all_lbl_train = np.array(all_lbl_train)
    all_ftr_valid = np.array(all_ftr_valid)
    all_lbl_valid = np.array(all_lbl_valid)

    print(all_ftr_train.shape, all_lbl_train.shape)
    print(all_ftr_valid.shape, all_lbl_valid.shape)
    misc_utils.save_file(ftr_file_name_train, all_ftr_train)
    misc_utils.save_file(lbl_file_name_train, all_lbl_train)
    misc_utils.save_file(ftr_file_name_valid, all_ftr_valid)
    misc_utils.save_file(lbl_file_name_valid, all_lbl_valid)
