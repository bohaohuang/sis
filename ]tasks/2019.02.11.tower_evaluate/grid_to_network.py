import os
import numpy as np
import sis_utils
from rst_utils import misc_utils
from post_processing_utils import get_tower_truth_pred


def load_data(dirs, city_id, tile_id):
    line_gt = misc_utils.load_file(os.path.join(dirs['line'], '{}{}_GT.png'.format(city_list[city_id].split('_')[1],
                                                                                   tile_id)))
    tower_gt = get_tower_truth_pred(dirs, city_id, tile_id)
    return line_gt, tower_gt


def find_point_id(centers, point):
    dist = np.linalg.norm(np.array(centers) - np.array(point), axis=1)
    assert np.min(dist) < 20
    return np.argmin(dist)


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
    city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']
    tile_id_list = [12, 15, 8, 12]

    # settings
    model_name = 'faster_rcnn'
    merge_range = 100

    for city_id in range(4):
        graph_train, graph_valid = [], []
        centers_train, centers_valid = [], []
        for tile_id in range(1, tile_id_list[city_id] + 1):
            centers = []
            print('Evaluating city {} tile {}'.format(city_id, tile_id))
            save_file_name = os.path.join(dirs['task'], '{}_{}_cp.npy'.format(city_list[city_id], tile_id))

            # load data
            line_gt, tower_gt = load_data(dirs, city_id, tile_id)

            # get tower connection info
            connected_pair = misc_utils.load_file(save_file_name)

            n_node = len(tower_gt)
            graph = np.zeros((n_node, n_node))
            for cp in connected_pair:
                graph[cp[0], cp[1]] = 1
                graph[cp[1], cp[0]] = 1

            for i in range(n_node):
                centers.append(np.array(tower_gt[i]) / np.array(line_gt.shape))

            if tile_id <= 3:
                graph_valid.append(graph)
                centers_valid.append(centers)
            else:
                graph_train.append(graph)
                centers_train.append(centers)

        save_file_name = os.path.join(dirs['task'], 'train_graph_{}.npy'.format(city_list[city_id].split('_')[1]))
        misc_utils.save_file(save_file_name, graph_train)
        save_file_name = os.path.join(dirs['task'], 'valid_graph_{}.npy'.format(city_list[city_id].split('_')[1]))
        misc_utils.save_file(save_file_name, graph_valid)
        save_file_name = os.path.join(dirs['task'], 'train_centers_{}.npy'.format(city_list[city_id].split('_')[1]))
        misc_utils.save_file(save_file_name, centers_train)
        save_file_name = os.path.join(dirs['task'], 'valid_centers_{}.npy'.format(city_list[city_id].split('_')[1]))
        misc_utils.save_file(save_file_name, centers_valid)
