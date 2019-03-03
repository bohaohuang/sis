import os
import numpy as np
import sis_utils
from rst_utils import misc_utils
from post_processing_utils import load_data, order_pair, visualize_results
from line_length_stats import read_line_csv_data, add_point_if_not_nearby


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

    # settings
    model_name = 'faster_rcnn'
    merge_range = 100

    for city_id in range(4):
        for tile_id in [1, 2, 3]:
            print('Evaluating city {} tile {}'.format(city_id, tile_id))
            save_file_name = os.path.join(dirs['task'], '{}_{}_cp.npy'.format(city_list[city_id], city_id))

            # load data
            preds, raw_rgb, conf_img, line_gt, tower_gt, tower_pred, tower_conf = \
                load_data(dirs, model_name, city_id, tile_id, merge_range=merge_range)

            # get tower connection info
            connected_pair = []
            csv_file_name = os.path.join(dirs['raw'], 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
            for start, stop, online_points in read_line_csv_data(csv_file_name, tower_gt):
                centers = add_point_if_not_nearby(start, stop, [tower_gt[a] for a in online_points])

                for i in range(len(centers) - 1):
                    # find the corresponding gt
                    try:
                        connected_pair.append(order_pair(find_point_id(tower_gt, centers[i]),
                                                         find_point_id(tower_gt, centers[i+1])))
                    except AssertionError:
                        pass


            #visualize_results(img_dir, city_id, tile_id, raw_rgb, line_gt, tower_gt, tower_gt, connected_pair,
            #                  None, None)
            misc_utils.save_file(save_file_name, connected_pair)
