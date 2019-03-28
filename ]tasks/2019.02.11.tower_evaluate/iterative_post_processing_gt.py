import os
import numpy as np
import sis_utils
import ersa_utils
from evaluate_utils import local_maxima_suppression
from post_processing_utils import load_data, get_edge_info, connect_lines, prune_lines, prune_towers, \
    visualize_results, towers_online, linked_length, break_lines, get_samples_between, load_model, \
    run_inference_for_single_image, update_connected_pairs


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

    # settings
    merge_range = 100
    radius = [2000]
    width = 7
    th = 5
    step = 5
    patch_size = (500, 500)

    for city_id in range(4):
        for tile_id in [1, 2, 3]:
            print('Evaluating city {} tile {}'.format(city_id, tile_id))

            # load data
            _, raw_rgb, conf_img, line_gt, tower_gt, _, tower_conf = \
                load_data(dirs, 'faster_rcnn', city_id, tile_id, merge_range=merge_range)
            tower_pred = np.array(tower_gt)

            # get line confidences
            connected_pairs, connected_towers, unconnected_towers = None, None, None
            for r in radius:
                tower_pairs, tower_dists, line_confs = \
                    get_edge_info(tower_pred, conf_img, radius=r, width=width,
                                  tile_min=(0, 0), tile_max=raw_rgb.shape)

                # connect lines
                connected_pairs = connect_lines(tower_pairs, line_confs, th, cut_n=2)
                connected_pairs, unconnected_pairs = prune_lines(connected_pairs, tower_pred)

                # get towers that are not connected
                connected_towers, unconnected_towers = prune_towers(connected_pairs, tower_pred)

                # search line
                try:
                    connected_towers, unconnected_towers, connected_pairs = \
                        towers_online(tower_pred, connected_towers, unconnected_towers, connected_pairs)
                except ValueError:
                    pass

                # update towers
                break_lines(connected_pairs, tower_pred)
                # tower_pred = [tower_pred[a] for a in connected_towers]
                # tower_conf = [tower_conf[a] for a in connected_towers]

            save_file_name = os.path.join(task_dir, 'overall_post_gt_{}_{}_pred.npy'.format(city_id, tile_id))
            ersa_utils.save_file(save_file_name, tower_pred)
            save_file_name = os.path.join(task_dir, 'overall_post_gt_{}_{}_conn.npy'.format(city_id, tile_id))
            ersa_utils.save_file(save_file_name, connected_pairs)
