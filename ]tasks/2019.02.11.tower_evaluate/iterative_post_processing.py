import utils
from post_processing_utils import load_data, get_edge_info, connect_lines, prune_lines, prune_towers, visualize_results


if __name__ == '__main__':
    # directories
    img_dir, task_dir = utils.get_task_img_folder()
    dirs = {
        'task': task_dir,
        'image': img_dir,
        'raw': r'/home/lab/Documents/bohao/data/transmission_line/raw',
        'conf': r'/media/ei-edl01/user/bh163/tasks/2018.11.16.transmission_line/'
                r'confmap_uab_UnetCrop_lines_pw30_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32',
        'line': r'/media/ei-edl01/data/uab_datasets/lines/data/Original_Tiles'
    }
    model_name = 'faster_rcnn'

    # settings
    merge_range = 100
    radius = [2000]
    width = 7
    th = 5

    for city_id in [1]:
        for tile_id in [3]:
            # load data
            preds, raw_rgb, conf_img, line_gt, tower_gt, tower_pred, tower_conf = \
                load_data(dirs, model_name, city_id, tile_id, merge_range=merge_range)

            # get line confidences
            for r in radius:
                tower_pairs, tower_dists, line_confs = \
                    get_edge_info(tower_pred, conf_img, radius=r, width=width,
                                  tile_min=(0, 0), tile_max=raw_rgb.shape)

                # connect lines
                connected_pairs = connect_lines(tower_pairs, line_confs, th, cut_n=2)
                connected_pairs, unconnected_pairs = prune_lines(connected_pairs, tower_pred)

                # get towers that are not connected
                connected_towers, unconnected_towers = prune_towers(connected_pairs, tower_pred)

                # visualize results
                visualize_results(dirs, city_id, tile_id, raw_rgb, line_gt, tower_pred, tower_gt, connected_pairs,
                                  connected_towers, unconnected_towers, save_fig=False)

                tower_pred = [tower_pred[a] for a in connected_towers]
