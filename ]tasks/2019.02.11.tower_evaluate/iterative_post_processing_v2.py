import os
import scipy.misc
import sis_utils
import ersa_utils
from rst_utils import misc_utils
from evaluate_utils import local_maxima_suppression
from post_processing_utils import get_edge_info, connect_lines, prune_lines, prune_towers, \
    visualize_results, towers_online, linked_length, break_lines, get_samples_between, load_model, \
    run_inference_for_single_image, get_tower_truth_pred, update_connected_pairs

city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']


def load_data(dirs, model_name, city_id, tile_id, merge_range=100):
    conf_dict = {0: 2, 1: 1, 2: 0, 3: 3}
    pred_file_name = os.path.join(dirs['task'], model_name + '_v2', 'USA_{}_{}.txt'.format(city_list[city_id], tile_id))
    preds = ersa_utils.load_file(pred_file_name)
    raw_rgb = ersa_utils.load_file(os.path.join(dirs['raw'], 'USA_{}_{}.tif'.format(city_list[city_id], tile_id)))
    conf_img = ersa_utils.load_file(os.path.join(dirs['conf'].format(conf_dict[city_id]),
                                                 '{}{}.png'.format(city_list[city_id].split('_')[1], tile_id)))
    line_gt = ersa_utils.load_file(os.path.join(dirs['line'], '{}{}_GT.png'.format(city_list[city_id].split('_')[1],
                                                                                   tile_id)))
    tower_gt = get_tower_truth_pred(dirs, city_id, tile_id)
    tower_pred, tower_conf, _ = local_maxima_suppression(preds, th=merge_range)
    conf_img = scipy.misc.imresize(conf_img, line_gt.shape)
    return preds, raw_rgb, conf_img, line_gt, tower_gt, tower_pred, tower_conf


if __name__ == '__main__':
    # directories
    img_dir, task_dir = sis_utils.get_task_img_folder()
    ersa_utils.make_dir_if_not_exist(os.path.join(img_dir, 'new_annotation'))
    dirs = {
        'task': task_dir,
        'image': os.path.join(img_dir, 'new_annotation'),
        'raw': r'/home/lab/Documents/bohao/data/transmission_line/raw2',
        'conf': r'/media/ei-edl01/user/bh163/tasks/2018.11.16.transmission_line/confmap_uab_UnetCrop_linesv3_city0_'
                r'pw50_0_PS(572, 572)_BS5_EP100_LR0.0001_DS80_DR0.1_SFN32',
        'line': r'/media/ei-edl01/data/uab_datasets/lines_v3/data/Original_Tiles'
    }

    # settings
    merge_range = 100
    radius = [1500]
    width = 5
    th = 8
    step = 5
    patch_size = (500, 500)
    for model_name in ['faster_rcnn']:
        model_list_1 = [
            '{}_Tucson_2019-06-30_12-51-03'.format(model_name),
            '{}_Colwich_2019-07-01_10-47-22'.format(model_name),
            '{}_NZ_2019-07-01_18-33-39'.format(model_name),
        ]

        model_list_2 = [
            '{}_Tucson_2019-07-03_00-05-04'.format(model_name),
            '{}_Colwich_2019-07-03_00-04-11'.format(model_name),
            '{}_NZ_2019-07-01_18-33-39'.format(model_name),
        ]

        model_list_3 = [
            '{}_Tucson_2019-07-02_14-53-17'.format(model_name),
            '{}_Colwich_2019-07-02_18-10-38'.format(model_name),
            '{}_NZ_2019-07-01_18-33-39'.format(model_name),
        ]

        if model_name == 'faster_rcnn':
            model_list = model_list_1
        elif model_name == 'faster_rcnn_res101':
            model_list = model_list_2
        elif model_name == 'faster_rcnn_res50':
            model_list = model_list_3

        '''model_list = [
            '{}_2019-02-12_09-30-35'.format(model_name),
            '{}_2019-02-12_09-33-08'.format(model_name),
            '{}_2019-02-12_09-34-45'.format(model_name),
            '{}_2019-02-12_09-36-15'.format(model_name),
        ]'''
        '''model_list = [
            '{}_2019-02-12_09-30-35'.format(model_name),
            '{}_2019-02-12_09-33-08'.format(model_name),
            '{}_2019-02-12_09-34-45'.format(model_name),
            '{}_2019-02-12_09-36-15'.format(model_name),
        ]'''
        model_id = 25000
        gpu = 1

        for city_id in range(2):
            detection_graph, category_index = load_model(model_list[city_id], model_id, gpu)
            for tile_id in [1, 2, 3]:
                print('Evaluating city {} tile {}'.format(city_id, tile_id))

                # load data
                preds, raw_rgb, conf_img, line_gt, tower_gt, tower_pred, tower_conf = \
                    load_data(dirs, model_name, city_id, tile_id, merge_range=merge_range)

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

                # check the connection length
                line_length_list, attention_pair = linked_length(tower_pred, connected_pairs, dirs, city_id, tile_id)
                for ap in attention_pair:
                    pred = []
                    for sample_patch, top_left in get_samples_between(raw_rgb, tower_pred[ap[0]], tower_pred[ap[1]], step, patch_size):
                        # Actual detection.
                        output_dict = run_inference_for_single_image(sample_patch, detection_graph)

                        for db, dc, ds in zip(output_dict['detection_boxes'], output_dict['detection_classes'],
                                              output_dict['detection_scores']):
                            left = int(db[1] * patch_size[1]) + top_left[1]
                            top = int(db[0] * patch_size[0]) + top_left[0]
                            right = int(db[3] * patch_size[1]) + top_left[1]
                            bottom = int(db[2] * patch_size[0]) + top_left[0]
                            confidence = ds
                            class_name = category_index[dc]['name']
                            if confidence > 0.5:
                                pred.append('{} {} {} {} {} {}\n'.format(class_name, confidence, left, top, right, bottom))
                    center_list, conf_list, _ = local_maxima_suppression(pred, th=20)
                    tower_pred.extend(center_list)
                    tower_conf.extend(conf_list)

                tower_pairs, tower_dists, line_confs = \
                    get_edge_info(tower_pred, conf_img, radius=radius[-1], width=width,
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
                '''break_lines(connected_pairs, tower_pred)
                connected_pairs = update_connected_pairs(connected_pairs, tower_pred, connected_towers)
                tower_pred = [tower_pred[a] for a in connected_towers]
                tower_conf = [tower_conf[a] for a in connected_towers]

                assert len(tower_pred) == len(tower_conf)
                save_file_name = os.path.join(task_dir, 'post_{}_{}_{}_pred_v2.npy'.format(model_name, city_id, tile_id))
                misc_utils.save_file(save_file_name, tower_pred)
                save_file_name = os.path.join(task_dir, 'post_{}_{}_{}_conf_v2.npy'.format(model_name, city_id, tile_id))
                misc_utils.save_file(save_file_name, tower_conf)
                save_file_name = os.path.join(task_dir, 'post_{}_{}_{}_conn_v2.npy'.format(model_name, city_id, tile_id))
                misc_utils.save_file(save_file_name, connected_pairs)'''

                # visualize results
                visualize_results(dirs['image'], city_id, tile_id, raw_rgb, line_gt, tower_pred, tower_gt, connected_pairs,
                                  connected_towers, unconnected_towers, save_fig=True, post_str='_{}'.format(model_name), close_file=True)
