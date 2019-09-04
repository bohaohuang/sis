import os
import scipy.misc
import sis_utils
import ersa_utils
from rst_utils import misc_utils
from evaluate_utils import local_maxima_suppression
from post_processing_utils import get_edge_info, connect_lines, prune_lines, prune_towers, \
    visualize_results, towers_online, linked_length, break_lines, get_samples_between, load_model, \
    run_inference_for_single_image, update_connected_pairs, read_polygon_csv_data, get_center_point

city_list = ['Dunedin', 'Gisborne', 'Palmerston North', 'Rotorua', 'Tauranga']


def get_tower_truth_pred(dirs, city_id, tile_id):
    gt_list = []
    csv_file_name = os.path.join(dirs['raw'], 'NZ_{}_{}_resize.csv'.format(city_list[city_id], tile_id))
    for label, bbox in read_polygon_csv_data(csv_file_name):
        y, x = get_center_point(*bbox)
        gt_list.append([y, x])
    return gt_list


def load_data(dirs, model_name, city_id, tile_id, merge_range=100):
    conf_dict = {0: 2, 1: 1, 2: 0, 3: 3}
    pred_file_name = os.path.join(dirs['task'], model_name + '_v2', 'NZ_{}_{}_resize.txt'.format(city_list[city_id], tile_id))
    preds = ersa_utils.load_file(pred_file_name)
    raw_rgb = ersa_utils.load_file(os.path.join(dirs['raw'], 'NZ_{}_{}_resize.tif'.format(city_list[city_id], tile_id)))
    conf_img = ersa_utils.load_file(os.path.join(dirs['conf'],
                                                 '{}{}.png'.format(city_list[city_id].replace(' ', ''), tile_id)))
    line_gt = ersa_utils.load_file(os.path.join(dirs['line'], '{}{}_GT.png'.format(city_list[city_id].replace(' ', ''),
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
    # model_name = 'faster_rcnn_res50'
    '''model_list = [
        '{}_NZ_2019-07-01_18-33-39'.format(model_name),
    ]'''

    '''model_list = [
        '{}_NZ_2019-07-03_11-00-34'.format(model_name),
    ]'''

    '''model_list = [
        '{}_NZ_2019-07-02_19-13-37'.format(model_name),
    ]'''

    for model_name, model_list in zip(['faster_rcnn',], # 'faster_rcnn_res101', 'faster_rcnn_res50'],
                                      [['{}_NZ_2019-07-01_18-33-39'],]):
                                       #['{}_NZ_2019-07-03_11-00-34'],
                                       #['{}_NZ_2019-07-02_19-13-37']]):
        model_list[0] = model_list[0].format(model_name)
        model_id = 25000
        gpu = 1

        for city_id in range(len(city_list)):
            detection_graph, category_index = load_model(model_list[0], model_id, gpu)
            for tile_id in [1, 2, 3]:
                print('Evaluating city {} tile {}'.format(city_id, tile_id))

                # load data
                try:
                    preds, raw_rgb, conf_img, line_gt, tower_gt, tower_pred, tower_conf = \
                        load_data(dirs, model_name, city_id, tile_id, merge_range=merge_range)
                except OSError:
                    continue

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
                        sample_patch = sample_patch[:, :, :3]
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
                save_file_name = os.path.join(task_dir, 'post_{}_NZ_{}_{}_pred_v2.npy'.format(model_name, city_id, tile_id))
                misc_utils.save_file(save_file_name, tower_pred)
                save_file_name = os.path.join(task_dir, 'post_{}_NZ_{}_{}_conf_v2.npy'.format(model_name, city_id, tile_id))
                misc_utils.save_file(save_file_name, tower_conf)
                save_file_name = os.path.join(task_dir, 'post_{}_NZ_{}_{}_conn_v2.npy'.format(model_name, city_id, tile_id))
                misc_utils.save_file(save_file_name, connected_pairs)'''

                # visualize results
                visualize_results(dirs['image'], city_id, tile_id, raw_rgb, line_gt, tower_pred, tower_gt, connected_pairs,
                                  connected_towers, unconnected_towers, save_fig=True, post_str='NZ_{}'.format(model_name), close_file=True)
