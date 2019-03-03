import os
import sis_utils
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import ersa_utils
from evaluate_utils import local_maxima_suppression
from evaluate_tower_performance import radius_scoring
from integrate_conf2 import get_edge_info, connect_lines, prune_pairs
from evaluate_tower_performance import get_center_point, read_polygon_csv_data


if __name__ == '__main__':
    # directories
    img_dir, task_dir = sis_utils.get_task_img_folder()
    data_dir = r'/home/lab/Documents/bohao/data/transmission_line'
    raw_dir = os.path.join(data_dir, 'raw')
    conf_dir = r'/media/ei-edl01/user/bh163/tasks/2018.11.16.transmission_line/' \
               r'confmap_uab_UnetCrop_lines_pw30_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'
    lines_dir = r'/media/ei-edl01/data/uab_datasets/lines/data/Original_Tiles'

    city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']
    model_name_dict = {'faster_rcnn': 'Inception', 'faster_rcnn_res101': 'ResNet101',
                       'faster_rcnn_res50': 'ResNet50'}

    # settings
    radius = 1500
    width = 7
    th = 5
    link_r = 20
    plt.figure(figsize=(10, 8))
    cmap = ersa_utils.get_default_colors()

    for city_id in range(4):
        for model_cnt, (model_name, model_label) in enumerate(model_name_dict.items()):
            gt_list_all = []
            pred_list_all = []
            cf_list_all = []
            pred_list_all_post = []
            cf_list_all_post = []

            plt.subplot(221 + city_id)
            for tile_id in [1, 2, 3]:
                # load data
                pred_dir = r'/media/ei-edl01/user/bh163/tasks/2019.02.11.tower_evaluate'
                pred_file_name = os.path.join(pred_dir, model_name, 'USA_{}_{}.txt'.format(city_list[city_id], tile_id))
                preds = ersa_utils.load_file(pred_file_name)
                csv_file_name = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
                raw_rgb = ersa_utils.load_file(os.path.join(raw_dir, 'USA_{}_{}.tif'.format(city_list[city_id], tile_id)))
                conf_img = ersa_utils.load_file(os.path.join(conf_dir, '{}{}.png'.format(city_list[city_id].split('_')[1],
                                                                                         tile_id)))
                line_gt = ersa_utils.load_file(os.path.join(lines_dir, '{}{}_GT.png'.format(city_list[city_id].split('_')[1],
                                                                                            tile_id)))

                # get tower preds
                center_list, conf_list, _ = local_maxima_suppression(preds, 100)

                # get tower truth
                gt_list = []
                csv_file_name = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
                for label, bbox in read_polygon_csv_data(csv_file_name):
                    y, x = get_center_point(*bbox)
                    gt_list.append([y, x])

                # get line confidences
                pairs, dists, confs = get_edge_info(center_list, conf_img, radius=radius, width=width,
                                                    tile_min=(0, 0), tile_max=raw_rgb.shape)

                # connect lines
                connected_pairs = connect_lines(pairs, confs, th)
                connected_pairs, unconnected_pairs = prune_pairs(connected_pairs, center_list)

                # get towers that are not connected
                center_list = [a.tolist() for a in center_list]
                connected_towers = []
                for p in connected_pairs:
                    connected_towers.append(p[0])
                    connected_towers.append(p[1])
                connected_towers = list(set(connected_towers))
                unconnected_towers = [a for a in range(len(center_list)) if a not in connected_towers]

                # get connected centers
                connected_centers = [center_list[a] for a in connected_towers]
                connected_confs = [conf_list[a] for a in connected_towers]

                # add to list
                gt_list_all.extend(gt_list)
                pred_list_all.extend(center_list)
                cf_list_all.extend(conf_list)
                pred_list_all_post.extend(connected_centers)
                cf_list_all_post.extend(connected_confs)

            _, y_true, y_score = radius_scoring(pred_list_all, gt_list_all, cf_list_all, link_r)
            ap = average_precision_score(y_true, y_score)
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            plt.step(recall[1:], precision[1:], color=cmap[model_cnt], linestyle='--', alpha=1, where='post',
                     label='{} Orig AP={:.2f}'.format(model_label, ap))

            _, y_true_post, y_score_post = radius_scoring(pred_list_all_post, gt_list_all, cf_list_all_post, link_r)
            ap_post = average_precision_score(y_true_post, y_score_post)
            precision_post, recall_post, _ = precision_recall_curve(y_true_post, y_score_post)
            plt.step(recall_post[1:], precision_post[1:], color=cmap[model_cnt], alpha=1, where='post',
                     label='{} Post AP={:.2f}'.format(model_label, ap_post))

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('{} Performance Comparison'.format(city_list[city_id]))
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0, 2, 4, 1, 3, 5]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower right', ncol=2,
                   fontsize=8)
        # plt.legend(loc='lower right', ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'cmp_pr_post.png'))
    plt.show()
