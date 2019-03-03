import os
import numpy as np
import matplotlib.pyplot as plt
import sis_utils
import ersa_utils
import util_functions
from evaluate_utils import local_maxima_suppression
from evaluate_tower_performance import get_center_point, read_polygon_csv_data


def add_points(center_points, color='r', size=20, marker='o', alpha=0.5, edgecolor='face'):
    center_points = np.array(center_points)
    plt.scatter(center_points[:, 1], center_points[:, 0], c=color, s=size, marker=marker, alpha=alpha,
                edgecolors=edgecolor)


if __name__ == '__main__':
    # directories
    img_dir, task_dir = sis_utils.get_task_img_folder()
    data_dir = r'/home/lab/Documents/bohao/data/transmission_line'
    raw_dir = os.path.join(data_dir, 'raw')
    conf_dir = r'/media/ei-edl01/user/bh163/tasks/2018.11.16.transmission_line/' \
               r'confmap_uab_UnetCrop_lines_pw30_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'
    lines_dir = r'/media/ei-edl01/data/uab_datasets/lines/data/Original_Tiles'

    city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']
    model_name = 'faster_rcnn'

    city_id, tile_id = 0, 3
    for city_id in range(4):
        for tile_id in [1, 2, 3]:
            # load data
            pred_dir = r'/media/ei-edl01/user/bh163/tasks/2019.02.11.tower_evaluate'
            pred_file_name = os.path.join(pred_dir, model_name, 'USA_{}_{}.txt'.format(city_list[city_id], tile_id))
            preds = ersa_utils.load_file(pred_file_name)
            csv_file_name = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
            raw_rgb = ersa_utils.load_file(os.path.join(raw_dir, 'USA_{}_{}.tif'.format(city_list[city_id], tile_id)))
            line_gt = ersa_utils.load_file(os.path.join(lines_dir, '{}{}_GT.png'.format(city_list[city_id].split('_')[1],
                                                                                        tile_id)))

            # get truth locations
            gt_list = []
            csv_file_name = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
            for label, bbox in read_polygon_csv_data(csv_file_name):
                y, x = get_center_point(*bbox)
                gt_list.append([y, x])

            # get pred locations
            center_list, conf_list, _ = local_maxima_suppression(preds, 100)

            # add line mask
            img_with_line = util_functions.add_mask(raw_rgb, line_gt, [0, 255, 0], 1)

            plt.figure(figsize=(9, 8))
            plt.imshow(img_with_line)
            add_points(gt_list, 'b', marker='s', size=80, alpha=1, edgecolor='k')
            add_points(center_list, 'r', marker='o', alpha=1, edgecolor='k')
            plt.axis('off')
            plt.tight_layout()
            #plt.savefig(os.path.join(img_dir, '{}_{}_cmp.png'.format(city_list[city_id], tile_id)))
            plt.show()
