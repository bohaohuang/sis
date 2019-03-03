import os
import numpy as np
from rst_utils import misc_utils, metrics

truth_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles'
pred_dir = '/home/lab/Documents/bohao/data/temp_results'

model_name = ['CTK', 'CTKA', 'CTK+STN_A']
for mn in model_name:
    iou_a = []
    iou_b = []
    for tile_id in range(1, 6):
        tile_name = 'vienna{}_GT.tif'.format(tile_id)
        gt = misc_utils.load_file(os.path.join(truth_dir, tile_name)) // 255
        pred = misc_utils.load_file(os.path.join(pred_dir, mn, 'inria', 'pred', 'vienna{}.png'.format(tile_id)))

        iou_a_temp, iou_b_temp = metrics.iou_metric(gt, pred, divide_flag=True)
        iou_a.append(iou_a_temp)
        iou_b.append(iou_b_temp)
    print('{} IoU={:.3f}'.format(mn, np.sum(iou_a)/np.sum(iou_b)))
