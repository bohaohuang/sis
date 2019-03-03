import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import sis_utils
import ersa_utils
import object_score

if __name__ == '__main__':
    data_dir = r'/media/ei-edl01/data/aemo/TILES_WITH_GT_ALL'
    for fold_cnt, fold_id in enumerate([1, 3, 5]):
        gt_files = glob(os.path.join(data_dir, '0584270470{}0_01'.format(fold_id), '*_comb.tif'))
        for gt_file in gt_files:
            print('Processing: {}'.format(gt_file))
            prefix = os.path.basename(gt_file)[:-8] + 'group.tif'

            gt = ersa_utils.load_file(gt_file)

            cmos = object_score.ConfMapObjectScoring()
            gt_obj = cmos.add_object_wise_info(gt)
            gt_obj = cmos.group_objects(gt_obj, gt)

            img = cmos.get_image_group_id(gt.shape, gt_obj).astype(np.uint16)
            ersa_utils.save_file(os.path.join(data_dir, '0584270470{}0_01'.format(fold_id), prefix), img)

            img = ersa_utils.load_file(os.path.join(data_dir, '0584270470{}0_01'.format(fold_id), prefix))
