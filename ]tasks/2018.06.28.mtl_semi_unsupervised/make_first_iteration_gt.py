import os
import imageio
import numpy as np
from glob import glob
from tqdm import tqdm
import sis_utils
import util_functions
import uab_collectionFunctions
import uabUtilreader


if __name__ == '__main__':
    city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
    model_name = 'unet'
    blCol = uab_collectionFunctions.uabCollection('inria')
    img_mean = blCol.getChannelMeans([0, 1, 2])
    img_dir, task_dir = sis_utils.get_task_img_folder()
    threshold = 0.1

    for city_num in tqdm(range(5)):
        pred_dir = r'/hdd/Results/domain_selection/UnetPredict_inria_loo_mtl_cust_{}_0_PS(572, 572)_BS5_' \
                   r'EP100_LR0.0001_DS60_DR0.1_SFN32/inria_all/pred'.format(city_num)

        if model_name == 'unet':
            patch_size = (572, 572)
            overlap = 92
            pad = 92
        else:
            patch_size = (321, 321)
            overlap = 0
            pad = 0

        # extract patches
        tile_files = sorted(glob(os.path.join(pred_dir, '{}*.png'.format(city_list[city_num]))))
        pred_building_binary = []
        for file in tile_files:
            gt = imageio.imread(file)
            gt = np.expand_dims(gt, axis=2)
            gt = uabUtilreader.pad_block(gt, np.array([pad, pad]))
            for patch in uabUtilreader.patchify(gt, (5184, 5184), patch_size, overlap):
                if model_name == 'deeplab':
                    pred_raw = np.sum(patch) / (patch_size[0] * patch_size[1])
                else:
                    pred_raw = np.sum(util_functions.crop_center(patch, 388, 388)) / (patch_size[0] * patch_size[1])
                if pred_raw > threshold:
                    pred_building_binary.append(1)
                else:
                    pred_building_binary.append(0)

        np.save(os.path.join(task_dir, '1iter_pred_building_binary_{}.npy'.format(city_num)), pred_building_binary)
