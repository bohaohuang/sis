import os
import imageio
import numpy as np
from sklearn.metrics import roc_curve
import utils
import uabUtilreader

img_dir, task_dir = utils.get_task_img_folder()
city_dict = {'austin': 0, 'chicago': 1, 'kitsap': 2, 'tyrol-w': 3, 'vienna': 4}
gt_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles'

pred_building = []
g_building = []
for city_name in ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']:
    pred_dir = r'/hdd/Results/domain_selection/DeeplabV3_inria_aug_train_leave_{}_0_PS(321, 321)_BS5_EP100_' \
               r'LR1e-05_DS40_DR0.1_SFN32/inria/pred'.format(city_dict[city_name])
    for i in range(5):
        img_name = os.path.join(pred_dir, '{}{}.png'.format(city_name, i + 1))
        gt_name = os.path.join(gt_dir, '{}{}_GT.tif'.format(city_name, i + 1))
        img = np.expand_dims(imageio.imread(img_name), axis=2)
        gt = np.expand_dims(imageio.imread(gt_name), axis=2)
        gt = gt / 255
        for patch, g_patch in zip(uabUtilreader.patchify(img, [5000, 5000], [321, 321]),
                                  uabUtilreader.patchify(gt, [5000, 5000], [321, 321])):
            if np.sum(patch)/(321 * 321) > 0.2:
                pred_building.append(1)
            else:
                pred_building.append(0)
            if np.sum(g_patch)/(321 * 321) > 0.2:
                g_building.append(1)
            else:
                g_building.append(0)
pred_building = np.array(pred_building)
g_building = np.array(g_building)

fpr_rf, tpr_rf, _ = roc_curve(g_building, pred_building)
print(fpr_rf[1], tpr_rf[1])
