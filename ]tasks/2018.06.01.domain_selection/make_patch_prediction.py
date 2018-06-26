import os
import imageio
import numpy as np
from tqdm import tqdm
import utils
import uabUtilreader
import uabCrossValMaker
import uab_collectionFunctions
import uab_DataHandlerFunctions


def center_crop(img, pad):
    return img[pad[0]:-pad[0], pad[1]:-pad[1]]


# settings
cnn_name = 'unet'
city_num = 4
img_dir, task_dir = utils.get_task_img_folder()
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
blCol = uab_collectionFunctions.uabCollection('inria')
img_mean = blCol.getChannelMeans([0, 1, 2])

if cnn_name == 'deeplab':
    c_size = 321
    extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4],
                                                    cSize=(c_size, c_size),
                                                    numPixOverlap=0,
                                                    extSave=['jpg', 'jpg', 'jpg', 'png'],
                                                    isTrain=True,
                                                    gtInd=3,
                                                    pad=0)
    pred_file_dir = r'/hdd/Results/domain_loo_all/DeeplabV3_inria_aug_train_leave_{}_0_PS(321, 321)_BS5_EP100_LR1e-05_' \
                    r'DS40_DR0.1_SFN32/inria/pred'.format(city_num)
    pad = 0
    overlap = 0
    size = 321
else:
    c_size = 572
    extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4],
                                                    cSize=(c_size, c_size),
                                                    numPixOverlap=184,
                                                    extSave=['jpg', 'jpg', 'jpg', 'png'],
                                                    isTrain=True,
                                                    gtInd=3,
                                                    pad=92)
    pred_file_dir = r'/hdd/Results/domain_loo_all/UnetCrop_inria_aug_leave_{}_0_PS(572, 572)_BS5_EP100_LR0.0001_' \
                    r'DS60_DR0.1_SFN32/inria/pred'.format(city_num)
    pad = 92
    overlap = 184
    size = 388

patchDir = extrObj.run(blCol)
chipFiles = os.path.join(patchDir, 'fileList.txt')
idx, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'city')
file_list_valid = uabCrossValMaker.make_file_list_by_key(idx, file_list, [0])

building_percent = np.zeros(len(file_list_valid))
for cnt, files in enumerate(tqdm(file_list_valid)):
    gt_file_name = files[-1]
    gt_parts = gt_file_name.split('_')
    city_id = int(''.join([i for i in gt_parts[0] if i.isdigit()]))
    y_str, x_str = gt_parts[1].split('x')
    y = int(''.join([i for i in y_str if i.isdigit()]))
    x = int(''.join([i for i in x_str if i.isdigit()]))

    pred_file_name = os.path.join(pred_file_dir, '{}{}.png'.format(city_list[city_num], city_id))
    pred = imageio.imread(pred_file_name)
    if pad > 0:
        pred = np.squeeze(uabUtilreader.pad_block(np.expand_dims(pred, axis=2), np.array([pad, pad])), axis=2)
    patch = pred[y:y+c_size, x:x+c_size]
    if pad > 0:
        patch = center_crop(patch, (pad, pad))

    building_percent[cnt] = np.sum(patch) / (size * size)
np.save(os.path.join(task_dir, '{}_{}_building_record.npy'.format(cnn_name, city_list[city_num])), building_percent)
