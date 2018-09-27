import os
import numpy as np
import utils
import util_functions
import ersa_utils

img_dir, task_dir = utils.get_task_img_folder()
pred_dir = os.path.join(task_dir, 'unet_patch_test_5')
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']

'''for step_size in range(1, 8):
    save_file_name = os.path.join(task_dir, 'iou_record_step_{}.npy'.format(step_size))
    iou_record = np.zeros((25, 2))
    patch_select = list(range(0, 16, step_size))

    cnt = 0
    for city_name in city_list:
        for city_id in range(1, 6):
            pred_map = np.zeros((5000, 4396))
            for i in patch_select:
                ref_dir = os.path.join(pred_dir, 'slide_step_{}'.format(i), 'pred')
                fig_name = '{}{}.png'.format(city_name, city_id)
                pred = ersa_utils.load_file(os.path.join(ref_dir, fig_name))

                pred_map = pred_map + pred
            pred_map = pred_map / len(patch_select)
            pred_map = (pred_map > 0.5).astype(np.int)

            truth_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles'
            gt = ersa_utils.load_file(os.path.join(truth_dir, '{}{}_GT.tif'.format(city_name, city_id))) / 255
            gt = gt[:, 32:-572]

            a, b = util_functions.iou_metric(gt, pred_map, divide_flag=True)
            iou_record[cnt, :] = np.array((a, b))
            cnt += 1

            print('{}{}: {:.4f}'.format(city_name, city_id, a/b))

    ersa_utils.save_file(save_file_name, iou_record)'''

'''save_file_name = os.path.join(task_dir, 'iou_record_ref.npy')
iou_record = np.zeros((25, 2))
ref_dir = r'/hdd/Results/inria_decay/UnetCrop_inria_decay_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60.0_DR0.1_SFN32/inria/pred'
cnt = 0
for city_name in city_list:
    for city_id in range(1, 6):
        fig_name = '{}{}.png'.format(city_name, city_id)
        pred = ersa_utils.load_file(os.path.join(ref_dir, fig_name))

        truth_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles'
        gt = ersa_utils.load_file(os.path.join(truth_dir, '{}{}_GT.tif'.format(city_name, city_id))) / 255
        gt = gt[:, 32:-572]

        a, b = util_functions.iou_metric(gt, pred[:, 32:-572], divide_flag=True)
        iou_record[cnt, :] = np.array((a, b))
        cnt += 1

        print('{}{}: {:.4f}'.format(city_name, city_id, a/b))

ersa_utils.save_file(save_file_name, iou_record)'''
