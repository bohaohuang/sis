import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import uabRepoPaths
import sis_utils

run_ids = [0, 1, 2, 3, 4]
batch_sizes = [10,  9,   8,   7,   6,   5,   4,   3,   2,   1]
patch_sizes = [460, 476, 492, 508, 540, 572, 620, 684, 796, 1052]


def get_results(file_str, batch_sizes, patch_sizes):
    result_all = np.zeros((len(batch_sizes), len(run_ids)))
    city_res_A = np.zeros((len(batch_sizes), 5, len(run_ids)))
    city_res_B = np.zeros((len(batch_sizes), 5, len(run_ids)))
    city_dict = {'Fresn': 0, 'Modest': 1, 'Stockto': 2}
    for cnt_1, run_id in enumerate(run_ids):
        for cnt_2, (batch_size, patch_size) in enumerate(zip(batch_sizes, patch_sizes)):
            model_name = \
                'UnetCrop_spca_aug_psbs_{}_PS({}, {})_BS{}_EP100_LR1e-05_DS60_DR0.1_SFN32'. \
                    format(run_id, patch_size, patch_size, batch_size)
            res_path = os.path.join(uabRepoPaths.evalPath, file_str, model_name, 'spca', 'result.txt')
            with open(res_path, 'r') as f:
                results = f.readlines()

            mean_iou_A = 0
            mean_iou_B = 0
            for item in results:
                city_name = item.split(' ')[0]
                city_name = ''.join([i for i in city_name if not i.isdigit()])
                if len(item.split(' ')) == 1:
                    continue
                A, B = item.split('(')[1].strip().strip(')').split(',')
                mean_iou_A += float(A)
                mean_iou_B += float(B)
                if float(B) != 0:
                    city_res_A[cnt_2, city_dict[city_name[:-1]], cnt_1] += float(A)
                    city_res_B[cnt_2, city_dict[city_name[:-1]], cnt_1] += float(B)
            mean_iou = mean_iou_A / mean_iou_B
            result_all[cnt_2, cnt_1] = mean_iou
    result_mean = np.mean(result_all, axis=1) * 100
    result_var = np.var(result_all, axis=1) * 100
    result_up = np.max(result_all, axis=1)
    result_down = np.min(result_all, axis=1)
    return result_mean, result_var, result_up, result_down


matplotlib.rcParams.update({'font.size': 18})
plt.figure(figsize=(12, 5))
ind = np.arange(len(batch_sizes))

ax1 = plt.subplot()
result_mean, result_var, result_up, result_down = get_results('fix_pixel_fix_test', batch_sizes, patch_sizes)
ax1.errorbar(patch_sizes, result_mean, yerr=result_var, uplims=result_up, lolims=result_down, label='test size=1052')

ax2 = ax1.twinx()
ax2.plot(patch_sizes, np.array(batch_sizes)*(np.array(patch_sizes)-184)**2, 'g.--')
#ax2.set_ylim(720000, 780000)
ax2.tick_params('y', colors='g')
ax2.set_ylabel('#pixels', color='g')

plt.grid('on')
plt.xticks(patch_sizes, patch_sizes)
ax1.tick_params(axis='x', labelsize=14)
ax1.set_xlabel('Patch Size')
ax1.set_ylabel('Mean IoU')
plt.title('Unet on Inria')
ax1.legend()
plt.tight_layout()

img_dir, task_dir = sis_utils.get_task_img_folder()
#plt.savefig(os.path.join(img_dir, 'unet_spca_fixpixel.png'))
#with open(os.path.join(task_dir, 'unet_spca_fixpixel.npy'), 'wb') as pk:
#    pickle.dump([result_mean, result_var, result_up, result_down, batch_sizes, patch_sizes], pk)

plt.show()
