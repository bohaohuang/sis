import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import uabRepoPaths
import utils

run_ids = [0, 1, 2, 3, 4]
batch_sizes = [10,  8,   7,   6,   5,   4,   3,   2]
patch_sizes = [232, 248, 264, 276, 300, 321, 368, 424, 520]
result_all = np.zeros((len(batch_sizes), len(run_ids)))
city_res = np.zeros((len(batch_sizes), 5, len(run_ids)))
city_dict = {'austin':0, 'chicago':1, 'kitsap':2, 'tyrol-w':3, 'vienna':4}


def get_results(file_str):
    for cnt_1, run_id in enumerate(run_ids):
        for cnt_2, (batch_size, patch_size) in enumerate(zip(batch_sizes, patch_sizes)):
            model_name = \
                'DeepLab_inria_aug_psbs_{}_PS({}, {})_BS{}_EP100_LR0.0001_DS60_DR0.1_SFN32'.\
                    format(run_id, patch_size, patch_size, batch_size)
            res_path = os.path.join(uabRepoPaths.evalPath, file_str, model_name, 'inria', 'result.txt')
            with open(res_path, 'r') as f:
                results = f.readlines()

            mean_iou = 0
            for item in results:
                city_name = item.split(' ')[0]
                if len(item.split(' ')) == 1:
                    mean_iou = float(item) * 100
                    continue
                A, B = item.split('(')[1].strip().strip(')').split(',')
                iou = float(A)/float(B) * 100
                city_res[cnt_2, city_dict[city_name[:-1]], cnt_1] = iou
            result_all[cnt_2, cnt_1] = mean_iou
    result_mean = np.mean(result_all, axis=1)
    result_var = np.var(result_all, axis=1)
    result_up = np.max(result_all, axis=1)
    result_down = np.min(result_all, axis=1)
    return result_mean, result_var, result_up, result_down


matplotlib.rcParams.update({'font.size': 18})
plt.figure(figsize=(12, 5))
ind = np.arange(len(batch_sizes))

ax1 = plt.subplot()
result_mean, result_var, result_up, result_down = get_results('fix_pixel_fix_test')
ax1.errorbar(patch_sizes, result_mean, yerr=result_var, uplims=result_up, lolims=result_down, label='test size=496')
result_mean, result_var, result_up, result_down = get_results('fix_pixel')
ax1.errorbar(patch_sizes, result_mean, yerr=result_var, uplims=result_up, lolims=result_down, label='test size=train size')

ax2 = ax1.twinx()
ax2.plot(patch_sizes, (np.array(batch_sizes)*np.array(patch_sizes)/4)**2, 'g.--')
# ax2.set_ylim(240000, 270000)
ax2.tick_params('y', colors='g')
ax2.set_ylabel('#pixels', color='g')

plt.grid('on')
plt.xticks(patch_sizes, patch_sizes)
ax1.set_xlabel('Patch Size')
ax1.set_ylabel('Mean IoU')
plt.title('DeepLab on Inria')
ax1.legend()
plt.tight_layout()

img_dir, task_dir = utils.get_task_img_folder()
plt.savefig(os.path.join(img_dir, 'deeplab_inria_fixpixel.png'))

plt.show()
