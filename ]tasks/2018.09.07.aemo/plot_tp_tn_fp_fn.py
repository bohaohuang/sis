import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve
import utils
import ersa_utils

'''panel_num = np.array([
    [8, 63, 108, 164, 153, 105, 69, 48, 39, 26, 26, 20, 6, 5, 2, 1, 0, 2, 2, 1, 0, 0, 0, 0, 0],
    [14, 44, 81, 99, 67, 53, 60, 32, 32, 21, 13, 12, 5, 4, 6, 10, 0, 1, 1, 0, 1, 1, 1, 0, 0],
    [34, 119, 236, 207, 143, 86, 82, 52, 40, 39, 24, 6, 9, 6, 6, 3, 3, 3, 2, 1, 1, 0, 0, 1, 2],
    [56, 226, 425, 470, 363, 244, 211, 132, 111, 86, 63, 38, 20, 15, 14, 14, 3, 6, 5, 2, 2, 1, 1, 1, 2],
])
panel_num = panel_num[:, :len(size_list)+1]'''

img_dir, task_dir = utils.get_task_img_folder()
size_list = list(range(0, 760, 40)) + list(range(760, 1000, 120))
step_list = np.concatenate([40 * np.ones(len(list(range(0, 760, 40))), dtype=np.int),
                            120 * np.ones(len(list(range(760, 1000, 120))), dtype=np.int)])

tn_list = np.zeros((4, len(size_list)))
fp_list = np.zeros((4, len(size_list)))
fn_list = np.zeros((4, len(size_list)))
tp_list = np.zeros((4, len(size_list)))

model_dir = ['confmap_uab_UnetCrop_aemo_comb_xfold0_1_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32',
             'confmap_uab_UnetCrop_aemo_comb_xfold1_1_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32',
             'confmap_uab_UnetCrop_aemo_comb_xfold2_1_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32',
             ]

for cnt, min_th in enumerate(size_list):
    max_th = min_th + step_list[cnt]
    save_file_name = os.path.join(task_dir, 'confmat_comb_{}-{}.npy'.format(min_th, max_th))
    conf_mat = ersa_utils.load_file(save_file_name)
    for i in range(4):
        tn_list[i, cnt], fp_list[i, cnt], fn_list[i, cnt], tp_list[i, cnt] = conf_mat[i, :] / np.sum(conf_mat[i, :])

tile_list = ['Region 0', 'Region 1', 'Region 2', 'Aggregate']
colors = ersa_utils.get_default_colors()
for i in range(4):
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(211)
    plt.plot(size_list, tp_list[i, :], '-o', color=colors[0], label='TP')
    plt.xticks(size_list, ['{}~\n{}'.format(a, a+40) for a in size_list])
    plt.ylabel('%')
    plt.grid(True)
    plt.title(tile_list[i])
    plt.legend(loc='upper left')

    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(size_list, fp_list[i, :], '-o', color=colors[1], label='FP')
    plt.plot(size_list, fn_list[i, :], '-o', color=colors[2], label='FN')
    plt.xlabel('Panel Size')
    plt.ylabel('%')
    plt.grid(True)
    plt.title(tile_list[i])
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'tp_tn_fp_fn_plots_{}.png'.format(tile_list[i])))
    plt.show()
