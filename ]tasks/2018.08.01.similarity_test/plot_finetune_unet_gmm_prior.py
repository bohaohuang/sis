import os
import numpy as np
import matplotlib.pyplot as plt
import utils
import util_functions
from gmm_cluster import softmax


def read_iou(model_dir):
    city_iou_a = np.zeros(6)
    city_iou_b = np.zeros(6)
    result_file = os.path.join(model_dir, 'result.txt')
    with open(result_file, 'r') as f:
        result_record = f.readlines()
    for cnt, line in enumerate(result_record[:-1]):
        A, B = line.split('(')[1].strip().strip(')').split(',')
        city_iou_a[cnt // 5] += float(A)
        city_iou_b[cnt // 5] += float(B)
    city_iou_a[-1] = np.sum(city_iou_a[:-1])
    city_iou_b[-1] = np.sum(city_iou_b[:-1])
    city_iou = city_iou_a / city_iou_b * 100
    return city_iou


img_dir, task_dir = utils.get_task_img_folder()
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
model_type = 'unet'
colors = util_functions.get_default_colors()
T = [100, 500, 1500, 3000, 5000, 10000, 20000]
save_fig = False
target_iou = np.zeros((6, len(T)))
softmax_llh = np.zeros((5, len(T)))
llh_all_record = np.load(os.path.join(task_dir, 'llh_unet_inria_n50.npy'))


for city_id in [2]:
    llh_all = llh_all_record[city_id, :]
    for cnt, t in enumerate(T):
        model_dir = r'/hdd/Results/Inria_Domain_Selection/UnetCrop_inria_{}_t{:.1f}_0_PS(572, 572)_BS5_' \
                     r'EP30_LR1e-05_DS20_DR0.1_SFN32/inria'.format(city_list[city_id], t)
        city_iou = read_iou(model_dir)
        target_iou[:, cnt] = city_iou

        # make softmax llh matrix
        softmax_llh[:, cnt] = softmax(llh_all, t)

    plt.figure(figsize=(18, 8))

    plt.subplot(211)
    width = 0.15
    X = np.arange(len(T))
    for plt_cnt in range(5):
        plt.bar(X + width * plt_cnt, softmax_llh[plt_cnt, :], width=width, color=colors[0], edgecolor='k')
        for cnt, llh in enumerate(softmax_llh[plt_cnt, :]):
            plt.text(X[cnt] + width * (plt_cnt - 0.4), llh, '{:.1f}'.format(llh*100), fontsize=8)
    for plt_cnt in range(len(T)):
        plt.vlines(X[plt_cnt] - width, 0, 1.2)
        plt.vlines(X[plt_cnt] + 5*width, 0, 1.2)
    plt.xticks(X + width * 2, ['T={}'.format(a) for a in T])
    plt.ylim([0, 1.2])
    plt.xlabel('Temperature')
    plt.ylabel('Prior')
    plt.title('Finetune on {}'.format(city_list[city_id]))

    plt.subplot(212)
    width = 0.1
    X = np.arange(6)
    for plt_cnt, t in enumerate(T):
        plt.bar(X + width * plt_cnt, target_iou[:, plt_cnt], width=width, label='T={}'.format(t),
                color=colors[plt_cnt + 1])
        plt.xticks(X + width * (len(T) / 2 - 0.5), city_list + ['Over All'])
        plt.xlabel('City')
        plt.ylabel('IoU')
        for cnt, iou in enumerate(target_iou[:, plt_cnt]):
            plt.text(X[cnt] + width * (plt_cnt - 0.5), iou, '{:.1f}'.format(iou), fontsize=8)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=9, fancybox=True, shadow=True)
    plt.ylim([55, 85])
    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(img_dir, 'finetune_t_{}_comparison_on_{}.png'.format(
            '_'.join([str(a) for a in T]), city_list[city_id])))
    plt.show()
