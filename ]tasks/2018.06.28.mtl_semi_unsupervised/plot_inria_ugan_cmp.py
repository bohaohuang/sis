import os
import numpy as np
import matplotlib.pyplot as plt
import sis_utils
import util_functions
from gmm_cluster import softmax


def read_iou(model_dir, target_city=None):
    result_file = os.path.join(model_dir, 'result.txt')
    city_iou_a = np.zeros(6)
    city_iou_b = np.zeros(6)
    with open(result_file, 'r') as f:
        result_record = f.readlines()
    if target_city is None:
        for cnt, line in enumerate(result_record[:-1]):
            A, B = line.split('(')[1].strip().strip(')').split(',')
            city_iou_a[cnt // 5] += float(A)
            city_iou_b[cnt // 5] += float(B)
        city_iou_a[-1] = np.sum(city_iou_a[:-1])
        city_iou_b[-1] = np.sum(city_iou_b[:-1])
        city_iou = city_iou_a / city_iou_b * 100
    else:
        for cnt, line in enumerate(result_record[:-1]):
            if cnt // 5 == target_city:
                A, B = line.split('(')[1].strip().strip(')').split(',')
                city_iou_a[cnt % 5] += float(A)
                city_iou_b[cnt % 5] += float(B)
        city_iou_a[-1] = np.sum(city_iou_a[:-1])
        city_iou_b[-1] = np.sum(city_iou_b[:-1])
        city_iou = city_iou_a / city_iou_b * 100
    return city_iou


def read_loo_iou(model_dir):
    city_iou_a_all = np.zeros(6)
    city_iou_b_all = np.zeros(6)
    for target_city in range(5):
        load_dir = model_dir.format(target_city)
        result_file = os.path.join(load_dir, 'result.txt')
        city_iou_a = np.zeros(5)
        city_iou_b = np.zeros(5)
        with open(result_file, 'r') as f:
            result_record = f.readlines()
        for cnt, line in enumerate(result_record[:-1]):
            if cnt // 5 == target_city:
                A, B = line.split('(')[1].strip().strip(')').split(',')
                city_iou_a[cnt % 5] += float(A)
                city_iou_b[cnt % 5] += float(B)
        city_iou_a_all[target_city] = np.sum(city_iou_a)
        city_iou_b_all[target_city] = np.sum(city_iou_b)
    city_iou_a_all[-1] = np.sum(city_iou_a_all[:-1])
    city_iou_b_all[-1] = np.sum(city_iou_b_all[:-1])
    city_iou = city_iou_a_all / city_iou_b_all * 100
    return city_iou


img_dir, task_dir = sis_utils.get_task_img_folder()
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
model_type = 'unet'
colors = util_functions.get_default_colors()
save_fig = True
LR = '1e-05'

plt.figure(figsize=(10, 6))
xtick_list = city_list + ['Overall']
legend_list = ['LOO', 'UGan', 'Base']

city_ious = np.zeros((len(legend_list), 6))
model_dir_loo = r'/hdd/Results/domain_selection/UnetCrop_inria_aug_leave_{}_0_PS(572, 572)_BS5_' \
                r'EP100_LR0.0001_DS60_DR0.1_SFN32/inria'
city_ious[0, :] = read_loo_iou(model_dir_loo)

model_dir_ugan = r'/hdd/Results/ugan/UnetGAN_V3_inria_gan_xregion_0_PS(572, 572)_BS20_' \
                 r'EP30_LR0.0001_1e-06_1e-06_DS30.0_30.0_30.0_DR0.1_0.1_0.1/inria'
city_ious[1, :] = read_loo_iou(model_dir_ugan)

model_dir_base = r'/hdd/Results/domain_selection/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_' \
                 r'EP100_LR0.0001_DS60_DR0.1_SFN32/inria'
city_ious[2, :] = read_iou(model_dir_base, target_city=None)

width = 0.25
X = np.arange(6)
for plt_cnt in range(len(legend_list)):
    plt.bar(X + width * plt_cnt, city_ious[plt_cnt, :], width=width, color=colors[plt_cnt],
            label=legend_list[plt_cnt])
    for cnt, llh in enumerate(city_ious[plt_cnt, :]):
        plt.text(X[cnt] + width * (plt_cnt - 0.5), llh, '{:.1f}'.format(llh), fontsize=6)
plt.xticks(X + width * 1, xtick_list, fontsize=10)
plt.ylim([45, 85])
plt.xlabel('City Name')
plt.ylabel('IoUs')
plt.legend(ncol=len(legend_list))
plt.title('Performance Comparison')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'ugan_cmp_v3_xregion.png'))
plt.show()
