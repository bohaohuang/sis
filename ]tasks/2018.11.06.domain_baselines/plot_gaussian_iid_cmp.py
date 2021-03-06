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


def read_loo_iou2(model_dir):
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
            A, B = line.split('(')[1].strip().strip(')').split(',')
            city_iou_a[cnt] += float(A)
            city_iou_b[cnt] += float(B)
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
legend_list = ['LOO', 'IID', 'DTM', 'MMD', 'DIS', 'XRegion']

city_ious = np.zeros((len(legend_list), 6))
model_dir_loo = r'/hdd/Results/domain_selection/UnetCrop_inria_aug_leave_{}_0_PS(572, 572)_BS5_' \
                r'EP100_LR0.0001_DS60_DR0.1_SFN32/inria'
city_ious[0, :] = read_loo_iou(model_dir_loo)

model_dir_iid = r'/hdd/Results/domain_baseline2/UnetCrop_inria_aug_leave_{}_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32/inria'
city_ious[1, :] = read_loo_iou2(model_dir_iid)

model_dir_iid = r'/hdd/Results/domain_baseline2/UnetDTDA_inria_aug_leave_{}_0_iid_PS(572, 572)_BS8_EP100_LR1e-06_DS40.0_DR0.1/inria'
city_ious[2, :] = read_loo_iou2(model_dir_iid)

model_dir_mmd = r'/hdd/Results/mmd/UnetCrop_inria_mmd_loo_5050_{}_1_PS(572, 572)_BS5_EP40_LR'+LR+'_DS30_DR0.1_SFN32/inria'
city_ious[3, :] = read_loo_iou(model_dir_mmd)

model_dir_dis = r'/hdd/Results/mmd/UnetCrop_inria_distance_loo_5050_{}_1_PS(572, 572)_BS5_EP40_LR'+LR+'_DS30_DR0.1_SFN32/inria'
city_ious[4, :] = read_loo_iou(model_dir_dis)

model_dir_base = r'/hdd/Results/domain_selection/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_' \
                 r'EP100_LR0.0001_DS60_DR0.1_SFN32/inria'
city_ious[5, :] = read_iou(model_dir_base, target_city=None)

width = 0.15
X = np.arange(6)
for plt_cnt in range(len(legend_list)):
    plt.bar(X + width * plt_cnt, city_ious[plt_cnt, :], width=width, color=colors[plt_cnt],
            label=legend_list[plt_cnt])
    for cnt, llh in enumerate(city_ious[plt_cnt, :]):
        plt.text(X[cnt] + width * (plt_cnt - 0.5), llh, '{:.1f}'.format(llh), fontsize=6)
plt.xticks(X + width * 2.5, xtick_list, fontsize=10)
plt.ylim([50, 85])
plt.xlabel('City Name')
plt.ylabel('IoUs')
plt.legend(ncol=len(legend_list))
plt.title('Performance Comparison (U-Net)')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'dtm_iid_comp.png'))
plt.show()
