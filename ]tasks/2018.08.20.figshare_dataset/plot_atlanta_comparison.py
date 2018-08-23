import os
import numpy as np
import matplotlib.pyplot as plt
import utils
import util_functions


def read_iou(model_dir):
    result_file = os.path.join(model_dir, 'result.txt')
    city_iou_a = np.zeros(4)
    city_iou_b = np.zeros(4)
    with open(result_file, 'r') as f:
        result_record = f.readlines()
    for cnt, line in enumerate(result_record[:-1]):
        A, B = line.split('(')[1].strip().strip(')').split(',')
        city_iou_a[cnt] = float(A)
        city_iou_b[cnt] = float(B)
    city_iou_a[-1] = np.sum(city_iou_a[:-1])
    city_iou_b[-1] = np.sum(city_iou_b[:-1])
    city_iou = city_iou_a / city_iou_b * 100
    return city_iou


img_dir, task_dir = utils.get_task_img_folder()
city_list = ['atlanta1', 'atlanta2', 'atlanta3']
model_type = 'unet'
colors = util_functions.get_default_colors()
save_fig = True

xtick_list = ['{}{}'.format('atlanta'.capitalize(), a+1) for a in range(3)] + ['Overall']
legend_list = ['Base', 'MMD', 'DIS']

city_ious = np.zeros((len(legend_list), 4))

if model_type == 'unet':
    model_dir_base = r'/hdd/Results/kyle/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_' \
                     r'EP100_LR0.0001_DS60_DR0.1_SFN32/atlanta'
    city_ious[0, :] = read_iou(model_dir_base)

    model_dir_mmd = r'/hdd/Results/kyle/UnetCrop_inria_mmd_xregion_5050_atlanta_1_PS(572, 572)_BS5_' \
                    r'EP40_LR1e-05_DS30_DR0.1_SFN32/atlanta'
    city_ious[1, :] = read_iou(model_dir_mmd)

    model_dir_dis = r'/hdd/Results/kyle/UnetCrop_inria_distance_xregion_5050_atlanta_1_PS(572, 572)_BS5_' \
                    r'EP40_LR1e-05_DS30_DR0.1_SFN32/atlanta'
    city_ious[2, :] = read_iou(model_dir_dis)

plt.figure(figsize=(8, 4))
width = 0.3
X = np.arange(4)
for plt_cnt in range(len(legend_list)):
    plt.bar(X + width * plt_cnt, city_ious[plt_cnt, :], width=width, color=colors[plt_cnt],
            label=legend_list[plt_cnt])
    for cnt, llh in enumerate(city_ious[plt_cnt, :]):
        plt.text(X[cnt] + width * (plt_cnt - 0.5), llh, '{:.1f}'.format(llh), fontsize=8)
plt.xticks(X + width, xtick_list)
plt.ylim([67, 71])
plt.xlabel('City Name')
plt.ylabel('IoUs')
plt.legend()
plt.title('Test on Atlanta')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, '{}_mmd_dis_iou_compare_5050_{}.png'.format(model_type, 'atlanta')))
plt.show()
