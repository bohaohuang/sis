import os
import numpy as np
import matplotlib.pyplot as plt
import utils
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


img_dir, task_dir = utils.get_task_img_folder()
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
model_type = 'unet'
colors = util_functions.get_default_colors()

plt.figure(figsize=(8, 6))
for city_id in [1]:
    xtick_list = ['{}{}'.format(city_list[city_id].capitalize(), a+1) for a in range(5)] + ['Overall']
    legend_list = ['LOO', 'UGan', 'Base']

    city_ious = np.zeros((3, 6))
    model_dir_loo = r'/hdd/Results/domain_selection/UnetCrop_inria_aug_leave_{}_0_PS(572, 572)_BS5_' \
                    r'EP100_LR0.0001_DS60_DR0.1_SFN32/inria'.format(city_id)
    city_ious[0, :] = read_iou(model_dir_loo, target_city=city_id)

    model_dir_ugan = r'/hdd/Results/ugan/UnetGAN_V3Shrink_inria_gan_loo_{}_0_PS(572, 572)_BS20_EP30_' \
                     r'LR0.0001_1e-05_1e-05_DS30.0_30.0_30.0_DR0.1_0.1_0.1/inria'.format(city_id)
    city_ious[1, :] = read_iou(model_dir_ugan, target_city=city_id)

    model_dir_base = r'/hdd/Results/domain_selection/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_' \
                     r'EP100_LR0.0001_DS60_DR0.1_SFN32/inria'
    city_ious[2, :] = read_iou(model_dir_base, target_city=city_id)

    width = 0.3
    X = np.arange(6)
    for plt_cnt in range(3):
        plt.bar(X + width * plt_cnt, city_ious[plt_cnt, :], width=width, color=colors[plt_cnt],
                label=legend_list[plt_cnt])
        for cnt, llh in enumerate(city_ious[plt_cnt, :]):
            plt.text(X[cnt] + width * (plt_cnt - 0.5), llh, '{:.1f}'.format(llh), fontsize=8)
    plt.xticks(X + width * 2, xtick_list)
    plt.ylim([50, 75])
    plt.xlabel('City Name')
    plt.ylabel('IoUs')
    plt.legend()
    plt.title('Finetune on {}'.format(city_list[city_id].capitalize()))
    plt.tight_layout()
    #plt.savefig(os.path.join(img_dir, 'ugan_iou_compare_{}_{}.png'.format(city_list[city_id],
    #                                                                      model_dir_ugan.split('/')[-2])))
    plt.show()
