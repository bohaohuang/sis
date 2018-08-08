import os
import numpy as np
import matplotlib.pyplot as plt
import utils
import util_functions


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
T = [100, 500, 1500, 3000, 5000, 10000, 20000, 40000]
target_iou = np.zeros((6, len(T)))


for city_id in [1]:
    for cnt, t in enumerate(T):
        model_dir = r'/hdd/Results/Inria_Domain_Selection/UnetCrop_inria_{}_t{:.1f}_0_PS(572, 572)_BS5_' \
                     r'EP40_LR1e-05_DS30_DR0.1_SFN32/inria'.format(city_list[city_id], t)
        city_iou = read_iou(model_dir)
        target_iou[:, cnt] = city_iou

    plt.plot(np.arange(len(T)), target_iou[city_id, :], '-o', label=city_list[city_id])
    other_city = [i for i in range(5) if i != city_id]
    '''for i in other_city:
        plt.plot(T, target_iou[i, :], '--o', label=city_list[i])'''
    #plt.plot(T, target_iou[-1, :], '--o', label='overall')
    plt.legend()
    plt.xticks(np.arange(len(T)), T)
    plt.xlabel('Temperature')
    plt.ylabel('IoU')
    plt.grid('on')
    plt.tight_layout()
    plt.show()

'''model_list = [
    r'/hdd/Results/domain_selection/UnetCrop_inria_aug_leave_0_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32',
    r'/hdd/Results/domain_selection/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32',
    #r'/hdd/Results/domain_selection/UnetCrop_inria_austin_loo_patch_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32',
    r'/hdd/Results/domain_selection/UnetPredict_inria_loo_mtl_0_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32',
]
model_name_show = ['LOO', 'Base', 'MTL']

fig = plt.figure()
for plt_cnt, model_name in enumerate(model_list):
    city_iou_a = np.zeros(6)
    city_iou_b = np.zeros(6)

    model_dir = model_name + '/inria'
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

    width = 0.3
    X = np.arange(6)
    plt.bar(X + width * plt_cnt, city_iou, width=width, label=model_name_show[plt_cnt])
    plt.xticks(X + width, city_list + ['Over All'])
    plt.xlabel('City')
    plt.ylabel('IoU')
plt.legend(loc='upper right')
plt.ylim([50, 85])
plt.title('IoU Comparison UNet Austin')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'unet_austin_cmp_loo_mtl.png'))
plt.show()'''
