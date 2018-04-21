import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import uabRepoPaths
import utils


def set_box_color(bp, color, c2):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=c2)


run_ids = [0, 1, 2, 3, 4]
run_types = ['incity', 'xcity']
result_all_deeplab = np.zeros((len(run_types), len(run_ids)))
city_res_deeplab = np.zeros((len(run_types), 5, len(run_ids)))
city_dict = {'austin':0, 'chicago':1, 'kitsap':2, 'tyrol-w':3, 'vienna':4}

for cnt_1, run_id in enumerate(run_ids):
    for cnt_2, model_type in enumerate(run_types):
        model_name = \
            'DeeplabV3_inria_aug_{}_{}_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32'.format(model_type, run_id)
        res_path = os.path.join(uabRepoPaths.evalPath, 'xcity_vs_incity', model_name, 'inria', 'result.txt')
        with open(res_path, 'r') as f:
            results = f.readlines()

        mean_iou_A = 0
        mean_iou_B = 0
        for item in results:
            city_name = item.split(' ')[0]
            if len(item.split(' ')) == 1:
                mean_iou = float(item) * 100
                continue
            A, B = item.split('(')[1].strip().strip(')').split(',')
            iou = float(A)/float(B) * 100
            mean_iou_A += float(A)
            mean_iou_B += float(B)
            city_res_deeplab[cnt_2, city_dict[city_name[:-1]], cnt_1] = iou
        result_all_deeplab[cnt_2, cnt_1] = mean_iou_A/mean_iou_B * 100

run_ids = [1, 2, 3, 4, 5]
run_types = ['incity', 'xcity']
result_all_unet = np.zeros((len(run_types), len(run_ids)))
city_res_unet = np.zeros((len(run_types), 5, len(run_ids)))

for cnt_1, run_id in enumerate(run_ids):
    for cnt_2, model_type in enumerate(run_types):
        model_name = \
            'UnetCrop_inria_aug_{}_{}_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'.format(model_type, run_id)
        res_path = os.path.join(uabRepoPaths.evalPath, 'xcity_vs_incity', model_name, 'inria', 'result.txt')
        with open(res_path, 'r') as f:
            results = f.readlines()

        mean_iou_A = 0
        mean_iou_B = 0
        for item in results:
            city_name = item.split(' ')[0]
            if len(item.split(' ')) == 1:
                mean_iou = float(item) * 100
                continue
            A, B = item.split('(')[1].strip().strip(')').split(',')
            mean_iou_A += float(A)
            mean_iou_B += float(B)
            iou = float(A)/float(B) * 100
            city_res_unet[cnt_2, city_dict[city_name[:-1]], cnt_1] = iou
        result_all_unet[cnt_2, cnt_1] = mean_iou_A/mean_iou_B * 100

iou_unet = []
iou_deeplab = []
for i in range(5):
    file_name = r'/hdd/Results/xgroup/DeeplabV3_inria_control_patch_deeplab_inria_fileList_{}_PS(321, 321)_' \
                r'BS5_EP100_LR1e-05_DS40_DR0.1_SFN32/inria/result.txt'.format(i)
    with open(file_name, 'r') as f:
        results = f.readlines()
    iou_deeplab.append(float(results[-1])*100)
for i in range(5, 10):
    file_name = r'/hdd/Results/xgroup/UnetCrop_inria_control_patch_unet_inria_fileList_{}_PS(572, 572)_' \
                r'BS5_EP100_LR0.0001_DS40_DR0.1_SFN32/inria/result.txt'.format(i)
    with open(file_name, 'r') as f:
        results = f.readlines()
    iou_unet.append(float(results[-1])*100)

result_all_unet = np.append(np.array([[75.88011385, 75.85050917, 75.86489646, 75.9223303, 75.89194368]]),
                            result_all_unet, axis=0)
result_all_unet = np.append(result_all_unet, np.array([iou_unet]), axis=0)
result_all_deeplab = np.append(np.array([[76.0153697,  75.88163622, 76.16206155, 76.19242719, 76.13832969]]),
                               result_all_deeplab, axis=0)
result_all_deeplab = np.append(result_all_deeplab, np.array([iou_deeplab]), axis=0)

matplotlib.rcParams.update({'font.size': 14})
plt.figure(figsize=(8, 6))

plt.subplot(211)
positions = np.arange(4)
width = 0.3
bp1 = plt.boxplot(np.transpose(result_all_unet[0, :]), positions=[positions[0]],widths=width)
bp2 = plt.boxplot(np.transpose(result_all_unet[1, :]), positions=[positions[1]],widths=width)
bp3 = plt.boxplot(np.transpose(result_all_unet[2, :]), positions=[positions[2]],widths=width)
bp4 = plt.boxplot(np.transpose(result_all_unet[3, :]), positions=[positions[3]],widths=width)
bp5 = plt.boxplot(np.transpose(result_all_deeplab[0, :]), positions=[positions[0]+width], widths=width)
bp6 = plt.boxplot(np.transpose(result_all_deeplab[1, :]), positions=[positions[1]+width], widths=width)
bp7 = plt.boxplot(np.transpose(result_all_deeplab[2, :]), positions=[positions[2]+width], widths=width)
bp8 = plt.boxplot(np.transpose(result_all_deeplab[3, :]), positions=[positions[3]+width], widths=width)
set_box_color(bp1, cm.tab10(0), 'blue')
set_box_color(bp2, cm.tab10(0), 'red')
set_box_color(bp3, cm.tab10(0), 'green')
set_box_color(bp4, cm.tab10(0), 'purple')
set_box_color(bp5, cm.tab10(1), 'blue')
set_box_color(bp6, cm.tab10(1), 'red')
set_box_color(bp7, cm.tab10(1), 'green')
set_box_color(bp8, cm.tab10(1), 'purple')
plt.xlim([-0.25, 3.75])
plt.xticks(np.arange(len(run_types)+2)+width/2, ['base', 'low diversity', 'high diversity', 'higher diversity'])
plt.plot([], c=cm.tab10(0), label='U-Net')
plt.plot([], c=cm.tab10(1), label='Deeplab-CRF')
plt.ylim([75, 77])
plt.legend(loc='lower left')
plt.xlabel('Patch Extraction Type')
plt.ylabel('IoU')
plt.title('Overall IoU Comparison D1')


# spca
run_ids = [1, 2, 3, 4]
run_types = ['incity', 'xcity']
result_all_unet = np.zeros((len(run_types), len(run_ids)))
city_res_A = np.zeros((len(run_types), 3, len(run_ids)))
city_res_B = np.zeros((len(run_types), 3, len(run_ids)))
city_dict = {'Fresno': 0, 'Modesto': 1, 'Stockton': 2}

for cnt_1, run_id in enumerate(run_ids):
    for cnt_2, model_type in enumerate(run_types):
        model_name = \
            'UnetCrop_spca_aug_{}_{}_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'.format(model_type, run_id)
        res_path = os.path.join(uabRepoPaths.evalPath, 'xcity_vs_incity2', model_name, 'spca', 'result.txt')
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
                city_res_A[cnt_2, city_dict[city_name], cnt_1] += float(A)
                city_res_B[cnt_2, city_dict[city_name], cnt_1] += float(B)
        mean_iou = mean_iou_A/mean_iou_B
        result_all_unet[cnt_2, cnt_1] = mean_iou
city_res_unet = city_res_A/city_res_B


run_ids = [0, 1, 2, 3, 4]
run_types = ['incity', 'xcity']
result_all_deeplab = np.zeros((len(run_types), len(run_ids)))
city_res_A = np.zeros((len(run_types), 3, len(run_ids)))
city_res_B = np.zeros((len(run_types), 3, len(run_ids)))
city_dict = {'Fresno': 0, 'Modesto': 1, 'Stockton': 2}

for cnt_1, run_id in enumerate(run_ids):
    for cnt_2, model_type in enumerate(run_types):
        model_name = \
            'DeeplabV3_spca_aug_{}_{}_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32'.format(model_type, run_id)
        res_path = os.path.join(uabRepoPaths.evalPath, 'xcity_vs_incity2', model_name, 'spca', 'result.txt')
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
                city_res_A[cnt_2, city_dict[city_name], cnt_1] += float(A)
                city_res_B[cnt_2, city_dict[city_name], cnt_1] += float(B)
        mean_iou = mean_iou_A/mean_iou_B
        result_all_deeplab[cnt_2, cnt_1] = mean_iou
city_res_deeplab = city_res_A/city_res_B

iou_unet = []
for i in range(5, 9):
    file_name = r'/hdd/Results/xgroup/UnetCrop_spca_aug_train_unet_spca_fileList_{}_PS(572, 572)_' \
                r'BS5_EP100_LR0.0001_DS40_DR0.1_SFN32/spca/result.txt'.format(i)
    with open(file_name, 'r') as f:
        results = f.readlines()
    iou_unet.append(float(results[-1])*100)
iou_deeplab = []
for i in range(5):
    file_name = r'//hdd/Results/xgroup/DeeplabV3_spca_aug_train_deeplab_spca_fileList_{}_PS(321, 321)_' \
                r'BS5_EP100_LR1e-05_DS40_DR0.1_SFN32/spca/result.txt'.format(i)
    with open(file_name, 'r') as f:
        results = f.readlines()
    iou_deeplab.append(float(results[-1])*100)

result_all_unet = np.append(np.array([[53.322488, 49.35155,  51.595617, 52.224888]]), result_all_unet*100, axis=0)
result_all_unet = np.append(result_all_unet, np.array([iou_unet]), axis=0)
result_all_deeplab = np.append(np.array([[50.620876, 52.950845, 53.664448, 54.913586, 50.957393]]), result_all_deeplab*100, axis=0)
result_all_deeplab = np.append(result_all_deeplab, np.array([iou_deeplab]), axis=0)

plt.subplot(212)
positions = np.arange(4)
width = 0.3
bp1 = plt.boxplot(np.transpose(result_all_unet[0, :]), positions=[positions[0]],widths=width)
bp2 = plt.boxplot(np.transpose(result_all_unet[1, :]), positions=[positions[1]],widths=width)
bp3 = plt.boxplot(np.transpose(result_all_unet[2, :]), positions=[positions[2]],widths=width)
bp4 = plt.boxplot(np.transpose(result_all_unet[3, :]), positions=[positions[3]],widths=width)
bp5 = plt.boxplot(np.transpose(result_all_deeplab[0, :]), positions=[positions[0]+width], widths=width)
bp6 = plt.boxplot(np.transpose(result_all_deeplab[1, :]), positions=[positions[1]+width], widths=width)
bp7 = plt.boxplot(np.transpose(result_all_deeplab[2, :]), positions=[positions[2]+width], widths=width)
bp8 = plt.boxplot(np.transpose(result_all_deeplab[3, :]), positions=[positions[3]+width], widths=width)
set_box_color(bp1, cm.tab10(0), 'blue')
set_box_color(bp2, cm.tab10(0), 'red')
set_box_color(bp3, cm.tab10(0), 'green')
set_box_color(bp4, cm.tab10(0), 'purple')
set_box_color(bp5, cm.tab10(1), 'blue')
set_box_color(bp6, cm.tab10(1), 'red')
set_box_color(bp7, cm.tab10(1), 'green')
set_box_color(bp8, cm.tab10(1), 'purple')
plt.xlim([-0.25, 3.75])
plt.xticks(np.arange(len(run_types)+2)+width/2, ['base', 'low diversity', 'high diversity', 'higher diversity'])
plt.plot([], c=cm.tab10(0), label='U-Net')
plt.plot([], c=cm.tab10(1), label='Deeplab-CRF')
plt.legend(loc='lower left')
plt.xlabel('Patch Extraction Type')
plt.ylabel('IoU')
plt.title('Overall IoU Comparison D2')

plt.tight_layout()
img_dir, task_dir = utils.get_task_img_folder()
plt.savefig(os.path.join(img_dir, 'all_diversity.png'))

plt.show()
