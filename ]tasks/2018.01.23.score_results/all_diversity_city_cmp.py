import os
import numpy as np
import matplotlib.pyplot as plt
import uabRepoPaths
import utils


def plot_bar(city_res, all_res, xtick_str, title_str, width=0.15):
    run_type_cnt, city_num, run_cnt = city_res.shape
    width = width
    N = np.arange(city_num+1)
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})
    for i in range(run_type_cnt):
        data = np.concatenate((city_res[i, :, :], np.array([all_res[i, :]])))
        data_mean = np.mean(data, axis=1)
        data_std = np.std(data, axis=1)
        plt.bar(N+i*width, data_mean, width, yerr=data_std)
    plt.xticks(N+(run_type_cnt/2-0.5)*width, xtick_str)
    plt.xlabel('City Name')
    plt.ylabel('IoU')
    plt.title(title_str)


def plot_bar2(city_res, all_res, xtick_str, title_str, width=0.15):
    run_type_cnt, city_num, run_cnt = city_res.shape
    width = width
    ylim_left = [100 for i in range(city_num+1)]
    plt.figure(figsize=(12, 5))
    plt.rcParams.update({'font.size': 14})
    for i in range(run_type_cnt):
        data = np.concatenate((city_res[i, :, :], np.array([all_res[i, :]])))
        data_mean = np.mean(data, axis=1)
        data_std = np.std(data, axis=1)
        for j in range(city_num+1):
            plt.subplot(1, (city_num + 1), j+1)
            plt.bar(np.arange(1)+i*width, data_mean[j], width, yerr=data_std[j])
            plt.xticks(np.arange(1)+(run_type_cnt/2-0.5)*width, [xtick_str[j]])
            if 0.9 * (data_mean[j] - data_std[j]) < ylim_left[j]:
                ylim_left[j] = 0.9 * (data_mean[j] - data_std[j])
            plt.ylim(bottom=ylim_left[j])
    plt.tight_layout()
    plt.suptitle(title_str)


img_dir, task_dir = utils.get_task_img_folder()

# Inria
tick_str = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna', 'avg']
run_ids = [0, 1, 2, 3, 4]
run_types = ['grid', 'incity', 'xcity', 'xgroup']
result_all_unet = np.zeros((len(run_types), len(run_ids)))
city_res_unet = np.zeros((len(run_types), 5, len(run_ids)))
city_dict = {'austin':0, 'chicago':1, 'kitsap':2, 'tyrol-w':3, 'vienna':4}
for cnt_1, run_id in enumerate(run_ids):
    for cnt_2, model_type in enumerate(run_types):
        model_name = \
            'UnetCrop_inria_aug_{}_{}_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'.format(model_type, run_id)
        res_path = os.path.join(uabRepoPaths.evalPath, 'diversity_final', model_name, 'inria', 'result.txt')
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
            city_res_unet[cnt_2, city_dict[city_name[:-1]], cnt_1] = iou
        result_all_unet[cnt_2, cnt_1] = mean_iou_A/mean_iou_B * 100
title_str = 'U-Net Citywise Comparison D1 (2)'
plot_bar2(city_res_unet, result_all_unet, tick_str, title_str)
plt.savefig(os.path.join(img_dir, '{}.png'.format('_'.join(title_str.split()))))
plt.show()

result_all_deeplab = np.zeros((len(run_types), len(run_ids)))
city_res_deeplab = np.zeros((len(run_types), 5, len(run_ids)))
city_dict = {'austin':0, 'chicago':1, 'kitsap':2, 'tyrol-w':3, 'vienna':4}
for cnt_1, run_id in enumerate(run_ids):
    for cnt_2, model_type in enumerate(run_types):
        model_name = \
            'DeeplabV3_inria_aug_{}_{}_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32'.format(model_type, run_id)
        res_path = os.path.join(uabRepoPaths.evalPath, 'diversity_final', model_name, 'inria', 'result.txt')
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
title_str = 'Deeplab-CRF Citywise Comparison D1 (2)'
plot_bar2(city_res_deeplab, result_all_deeplab, tick_str, title_str)
plt.savefig(os.path.join(img_dir, '{}.png'.format('_'.join(title_str.split()))))
plt.show()

# spca
tick_str = ['fresno', 'modesto', 'stockton', 'avg']
run_ids = [0, 1, 2, 3, 4]
run_types = ['grid', 'incity', 'xcity', 'xgroup']
result_all_unet = np.zeros((len(run_types), len(run_ids)))
city_res_A = np.zeros((len(run_types), 3, len(run_ids)))
city_res_B = np.zeros((len(run_types), 3, len(run_ids)))
city_dict = {'Fresno': 0, 'Modesto': 1, 'Stockton': 2}
for cnt_1, run_id in enumerate(run_ids):
    for cnt_2, model_type in enumerate(run_types):
        model_name = \
            'UnetCrop_spca_aug_{}_{}_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'.format(model_type, run_id)
        res_path = os.path.join(uabRepoPaths.evalPath, 'diversity_final', model_name, 'spca', 'result.txt')
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
title_str = 'U-Net Citywise Comparison D2'
plot_bar(city_res_unet, result_all_unet, tick_str, title_str)
plt.savefig(os.path.join(img_dir, '{}.png'.format('_'.join(title_str.split()))))
plt.show()


run_ids = [0, 1, 2, 3, 4]
run_types = ['grid', 'incity', 'xcity', 'xgroup']
result_all_deeplab = np.zeros((len(run_types), len(run_ids)))
city_res_A = np.zeros((len(run_types), 3, len(run_ids)))
city_res_B = np.zeros((len(run_types), 3, len(run_ids)))
city_dict = {'Fresno': 0, 'Modesto': 1, 'Stockton': 2}
for cnt_1, run_id in enumerate(run_ids):
    for cnt_2, model_type in enumerate(run_types):
        model_name = \
            'DeeplabV3_spca_aug_{}_{}_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32'.format(model_type, run_id)
        res_path = os.path.join(uabRepoPaths.evalPath, 'diversity_final', model_name, 'spca', 'result.txt')
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
title_str = 'Deeplab-CRF Citywise Comparison D2'
plot_bar(city_res_deeplab, result_all_deeplab, tick_str, title_str)
plt.savefig(os.path.join(img_dir, '{}.png'.format('_'.join(title_str.split()))))
plt.show()
