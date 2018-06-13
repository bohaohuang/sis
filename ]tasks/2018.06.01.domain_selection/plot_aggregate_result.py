import os
import numpy as np
import matplotlib.pyplot as plt
import utils
import util_functions

img_dir, task_dir = utils.get_task_img_folder()
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
model_type = 'deeplab'
colors = util_functions.get_default_colors()

if model_type == 'deeplab':
    model_list = [
                  r'/hdd/Results/domain_selection/DeeplabV3_inria_aug_grid_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32',
                  r'/hdd/Results/domain_selection/DeeplabV3_inria_{}_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32',
    ]
    model_name_show = ['Base', 'Agg']

    fig = plt.figure()
    for plt_cnt, model_name in enumerate(model_list):
        if plt_cnt == 0:
            city_iou_a = np.zeros(6)
            city_iou_b = np.zeros(6)

            model_dir = model_name + '/inria'
            result_file = os.path.join(model_dir, 'result.txt')
            with open(result_file, 'r') as f:
                result_record = f.readlines()
            for cnt, line in enumerate(result_record[:-1]):
                A, B = line.split('(')[1].strip().strip(')').split(',')
                city_iou_a[cnt//5] += float(A)
                city_iou_b[cnt//5] += float(B)
            city_iou_a[-1] = np.sum(city_iou_a[:-1])
            city_iou_b[-1] = np.sum(city_iou_b[:-1])
            city_iou = city_iou_a/city_iou_b * 100
        else:
            city_iou_a_all = np.zeros(6)
            city_iou_b_all = np.zeros(6)
            for city_cnt, city_name in enumerate(city_list):
                city_iou_a = np.zeros(6)
                city_iou_b = np.zeros(6)

                model_dir = model_name + '/inria'
                result_file = os.path.join(model_dir.format(city_name), 'result.txt')
                with open(result_file, 'r') as f:
                    result_record = f.readlines()
                for cnt, line in enumerate(result_record[:-1]):
                    A, B = line.split('(')[1].strip().strip(')').split(',')
                    city_iou_a[cnt // 5] += float(A)
                    city_iou_b[cnt // 5] += float(B)
                city_iou_a_all[city_cnt] = city_iou_a[city_cnt]
                city_iou_b_all[city_cnt] = city_iou_b[city_cnt]
            city_iou_a_all[-1] = np.sum(city_iou_a_all[:-1])
            city_iou_b_all[-1] = np.sum(city_iou_b_all[:-1])
            city_iou = city_iou_a_all / city_iou_b_all * 100

        width = 0.35
        X = np.arange(6)
        plt.bar(X+width*plt_cnt, city_iou, width=width, label=model_name_show[plt_cnt],
                color=colors[plt_cnt + 1])
        plt.xticks(X + width * 0.5, city_list+['Over All'])
        plt.xlabel('City')
        plt.ylabel('IoU')
        for cnt, iou in enumerate(city_iou):
            plt.text(X[cnt] +  width * (plt_cnt - 0.5), iou, '{:.1f}'.format(iou))
    plt.legend(loc='upper right')
    plt.ylim([50, 85])
    plt.title('IoU Comparison Deeplab Austin')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'deeplab_agg_cmp.png'))
    plt.show()
else:
    model_list = [
        r'/hdd/Results/domain_selection/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32',
        r'/hdd/Results/domain_selection/UnetCrop_inria_{}_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32',
    ]
    model_name_show = ['Base', 'Agg']

    fig = plt.figure()
    for plt_cnt, model_name in enumerate(model_list):
        if plt_cnt == 0:
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
        else:
            city_iou_a_all = np.zeros(6)
            city_iou_b_all = np.zeros(6)
            for city_cnt, city_name in enumerate(city_list):
                city_iou_a = np.zeros(6)
                city_iou_b = np.zeros(6)

                model_dir = model_name + '/inria'
                result_file = os.path.join(model_dir.format(city_name), 'result.txt')
                with open(result_file, 'r') as f:
                    result_record = f.readlines()
                for cnt, line in enumerate(result_record[:-1]):
                    A, B = line.split('(')[1].strip().strip(')').split(',')
                    city_iou_a[cnt // 5] += float(A)
                    city_iou_b[cnt // 5] += float(B)
                city_iou_a_all[city_cnt] = city_iou_a[city_cnt]
                city_iou_b_all[city_cnt] = city_iou_b[city_cnt]
            city_iou_a_all[-1] = np.sum(city_iou_a_all[:-1])
            city_iou_b_all[-1] = np.sum(city_iou_b_all[:-1])
            city_iou = city_iou_a_all / city_iou_b_all * 100

        width = 0.35
        X = np.arange(6)
        plt.bar(X + width * plt_cnt, city_iou, width=width, label=model_name_show[plt_cnt],
                color=colors[plt_cnt + 1])
        plt.xticks(X + width * 0.5, city_list + ['Over All'])
        plt.xlabel('City')
        plt.ylabel('IoU')
        for cnt, iou in enumerate(city_iou):
            plt.text(X[cnt] +  width * (plt_cnt - 0.5), iou, '{:.1f}'.format(iou))
    plt.legend(loc='upper right')
    plt.ylim([50, 85])
    plt.title('IoU Comparison UNet Aggregate')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'unet_agg_cmp.png'))
    plt.show()
