import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import uabRepoPaths
import utils

run_ids = [1, 2, 3, 4, 5]
run_types = ['random', 'grid']
result_all = np.zeros((len(run_types), len(run_ids)))
city_res = np.zeros((len(run_types), 5, len(run_ids)))
city_dict = {'austin':0, 'chicago':1, 'kitsap':2, 'tyrol-w':3, 'vienna':4}

for cnt_1, run_id in enumerate(run_ids):
    for cnt_2, model_type in enumerate(run_types):
        model_name = \
            'UnetCrop_inria_aug_{}_{}_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'.format(model_type, run_id)
        res_path = os.path.join(uabRepoPaths.evalPath, 'grid_vs_random', model_name, 'inria', 'result.txt')
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
            city_res[cnt_2, city_dict[city_name[:-1]], cnt_1] = iou
        result_all[cnt_2, cnt_1] = mean_iou_A/mean_iou_B * 100

matplotlib.rcParams.update({'font.size': 18})
plt.figure(figsize=(11, 6))

plt.subplot(121)
bp = plt.boxplot(np.transpose(result_all))
plt.setp(bp['boxes'][0], color='red')
plt.setp(bp['boxes'][1], color='green')
plt.xticks(np.arange(len(run_types))+1, run_types)
for cnt, b in enumerate(bp['boxes']):
    b.set_label(run_types[cnt])
plt.legend()
plt.xlabel('Patch Extraction Type')
plt.ylabel('IoU')
plt.title('Overall IoU Comparison')

plt.subplot(122)
positions = np.arange(5)
width = 0.35
bp1 = plt.boxplot(np.transpose(city_res[0, :, :]), positions=positions, widths=width)
bp2 = plt.boxplot(np.transpose(city_res[1, :, :]), positions=positions+width, widths=width)
plt.setp(bp1['boxes'], color='red')
plt.setp(bp2['boxes'], color='green')
plt.title('City-wise IoU Comparison')
plt.xticks(positions+width/2, ['Austin', 'Chicago', 'Kitsap', 'Tyrol-w', 'Vienna'], rotation=-12)
plt.xlabel('City Name')
plt.ylabel('IoU')
plt.tight_layout()

img_dir, task_dir = utils.get_task_img_folder()
plt.savefig(os.path.join(img_dir, 'unet_grid_vs_random_inria_fixed.png'))

plt.show()
