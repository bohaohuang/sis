import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import uabRepoPaths
import utils

if __name__ == '__main__':
    file = r'/hdd/Results/grid_vs_random/UnetCrop_spca_aug_grid_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32/spca/result.txt'
    with open(file, 'r') as f:
        records = f.readlines()
    records = records[:-1]
    verify = {'Fresno': np.zeros(2), 'Modesto': np.zeros(2), 'Stockton': np.zeros(2)}
    for line in records:
        city_name = line.split(' ')[0][:-3]
        A, B = line.split('(')[1].strip().strip(')').split(',')
        verify[city_name][0] += float(A)
        verify[city_name][1] += float(B)
    #print(verify['Fresno'])
    #print(verify['Modesto'])
    #print(verify['Stockton'])

run_ids = [0, 1, 2, 3, 4]
run_types = ['random', 'grid']
result_all = np.zeros((len(run_types), len(run_ids)))
city_res_A = np.zeros((len(run_types), 3, len(run_ids)))
city_res_B = np.zeros((len(run_types), 3, len(run_ids)))
city_dict = {'Fresno': 0, 'Modesto': 1, 'Stockton': 2}

for cnt_1, run_id in enumerate(run_ids):
    for cnt_2, model_type in enumerate(run_types):
        model_name = \
            'DeeplabV3_spca_aug_{}_{}_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32'.format(model_type, run_id)
        res_path = os.path.join(uabRepoPaths.evalPath, 'grid_vs_random', model_name, 'spca', 'result.txt')
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
        result_all[cnt_2, cnt_1] = mean_iou

#plt.imshow(city_res_A[0, :, :]/city_res_B[0, :, :])
#plt.show()

city_res = city_res_A/city_res_B
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
positions = np.arange(3)
width = 0.35
bp1 = plt.boxplot(np.transpose(city_res[0, :, :]), positions=positions, widths=width)
bp2 = plt.boxplot(np.transpose(city_res[1, :, :]), positions=positions+width, widths=width)
plt.setp(bp1['boxes'], color='red')
plt.setp(bp2['boxes'], color='green')
plt.title('City-wise IoU Comparison')
plt.xticks(positions+width/2, ['Fresno', 'Modesto', 'Stockton'], rotation=-12)
plt.xlabel('City Name')
plt.ylabel('IoU')
plt.tight_layout()

img_dir, task_dir = utils.get_task_img_folder()
plt.savefig(os.path.join(img_dir, 'deeplab_grid_vs_random_spca.png'))

plt.show()
