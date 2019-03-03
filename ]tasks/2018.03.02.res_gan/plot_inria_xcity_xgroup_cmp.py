import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import sis_utils

city_dir = r'/hdd/Results/xcity_vs_incity'
group_dir = r'/hdd/Results/xgroup'
incity_dir = glob(os.path.join(city_dir, 'UnetCrop_inria_aug_incity*/'))
xcity_dir = glob(os.path.join(city_dir, 'UnetCrop_inria_aug_xcity*/'))
xgroup_dir = glob(os.path.join(group_dir, 'UnetCrop_inria_aug_train_unet_inria_fileList*/'))
city_dict = {'aus': 0, 'chi': 1, 'kit': 2, 'tyr': 3, 'vie': 4}
img_dir, task_dir = sis_utils.get_task_img_folder()

incity_record_deeplab = np.zeros((len(incity_dir), 6))
xcity_record_deeplab = np.zeros((len(xcity_dir), 6))
xgroup_record_deeplab = np.zeros((len(xgroup_dir), 6))

for cnt, fold in enumerate(incity_dir):
    result_file = os.path.join(fold, 'inria', 'result.txt')
    with open(result_file, 'r') as f:
        lines = f.readlines()
    # incity_record[cnt] = float(lines[-1])
    for line in lines[:-1]:
        A, B = line.split('(')[1].strip().strip(')').split(',')
        incity_record_deeplab[cnt][city_dict[line[:3]]] = float(A)/float(B)
    incity_record_deeplab[cnt][-1] += float(lines[-1])

for cnt, fold in enumerate(xcity_dir):
    result_file = os.path.join(fold, 'inria', 'result.txt')
    with open(result_file, 'r') as f:
        lines = f.readlines()
    # xcity_record[cnt] = float(lines[-1])
    for line in lines[:-1]:
        A, B = line.split('(')[1].strip().strip(')').split(',')
        xcity_record_deeplab[cnt][city_dict[line[:3]]] = float(A)/float(B)
    xcity_record_deeplab[cnt][-1] += float(lines[-1])

for cnt, fold in enumerate(xgroup_dir):
    result_file = os.path.join(fold, 'inria', 'result.txt')
    with open(result_file, 'r') as f:
        lines = f.readlines()
    # xgroup_record[cnt] = float(lines[-1])
    for line in lines[:-1]:
        A, B = line.split('(')[1].strip().strip(')').split(',')
        xgroup_record_deeplab[cnt][city_dict[line[:3]]] = float(A)/float(B)
    xgroup_record_deeplab[cnt][-1] += float(lines[-1])

print(xgroup_record_deeplab)

plt.figure(figsize=(8, 4))
plt.boxplot([xgroup_record_deeplab[:, -1], incity_record_deeplab[:, -1], xcity_record_deeplab[:, -1]])
plt.xticks(np.arange(3)+1, ['InCity', 'XCity', 'XGroup'])
plt.xlabel('Train Type')
plt.ylabel('IoU')
plt.grid(linestyle='dotted')
plt.title('Overall Performance Comparison')
plt.tight_layout()
#plt.savefig(os.path.join(img_dir, 'overall_performance_cmp.png'))
#plt.show()

'''plt.figure(figsize=(8, 5))
ind = np.arange(6)
width = 0.2
plt.bar(ind, np.mean(incity_record, axis=0), width=width, label='InCity')
plt.bar(ind+width, np.mean(xcity_record, axis=0), width=width, label='XCity')
plt.bar(ind+width*2, np.mean(xgroup_record, axis=0), width=width, label='XGroup')
plt.xticks(ind+width, ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna', 'Avg'])
plt.legend(loc='upper left')
plt.xlabel('City')
plt.ylabel('IoU')
#plt.ylim(0.65, 0.85)
plt.title('Citywise Performance Comparison')
plt.tight_layout()
#plt.savefig(os.path.join(img_dir, 'citywise_performance_cmp.png'))'''
plt.show()
