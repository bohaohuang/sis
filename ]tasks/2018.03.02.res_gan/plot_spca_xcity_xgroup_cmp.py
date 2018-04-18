import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import utils

city_dir = r'/hdd/Results/xcity_vs_incity'
group_dir = r'/hdd/Results/xgroup'
incity_dir = glob(os.path.join(city_dir, 'DeeplabV3_spca_aug_incity*/'))
xcity_dir = glob(os.path.join(city_dir, 'DeeplabV3_spca_aug_xcity*/'))
xgroup_dir = glob(os.path.join(group_dir, 'DeeplabV3_spca_aug_train_deeplab_spca_fileList*/'))
city_dict = {'Fre': 0, 'Mod': 1, 'Sto': 2}
img_dir, task_dir = utils.get_task_img_folder()

incity_record_a = np.zeros((len(incity_dir), 4))
incity_record_b = np.zeros((len(incity_dir), 4))
xcity_record_a = np.zeros((len(xcity_dir), 4))
xcity_record_b = np.zeros((len(xcity_dir), 4))
xgroup_record_a = np.zeros((len(xgroup_dir), 4))
xgroup_record_b = np.zeros((len(xgroup_dir), 4))

for cnt, fold in enumerate(incity_dir):
    result_file = os.path.join(fold, 'spca', 'result.txt')
    with open(result_file, 'r') as f:
        lines = f.readlines()
    # incity_record[cnt] = float(lines[-1])
    for line in lines[:-1]:
        A, B = line.split('(')[1].strip().strip(')').split(',')
        incity_record_a[cnt][city_dict[line[:3]]] += float(A)
        incity_record_b[cnt][city_dict[line[:3]]] += float(B)
    incity_record_a[cnt][-1] = np.sum(incity_record_a[cnt][:-1])
    incity_record_b[cnt][-1] = np.sum(incity_record_b[cnt][:-1])
incity_record = incity_record_a/incity_record_b

for cnt, fold in enumerate(xcity_dir):
    result_file = os.path.join(fold, 'spca', 'result.txt')
    with open(result_file, 'r') as f:
        lines = f.readlines()
    # xcity_record[cnt] = float(lines[-1])
    for line in lines[:-1]:
        A, B = line.split('(')[1].strip().strip(')').split(',')
        xcity_record_a[cnt][city_dict[line[:3]]] += float(A)
        xcity_record_b[cnt][city_dict[line[:3]]] += float(B)
    xcity_record_a[cnt][-1] = np.sum(xcity_record_a[cnt][:-1])
    xcity_record_b[cnt][-1] = np.sum(xcity_record_b[cnt][:-1])
xcity_record = xcity_record_a / xcity_record_b

for cnt, fold in enumerate(xgroup_dir):
    result_file = os.path.join(fold, 'spca', 'result.txt')
    with open(result_file, 'r') as f:
        lines = f.readlines()
    # xgroup_record[cnt] = float(lines[-1])
    for line in lines[:-1]:
        A, B = line.split('(')[1].strip().strip(')').split(',')
        xgroup_record_a[cnt][city_dict[line[:3]]] += float(A)
        xgroup_record_b[cnt][city_dict[line[:3]]] += float(B)
    xgroup_record_a[cnt][-1] = np.sum(xgroup_record_a[cnt][:-1])
    xgroup_record_b[cnt][-1] = np.sum(xgroup_record_b[cnt][:-1])
xgroup_record = xgroup_record_a / xgroup_record_b
print(xgroup_record)

'''city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna', 'Avg']
position = np.arange(6)
width = 0.2
bx1 = plt.boxplot(incity_record, positions=position, widths=width)
plt.boxplot(xcity_record, positions=position+width, widths=width)
plt.boxplot(xgroup_record, positions=position+2*width, widths=width)
plt.grid('on')
plt.xticks(np.arange(6)+width, city_list)
plt.show()'''

plt.figure(figsize=(8, 4))
plt.boxplot([incity_record[:, -1], xcity_record[:, -1], xgroup_record[:, -1]])
plt.xticks(np.arange(3)+1, ['InCity', 'XCity', 'XGroup'])
plt.xlabel('Train Type')
plt.ylabel('IoU')
plt.grid(linestyle='dotted')
plt.title('Overall Performance Comparison')
plt.tight_layout()
#plt.savefig(os.path.join(img_dir, 'overall_performance_cmp.png'))
#plt.show()

plt.figure(figsize=(8, 5))
ind = np.arange(4)
width = 0.2
plt.bar(ind, np.mean(incity_record, axis=0), width=width, label='InCity')
plt.bar(ind+width, np.mean(xcity_record, axis=0), width=width, label='XCity')
plt.bar(ind+width*2, np.mean(xgroup_record, axis=0), width=width, label='XGroup')
plt.xticks(ind+width, ['Fresno', 'Modesto', 'Stockton', 'Avg'])
plt.legend(loc='upper left')
plt.xlabel('City')
plt.ylabel('IoU')
#plt.ylim(0.65, 0.85)
plt.title('Citywise Performance Comparison')
plt.tight_layout()
#plt.savefig(os.path.join(img_dir, 'citywise_performance_cmp.png'))
plt.show()
