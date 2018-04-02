import os
import numpy as np
import matplotlib.pyplot as plt
import utils

img_dir, task_dir = utils.get_task_img_folder()
iou_cmp = np.array([77.11158352, 70.14007803, 65.96846293, 79.0137734, 80.01516755, 75.8801136025218])
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
run_list = [#[0, 1, 2, 3, 4],
            [0, 1, 2, 3],
            [0, 1, 2, 4],
            [0, 1, 3, 4],
            [0, 2, 3, 4],
            [1, 2, 3, 4],
            [2, 3, 4],
            [1, 3, 4],
            [1, 2, 4],
            [1, 2, 3],
            [0, 3, 4],
            [0, 2, 4],
            [0, 2, 3],
            [0, 1, 4],
            [0, 1, 3],
            [0, 1, 2]
            ]
ylabel_list = []
cmp_cm = np.zeros((len(run_list), 6))

for cnt_line, run_id in enumerate(run_list):
    city_iou_a = np.zeros(6)
    city_iou_b = np.zeros(6)

    city_str = '_'.join([city_list[a] for a in run_id])
    city_str_show = 'No ' + ' '.join([city_list[a] for a in range(5) if a not in run_id])
    ylabel_list.append(city_str_show)
    model_dir = r'/hdd/Results/control_city/DeeplabV3_inria_aug_train_{}_' \
                r'PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32/inria'.format(city_str)
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

    cmp_cm[cnt_line, :] = city_iou - iou_cmp

cm_min = np.min(cmp_cm)
cm_max = np.max(cmp_cm)
threshold = np.max([-cm_min, cm_max])
X = np.arange(6)
Y = np.arange(len(run_list))

plt.figure(figsize=(6.5, 8.5))
plt.imshow(cmp_cm, vmin=-threshold, vmax=threshold, cmap=plt.get_cmap('bwr'))
plt.xticks(X, city_list+['Over All'])
plt.yticks(Y, ylabel_list)
plt.colorbar()
for i in range(len(run_list)):
    plt.text(5-0.4, i, '{:.2f}'.format(cmp_cm[i, -1]))
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'city_cmp.png'))
plt.show()
