import os
import numpy as np
import matplotlib.pyplot as plt
import sis_utils

img_dir, task_dir = sis_utils.get_task_img_folder()
iou_cmp = [77.11158352, 70.14007803, 65.96846293, 79.0137734, 80.01516755, 75.8801136025218]
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

for run_id in run_list:
    city_iou_a = np.zeros(6)
    city_iou_b = np.zeros(6)

    city_str = '_'.join([city_list[a] for a in run_id])
    city_str_show = 'No ' + ' '.join([city_list[a] for a in range(5) if a not in run_id])
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

    width = 0.35
    X = np.arange(6)
    fig = plt.figure()
    plt.bar(X, city_iou, width=width, label=city_str_show)
    plt.bar(X+width, iou_cmp, width=width, label='baseline')
    plt.xticks(X, city_list+['Over All'])
    plt.xlabel('City')
    plt.ylabel('IoU')
    plt.legend(loc='upper right')
    plt.ylim([50, 85])
    plt.title('IoU Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '{}.png'.format(city_str)))
    plt.close(fig)
    #plt.show()
