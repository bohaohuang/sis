import os
import numpy as np
import matplotlib.pyplot as plt
import uabRepoPaths


def get_single_iou(line):
    a = int(line.split(',')[0].split('(')[1])
    b = int(line.split(',')[1].split(')')[0])
    return a, b


def get_avg_iou(lines):
    a = 0
    b = 0
    for line in lines:
        a_temp, b_temp = get_single_iou(line)
        a += a_temp
        b += b_temp
    return a/b


def get_ious(results):
    overall_iou = np.zeros(6)
    for i in range(5):
        iou = get_avg_iou(results[i*5:(i+1)*5])
        overall_iou[i] = iou
    overall_iou[-1] = float(results[-1])
    return overall_iou


model_dir = os.path.join(uabRepoPaths.evalPath, 'LOO_PORTION')
portion = [5, 10, 15, 20, 25, 30, 35]
x_ticks = 36 - np.array(portion)
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']

for city_id in [0]:
    target_city_iou = np.zeros(len(portion))
    for p_cnt, p in enumerate(portion):
        folder_name = r'UnetCrop_inria_loo_portion_{}_{}_0_PS(572, 572)_BS5_' \
                      r'EP100_LR0.0001_DS60_DR0.1_SFN32'.format(city_id, p)
        result_file = os.path.join(model_dir, '{}'.format(p), folder_name, 'inria', 'result.txt')
        with open(result_file, 'r') as f:
            results = f.readlines()
        ious = get_ious(results)
        target_city_iou[p_cnt] = ious[city_id]

    plt.figure(figsize=(8, 4))
    plt.plot(x_ticks, target_city_iou, '-o')
    plt.xlabel('#Tiles Trained')
    plt.ylabel('IoU')
    plt.xticks(x_ticks, x_ticks)
    plt.grid()
    plt.tight_layout()
    plt.title(city_list[city_id].capitalize())
    plt.show()
