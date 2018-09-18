import os
import numpy as np
import matplotlib.pyplot as plt
import utils


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


img_dir, task_dir = utils.get_task_img_folder()
slide = range(32)
ious = np.zeros((6, len(slide)))

for cnt, s in enumerate(slide):
    result_file = os.path.join(task_dir, 'unet_patch_test_4', 'slide_step_{}'.format(s), 'result.txt')
    with open(result_file, 'r') as f:
        results = f.readlines()
        ious[:, cnt] = get_ious(results)

city_list = ['austin', 'chicago', 'kitsap', 'tyrol', 'vienna', 'overall']
plt.figure(figsize=(12, 6))
for i in range(6):
    plt.subplot(231+i)
    plt.plot(slide, ious[i, :], '-o')
    ymin, ymax = plt.ylim()
    plt.vlines(15, ymin, ymax, colors='r', linestyles='--')
    plt.title(city_list[i].capitalize())
    if i % 3 == 0:
        plt.ylabel('IoU')
    if i // 3 == 1:
        plt.xlabel('Slide Stride')
plt.tight_layout()
#plt.savefig(os.path.join(img_dir, 'shift_variance_ious.png'))
plt.show()
