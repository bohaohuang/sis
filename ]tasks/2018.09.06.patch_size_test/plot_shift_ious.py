import os
import numpy as np
import matplotlib.pyplot as plt
import sis_utils


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
    return a/b, a, b


def get_ious(results):
    overall_iou = np.zeros(6)
    a_all = 0
    b_all = 0
    for i in range(5):
        iou, a, b = get_avg_iou(results[i*5:(i+1)*5])
        a_all += a
        b_all += b
        overall_iou[i] = iou
    overall_iou[-1] = float(a_all / b_all)
    return overall_iou


result_dir = r'/hdd/Results/Road/UnetCrop_road_0_PS(572, 572)_BS5_EP80_LR0.0001_DS60_DR0.1_SFN32/Mass_road'
with open(os.path.join(result_dir, 'result.txt'), 'r') as f:
    results = f.readlines()


img_dir, task_dir = sis_utils.get_task_img_folder()
slide = range(32)
ious = np.zeros((6, len(slide)))

for cnt, s in enumerate(slide):
    result_file = os.path.join(task_dir, 'unet_patch_test_7', 'slide_step_{}'.format(s), 'result.txt')
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
plt.savefig(os.path.join(img_dir, 'shift_steps_ious.png'))
plt.show()

plt.figure(figsize=(12, 6))
for i in range(6):
    plt.subplot(231+i)
    plt.plot(slide[:16], ious[i, :16]-ious[i, -16:], '-o')
    plt.title(city_list[i].capitalize())
    if i % 3 == 0:
        plt.ylabel('IoU')
    if i // 3 == 1:
        plt.xlabel('Slide Stride')
    plt.ylim(-0.001, 0.001)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'shift_steps_diff.png'))
plt.show()
