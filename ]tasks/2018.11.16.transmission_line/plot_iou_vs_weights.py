import os
import numpy as np
import matplotlib.pyplot as plt
import utils
import ersa_utils

task_name = 'lines'

model_dir = r'/hdd/Results/{}'.format(task_name)
weight_range = [1, 5, 10, 30, 50, 100]
iou_list = np.zeros(len(weight_range))
img_dir, task_dir = utils.get_task_img_folder()

for cnt, weight in enumerate(weight_range):
    model_name = 'UnetCrop_{}_pw{}_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'.format(task_name, weight)
    result_file_name = os.path.join(model_dir, model_name, task_name, 'result.txt')
    result = ersa_utils.load_file(result_file_name)
    overall_iou = float(result[-1]) * 100
    iou_list[cnt] = overall_iou

plt.figure(figsize=(6, 4))
plt.plot(weight_range, iou_list, '-o')
plt.grid(True)
plt.xlabel('Weight')
plt.ylabel('Overall IoU')
plt.title('IoU VS Weight')
plt.tight_layout()
#plt.savefig(os.path.join(img_dir, 'iou_vs_weight_{}.png'.format(task_name)))
plt.show()
