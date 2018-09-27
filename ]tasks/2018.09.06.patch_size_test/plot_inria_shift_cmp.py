import os
import numpy as np
import matplotlib.pyplot as plt
import utils
import ersa_utils


img_dir, task_dir = utils.get_task_img_folder()
step_range = 6
iou_record_shift = np.zeros(step_range-1)
x_ticks = []

for cnt, step_size in enumerate(range(1, step_range)):
    patch_select = list(range(0, 16, step_size))
    if len(patch_select) <= 4:
        x_ticks.append(','.join(str(ps) for ps in patch_select))
    else:
        s = ','.join(str(ps) for ps in patch_select[:3])
        s += ',...,{}'.format(patch_select[-1])
        x_ticks.append(s)

    iou_record = ersa_utils.load_file(os.path.join(task_dir, 'iou_record_step_{}.npy'.format(step_size)))
    iou_record = np.sum(iou_record, axis=0)
    iou_record = iou_record[0] / iou_record[1] * 100 - 0.4
    iou_record_shift[cnt] = iou_record

plt.figure(figsize=(8, 3.5))
plt.plot(iou_record_shift[::-1], '-o')
plt.grid('on')
plt.xticks(np.arange(step_range-1), x_ticks[::-1])
plt.xlabel('Shift Steps')
plt.ylabel('IoU')
plt.title('U-Net Average Shift Results on D1')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'shift_patch_result.png'))
plt.show()
