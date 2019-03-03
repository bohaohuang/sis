import os
import numpy as np
import matplotlib.pyplot as plt
import sis_utils

from matplotlib import rc
rc('text', usetex=True)

img_dir, task_dir = sis_utils.get_task_img_folder()
save_dir = os.path.join(task_dir, 'train_patch')

plt.rcParams.update({'font.size': 14})
plt.rc('grid', linestyle='--')
model_name2show = ['U-Net', 'DeepLabV2']

dataset = 'inria'
plt.subplot(211)
for cnt_m, model_name in enumerate(['unet', 'deeplab']):
    save_name = '{}_{}_test_patch.npy'.format(dataset, model_name)
    [sizes, iou_record, duration_record] = np.load(os.path.join(save_dir, save_name))
    if model_name == 'deeplab':
        plt.plot(sizes[1:], iou_record[0][1:], '-o', label=model_name2show[cnt_m])
    else:
        plt.plot(sizes, iou_record[0], '-o', label=model_name2show[cnt_m])
plt.title('D1')
plt.grid()
plt.ylabel('IoU')
plt.legend(loc='upper left')

dataset = 'spca'
plt.subplot(212)
for cnt_m, model_name in enumerate(['unet', 'deeplab']):
    save_name = '{}_{}_test_patch.npy'.format(dataset, model_name)
    [sizes, iou_record, duration_record] = np.load(os.path.join(save_dir, save_name))
    if model_name == 'deeplab':
        plt.plot(sizes[1:], iou_record[0][1:], '-o', label=model_name2show[cnt_m])
    else:
        plt.plot(sizes, iou_record[0], '-o', label=model_name2show[cnt_m])
plt.title('D2')
plt.grid()
plt.xlabel('Input Size')
plt.ylabel('IoU')

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'test_patch_iou_cmp_2.png'))
plt.show()

dataset = 'inria'
plt.subplot(211)
for cnt_m, model_name in enumerate(['unet', 'deeplab']):
    save_name = '{}_{}_test_patch.npy'.format(dataset, model_name)
    [sizes, iou_record, duration_record] = np.load(os.path.join(save_dir, save_name))
    if model_name == 'deeplab':
        plt.plot(sizes[1:], duration_record[0][1:], '-o', label=model_name2show[cnt_m])
    else:
        plt.plot(sizes, duration_record[0], '-o', label=model_name2show[cnt_m])
plt.title('D1')
plt.grid()
plt.ylabel('Time:s')
plt.legend()

dataset = 'spca'
plt.subplot(212)
for cnt_m, model_name in enumerate(['unet', 'deeplab']):
    save_name = '{}_{}_test_patch.npy'.format(dataset, model_name)
    [sizes, iou_record, duration_record] = np.load(os.path.join(save_dir, save_name))
    if model_name == 'deeplab':
        plt.plot(sizes[1:], duration_record[0][1:], '-o', label=model_name2show[cnt_m])
    else:
        plt.plot(sizes, duration_record[0], '-o', label=model_name2show[cnt_m])
plt.title('D2')
plt.grid()
plt.xlabel('Input Size')
plt.ylabel('Time:s')

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'test_patch_time_cmp_2.png'))
plt.show()
