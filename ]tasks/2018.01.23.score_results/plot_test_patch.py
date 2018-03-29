import os
import numpy as np
import matplotlib.pyplot as plt
import utils

from matplotlib import rc
rc('text', usetex=True)

img_dir, task_dir = utils.get_task_img_folder()
save_dir = os.path.join(task_dir, 'train_patch')

plt.rcParams.update({'font.size': 14})
plt.rc('grid', linestyle='--')

dataset = 'inria'
plt.subplot(211)
for cnt_m, model_name in enumerate(['unet', 'deeplab']):
    save_name = '{}_{}_test_patch.npy'.format(dataset, model_name)
    [sizes, iou_record, duration_record] = np.load(os.path.join(save_dir, save_name))
    if model_name == 'deeplab':
        plt.plot(sizes[1:], iou_record[0][1:]-iou_record[0][1], '-o', label=model_name)
    else:
        plt.plot(sizes, iou_record[0]-iou_record[0][0], '-o', label=model_name)
plt.title('Inria')
plt.grid()
plt.ylabel('$\Delta$IoU')
plt.legend()

dataset = 'spca'
plt.subplot(212)
for cnt_m, model_name in enumerate(['unet', 'deeplab']):
    save_name = '{}_{}_test_patch.npy'.format(dataset, model_name)
    [sizes, iou_record, duration_record] = np.load(os.path.join(save_dir, save_name))
    if model_name == 'deeplab':
        plt.plot(sizes[1:], iou_record[0][1:]-iou_record[0][1], '-o', label=model_name)
    else:
        plt.plot(sizes, iou_record[0]-iou_record[0][0], '-o', label=model_name)
plt.title('Solar Panel')
plt.grid()
plt.xlabel('Input Size')
plt.ylabel('$\Delta$IoU')

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'test_patch_iou_cmp2.png'))
plt.show()

dataset = 'inria'
plt.subplot(211)
for cnt_m, model_name in enumerate(['unet', 'deeplab']):
    save_name = '{}_{}_test_patch.npy'.format(dataset, model_name)
    [sizes, iou_record, duration_record] = np.load(os.path.join(save_dir, save_name))
    if model_name == 'deeplab':
        plt.plot(sizes[1:], duration_record[0][1:]-duration_record[0][1], '-o', label=model_name)
    else:
        plt.plot(sizes, duration_record[0]-duration_record[0][0], '-o', label=model_name)
plt.title('Inria')
plt.grid()
plt.ylabel('\%Time:s')
plt.legend()

dataset = 'spca'
plt.subplot(212)
for cnt_m, model_name in enumerate(['unet', 'deeplab']):
    save_name = '{}_{}_test_patch.npy'.format(dataset, model_name)
    [sizes, iou_record, duration_record] = np.load(os.path.join(save_dir, save_name))
    if model_name == 'deeplab':
        plt.plot(sizes[1:], duration_record[0][1:]-duration_record[0][1], '-o', label=model_name)
    else:
        plt.plot(sizes, duration_record[0]-duration_record[0][0], '-o', label=model_name)
plt.title('Solar Panel')
plt.grid()
plt.xlabel('Input Size')
plt.ylabel('\%Time:s')

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'test_patch_time_cmp2.png'))
plt.show()
