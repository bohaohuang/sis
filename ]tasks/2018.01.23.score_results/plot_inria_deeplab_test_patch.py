import os
import sis_utils
import numpy as np
import matplotlib.pyplot as plt


img_dir, task_dir = sis_utils.get_task_img_folder()
save_dir = os.path.join(task_dir, 'train_patch')
runs = [0]
sizes = [321, 520, 736, 832, 1088, 1344, 1600, 1856, 2096, 2640]
iou_record = np.zeros((len(runs), len(sizes)))
duration_record = np.zeros((len(runs), len(sizes)))

for cnt_run, run_repeat in enumerate(runs):
    for cnt_size, size in enumerate(sizes):
        file_name = 'TDeeplabV3_res101_inria_aug_grid_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32_{}_{}.npy'.format(
            size, run_repeat
        )
        iou, duration = np.load(os.path.join(save_dir, file_name))

        A = 0
        B = 0
        for item in iou:
            A += iou[item][0]
            B += iou[item][1]

        iou_record[cnt_run][cnt_size] = A/B*100
        duration_record[cnt_run][cnt_size] = duration

np.save(os.path.join(save_dir, 'inria_deeplab_test_patch.npy'), [sizes, iou_record, duration_record])

plt.rcParams.update({'font.size': 14})
plt.rc('grid', linestyle='--')
plt.subplot(211)
plt.plot(sizes, iou_record[0]-iou_record[0][0], '-o')
plt.grid()
plt.ylabel('delta IoU')

plt.subplot(212)
plt.plot(sizes, duration_record[0], '-o')
plt.grid()
plt.xlabel('Input Size')
plt.ylabel('Time:s')

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'test_patch_frrn.png'))
plt.show()
