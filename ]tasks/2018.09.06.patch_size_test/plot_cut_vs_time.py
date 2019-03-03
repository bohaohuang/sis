import os
import numpy as np
import matplotlib.pyplot as plt
import sis_utils

img_dir, task_dir = sis_utils.get_task_img_folder()
save_file_name_dp = os.path.join(task_dir, 'deeplab_crop_records.npy')
dp_record = np.load(os.path.join(save_file_name_dp))
save_file_name_un = os.path.join(task_dir, 'unet_crop_records.npy')
un_record = np.load(os.path.join(save_file_name_un))

cut = np.arange(5, 36, 5)
plt.figure(figsize=(8, 6))
ax1 = plt.subplot(211)
plt.plot(cut, un_record[0, :]*100, '-o')
plt.plot(cut, dp_record[0, :]*100, '-o')
plt.ylabel('IoU')
plt.grid(True)
plt.title('IoU Comparison on D1')
ax2 = plt.subplot(212)
plt.plot(cut, un_record[1, :], '-o', label='U-Net')
plt.plot(cut, dp_record[1, :], '-o', label='DeepLab-CRF')
plt.grid(True)
plt.xlabel('#Extra Cutting Pixels')
plt.ylabel('Time:s')
plt.title('Run Time Comparison on D1')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'cut_vs_time.png'))
plt.show()
