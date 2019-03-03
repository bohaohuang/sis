import os
import numpy as np
import matplotlib.pyplot as plt
import sis_utils

img_dir, task_dir = sis_utils.get_task_img_folder()
data = np.load(os.path.join(task_dir, 'iou_records.npy'))
large_mean = np.mean(data[0])
small_mean = np.mean(data[1])

bp = plt.boxplot(np.transpose(data))
plt.xticks(np.arange(2)+1, ['large:{:.3f}'.format(large_mean), 'small{:.3f}'.format(small_mean)])
plt.ylim(65, 90)
plt.show()
