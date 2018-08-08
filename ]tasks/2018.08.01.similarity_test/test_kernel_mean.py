import os
import numpy as np
import utils


img_dir, task_dir = utils.get_task_img_folder()
for target_city in range(5):
    save_file_name = os.path.join(task_dir, 'target_{}_weight.npy'.format(target_city))
    weight = np.load(save_file_name)

    print(weight.shape)
