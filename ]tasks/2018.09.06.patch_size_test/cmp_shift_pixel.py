import os
import numpy as np
import matplotlib.pyplot as plt
import sis_utils
import ersa_utils
from visualize import visualize_utils


img_dir, task_dir = sis_utils.get_task_img_folder()
pred_dir = os.path.join(task_dir, 'unet_patch_test_5')

for step_size in range(1, 8):


    for i in range(16):
        ref_dir_1 = os.path.join(pred_dir, 'slide_step_{}'.format(i), 'pred')
        ref_dir_2 = os.path.join(pred_dir, 'slide_step_{}'.format(i + 16), 'pred')

        fig_name = 'austin1.png'
        img_1 = ersa_utils.load_file(os.path.join(ref_dir_1, fig_name))
        img_2 = ersa_utils.load_file(os.path.join(ref_dir_2, fig_name))

        print(img_1.shape, img_2.shape)

        visualize_utils.compare_two_figure(img_1, img_1-img_2)
        #print(np.sum(img_1-img_2))

        diff = np.sum(np.abs(img_1-img_2), axis=0)
        plt.plot(diff)
        plt.show()
