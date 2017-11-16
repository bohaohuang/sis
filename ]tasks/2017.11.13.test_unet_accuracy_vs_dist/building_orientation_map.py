import os
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import utils
from dataReader import patch_extractor


IMG_SAVE_DIR = r'/media/ei-edl01/user/bh163/figs'
truth_image_dir = r'/media/ei-edl01/data/remote_sensing_data/inria/truth'
SIZE = 224

save_dir = utils.make_task_img_folder(IMG_SAVE_DIR)
for i in range(5):
    truth_sum = np.zeros((SIZE, SIZE))

    truth_image_name = 'austin{}.tif'.format(i+1)
    truth_image = scipy.misc.imread(os.path.join(truth_image_dir, truth_image_name))

    for label_patch in patch_extractor.patchify(
            np.expand_dims(truth_image, axis=2),
            (5000, 5000),
            (SIZE, SIZE)):
        label = label_patch[:, :, 0]
        truth_sum += label

    plt.figure()
    plt.imshow(truth_sum/np.max(truth_sum), cmap='Greys')
    plt.title('austin{} Size={}'.format(i+1, SIZE))
    plt.savefig(os.path.join(save_dir, 'austin{}_building_PS-{}.png'.format(i+1, SIZE)))
