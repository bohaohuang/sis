import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils

img_dir, task_dir = utils.get_task_img_folder()
img_save_dir = os.path.join(img_dir, 'dist_error')

error_cnt = []
input_sizes = [572, 828, 1084, 1340, 1596, 1852, 2092, 2332, 2636]
cnt = 0
for size in tqdm(input_sizes):
    error_map = np.zeros((5000, 5000))
    for city in ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']:
        for tile_num in range(5):
            error_img_name = '{}_{}{}.png'.format(size, city, tile_num+1)
            error_img = imageio.imread(os.path.join(img_save_dir, error_img_name))
            error_map += error_img
    error_cnt.append(np.sum(error_map))

    cnt += 1
    plt.subplot(330+cnt)
    plt.imshow(error_map, cmap=plt.cm.get_cmap('Reds'))
    plt.axis('off')

#plt.plot(input_sizes, error_cnt)
plt.tight_layout()
plt.show()
