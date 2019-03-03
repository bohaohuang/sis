import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import sis_utils

img_dir, task_dir = sis_utils.get_task_img_folder()
img_save_dir = os.path.join(img_dir, 'dist_error')

error_cnt = []
input_sizes = [572, 828, 1084, 1340, 1596, 1852, 2092, 2332, 2636]
cnt = 0
error_map_ref = None

for size in tqdm(input_sizes):
    error_map = np.zeros((5000, 5000))
    for city in ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']:
        for tile_num in range(5):
            error_img_name = '{}_{}{}.png'.format(size, city, tile_num+1)
            error_img = imageio.imread(os.path.join(img_save_dir, error_img_name))
            error_map += error_img
    error_cnt.append(np.sum(error_map))

    '''if size == 572:
        error_map_ref = error_map
    else:
        cnt += 1
        #plt.subplot(240+cnt)
        plt.imshow(error_map, cmap=plt.cm.get_cmap('Greys'), vmin=0)
        #plt.imshow(error_map_ref-error_map, cmap=plt.cm.get_cmap('bwr'), vmin=-1300, vmax=1300)
        #plt.plot(np.sum(error_map_ref-error_map, axis=0))
        #plt.ylim(-100000, 100000)
        #plt.axis('off')
        plt.title(size)
        #plt.colorbar()
        #plt.tight_layout()
        plt.show()'''

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(5000)
    Y = np.arange(5000)
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, error_map, cmap=plt.cm.get_cmap('coolwarm'),
                           linewidth=0)
    fig.colorbar(surf, shrink=0.5)
    plt.show()

#plt.plot(input_sizes, error_cnt)
'''utils.set_full_screen_img()
plt.tight_layout()
plt.show()'''
