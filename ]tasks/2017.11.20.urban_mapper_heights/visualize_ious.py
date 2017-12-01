from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import os
from glob import glob

grid_no_rescale = np.array([[0.72, 0.705, 0.701, 0.689, 0.667, 0.619, 0.617],
                            [0.715, 0.719, 0.703, 0.666, 0.651, 0.616, 0.556],
                            [0.731, 0.659, 0.603, 0.353, 0.659, 0.603, 0],
                            [0, 0, 0, 0, 0.523, 0.554, 0]])

'''fig = plt.figure()
ax =fig.add_subplot(111, projection='3d')
layers, items = grid_no_rescale.shape
for i in range(layers):
    x = np.arrange()
    ax.bar()'''

if __name__ == '__main__':
    data_dir = r'/media/ei-edl01/data/remote_sensing_data/urban_mapper/truthWithEdgeClass'
    files = glob(os.path.join(data_dir, '*_GTC.tif'))

    img = scipy.misc.imread(r'/home/lab/Documents/bohao/data/urban_mapper/PS_(572, 572)-OL_0-AF_valid_augfr_um_npy_mult/JAX004_label_00000.png')
    plt.imshow(img)
    plt.colorbar()
    plt.show()

    for file in files:
        img = scipy.misc.imread(file)

        img = scipy.misc.toimage(img, high=np.max(img), low=np.min(img), mode='I')
        img.save('test.png')

        img_load = scipy.misc.imread('test.png')
        plt.imshow(img_load)
        plt.colorbar()
        plt.show()
