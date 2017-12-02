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
    data_dir = r'/media/ei-edl01/data/remote_sensing_data/urban_mapper/truthWithBigAndSmallClass'
    files = glob(os.path.join(data_dir, '*_GTC.tif'))

    for file in files:
        img = scipy.misc.imread(file)

        plt.imshow(img)
        plt.colorbar()
        plt.show()
