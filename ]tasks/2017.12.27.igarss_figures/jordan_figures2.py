import os
import imageio
import numpy as np
import matplotlib.pyplot as plt

img_dir = r'/media/ei-edl01/user/bh163/figs'
save_dir = r'/media/ei-edl01/user/bh163/figs/2017.12.27.igarss_figures/jordan'
img = imageio.imread(os.path.join(img_dir, 'austin_1out_patch2.png'))

x = np.array([1669.5, 1669.5])
y = np.array([20, 946])
x2 = np.array([1287, 1836])
y2 = np.array([210.5, 210.5])

plt.imshow(img)
plt.plot(x, y, 'r--', linewidth=2)
plt.plot(x-224, y, 'r--', linewidth=2)
plt.plot(x2, y2, 'r--', linewidth=2)
plt.plot(x2, y2+224, 'r--', linewidth=2)
plt.plot(x2, y2+448, 'r--', linewidth=2)
plt.plot(x2, y2+672, 'r--', linewidth=2)
plt.box('off')
plt.xticks([], [])
plt.yticks([], [])
plt.show()
