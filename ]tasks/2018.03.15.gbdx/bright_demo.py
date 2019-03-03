import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import sis_utils

rc('text', usetex=True)
img_dir, task_dir = sis_utils.get_task_img_folder()

intensity = np.arange(1, 256)
gamma_range = [0.1, 0.5, 2, 3, 4, 5]

for gamma in gamma_range:
    intensity_new = (intensity/255) ** (1/gamma) * 255
    plt.plot(intensity, intensity_new, label='$\gamma$={}'.format(gamma))
intensity_new = (intensity / 255) ** (1 / 1) * 255
plt.plot(intensity, intensity_new, '--k', label='$\gamma$={}'.format(1))
plt.legend()
plt.xlabel('Origin Pixel')
plt.ylabel('New Pixel')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'bright_demo.png'))
plt.show()
