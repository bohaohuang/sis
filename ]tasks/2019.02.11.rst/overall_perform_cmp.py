import os
import numpy as np
import matplotlib.pyplot as plt
import utils

img_dir, task_dir = utils.get_task_img_folder()

perform = np.array(
    [
        [78.04, 69.50, 67.17, 76.67, 79.66, 75.51],             # uab unet
        [81.35, 79.98, 75.00, 72.92, 72.07, 78.13],             # rst unet
        [79.47, 76.92, 73.41, 71.54, 70.78, 76.90],             # link
        [70.74, 68.34, 66.71, 65.24, 64.79, 69.99],             # psp
        [74.43, 71.77, 67.80, 65.01, 63.67, 71.60],             # fpn
    ]
)

N = 6
X = np.arange(N)
width = 0.18
names = ['UNet', 'UNet*Res50', 'Link*Res50', 'PSP*Res50', 'FPN*Res50']
xtick_names = ['Austin', 'Chicago', 'Kitsap', 'Tyrol-w', 'Vienna', 'Overall']

fig = plt.figure(figsize=(10, 5))
for i in range(5):
    plt.bar(X+i*width, perform[i, :], width=width, label=names[i])
plt.ylim([60, 85])
plt.xticks(X+width*2, xtick_names)
plt.xlabel('City Name')
plt.ylabel('IoU')
plt.legend(ncol=5)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'perf_cmp.png'))
plt.show()
