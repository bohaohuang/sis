import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import ersa_utils
import utils

img_dir, task_dir = utils.get_task_img_folder()

plt.figure(figsize=(6, 10))
for sample_id in range(1, 4):
    file_name = os.path.join(task_dir, 'sample_{}_dist.npy'.format(sample_id))

    if not os.path.exists(file_name):
        dist = np.zeros((3, 256))
        data_dir = r'/media/ei-edl01/data/aemo/samples/0584007740{}0_01'.format(sample_id)
        files = sorted(glob(os.path.join(data_dir, 'TILES', '*.tif')))

        for f in files:
            img = ersa_utils.load_file(f)
            for i in range(3):
                for pixel in img[:, :, i]:
                    dist[i, pixel] += 1
        np.save(file_name, dist)
    else:
        dist = np.load(file_name)

    labels = ['r', 'g', 'b']
    plt.subplot(310+sample_id)
    for i in range(3):
        plt.plot(dist[i, :], color=labels[i], label=labels[i])
    plt.legend()
    plt.title('Sample {}'.format(sample_id))

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'samples_channel_distribution.png'))
plt.show()
