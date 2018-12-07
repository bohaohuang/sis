import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
import ersa_utils

img_dir, task_dir = utils.get_task_img_folder()
img_name = 'demo_2.jpg'

raw = ersa_utils.load_file(os.path.join(img_dir, img_name))

blue_chan = raw[:, :, 2]


gt = (blue_chan > 150).astype(np.uint8) * 255

# remove small objects
nb_componets, output, stats, centroids = cv2.connectedComponentsWithStats(gt, connectivity=8)
sizes = stats[1:, -1]
nb_componets = nb_componets - 1
min_size = 100

img2 = np.zeros_like(gt)
for i in range(nb_componets):
    if sizes[i] >= min_size:
        img2[output == i + 1] = 255

# closing and opening
#img2 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=2)
img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, np.ones((2, 2), dtype=np.uint8), iterations=2)

plt.imshow(img2)
plt.show()

ersa_utils.save_file(os.path.join(img_dir, 'demo_2_binary_post.png'), img2)
