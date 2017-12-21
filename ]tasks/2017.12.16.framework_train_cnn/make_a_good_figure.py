import os
import imageio
import numpy as np
import matplotlib.pyplot as plt

pred_dir = r'/home/lab/Documents/bohao/tasks/2017.12.11.inria_unet_test/fuse_2_1+-1'
image_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles'
gt_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles'

img = imageio.imread(os.path.join(image_dir, 'chicago2_RGB.tif'))
gt = imageio.imread(os.path.join(gt_dir, 'chicago2_GT.tif'))
pred = imageio.imread(os.path.join(pred_dir, 'chicago_2.png'))

gt2show = np.copy(img)
gtmask = np.where(gt == 255)
padind = np.ones((gtmask[0].shape[0],), dtype=np.uint8)
gtmask = (gtmask[0], gtmask[1], padind)
gt2show[gtmask] = 255
pred2show = np.copy(img)
predmask = np.where(pred == 255)
padind = np.zeros((predmask[0].shape[0],), dtype=np.uint8)
predmask = (predmask[0], predmask[1], padind)
pred2show[predmask] = 255

plt.figure(figsize=(12, 6))
ax1 = plt.subplot(121)
ax1.imshow(gt2show)
plt.axis('off')
ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
ax2.imshow(pred2show)
plt.axis('off')
plt.show()
