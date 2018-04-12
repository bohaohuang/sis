import os
import cv2
import imageio
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

result_dir = r'/media/ei-edl01/user/as667/BOHAO/gbdx_results_v1'
orig_img_dir = r'/media/ei-edl01/user/as667'
task_id = '104001001099F800_rechunked'
img_id = '000795_se'

gamma = 2.5
invGamma = 1.0 / gamma
table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')

orig_img_name = os.path.join(orig_img_dir, task_id, '{}.tif'.format(img_id))
pred_img_name = os.path.join(result_dir, task_id, 'sp_{}.tif'.format(img_id))

orig = imageio.imread(orig_img_name)
orig = scipy.misc.imresize(orig, [2541, 2541])
orig = cv2.LUT(orig, table)
pred = imageio.imread(pred_img_name)

plt.figure(figsize=(12, 6))
ax1 = plt.subplot(121)
plt.imshow(orig)
plt.axis('off')
ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
plt.imshow(pred)
plt.axis('off')
plt.tight_layout()
plt.show()
