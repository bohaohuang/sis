import os
import cv2
import imageio
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

result_dir = r'/media/ei-edl01/user/jmm123/gbdx_pred/'
orig_img_dir = r'/media/ei-edl01/user/as667'
task_list = ['104001001099F800', '1040010021B61200', '1040010033CCDF00']
img_id = '000795_ne'
detctor = 'building'
gamma = 2.5
invGamma = 1.0 / gamma
table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')

plt.figure(figsize=(12, 10))
for cnt, task in enumerate(task_list):
    task_id = '{}_rechunked'.format(task)
    year = cnt + 2015

    orig_img_name = os.path.join(orig_img_dir, task_id, '{}.tif'.format(img_id))
    pred_img_name_unet = os.path.join(result_dir, str(year), detctor, 'UNET', 'sp_{}.tif'.format(img_id))
    pred_img_name_deeplab = os.path.join(result_dir, str(year), detctor, 'DEEPLAB', 'sp_{}.tif'.format(img_id))

    orig = imageio.imread(orig_img_name)
    orig = scipy.misc.imresize(orig, [2541, 2541])
    orig = cv2.LUT(orig, table)
    pred_unet = imageio.imread(pred_img_name_unet)
    pred_deeplab = imageio.imread(pred_img_name_deeplab)

    ax1 = plt.subplot(331+cnt*3)
    plt.imshow(orig)
    plt.title(year)
    plt.axis('off')
    ax2 = plt.subplot(331+cnt*3+1, sharex=ax1, sharey=ax1)
    plt.imshow(pred_unet)
    plt.title('Unet')
    plt.axis('off')
    ax3 = plt.subplot(331+cnt*3+2, sharex=ax1, sharey=ax1)
    plt.imshow(pred_deeplab)
    plt.title('DEEPLAB')
    plt.axis('off')

plt.tight_layout()
plt.show()
