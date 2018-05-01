import os
import cv2
import imageio
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt


def pre_processing(orig_img, gamma=2.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    orig_img = scipy.misc.imresize(orig_img, [2541, 2541])
    return cv2.LUT(orig_img, table)


def draw_contour(orig_img, post_img, truth_val=255):
    post_img = post_img / truth_val
    im2, contours, hierarchy = cv2.findContours(np.uint8(post_img), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(orig_img, contours, -1, (255, 0, 0), 5)
    return img


post_dir_1 = r'/media/ei-edl01/user/as667/BOHAO/gbdx_results_v1/1040010021B61200_rechunked_postproc'
orig_dir_1 = r'/media/ei-edl01/user/as667/1040010021B61200_rechunked'
post_dir_2 = r'/media/ei-edl01/user/as667/BOHAO/gbdx_results_v1/1040010033CCDF00_rechunked_postproc'
orig_dir_2 = r'/media/ei-edl01/user/as667/1040010033CCDF00_rechunked'
save_dir = r'/media/ei-edl01/user/bh163/figs/2018.03.15.gbdx/blog_figures'

img_id = '010825_se'
post_name_1 = os.path.join(post_dir_1, 'sp_{}.tif'.format(img_id))
orig_name_1 = os.path.join(orig_dir_1, '{}.tif'.format(img_id))
post_name_2 = os.path.join(post_dir_2, 'sp_{}.tif'.format(img_id))
orig_name_2 = os.path.join(orig_dir_2, '{}.tif'.format(img_id))

post_img_1 = imageio.imread(post_name_1)
orig_img_1 = imageio.imread(orig_name_1)
orig_img_1 = pre_processing(orig_img_1)
mask_img_1 = np.copy(orig_img_1)
cout_img_1 = draw_contour(mask_img_1, post_img_1)

print(post_name_2)
post_img_2 = scipy.misc.imread(post_name_2)
orig_img_2 = imageio.imread(orig_name_2)
orig_img_2 = pre_processing(orig_img_2)
mask_img_2 = np.copy(orig_img_2)
cout_img_2 = draw_contour(mask_img_2, post_img_2)

region = [600, 1600, 1000, 2000]

plt.figure(figsize=(15, 8))
ax1 = plt.subplot(121)
plt.imshow(cout_img_1[region[0]:region[1], region[2]:region[3], :])
#plt.axis('off')
ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
plt.imshow(cout_img_2[region[0]:region[1], region[2]:region[3], :])
#plt.axis('off')
plt.tight_layout()
plt.show()

#imageio.imsave(os.path.join(save_dir, 'header_1.png'), cout_img[region[0]:region[1], region[2]:region[3], :])
