import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import utils
import util_functions

orig_dir = r'/media/ei-edl01/user/as667/BOHAO/gbdx_results'
rgb_orig_dir = r'/media/ei-edl01/user/as667'
task_id = 'Honolulu_chunks'
tile_id = '104001000832C600_0'
img_name = os.path.join(rgb_orig_dir, task_id, '{}.tif'.format(tile_id))
bgt_name = os.path.join(orig_dir, task_id, 'building_{}.tif'.format(tile_id))
sgt_name = os.path.join(orig_dir, task_id, 'sp_{}.tif'.format(tile_id))
img_dir, task_dir = utils.get_task_img_folder()

img = imageio.imread(img_name)
bmask = np.copy(img)
smask = np.copy(img)
bgt = imageio.imread(bgt_name)
sgt = imageio.imread(sgt_name)
b_mask = util_functions.add_mask(bmask, bgt, [None, 255, None], mask_1=255)
s_mask = util_functions.add_mask(smask, sgt, [255, None, None], mask_1=255)

plt.figure(figsize=(12, 6))
ax1 = plt.subplot(121)
plt.imshow(b_mask)
plt.axis('off')
ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
plt.imshow(s_mask)
plt.axis('off')
plt.show()
