import os
import imageio
import scipy.misc
import matplotlib.pyplot as plt
import utils

img_dir, task_dir = utils.get_task_img_folder()

tile_id = '945785'
data_path = r'/media/ei-edl01/user/as667/CT_9tile_trees'
large_tile = os.path.join(data_path, '9tile_{}_sw.jpg'.format(tile_id))
result = os.path.join(data_path, '{}_sw_scrnshot.JPG'.format(tile_id))

lt = imageio.imread(large_tile)
rs = imageio.imread(result)
rs = scipy.misc.imresize(rs, [7623, 7623])

plt.figure(figsize=(16, 8))
ax1 = plt.subplot(121)
plt.imshow(lt)
plt.axhline(2541, 0, 7623, color='r', linestyle='--')
plt.axhline(5082, 0, 7623, color='r', linestyle='--')
plt.axvline(2541, 0, 7623, color='r', linestyle='--')
plt.axvline(5082, 0, 7623, color='r', linestyle='--')
plt.axis('off')
ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
plt.imshow(rs)
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, '{}_cmp.png'.format(tile_id)))
plt.show()
