import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid

img_ids = [30566, 40325, 30022, 42195]
input_size = 321
patchDir = r'/hdd/uab_datasets/Results/PatchExtr/inria/chipExtrReg_cSz321x321_pad0'

file_name = os.path.join(patchDir, 'fileList.txt')
with open(file_name, 'r') as f:
    files = f.readlines()

fig = plt.figure(figsize=(5, 5))
grid = Grid(fig, rect=111, nrows_ncols=(2, 2), axes_pad=0.05, label_mode='L')
for plt_cnt, ax in enumerate(grid):
    iid = img_ids[plt_cnt]
    patch_name = files[iid].split('.')[0][:-5]
    img = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    for cnt, file in enumerate(files[iid].strip().split(' ')[:3]):
        img[:, :, cnt] = imageio.imread(os.path.join(patchDir, file))
    #plt.subplot(321+plt_cnt)
    ax.imshow(img)
    #plt.title(patch_name)
    ax.set_axis_off()
plt.tight_layout()
plt.savefig(os.path.join(r'/media/ei-edl01/user/bh163/figs/2018.03.02.res_gan', 'inria_deeplab_{}.png'.
            format('_'.join([str(a) for a in img_ids]))))
plt.show()
