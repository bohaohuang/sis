import os
import imageio
import numpy as np
import matplotlib.pyplot as plt

img_ids = [36342, 129581, 120370, 107843, 120530, 59984]
input_size = 321
patchDir = r'/hdd/uab_datasets/Results/PatchExtr/spca/chipExtrReg_cSz321x321_pad0'

file_name = os.path.join(patchDir, 'fileList.txt')
with open(file_name, 'r') as f:
    files = f.readlines()

plt.figure(figsize=(6, 8))
for plt_cnt, iid in enumerate(img_ids):
    patch_name = files[iid].split('.')[0][:-5]
    img = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    for cnt, file in enumerate(files[iid].strip().split(' ')[1:]):
        img[:, :, cnt] = imageio.imread(os.path.join(patchDir, file))
    plt.subplot(321+plt_cnt)
    plt.imshow(img)
    plt.title(patch_name)
    plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(r'/media/ei-edl01/user/bh163/figs/2018.03.02.res_gan/view_spca_deeplab', '{}.png'.
            format('_'.join([str(a) for a in img_ids]))))
plt.show()
