import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import sis_utils

img_dir, task_dir = sis_utils.get_task_img_folder()
npy_file_name = os.path.join(task_dir, 'encoded_res50_inria_spca.npy')
feature_encode, len1, len2 = np.load(npy_file_name)

img_ids = [41847, 18248, 13941, 114425, 98837, 78954]
input_size = 321

patchDir_inria = r'/hdd/uab_datasets/Results/PatchExtr/inria/chipExtrReg_cSz321x321_pad0'
file_name = os.path.join(patchDir_inria, 'fileList.txt')
with open(file_name, 'r') as f:
    files_inria = f.readlines()
patchDir_spca = r'/hdd/uab_datasets/Results/PatchExtr/spca/chipExtrReg_cSz321x321_pad0'
file_name = os.path.join(patchDir_spca, 'fileList.txt')
with open(file_name, 'r') as f:
    files_spca = f.readlines()

plt.figure(figsize=(6, 8))
for plt_cnt, iid in enumerate(img_ids):
    if iid < len1:
        # inria
        patch_name = files_inria[iid].split('.')[0][:-5]
        img = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        for cnt, file in enumerate(files_inria[iid].strip().split(' ')[:3]):
            img[:, :, cnt] = imageio.imread(os.path.join(patchDir_inria, file))
    else:
        # spca
        patch_name = files_spca[iid-len1].split('.')[0][:-5]
        img = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        for cnt, file in enumerate(files_spca[iid-len1].strip().split(' ')[1:]):
            img[:, :, cnt] = imageio.imread(os.path.join(patchDir_spca, file))
    plt.subplot(321 + plt_cnt)
    plt.imshow(img)
    plt.title(patch_name)
    plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(r'/media/ei-edl01/user/bh163/figs/2018.03.02.res_gan/view_inria_spca', '{}.png'.
            format('_'.join([str(a) for a in img_ids]))))
plt.show()
