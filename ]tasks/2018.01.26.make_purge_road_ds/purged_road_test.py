import os
import imageio
import numpy as np
import matplotlib.pyplot as plt


patchDir2 = r'/hdd/uab_datasets/Results/PatchExtr/inria/chipExtrRand0_cSz224x224_pad0'
files = os.path.join(patchDir2, 'fileList.txt')
with open(files, 'r') as f:
    file_list = f.readlines()

files = os.path.join(r'/media/lab/Michael(01)/chipExtrRegPurge_cSz572x572_pad184', 'state.txt')
with open(files, 'r') as f:
    text = f.readlines()
print(text)

'''for i in file_list[:5]:
    file_array = i.strip().split(' ')
    rgb = []
    for file in file_array[:3]:
        img = imageio.imread(os.path.join(patchDir2, file))
        rgb.append(img)
    rgb = np.dstack(rgb)
    gt = imageio.imread(os.path.join(patchDir2, file_array[-1]))

    plt.subplot(121)
    plt.imshow(rgb)
    plt.subplot(122)
    plt.imshow(gt)
    plt.show()'''
