import os
import numpy as np
import matplotlib.pyplot as plt
import utils


IMG_SAVE_DIR = r'/media/ei-edl01/user/bh163/figs'
with open('iou_vs_dist.txt', 'r') as f:
    lines = f.readlines()

patch_size = np.zeros(len(lines)).astype(np.int32)
ious = np.zeros((5, len(lines)))
for cnt, line in enumerate(lines):
    stats = line.strip('\n').split(' ')
    patch_size[cnt] = stats[0]
    ious[:, cnt] = np.array(stats[1:])

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
for i in range(5):
    plt.plot(patch_size, ious[i, :], '--', label='tile {}'.format(i+1))
plt.plot(patch_size, np.mean(ious, axis=0), 'r', linewidth=2, label='mean')
plt.xticks(patch_size[:-1])
ax.set_xticklabels(patch_size[:-1], rotation=45)
plt.legend(loc='lower right')
plt.title('IOU vs Patch Size')
save_dir = utils.make_task_img_folder(IMG_SAVE_DIR)
plt.savefig(os.path.join(save_dir, 'iou_vs_patch_size.png'))
plt.show()
