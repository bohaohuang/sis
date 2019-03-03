import os
import numpy as np
import matplotlib.pyplot as plt
import sis_utils
import ersa_utils

img_dir, task_dir = sis_utils.get_task_img_folder()

model_name = ['fold_0', 'fold_1', 'fold_2']

area, ious = [], []

for cnt, mn in enumerate(model_name):
    save_name = os.path.join(task_dir, '{}_area_vs_iou_stats.npy'.format(mn))
    stats = ersa_utils.load_file(save_name)

    area.append(stats[:, 0])
    ious.append(stats[:, 1])

    H, x_edges, y_edges = np.histogram2d(stats[:, 0], stats[:, 1], bins=(25, 50), range=[[0, 1000], [0, 1]])
    H = H.T

    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, H, cmap='Blues', vmin=0, vmax=40)
    plt.colorbar()
    plt.xlabel('Panel Size')
    plt.ylabel('IoU')
    plt.title('Region {}'.format(cnt))
    plt.tight_layout()
    #plt.savefig(os.path.join(img_dir, '{}_area_vs_iou_stats.png'.format(mn)))
    plt.close()

    new_H = np.zeros(H.shape[1])
    panel_num = np.zeros(H.shape[1])
    r, c = H.shape
    for i in range(c):
        if np.sum(H[:, i]) != 0:
            new_H[i] = np.sum(H[:, i] * y_edges[:-1]) / np.sum(H[:, i])
            panel_num[i] = np.sum(H[:, i])

    plt.figure(figsize=(8, 8))
    ax1 = plt.subplot(211)
    plt.bar(x_edges[:-1], new_H, width=40, align='edge', edgecolor='k')
    plt.xticks(x_edges[:-1], x_edges[:-1] * 0.09, fontsize=6)
    plt.xlabel('Panel Size ($m^2$)')
    plt.ylabel('mean IoU')
    plt.title('Region {}'.format(cnt))

    plt.subplot(212, sharex=ax1)
    plt.bar(x_edges[:-1], panel_num, width=40, align='edge', edgecolor='k')
    plt.xticks(x_edges[:-1], x_edges[:-1] * 0.09, fontsize=6)
    plt.xlabel('Panel Size ($m^2$)')
    plt.ylabel('#Panels')
    for i in range(len(panel_num)):
        plt.text(x_edges[i]+5, panel_num[i]+2, str(int(panel_num[i])), fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '{}_area_vs_iou_stats_mean2.png'.format(mn)))
    plt.show()

    '''a = stats[:, 0]
    i = stats[:, 1]
    plt.hist(a[np.where(i < 0.2)], bins=100, range=(0, 1000))
    plt.xlabel('Panel Size')
    plt.ylabel('Counts')
    plt.title('Missed Panel Size Distribution Region {}'.format(cnt))
    plt.ylim([0, 40])
    plt.tight_layout()
    #plt.savefig(os.path.join(img_dir, '{}_missed_panel_dist.png'.format(mn)))
    plt.close()'''

area = np.concatenate(area)
ious = np.concatenate(ious)
H, x_edges, y_edges = np.histogram2d(area, ious, bins=(25, 50), range=[[0, 1000], [0, 1]])
H = H.T

X, Y = np.meshgrid(x_edges, y_edges)
plt.pcolormesh(X, Y, H, cmap='Blues', vmin=0, vmax=80)
plt.colorbar()
plt.xlabel('Panel Size')
plt.ylabel('IoU')
plt.title('Aggregate')
plt.tight_layout()
#plt.savefig(os.path.join(img_dir, 'agg_area_vs_iou_stats.png'))
plt.close()

new_H = np.zeros(H.shape[1])
panel_num = np.zeros(H.shape[1])
_, c = H.shape
for i in range(c):
    if np.sum(H[:, i]) != 0:
        new_H[i] = np.sum(H[:, i] * y_edges[:-1]) / np.sum(H[:, i])
        panel_num[i] = np.sum(H[:, i])

plt.figure(figsize=(8, 8))
ax1 = plt.subplot(211)
plt.bar(x_edges[:-1], new_H, width=40, align='edge', edgecolor='k')
plt.xticks(x_edges[:-1], x_edges[:-1] * 0.09, fontsize=6)
plt.xlabel('Panel Size ($m^2$)')
plt.ylabel('mean IoU')
plt.title('Aggregate')

plt.subplot(212, sharex=ax1)
plt.bar(x_edges[:-1], panel_num, width=40, align='edge', edgecolor='k')
plt.xticks(x_edges[:-1], x_edges[:-1] * 0.09, fontsize=6)
plt.xlabel('Panel Size ($m^2$)')
plt.ylabel('#Panels')
for i in range(len(panel_num)):
    plt.text(x_edges[i] + 5, panel_num[i] + 2, str(int(panel_num[i])), fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'agg_area_vs_iou_stats_mean2.png'))
plt.show()

'''plt.hist(area[np.where(ious< 0.2)], bins=100, range=(0, 1000))
plt.xlabel('Panel Size')
plt.ylabel('Counts')
plt.title('Missed Panel Size Distribution Aggregate')
plt.ylim([0, 40])
plt.tight_layout()
#plt.savefig(os.path.join(img_dir, 'agg_missed_panel_dist.png'))
plt.close()

min_size_array = 1 * np.ones(100)
max_size_array = np.zeros(100)
for a, i in zip(area, ious):
    cnt = int(a // 10)
    cnt = min([cnt, 99])
    if i < min_size_array[cnt]:
        min_size_array[cnt] = i
    if i > max_size_array[cnt]:
        max_size_array[cnt] = i

plt.figure(figsize=(8, 6))
plt.subplot(211)
plt.bar(np.arange(100), min_size_array)
plt.subplot(212)
plt.bar(np.arange(100), max_size_array)
plt.show()'''
