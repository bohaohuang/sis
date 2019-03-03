import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sis_utils

img_dir, task_dir = sis_utils.get_task_img_folder()
pred_file_name = os.path.join(task_dir, 'predOut_20190211_170250.csv')
truth_file_name = os.path.join(task_dir, 'truthOut_20190211_170250.csv')

pred = pd.read_csv(pred_file_name).values
truth = pd.read_csv(truth_file_name).values
mse = np.mean(np.square(pred - truth), axis=1)

ids = [2245, 2829, 2035, 2131, 2133, 2266]
p = 40

plt.figure(figsize=(10, 8))
for i in range(6):
    plt.subplot(3, 2, i+1)
    plt.plot(truth[ids[i], :], label='truth')
    plt.plot(pred[ids[i], :], label='pred')
    plt.text(100, 0.5, 'id={}\nMSE={:.2e}'.format(ids[i], mse[ids[i]]))
    if i == 0:
        plt.legend()
    plt.xticks([], [])
    plt.yticks([], [])
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'specific_tsne_p{}_2.png'.format(p)))
plt.show()

'''sort_idx = np.argsort(mse)[:50]
sort_idx = np.random.permutation(sort_idx)[:12]
plt.figure(figsize=(10, 6))
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.plot(truth[sort_idx[i], :], label='truth')
    plt.plot(pred[sort_idx[i], :], label='pred')
    plt.text(100, 0.5, 'id={}\nMSE={:.2e}'.format(sort_idx[i], mse[sort_idx[i]]))
    if i == 0:
        plt.legend()
    plt.xticks([], [])
    plt.yticks([], [])
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'best_tsne.png'))
plt.show()'''
