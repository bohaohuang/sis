import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sis_utils
import ersa_utils
from run_tsne import run_tsne


def plot_tsne(feature_encode, mse, p, show_id=False):
    plt.figure(figsize=(8, 6))
    plt.set_cmap('Reds')
    plt.scatter(feature_encode[:, 0], feature_encode[:, 1], c=mse/max(mse), edgecolors='none')
    if show_id:
        len_ = feature_encode.shape[0]
        for i in range(len_):
            plt.text(feature_encode[i, 0], feature_encode[i, 1], str(i))

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('TSNE Projection Result (Perplex={})'.format(p))
    plt.colorbar()
    plt.tight_layout()


img_dir, task_dir = sis_utils.get_task_img_folder()
pred_file_name = os.path.join(task_dir, 'test_pred_20190218_182224.csv')
truth_file_name = os.path.join(task_dir, 'test_truth.csv')

pred = pd.read_csv(pred_file_name, sep=' ').values
truth = pd.read_csv(truth_file_name, sep=' ').values
mse = np.mean(np.square(pred - truth), axis=1)

perplex = 40
file_name = os.path.join(task_dir, 'dlm_p{}_2.npy'.format(perplex))
feature_encode = run_tsne(truth, file_name, perplex=perplex, force_run=False)
plot_tsne(feature_encode, mse, perplex, True)
# plt.savefig(os.path.join(img_dir, 'dlm_tsne_p{}_2.png'.format(perplex)))
plt.show()
