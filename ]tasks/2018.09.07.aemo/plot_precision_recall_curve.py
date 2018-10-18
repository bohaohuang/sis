import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import precision_recall_curve, roc_curve
import utils
import ersa_utils
from collection import collectionMaker

img_dir, task_dir = utils.get_task_img_folder()
model_name = 'unet_aemo_pad_PS(572, 572)_BS5_EP130_LR0.001_DS100_DR0.1'
conf_dir = os.path.join(task_dir, 'conf_map_{}'.format(model_name))
conf_files = sorted(glob(os.path.join(conf_dir, '*.npy')))

truth_dir = r'/home/lab/Documents/bohao/data/aemo/aemo_pad'
cm = collectionMaker.read_collection('aemo_pad')
truth_files = cm.load_files(field_name='aus50', field_id='', field_ext='.*gt_d255')
truth_files = [f[0] for f in truth_files[:2]]

conf_list = []
true_list = []

for conf, truth in zip(conf_files, truth_files):
    c = ersa_utils.load_file(conf)
    t = ersa_utils.load_file(truth)

    conf_list.append(c.flatten())
    true_list.append(t.flatten())

conf_list = np.concatenate(conf_list)
true_list = np.concatenate(true_list)

plt.figure(figsize=(12, 5))

plt.subplot(121)
precision, recall, _ = precision_recall_curve(true_list, conf_list)
plt.plot(recall, precision, linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('PR Curve')

plt.subplot(122)
fpr, tpr, thresholds = roc_curve(true_list, conf_list)
plt.plot(fpr, tpr, linewidth=2)
plt.xlabel('FPr')
plt.ylabel('TPr')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('ROC Curve')

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'pr_roc_curve.png'))
plt.show()
