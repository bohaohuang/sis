import os
import matplotlib.pyplot as plt
import sis_utils
import ersa_utils

img_dir, task_dir = sis_utils.get_task_img_folder()
f2_name = os.path.join(task_dir, 'f2s.npy')
lr_name = os.path.join(task_dir, 'lrs.npy')
model_name = ['Raw Finetune 1e-3', 'Raw Scratch 1e-3', 'Hist Finetune 1e-3', 'Hard Sample']

f2 = ersa_utils.load_file(f2_name)
lr = ersa_utils.load_file(lr_name)
ious = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


for cnt, mn in enumerate(model_name):
    plt.plot(ious, f2[:, cnt], marker='o', label=mn)
plt.legend()
plt.grid()
plt.title('F2 Score VS IoU Threshold')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'f2_vs_iou_th.png'))
plt.show()

for cnt, mn in enumerate(model_name):
    plt.plot(ious, lr[:, cnt], marker='o', label=mn)
plt.legend()
plt.grid()
plt.title('Largest Recall Score VS IoU Threshold')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'lr_vs_iou_th.png'))
plt.show()
