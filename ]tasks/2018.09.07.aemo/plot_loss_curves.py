import os
import matplotlib.pyplot as plt
import sis_utils
from ersa_utils import read_tensorboard_csv


img_dir, task_dir = sis_utils.get_task_img_folder()
file_name_temp = 'run_unet_aemo_pad_PS(572, 572)_BS5_EP130_LR0.001_DS100_DR0.1-tag-{}_loss.csv'
run_type = ['train', 'valid']

plt.figure(figsize=(8, 6))

for cnt, run in enumerate(run_type):
    ax = plt.subplot(211 + cnt)
    file_name = file_name_temp.format(run)
    step, value = read_tensorboard_csv(os.path.join(task_dir, file_name), 5, 2)
    plt.plot(step, value, linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Xent')
    plt.grid(True)
    plt.title('{} Loss'.format(run.capitalize()))

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'train_crve_pad_hist.png'))
plt.show()
