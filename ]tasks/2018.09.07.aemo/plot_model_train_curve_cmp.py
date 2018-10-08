import os
import matplotlib.pyplot as plt
import utils
from ersa_utils import read_tensorboard_csv


img_dir, task_dir = utils.get_task_img_folder()
file_name_temp = 'run_unet_aemo_PS(572, 572)_BS5_EP60_LR0.001_DS{}_DR0.1-tag-{}.csv'

plt.figure(figsize=(10, 6))

# IoU
ax1 = plt.subplot(211)
for ds in [20, 40]:
    file_name = file_name_temp.format(ds, 'IoU')
    step, value = read_tensorboard_csv(os.path.join(task_dir, file_name), 5, 2)

    plt.plot(step, value, label='decay step={}'.format(ds), linewidth=2)
plt.ylabel('IoU')
plt.grid(True)
plt.title('Training Comparison')

# learning rate
ax2 = plt.subplot(212, sharex=ax1)
for ds in [20, 40]:
    file_name = file_name_temp.format(ds, 'learning_rate_1')
    step, value = read_tensorboard_csv(os.path.join(task_dir, file_name), 3, 0)

    plt.plot(step, value, label='decay step={}'.format(ds), linewidth=2)
plt.legend()
plt.xlabel('Step Number')
plt.ylabel('Learning Rate')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'train_crve_cmp.png'))
plt.show()
