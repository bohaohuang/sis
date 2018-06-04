import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
import util_functions

img_dir, task_dir = utils.get_task_img_folder()
plot_labels = ['Base', 'Scratch', 'Finetune']
fields = ['Step', 'Value']
model_type = 'unet'

if model_type == 'deeplab':
    files = ['run_DeeplabV3_inria_decay_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40.0_DR0.1_SFN32-tag-iou_validation.csv',
             'run_DeeplabV3_inria_austin_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32-tag-iou_validation.csv',
             'run_DeeplabV3_inria_austin_0_PS(321, 321)_BS5_EP40_LR1e-06_DS20_DR0.1_SFN32-tag-iou_validation.csv']
else:
    files = ['run_UnetCrop_inria_decay_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60.0_DR0.1_SFN32-tag-iou_validation.csv',
             'run_UnetCrop_inria_austin_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32-tag-iou_validation.csv',
             'run_UnetCrop_inria_austin_0_PS(572, 572)_BS5_EP40_LR1e-05_DS20_DR0.1_SFN32-tag-iou_validation.csv']

plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(8, 4))
for cnt, file in enumerate(files):
    df = pd.read_csv(os.path.join(task_dir, file), skipinitialspace=True, usecols=fields)
    value = util_functions.savitzky_golay(np.array(df['Value']), 11, 2)
    plt.plot(value, label=plot_labels[cnt])
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()
plt.grid('on')
plt.title('Validation IoU in Training')
plt.tight_layout()
if model_type == 'deeplab':
    plt.savefig(os.path.join(img_dir, 'deeplab_austin_train_curve.png'))
else:
    plt.savefig(os.path.join(img_dir, 'unet_austin_train_curve.png'))
plt.show()
