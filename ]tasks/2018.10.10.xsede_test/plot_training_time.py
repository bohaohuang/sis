import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sis_utils

n_train = 8000
epoch = 100
resolution = 0.3 * 1e-3
img_dir, task_dir = sis_utils.get_task_img_folder()

time = [17*60*60+19*60+20, 21*60*60+24*60+14, 24*60*60+45*60+53]
time_inf = [342, 388, 396]
model_list = ['U-Net', 'DeepLabV2', 'PSPNet']
time_avg = np.zeros(len(time))
time_inf_avg = np.zeros(len(time))
area_inf = 5*2.25*5

for cnt, model in enumerate(model_list):
    if model == 'U-Net':
        size = 388
    elif model == 'DeepLabV2':
        size = 321
    else:
        size = 384

    area = n_train * epoch * (size * resolution) ** 2
    time_avg[cnt] = time[cnt] / area
    time_inf_avg[cnt] = time_inf[cnt] / area_inf

X = np.arange(len(time_avg))
plt.figure(figsize=(8, 6))
matplotlib.rcParams.update({'font.size': 12})

ax1 = plt.subplot(211)
plt.bar(X, time_avg, width=0.35)
plt.ylabel('Time:s')
plt.ylim([0, 12])
plt.title('Run Time Comparison (Training)')
for cnt, i in enumerate(X):
    plt.text(i-0.07, time_avg[i]+0.1, '{:.2f}s'.format(time_avg[i]), fontsize=12)

ax2 = plt.subplot(212, sharex=ax1)
plt.bar(X, time_inf_avg, width=0.35)
plt.xlabel('Area:km^2')
plt.ylabel('Time:s')
plt.xticks(X, model_list)
plt.ylim([4, 8])
plt.title('Run Time Comparison (Inference)')
for cnt, i in enumerate(X):
    plt.text(i-0.07, time_inf_avg[i]+0.1, '{:.2f}s'.format(time_inf_avg[i]), fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'run_time_cmp_2.png'))
plt.show()
