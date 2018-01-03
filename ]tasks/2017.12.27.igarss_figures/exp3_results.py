import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import utils


def exponential_smoothing(series, alpha=1):
    result = [series[0]] # first value is same as series
    for n in range(1, series.shape[0]):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return np.array(result)


img_dir, task_dir = utils.get_task_img_folder()
input_sizes = [508, 540, 572, 620, 684, 796]
city_dict = {'austin':0, 'chicago':1, 'kitsap':2, 'tyrol-w':3, 'vienna':4}
res_dir = r'/media/ei-edl01/user/bh163/tasks/2017.12.16.framework_train_cnn'
iou_record_city = np.zeros((5, 5, len(input_sizes)))
iou_record_all = np.zeros(len(input_sizes))

for cnt_1, input_size in enumerate(input_sizes):
    file_name = '{}_exp2.npy'.format(input_size)
    data = dict(np.load(os.path.join(res_dir, file_name)).tolist())

    iou = []
    for item in data.keys():
        if item != 'kitsap4' and item != 'time':
            iou.append(data[item])
            iou_record_city[city_dict[item[:-1]]][int(item[-1])-1][cnt_1] = data[item]
    iou_mean = np.mean(iou)
    iou_record_all[cnt_1] = iou_mean


plt.figure(figsize=(12, 6))
matplotlib.rcParams.update({'font.size': 14})
for cnt_1, city_name in enumerate(city_dict.keys()):
    plt.subplot(321+cnt_1)
    if city_name == 'kitsap':
        plt.boxplot(iou_record_city[cnt_1, [0, 1, 2, 4], :])
    else:
        plt.boxplot(iou_record_city[cnt_1, :, :])
    if cnt_1 >= 4:
        plt.xticks(np.arange(len(input_sizes))+1, input_sizes)
        plt.xlabel('Patch Size')
    else:
        plt.xticks([], [])
    if cnt_1%2 == 0:
        plt.ylabel('IoU')
    plt.title(city_name)

plt.subplot(326)
plt.plot(input_sizes, iou_record_all, 'darkorange')
plt.xlabel('Patch Size')
plt.ylabel('IoU')
plt.title('Overall')
#plt.savefig(os.path.join(img_dir, 'exp3_cmp.png'))
plt.show()

# plot validation curve
input_sizes = [508, 540, 572, 620, 684, 796, 1052]
fields = ['Step', 'Value']
plt.figure(figsize=(9, 4))
matplotlib.rcParams.update({'font.size': 14})
for cnt, input_size  in enumerate(input_sizes):
    file_name = 'run_exp2_UnetCrop_inria_aug_grid_1_PS({}, {})_BS{}_' \
                'EP100_LR0.0001_DS60_DR0.1_SFN32-tag-xent_validation.csv'.\
        format(input_size, input_size, len(input_sizes)-cnt)
    df = pd.read_csv(os.path.join(task_dir, os.path.join(task_dir, file_name)),
                     skipinitialspace=True, usecols=fields)
    value = exponential_smoothing(np.array(df['Value']))
    plt.plot(np.arange(100), value, label='PatchSize={}'.format(input_size))
plt.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
plt.xlabel('Epoch Number')
plt.ylabel('Cross Entropy')
plt.title('Patch Size in Training Comparison')
#plt.savefig(os.path.join(img_dir, 'exp3_val_curve.png'))
#plt.show()
