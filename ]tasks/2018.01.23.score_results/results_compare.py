import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import utils
from util_functions import iou_metric

unet_pred_dir = r'/hdd/Results/grid_vs_random/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32/inria/pred'
frrn_pred_dir = r'/hdd/Results/grid_vs_random/FRRN_inria_aug_grid_0_PS(224, 224)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32/inria/pred'
deeplab_pred_dir = r'/hdd/Results/DeeplabV3_res101_inria_aug_grid_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32/default/pred'
fpn_pred_dir = r'/hdd/Results/FPNRes101_inria_aug_grid_1_PS(224, 224)_BS10_EP100_LR1e-05_DS40_DR0.1_SFN32/default/pred'
data_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles'
img_dir, task_dir = utils.get_task_img_folder()

# plot city level score compare
model_names = ['U-Net', 'FRRN', 'DeepLab V2', 'FPN']
group_mean = np.zeros((4, 5))
group_err = np.zeros((4, 5))
ious = np.zeros(4)
for cnt, model in enumerate([unet_pred_dir, frrn_pred_dir, deeplab_pred_dir, fpn_pred_dir]):
    result_dir = os.path.join(model[:-5], 'result.txt')
    with open(result_dir, 'r') as file:
        iou_all = file.readlines()
    iou_mean = float(iou_all[-1])
    ious[cnt] = iou_mean*100
    city_name = []
    city_iou = []
    for line in iou_all[:-1]:
        city_name.append(line.split(' ')[0])
        iou_tuple = line.split('(')[1]
        A = int(iou_tuple.split(',')[0])
        B = int(iou_tuple.split(', ')[1].strip()[:-1])
        city_iou.append(A/B)
    for i in range(0, 25, 5):
        group_mean[cnt, int(i/5)] = np.mean(city_iou[i+5-1])
        group_err[cnt, int(i/5)] = np.std(city_iou[i+5-1])

fig = plt.figure(figsize=(6, 5))
width = 0.2
ind = np.arange(5)
for i in range(4):
    plt.bar(ind+width*i, group_mean[i,:], width, yerr=group_err[i,:], label='{} = {:.2f}'.format(model_names[i], ious[i]))
plt.xticks(ind+1.5*width, ['Austin', 'Chicago', 'Kitsap', 'Tyrol-w', 'Vienna'])
plt.legend(loc='upper left', fancybox=True, framealpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'city_level_cmp_thinner.png'))
plt.show()


city_name = 'kitsap4'
chip_size = 5000

img = imageio.imread(os.path.join(data_dir, city_name+'_RGB.tif'))[:chip_size, :chip_size, :]
gt = imageio.imread(os.path.join(data_dir, city_name+'_GT.tif'))[:chip_size, :chip_size]/255
unet_pred = imageio.imread(os.path.join(unet_pred_dir, city_name+'.png'))[:chip_size, :chip_size]/255
frrn_pred = imageio.imread(os.path.join(frrn_pred_dir, city_name+'.png'))[:chip_size, :chip_size]/255
deeplab_pred = imageio.imread(os.path.join(deeplab_pred_dir, city_name+'.png'))[:chip_size, :chip_size]/255
fpn_pred = imageio.imread(os.path.join(fpn_pred_dir, city_name+'.png'))[:chip_size, :chip_size]/255

unet_iou = iou_metric(gt, unet_pred, truth_val=1)*100
frrn_iou = iou_metric(gt, frrn_pred, truth_val=1)*100
deeplab_iou = iou_metric(gt, deeplab_pred, truth_val=1)*100
fpn_iou = iou_metric(gt, fpn_pred, truth_val=1)*100

plt.figure(figsize=(16, 4))
ax1 = plt.subplot(151)
plt.imshow(img)
plt.axis('off')
plt.title(city_name)
plt.subplot(152, sharex=ax1, sharey=ax1)
plt.imshow(unet_pred-gt, cmap='bwr')
plt.xticks([], [])
plt.yticks([], [])
plt.title('U-Net {:.2f}'.format(unet_iou))
plt.subplot(153, sharex=ax1, sharey=ax1)
plt.imshow(frrn_pred-gt, cmap='bwr')
plt.xticks([], [])
plt.yticks([], [])
plt.title('FRRN {:.2f}'.format(frrn_iou))
plt.subplot(154, sharex=ax1, sharey=ax1)
plt.imshow(deeplab_pred-gt, cmap='bwr')
plt.xticks([], [])
plt.yticks([], [])
plt.title('DeepLab V2 {:.2f}'.format(deeplab_iou))
plt.tight_layout()
plt.subplot(155, sharex=ax1, sharey=ax1)
plt.imshow(fpn_pred-gt, cmap='bwr')
plt.xticks([], [])
plt.yticks([], [])
plt.title('FPN {:.2f}'.format(fpn_iou))
plt.tight_layout()

img_name = '{}_{}.png'.format('results_compare_all', city_name)
plt.savefig(os.path.join(img_dir, img_name))
plt.show()
