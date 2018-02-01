import os
import imageio
import matplotlib.pyplot as plt
import utils
from util_functions import iou_metric

unet_pred_dir = r'/hdd/Results/grid_vs_random/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32/inria/pred'
frrn_pred_dir = r'/hdd/Results/grid_vs_random/FRRN_inria_aug_grid_0_PS(224, 224)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32/inria/pred'
deeplab_pred_dir = r'/hdd/Results/DeeplabV2_inria_aug_grid_0_PS(321, 321)_BS5_EP100_LR0.0005_DS40_DR0.1_SFN32/default/pred'
data_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles'

city_name = 'chicago1'
chip_size = 321
img_dir, task_dir = utils.get_task_img_folder()

img = imageio.imread(os.path.join(data_dir, city_name+'_RGB.tif'))[:chip_size, :chip_size, :]
gt = imageio.imread(os.path.join(data_dir, city_name+'_GT.tif'))[:chip_size, :chip_size]/255
unet_pred = imageio.imread(os.path.join(unet_pred_dir, city_name+'.png'))[:chip_size, :chip_size]/255
frrn_pred = imageio.imread(os.path.join(frrn_pred_dir, city_name+'.png'))[:chip_size, :chip_size]/255
deeplab_pred = imageio.imread(os.path.join(deeplab_pred_dir, city_name+'.png'))[:chip_size, :chip_size]/255

unet_iou = iou_metric(gt, unet_pred, truth_val=1)*100
frrn_iou = iou_metric(gt, frrn_pred, truth_val=1)*100
deeplab_iou = iou_metric(gt, deeplab_pred, truth_val=1)*100

plt.figure(figsize=(8, 8))
plt.subplot(221)
plt.imshow(img)
plt.axis('off')
plt.title(city_name)
plt.subplot(222)
plt.imshow(unet_pred-gt, cmap='bwr')
plt.xticks([], [])
plt.yticks([], [])
plt.title('U-Net {:.2f}'.format(unet_iou))
plt.subplot(223)
plt.imshow(frrn_pred-gt, cmap='bwr')
plt.xticks([], [])
plt.yticks([], [])
plt.title('FRRN {:.2f}'.format(frrn_iou))
plt.subplot(224)
plt.imshow(deeplab_pred-gt, cmap='bwr')
plt.xticks([], [])
plt.yticks([], [])
plt.title('DeepLab V2 {:.2f}'.format(deeplab_iou))
plt.tight_layout()

img_name = '{}_{}.png'.format('results_compare2', city_name)
plt.savefig(os.path.join(img_dir, img_name))
plt.show()
