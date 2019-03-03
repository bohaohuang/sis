import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sis_utils
from contour_maker import get_contour


def trimap_test(gt, pred, width=1):
    contour = get_contour(np.copy(gt), contour_length=width)
    pred = pred[contour == 1].astype(np.float32)
    gt = gt[contour == 1].astype(np.float32)
    err = [np.sum(np.abs(pred-gt)), pred.shape[0]]
    return np.array(err)


epoch = ['epoch_{}'.format(i) for i in range(0, 60, 10)] + ['Mass_road']
width = list(range(5, 25, 5))
ugan_err_list = np.zeros((len(width), len(epoch)))
img_dir, task_dir = sis_utils.get_task_img_folder()

gt_dir = r'/media/ei-edl01/data/uab_datasets/Mass_road/data/TilePreproc/MultChanOp_chans3_Divide_dF255p000'

model_dir_ugan = r'/hdd/Results/Road/UnetGAN_V3ShrinkRGB_road_gan_0_PS(572, 572)_BS20_EP60_' \
                 r'LR0.0001_1e-06_1e-06_DS60.0_60.0_60.0_DR0.1_0.1_0.1'

model_dir_base = r'/hdd/Results/Road/UnetGAN_V3Shrink_road_gan_0_PS(572, 572)_BS20_EP60_LR0.0001_1e-06_1e-06_' \
                 r'DS40.0_40.0_40.0_DR0.1_0.1_0.1/Mass_road/pred'

for cnt, epoch_num in enumerate(tqdm(epoch)):
    for cnt_width, test_width in enumerate(width):
        ugan_trimap = np.zeros(2)
        for tile_id in range(1, 50):
            gt_tile_name = 'roadTest{}_GT_Divide.tif'.format(tile_id)
            pred_tile_name = 'roadTest{}.png'.format(tile_id)

            gt_img = imageio.imread(os.path.join(gt_dir, gt_tile_name))
            ugan_img = imageio.imread(os.path.join(model_dir_ugan, epoch_num, 'pred', pred_tile_name))
            ugan_trimap += 1/50 * trimap_test(gt_img, ugan_img, width=test_width)
        ugan_err = ugan_trimap[0] / ugan_trimap[1] * 100
        ugan_err_list[cnt_width, cnt] = ugan_err

plt.figure(figsize=(8, 4))
for cnt, test_width in enumerate(width):
    plt.plot(np.arange(0, 65, 10), ugan_err_list[cnt, :], marker='s', label='Width={}'.format(test_width))
plt.xlabel('Epoch')
plt.ylabel('Pixelwise Classification Error [%]')
plt.grid('on')
plt.legend()
plt.title('Trimap Test on Mass Road')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'trimap_road_epoch.png'))
plt.show()
