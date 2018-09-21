import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils
from contour_maker import get_contour


def trimap_test(gt, pred, width=1):
    contour = get_contour(np.copy(gt), contour_length=width)
    pred = pred[contour == 1].astype(np.float32)
    gt = gt[contour == 1].astype(np.float32)
    err = [np.sum(np.abs(pred-gt)), pred.shape[0]]
    return np.array(err)


city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
trimap_width = np.arange(1, 21)
base_err_list = np.zeros(len(trimap_width))
ugan_err_list = np.zeros(len(trimap_width))
img_dir, task_dir = utils.get_task_img_folder()

gt_dir = r'/media/ei-edl01/data/uab_datasets/Mass_road/data/TilePreproc/MultChanOp_chans3_Divide_dF255p000'

model_dir_ugan = r'/hdd/Results/Road/UnetGAN_V3ShrinkRGB_road_gan_0_PS(572, 572)_BS20_EP60_' \
                 r'LR0.0001_1e-06_1e-06_DS60.0_60.0_60.0_DR0.1_0.1_0.1/Mass_road/pred'

model_dir_base = r'/hdd/Results/Road/UnetGAN_V3Shrink_road_gan_0_PS(572, 572)_BS20_EP60_LR0.0001_1e-06_1e-06_' \
                 r'DS40.0_40.0_40.0_DR0.1_0.1_0.1/Mass_road/pred'

for cnt, width in enumerate(tqdm(trimap_width)):
    base_trimap = np.zeros(2)
    ugan_trimap = np.zeros(2)
    for tile_id in range(1, 50):
        gt_tile_name = 'roadTest{}_GT_Divide.tif'.format(tile_id)
        pred_tile_name = 'roadTest{}.png'.format(tile_id)

        gt_img = imageio.imread(os.path.join(gt_dir, gt_tile_name))
        base_img = imageio.imread(os.path.join(model_dir_base, pred_tile_name)) // 255
        ugan_img = imageio.imread(os.path.join(model_dir_ugan, pred_tile_name))

        base_trimap += 1/50 * trimap_test(gt_img, base_img, width=width)
        ugan_trimap += 1/50 * trimap_test(gt_img, ugan_img, width=width)
    base_err = base_trimap[0] / base_trimap[1] * 100
    ugan_err = ugan_trimap[0] / ugan_trimap[1] * 100

    base_err_list[cnt] = base_err
    ugan_err_list[cnt] = ugan_err

plt.figure(figsize=(8, 4))
plt.plot(trimap_width, base_err_list, marker='o', label='U-Net')
plt.plot(trimap_width, ugan_err_list, marker='s', label='U-Net GAN')
plt.xlabel('Trimap Width [Pixels]')
plt.ylabel('Pixelwise Classification Error [%]')
plt.grid('on')
plt.legend()
plt.title('Trimap Test on Mass Road')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'trimap_road.png'))
plt.show()
