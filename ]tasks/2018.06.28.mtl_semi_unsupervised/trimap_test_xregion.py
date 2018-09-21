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

gt_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/TilePreproc/MultChanOp_chans3_Divide_dF255p000'

model_dir_ugan = r'/hdd/Results/ugan/UnetGAN_V3_inria_gan_xregion_0_PS(572, 572)_BS20_' \
                 r'EP30_LR0.0001_1e-06_1e-06_DS30.0_30.0_30.0_DR0.1_0.1_0.1/inria/pred'

model_dir_base = r'/hdd/Results/domain_selection/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_' \
                 r'EP100_LR0.0001_DS60_DR0.1_SFN32/inria/pred'

for cnt, width in enumerate(tqdm(trimap_width)):
    base_trimap = np.zeros(2)
    ugan_trimap = np.zeros(2)
    for city in city_list:
        for tile_id in range(1, 6):
            gt_tile_name = '{}{}_GT_Divide.tif'.format(city, tile_id)
            pred_tile_name = '{}{}.png'.format(city, tile_id)

            gt_img = imageio.imread(os.path.join(gt_dir, gt_tile_name))
            base_img = imageio.imread(os.path.join(model_dir_base, pred_tile_name)) // 255
            ugan_img = imageio.imread(os.path.join(model_dir_ugan, pred_tile_name))

            base_trimap += 1/25 * trimap_test(gt_img, base_img, width=width)
            ugan_trimap += 1/25 * trimap_test(gt_img, ugan_img, width=width)
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
plt.title('Trimap Test on Inria')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'trimap_inria_xregion.png'))
plt.show()
