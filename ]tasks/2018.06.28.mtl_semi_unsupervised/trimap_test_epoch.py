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
epoch = ['epoch_{}'.format(i) for i in range(0, 30, 5)] + ['inria']
width = list(range(5, 25, 5))
ugan_err_list = np.zeros((len(width), len(epoch)))
img_dir, task_dir = utils.get_task_img_folder()

gt_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/TilePreproc/MultChanOp_chans3_Divide_dF255p000'

model_dir_ugan = r'/hdd/Results/ugan/UnetGAN_V3_inria_gan_xregion_0_PS(572, 572)_BS20_' \
                 r'EP30_LR0.0001_1e-06_1e-06_DS30.0_30.0_30.0_DR0.1_0.1_0.1/'

for cnt, epoch_num in enumerate(tqdm(epoch)):
    for cnt_width, test_width in enumerate(width):
        ugan_trimap = np.zeros(2)
        for city in city_list:
            for tile_id in range(1, 6):
                gt_tile_name = '{}{}_GT_Divide.tif'.format(city, tile_id)
                pred_tile_name = '{}{}.png'.format(city, tile_id)

                gt_img = imageio.imread(os.path.join(gt_dir, gt_tile_name))
                ugan_img = imageio.imread(os.path.join(model_dir_ugan, epoch_num, 'pred', pred_tile_name))

                ugan_trimap += 1/25 * trimap_test(gt_img, ugan_img, width=test_width)
        ugan_err = ugan_trimap[0] / ugan_trimap[1] * 100
        ugan_err_list[cnt_width, cnt] = ugan_err

plt.figure(figsize=(8, 4))
for cnt, test_width in enumerate(width):
    plt.plot(np.arange(0, 35, 5), ugan_err_list[cnt, :], marker='s', label='Width={}'.format(test_width))
plt.xlabel('Epoch')
plt.ylabel('Pixelwise Classification Error [%]')
plt.grid('on')
plt.legend()
plt.title('Trimap Test on Inria')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'trimap_inria_xregion_epoch.png'))
plt.show()
