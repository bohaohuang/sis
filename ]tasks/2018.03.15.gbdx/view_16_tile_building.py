import os
import imageio
from glob import glob
from tqdm import tqdm
import utils
from util_functions import add_mask


if __name__ == '__main__':
    img_dir, task_dir = utils.get_task_img_folder()

    pred_dir_base = r'/hdd/Results/gbdx_cmp/DeeplabV3_res101_inria_aug_grid_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32/sp/pred'
    gt_dir = r'/media/ei-edl01/data/uab_datasets/sp/DATA_BUILDING_AND_PANEL'
    tile_list = [os.path.basename(x).split('.')[0] for x in glob(os.path.join(pred_dir_base, '*.png'))]

    for city_name in tqdm(tile_list):
        pred_file_base_name = os.path.join(pred_dir_base, '{}.png'.format(city_name))
        rgb_file_name = os.path.join(gt_dir, '{}_RGB.jpg'.format(city_name))

        pred_img_base = imageio.imread(pred_file_base_name)
        rgb_img = imageio.imread(rgb_file_name)

        masked_img = add_mask(rgb_img, pred_img_base, [255, None, None], mask_1=1)
        #plt.imshow(masked_img)
        #plt.show()

        img_name = os.path.join(img_dir, 'ca_building', 'deeplab', '{}_building_mask.png'.format(city_name))
        imageio.imsave(img_name, masked_img)
