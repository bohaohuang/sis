import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import utils
import util_functions

cnn_name = 'unet'
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
img_dir, task_dir = utils.get_task_img_folder()

truth_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles'
if cnn_name == 'unet':
    base_dir = r'/hdd/Results/domain_selection/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_' \
               r'EP100_LR0.0001_DS60_DR0.1_SFN32/inria/pred'
    loo_dir = r'/hdd/Results/domain_selection/UnetCrop_inria_aug_leave_{}_0_PS(572, 572)_BS5_' \
              r'EP100_LR0.0001_DS60_DR0.1_SFN32/inria/pred'
    mtl_dir = r'/hdd/Results/domain_selection/UnetPredict_inria_loo_mtl_{}_0_PS(572, 572)_BS5_' \
              r'EP100_LR0.0001_DS60_DR0.1_SFN32/inria/pred'
else:
    base_dir = r'/hdd/Results/domain_selection/DeeplabV3_inria_aug_grid_0_PS(321, 321)_BS5_' \
               r'EP100_LR1e-05_DS40_DR0.1_SFN32/inria/pred'
    loo_dir = r'/hdd/Results/domain_selection/DeeplabV3_inria_aug_train_leave_{}_0_PS(321, 321)_BS5_' \
              r'EP100_LR1e-05_DS40_DR0.1_SFN32/inria/pred'
    mtl_dir = None

fn_base, fp_base = np.zeros(5), np.zeros(5)
fn_loo, fp_loo = np.zeros(5), np.zeros(5)
fn_mtl, fp_mtl = np.zeros(5), np.zeros(5)
total_cnt = 0
for city_num in range(5):
    for val_img_cnt in range(1, 6):
        img_save_dir = os.path.join(img_dir, 'base_mtl_looidea', city_list[city_num])
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)

        rgb = imageio.imread(os.path.join(truth_dir, '{}{}_RGB.tif'.format(city_list[city_num], val_img_cnt)))

        truth_img_name = os.path.join(truth_dir, '{}{}_GT.tif'.format(city_list[city_num], val_img_cnt))
        base_img_name = os.path.join(base_dir, '{}{}.png'.format(city_list[city_num], val_img_cnt))
        loo_img_name = os.path.join(loo_dir.format(city_num), '{}{}.png'.format(city_list[city_num], val_img_cnt))
        mtl_img_name = os.path.join(mtl_dir.format(city_num), '{}{}.png'.format(city_list[city_num], val_img_cnt))

        truth, base, loo, mtl = imageio.imread(truth_img_name), imageio.imread(base_img_name), \
                                imageio.imread(loo_img_name), imageio.imread(mtl_img_name)
        truth = truth / 255
        base = base / 255

        fn_base[city_num] += np.sum((truth - base) == 1)
        fn_loo[city_num] += np.sum((truth - loo) == 1)
        fn_mtl[city_num] += np.sum((truth - mtl) == 1)

        fp_base[city_num] += np.sum((truth - base) == -1)
        fp_loo[city_num] += np.sum((truth - loo) == -1)
        fp_mtl[city_num] += np.sum((truth - mtl) == -1)

        total_cnt += 5000 * 5000

fig = plt.figure(figsize=(8, 12))
width = 0.3
X = np.arange(5)

plt.subplot(211)
plt.bar(X, fn_base, width=width, label='Base')
plt.bar(X + width, fn_loo, width=width, label='LOO')
plt.bar(X + width * 2, fn_mtl, width=width, label='MTL')
plt.xticks([])
plt.ylabel('CNT')
plt.legend()
plt.title('FN Comparison UNet MTL')

plt.subplot(212)
plt.bar(X, fp_base, width=width, label='Base')
plt.bar(X + width, fp_loo, width=width, label='LOO')
plt.bar(X + width * 2, fp_mtl, width=width, label='MTL')
plt.xticks(X + width, city_list)
plt.xlabel('City')
plt.ylabel('CNT')
plt.legend()
plt.title('FP Comparison UNet MTL')

plt.tight_layout()
#plt.savefig(os.path.join(img_dir, 'unet_agg_cmp_loo_mtl_base_fn_fp.png'))
plt.show()

plt.figure()
colors = util_functions.get_default_colors()
base_plt = []
loo_plt = []
mtl_plt = []
for i in range(5):
    base_plt.append(plt.scatter(fn_base[i] / total_cnt, fp_base[i] / total_cnt, c=colors[i], marker='o'))
    loo_plt.append(plt.scatter(fn_loo[i] / total_cnt, fp_loo[i] / total_cnt, c=colors[i], marker='v'))
    mtl_plt.append(plt.scatter(fn_mtl[i] / total_cnt, fp_mtl[i] / total_cnt, c=colors[i], marker='s'))
plt.xlim(0, 0.013)
plt.ylim(0, 0.012)
plt.legend((base_plt[0], loo_plt[0], mtl_plt[0], base_plt[1], loo_plt[1], mtl_plt[1],
            base_plt[2], loo_plt[2], mtl_plt[2], base_plt[3], loo_plt[3], mtl_plt[3],
            base_plt[4], loo_plt[4], mtl_plt[4]),
           ('Austin Base', 'Austin LOO', 'Austin MTL', 'Chicago Base', 'Chicago LOO', 'Chicago MTL',
            'Kitsap Base', 'Kitsap LOO', 'Kitsap MTL', 'Tyrol-w Base', 'Tyrol-w LOO', 'Tyrol-w MTL',
            'Vienna Base', 'Vienna LOO', 'Vienna MTL'), scatterpoints=1, ncol=5, fontsize=8,
           bbox_to_anchor=(1.1, 1.1), fancybox=True, shadow=True)
plt.xlabel('FNr')
plt.ylabel('FPr')
#plt.savefig(os.path.join(img_dir, 'unet_agg_cmp_loo_mtl_base_fn_fp_scatter.png'))
plt.show()
