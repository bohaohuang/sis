import os
import imageio
import scipy.misc
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from shutil import copyfile

exclude_id = '000795_se'
orig_gt_dir = r'/media/ei-edl01/user/jmm123/gbdx/solar_ground_truth'
orig_rgb_dir = r'/media/ei-edl01/user/as667/1040010021B61200_rechunked'
dest_dir = r'/media/ei-edl01/data/uab_datasets/gbdx/data/Original_Tiles'

gt_files = glob(os.path.join(orig_gt_dir, '*_truth.png'))
'''for gt_file in tqdm(gt_files):
    file_id = '_'.join(os.path.basename(gt_file).split('_')[:2])
    if file_id == exclude_id:
        continue
    rgb_orig_fname = file_id + '.tif'

    # copy file to dest
    copyfile(gt_file, os.path.join(dest_dir, file_id.replace('_', '-')+'_GT.png'))
    rgb_orig = imageio.imread(os.path.join(orig_rgb_dir, rgb_orig_fname))
    rgb_orig = scipy.misc.imresize(rgb_orig, [2541, 2541])
    imageio.imsave(os.path.join(dest_dir, file_id.replace('_', '-')+'_RGB.tif'), rgb_orig)'''

for gt_file in gt_files:
    file_id = '_'.join(os.path.basename(gt_file).split('_')[:2])
    rgb_orig_fname = file_id + '.tif'

    # check original file
    try:
        gt_orig = imageio.imread(gt_file)
        gt_dest = imageio.imread(os.path.join(dest_dir, file_id.replace('_', '-')+'_GT.png'))
        rgb_orig = imageio.imread(os.path.join(orig_rgb_dir, rgb_orig_fname))
        rgb_dest = imageio.imread(os.path.join(dest_dir, file_id.replace('_', '-')+'_RGB.tif'))
        print(gt_orig.shape, gt_dest.shape, rgb_orig.shape, rgb_dest.shape)

        plt.figure(figsize=(12, 9))
        ax1 = plt.subplot(221)
        plt.imshow(gt_orig)
        plt.title('gt orig')
        plt.axis('off')
        ax2 = plt.subplot(222, sharex=ax1, sharey=ax1)
        plt.imshow(rgb_orig)
        plt.title('rgb orig')
        plt.axis('off')
        ax3 = plt.subplot(223, sharex=ax1, sharey=ax1)
        plt.imshow(gt_dest)
        plt.title('gt dest')
        plt.axis('off')
        ax4 = plt.subplot(224, sharex=ax1, sharey=ax1)
        plt.imshow(rgb_dest)
        plt.title('rgb dest')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except OSError:
        continue
