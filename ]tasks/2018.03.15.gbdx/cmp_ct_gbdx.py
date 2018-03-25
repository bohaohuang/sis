import os
import imageio
import scipy.misc
import matplotlib.pyplot as plt
from glob import glob

ct_dir = r'/media/ei-edl01/data/uab_datasets/sp/DATA_BUILDING_AND_PANEL'
gbdx_dir = r'/media/ei-edl01/user/as667/ctims'

imgs = glob(os.path.join(gbdx_dir, '*.tif'))
tile_ids = sorted([os.path.basename(a).split('.')[0] for a in imgs])

for tile_name in tile_ids:
    ct_img = imageio.imread(os.path.join(ct_dir, '{}_RGB.jpg'.format(tile_name.replace('_', '-'))))
    gbdx_img = imageio.imread(os.path.join(gbdx_dir, '{}.tif'.format(tile_name)))
    gbdx_img = scipy.misc.imresize(gbdx_img, [2541, 2541])

    img_resize_name = os.path.join(gbdx_dir, '{}_RGB.jpg'.format(tile_name))
    #imageio.imsave(img_resize_name, gbdx_img)

    plt.subplot(121)
    plt.imshow(ct_img)
    plt.subplot(122)
    plt.imshow(gbdx_img)
    plt.show()
