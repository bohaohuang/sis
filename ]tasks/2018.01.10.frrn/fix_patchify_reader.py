import os
import imageio
from uabUtilreader import patchify

img = imageio.imread(os.path.join(r'/media/ei-edl01/user/bh163/figs/2017.12.27.igarss_figures/jordan',
                                  'austin1_img_00045.jpg'))
reader = patchify(img, tile_dim=(572, 572), patch_size=(572, 572), overlap=184)
for file in reader:
    print(file.shape)
