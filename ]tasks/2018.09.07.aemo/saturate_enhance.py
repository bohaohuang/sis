import os
import PIL.Image, PIL.ImageEnhance
import matplotlib.pyplot as plt
from glob import glob
import ersa_utils

# get test data
gammas = [2.5, 1, 2.5]
sample_id = 3
data_dir = r'/media/ei-edl01/data/aemo/samples/0584007740{}0_01'.format(sample_id)
files = sorted(glob(os.path.join(data_dir, 'TILES', '*.tif')))

# adjust gamma
gamma_save_dir = os.path.join(data_dir, 'gamma_adjust')
orig_files = sorted(glob(os.path.join(gamma_save_dir, '*{}.tif'.format(ersa_utils.float2str(gammas[sample_id-1])))))

for f in orig_files:
    img = PIL.Image.open(f)
    converter = PIL.ImageEnhance.Color(img)
    img2 = converter.enhance(2)

    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(121)
    plt.imshow(img)
    plt.axis('off')
    ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
    plt.imshow(img2)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
