import os
from glob import glob
import matplotlib.pyplot as plt
import ersa_utils

data_dir = r'/media/ei-edl01/data/aemo/samples/0584007740{}0_01'
image_id = 0

files = glob(os.path.join(data_dir.format(2), 'TILES', '*5000x5000*.tif'))
rgb_img = files[image_id]
print(rgb_img)
rgb = ersa_utils.load_file(rgb_img)

plt.figure(figsize=(12, 10))
ax1 = plt.subplot(221)
plt.imshow(rgb)
plt.axis('off')
for i in range(1, 4):
    if i == 2:
        files = glob(os.path.join(data_dir.format(i), 'bh_pred', '*5000x5000*.tif'))
    else:
        files = glob(os.path.join(data_dir.format(i), 'bh_pred', '*5000*5000*RGB.tif'))
    pred_img = files[image_id]
    print(pred_img)
    pred = ersa_utils.load_file(pred_img)

    ax2 = plt.subplot(221 + i, sharex=ax1, sharey=ax1)
    plt.imshow(pred)
    plt.axis('off')
    plt.title('Sample {}'.format(i))
plt.tight_layout()
plt.show()
