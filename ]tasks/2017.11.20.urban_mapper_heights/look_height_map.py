import os
import scipy.misc
import matplotlib.pyplot as plt


data_dir = r'/media/ei-edl01/data/remote_sensing_data/urban_mapper/image'

dsm = 'JAX_Tile_004_DSM.tif'
dtm = 'JAX_Tile_004_DTM.tif'
rgb = 'JAX_Tile_004_RGB.tif'

'''dsm_fig = scipy.misc.imread(os.path.join(data_dir, dsm))
dtm_fig = scipy.misc.imread(os.path.join(data_dir, dtm))
rgb_fig = scipy.misc.imread(os.path.join(data_dir, rgb))

plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.hist(dsm_fig.flatten())
plt.subplot(222)
plt.hist(dtm_fig.flatten())
plt.subplot(223)
plt.imshow(dsm_fig-dtm_fig)
plt.subplot(224)
plt.imshow(rgb_fig)
plt.show()

plt.figure()
ax1 = plt.subplot(121 )
ax1.imshow(dsm_fig-dtm_fig)
ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
ax2.imshow(rgb_fig)
plt.show()'''

print('flip' in 'flip,rotate')
