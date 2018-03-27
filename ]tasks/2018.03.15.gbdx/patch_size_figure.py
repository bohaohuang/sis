import os
import imageio
import matplotlib.pyplot as plt

fig_dir = r'/media/ei-edl01/user/bh163/figs/2018.03.02.res_gan'
small_name = '010780-sw_building_mask_321.png'
large_name = '010780-sw_building_mask_736.png'
small_img = imageio.imread(os.path.join(fig_dir, small_name))
large_img = imageio.imread(os.path.join(fig_dir, large_name))

plt.figure(figsize=(15, 8))
plt.subplot(121)
plt.imshow(small_img[0:800, 1000:1800, :])
plt.axis('off')
plt.title('Test Size = 321')
plt.axhline(320, 0, 1, linestyle='--', color='b')
plt.axhline(641, 0, 1, linestyle='--', color='b')
plt.axvline(271, 0, 1, linestyle='--', color='b')
plt.axvline(592, 0, 1, linestyle='--', color='b')
plt.subplot(122)
plt.imshow(large_img[0:800, 1000:1800, :])
plt.axhline(736, 0, 1, linestyle='--', color='b')
plt.axvline(472, 0, 1, linestyle='--', color='b')
plt.axis('off')
plt.title('Test Size = 736')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'patch_size_cmp.png'))
plt.show()
