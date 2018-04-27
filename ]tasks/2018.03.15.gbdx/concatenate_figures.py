import os
import imageio
import numpy as np
import matplotlib.pyplot as plt

save_dir = r'/media/ei-edl01/user/bh163/figs/2018.03.15.gbdx/blog_figures'
'''img_2016_1 = os.path.join(r'/media/ei-edl01/user/bh163/figs/2018.03.15.gbdx/blog_figures', '2016_1.png')
img_2017_1 = os.path.join(r'/media/ei-edl01/user/bh163/figs/2018.03.15.gbdx/blog_figures', '2017_1.png')
img_2016_2 = os.path.join(r'/media/ei-edl01/user/bh163/figs/2018.03.15.gbdx/blog_figures', '2016_2.png')
img_2017_2 = os.path.join(r'/media/ei-edl01/user/bh163/figs/2018.03.15.gbdx/blog_figures', '2017_2.png')
img_2016_1 = imageio.imread(img_2016_1)
img_2017_1 = imageio.imread(img_2017_1)
img_2016_2 = imageio.imread(img_2016_2)
img_2017_2 = imageio.imread(img_2017_2)

final_figure = np.zeros((1000, 1100, 3), dtype=np.uint8)
final_figure[0:500, 0:600, :] = img_2016_1[100:600, 250:850, :]
final_figure[500:1000, 0:600, :] = img_2017_1[100:600, 250:850, :]
final_figure[0:500, 600:, :] = img_2016_2
final_figure[500:1000, 600:, :] = img_2017_2

plt.imshow(final_figure)
plt.show()

imageio.imsave(os.path.join(save_dir, '2016top_2017bottom.png'), final_figure)'''

strip = np.zeros((400, 800, 3), dtype=np.uint8)
for cnt_1, city_name in enumerate(['CT', 'HW', 'IN', 'DZ']):
    for cnt_2, j in enumerate(range(2)):
        img_name = r'/media/ei-edl01/user/bh163/figs/2018.03.15.gbdx/blog_figures/{}_{}.png'.format(city_name, j+1)
        img = imageio.imread(img_name)
        strip[cnt_2*200:(cnt_2+1)*200, cnt_1*200:(cnt_1+1)*200, :] = img
plt.imshow(strip)
plt.show()
imageio.imsave(os.path.join(save_dir, 'left_CT_HW_IN_DZ_right.png'), strip)
