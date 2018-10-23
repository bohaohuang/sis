import os
import numpy as np
import matplotlib.pyplot as plt
import utils
import ersa_utils


def crop_center(img, cropx, cropy):
    y,x = img.shape[:2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    if len(img.shape) == 3:
        return img[starty:starty+cropy, startx:startx+cropx, :]
    else:
        return img[starty:starty+cropy, startx:startx+cropx]


# settings
img_dir, task_dir = utils.get_task_img_folder()
show_figure = False

aemo_img_dir = os.path.join(img_dir, 'aemo_patches')
aemo_ftr_dir = os.path.join(task_dir, 'aemo_patches')

spca_img_dir = r'/hdd/ersa/patch_extractor/spca_all'
spca_ftr_dir = os.path.join(task_dir, 'spca_patches')

# load aemo features
aemo_patch_name_file = os.path.join(aemo_ftr_dir, 'res50_patches.txt')
aemo_patch_names = ersa_utils.load_file(aemo_patch_name_file)

aemo_feature_name_file = os.path.join(aemo_ftr_dir, 'res50_feature.csv')
aemo_feature = np.genfromtxt(aemo_feature_name_file, delimiter=',')

# load spca features
spca_patch_name_file = os.path.join(spca_ftr_dir, 'res50_patches.txt')
spca_patch_names = ersa_utils.load_file(spca_patch_name_file)

spca_feature_name_file = os.path.join(spca_ftr_dir, 'res50_feature.csv')
spca_feature = np.genfromtxt(spca_feature_name_file, delimiter=',')

top_range = np.arange(20)
patch_select_record = np.zeros(len(top_range))
for top_num in [5]:
    select_patches = []
    for cnt in range(len(aemo_patch_names)):
        aemo_ftr = aemo_feature[cnt, :]

        dist = np.sum(np.square(spca_feature - aemo_ftr), axis=1)
        sort_idx = np.argsort(dist)

        if show_figure:
            aemo_img = ersa_utils.load_file(os.path.join(aemo_img_dir, aemo_patch_names[cnt].strip()))

            fig = plt.figure(figsize=(16, 3))
            plt.subplot(1, top_num+1, 1)
            plt.imshow(aemo_img)
            plt.axis('off')

            for i in range(top_num):
                spca_img = ersa_utils.load_file(spca_patch_names[sort_idx[i]].strip())
                plt.subplot(1, top_num+1, 2+i)
                plt.imshow(crop_center(spca_img, 224, 224))
                plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(img_dir, 'similar_aemo{}_spca_top{}.png'.format(cnt, top_num)))
            plt.show()

        select_patches.append(sort_idx[:top_num])

    select_patch_file = os.path.join(task_dir, 'top{}_select_patch.npy'.format(top_num))
    ersa_utils.save_file(select_patch_file, np.unique(select_patches))

    select_num = len(np.unique(select_patches))
    total_num = spca_feature.shape[0]
    patch_select_record[top_num] = select_num / total_num

'''plt.figure(figsize=(8, 4))
plt.plot(top_range, patch_select_record, '-o', linewidth=2)
plt.xlabel('#Top Patches')
plt.ylabel('%Patches Selected')
plt.title('Patch Selection Result in California')
plt.grid(True)
plt.tight_layout()
plt.show()'''
