import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sis_utils


def plot_patch_effect(res_dir, input_sizes, name, appendix='.npy', savefig=True):
    iou_record_all = np.zeros(len(input_sizes))
    time_record_all = np.zeros(len(input_sizes))
    for cnt_1, size in enumerate(input_sizes):
        file_name = '{}{}'.format(size, appendix)
        data = dict(np.load(os.path.join(res_dir, file_name)).tolist())
        iou = []
        for item in data.keys():
            if item != 'kitsap4' and item != 'time':
                iou.append(data[item]*100)
            elif item == 'time':
                time_record_all[cnt_1] = data[item]
        iou_record_all[cnt_1] = np.mean(iou)
    # plot the figure
    fig = plt.figure(figsize=(8, 4))
    matplotlib.rcParams.update({'font.size': 14})
    plt.subplot(211)
    plt.plot(np.array(input_sizes), iou_record_all)
    plt.xticks([], [])
    plt.ylabel('IoU')
    plt.title(name)
    plt.subplot(212)
    plt.plot(np.array(input_sizes), time_record_all)
    plt.xticks(input_sizes, input_sizes)
    plt.xlabel('Patch Size')
    plt.ylabel('Time:s')
    fig.tight_layout()
    if savefig:
        img_dir, _ = sis_utils.get_task_img_folder()
        plt.savefig(os.path.join(img_dir, 'paper2_{}.png'.format(name)))
    plt.show()


res_dir = r'/media/ei-edl01/user/bh163/tasks/2017.12.16.framework_train_cnn'

# unet crop
input_sizes = [572, 828, 1084, 1340, 1596, 1852, 2092, 2332, 2636]
appendix = '.npy'
plot_patch_effect(res_dir, input_sizes, 'Unet No Zero Padding', appendix)

# unet no crop
input_sizes = [576, 736, 992, 1248, 1504, 1760, 2016, 2272, 2528]
appendix = '_unet_no_crop.npy'
plot_patch_effect(res_dir, input_sizes, 'Unet Zero Padding', appendix)

# resfcn no crop
input_sizes = [224, 480, 736, 992, 1248, 1504, 1760, 2016, 2272, 2528]
appendix = '_resfcn.npy'
plot_patch_effect(res_dir, input_sizes, 'ResNet50', appendix)

import imageio
img_dir, _ = sis_utils.get_task_img_folder()
img1 = imageio.imread(os.path.join(img_dir, 'paper2_Unet No Zero Padding.png'))
img2 = imageio.imread(os.path.join(img_dir, 'paper2_Unet Zero Padding.png'))
img3 = imageio.imread(os.path.join(img_dir, 'paper2_ResNet50.png'))
img = np.concatenate([img1, img2, img3], axis=0)
imageio.imsave(os.path.join(img_dir, 'paper2_concat.png'), img)
