import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import utils


def get_patch_effect(res_dir, file_dir, input_sizes, model_name, appendix='.npy'):
    iou_record_all = np.zeros(len(input_sizes))
    time_record_all = np.zeros(len(input_sizes))
    for cnt_1, size in enumerate(input_sizes):
        file_name = '{}{}'.format(size, appendix)
        data = dict(np.load(os.path.join(res_dir, file_name)).tolist())
        iou = []
        for item in data.keys():
            #if item != 'kitsap4' and item != 'time':
            if item != 'time':
                iou.append(data[item]*100)
            elif item == 'time':
                time_record_all[cnt_1] = data[item]
        iou_record_all[cnt_1] = np.mean(iou)
    with open(os.path.join(file_dir, model_name, 'run_time.txt'), 'r') as f:
        record = f.readlines()
    time_record_all = [item.split(' ')[1] for item in record]
    return iou_record_all, time_record_all


res_dir = r'/media/ei-edl01/user/bh163/tasks/2017.12.16.framework_train_cnn'
file_dir = '/hdd/Temp/IGARSS2018'

# unet crop
input_sizes_unet = [572, 828, 1084, 1340, 1596, 1852, 2092, 2332, 2636]
appendix = '.npy'
unet_iou, unet_time = get_patch_effect(res_dir, file_dir, input_sizes_unet, 'UnetCrop', appendix)

# resfcn no crop
input_sizes_res = [224, 480, 736, 992, 1248, 1504, 1760, 2016, 2272, 2528]
appendix = '_resfcn.npy'
res_iou, res_time = get_patch_effect(res_dir, file_dir, input_sizes_res, 'ResFcn', appendix)

plt.rcParams.update({'font.size': 14})
plt.rc('grid', linestyle='--')
plt.figure(figsize=(9, 10))
plt.subplot(211)
plt.plot(input_sizes_unet, unet_iou-unet_iou[0], '-o', label='U-Net')
plt.plot(input_sizes_res, res_iou-res_iou[0], '-v', label='ResNet50')
plt.grid()
#plt.text(150, 0.7, '(a)')
plt.ylabel('delta IoU')

plt.subplot(212)
plt.plot(input_sizes_unet, unet_time, '-o', label='U-Net')
plt.plot(input_sizes_res, res_time, '-v', label='ResNet50')
plt.grid()
plt.legend()
plt.xlabel('Input Size')
plt.ylabel('Time:s')
#plt.text(150, 1100, '(b)')
save_dir = r'/media/ei-edl01/user/bh163/figs/2017.12.27.igarss_figures/jordan'
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'result4poster.png'), dpi=600)
plt.show()
