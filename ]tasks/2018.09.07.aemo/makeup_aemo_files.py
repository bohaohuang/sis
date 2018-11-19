import os
import numpy as np
from glob import glob
import utils
import ersa_utils
from preprocess import histMatching
from collection import collectionMaker, collectionEditor


def get_blank_regions(img):
    img_tmp = img.astype(np.float32)
    img_tmp = np.sum(img_tmp, axis=2)
    blank_mask = (img_tmp < 0.1).astype(np.int)
    return blank_mask


def process_different_situation(blank_region, orig_img, gt, code):
    h, w = blank_region.shape
    makeup = np.zeros((h, w, 3))
    makeup_gt = np.zeros((h, w))
    end_point = 0
    if code == 0:
        for i in range(w):
            if blank_region[h-1, i] != 0:
                end_point = i
        strip = orig_img[:, end_point:end_point*2, :]
        makeup[:, :end_point, :] = strip[:, ::-1, :]

        strip_gt = gt[:, end_point:end_point * 2]
        makeup_gt[:, :end_point] = strip_gt[:, ::-1]
    elif code == 1:
        for i in range(h-1, -1, -1):
            if blank_region[i, 0] != 0:
                end_point = i
        strip = orig_img[h-end_point*2:h-end_point, :, :]
        makeup[h-end_point:h, :, :] = strip[::-1, :, :]

        strip_gt = gt[h - end_point * 2:h - end_point, :]
        makeup_gt[h - end_point:h, :] = strip_gt[::-1, :]
    elif code == 2:
        for i in range(h):
            if blank_region[i, 0] != 0:
                end_point = i
        strip = orig_img[end_point:2*end_point, :, :]
        makeup[:end_point, :, :] = strip[::-1, :, :]

        strip_gt = gt[end_point:2 * end_point, :]
        makeup_gt[:end_point, :] = strip_gt[::-1, :]
    elif code == 3:
        for i in range(w):
            if blank_region[0, i] != 0:
                end_point = i
        strip = orig_img[:, end_point:end_point*2, :]
        makeup[:, :end_point, :] = strip[:, ::-1, :]

        strip_gt = gt[:, end_point:end_point * 2]
        makeup_gt[:, :end_point] = strip_gt[:, ::-1]
    elif code == 4:
        for i in range(w-1, -1, -1):
            if blank_region[0, i] != 0:
                end_point = i
        end_point = w - end_point
        strip = orig_img[:, w-end_point*2:w-end_point, :]
        makeup[:, w-end_point:, :] = strip[:, ::-1, :]

        strip_gt = gt[:, w - end_point * 2:w - end_point]
        makeup_gt[:, w - end_point:] = strip_gt[:, ::-1]
    elif code == 5:
        for i in range(h):
            if blank_region[i, w-1] != 0:
                end_point = i
        strip = orig_img[end_point:end_point*2, :, :]
        makeup[:end_point, :, :] = strip[::-1, :, :]

        strip_gt = gt[end_point:end_point * 2, :]
        makeup_gt[:end_point, :] = strip_gt[::-1, :]
    elif code == 6:
        for i in range(h-1, -1, -1):
            if blank_region[i, w-1] != 0:
                end_point = i
        end_point = h - end_point
        strip = orig_img[h-end_point*2:h-end_point, :, :]
        makeup[h-end_point:, :, :] = strip[::-1, :, :]

        strip_gt = gt[h - end_point * 2:h - end_point, :]
        makeup_gt[h - end_point:, :] = strip_gt[::-1, :]
    elif code == 7:
        for i in range(w-1, -1, -1):
            if blank_region[h-1, i] != 0:
                end_point = i
        strip = orig_img[:, w - end_point * 2:w - end_point, :]
        makeup[:, w - end_point:, :] = strip[:, ::-1, :]

        strip_gt = gt[:, w - end_point * 2:w - end_point]
        makeup_gt[:, w - end_point:] = strip_gt[:, ::-1]
    else:
        return orig_img.astype(np.uint8), gt.astype(np.uint8)
    for i in range(3):
        makeup[:, :, i] = makeup[:, :, i] * blank_region
    makeup_gt = makeup_gt * blank_region
    return (orig_img + makeup).astype(np.uint8), (gt + makeup_gt).astype(np.uint8)


def makeup_aemo_img(img, gt, code):
    for c in code:
        bm = get_blank_regions(img)
        img, gt = process_different_situation(bm, img, gt, int(c))

    return img, gt


'''suffix = 'aemo'
np.random.seed(1004)
img_dir, task_dir = utils.get_task_img_folder()
save_dir = r'/home/lab/Documents/bohao/data/aemo/aemo_pad'

cm = collectionMaker.read_collection(raw_data_path=r'/home/lab/Documents/bohao/data/aemo',
                                     field_name='aus10,aus30,aus50',
                                     field_id='',
                                     rgb_ext='.*rgb',
                                     gt_ext='.*gt',
                                     file_ext='tif',
                                     force_run=False,
                                     clc_name='aemo')
gt_d255 = collectionEditor.SingleChanMult(cm.clc_dir, 1/255, ['.*gt', 'gt_d255']).\
    run(force_run=False, file_ext='tif', d_type=np.uint8,)
cm.replace_channel(gt_d255.files, True, ['gt', 'gt_d255'])
# hist matching
ref_file = r'/media/ei-edl01/data/uab_datasets/spca/data/Original_Tiles/Fresno1_RGB.jpg'
ga = histMatching.HistMatching(ref_file, color_space='RGB', ds_name=suffix)
file_list = [f[0] for f in cm.meta_data['rgb_files']]
hist_match = ga.run(force_run=False, file_list=file_list)
cm.add_channel(hist_match.get_files(), '.*rgb_hist')
cm.print_meta_data()

aemo_files = cm.load_files(field_name='aus10,aus30,aus50', field_id='', field_ext='.*rgb,.*gt_d255')

print(aemo_files)

code_list = ['02', '6', '46', '6', '0', '8']

for i in range(6):
    test_file = aemo_files[i]
    print('processing {}'.format(test_file[0]))
    rgb = ersa_utils.load_file(test_file[0])
    gt = ersa_utils.load_file(test_file[1])
    rgb_new, gt_new = makeup_aemo_img(rgb, gt, code_list[i])
    save_name = os.path.join(save_dir, os.path.basename(test_file[0]))
    ersa_utils.save_file(save_name, rgb_new)
    save_name = os.path.join(save_dir, os.path.basename(test_file[1]))
    ersa_utils.save_file(save_name, gt_new)'''

'''data_dir = r'/home/lab/Documents/bohao/data/aemo/aemo_union'
rgb_files = sorted(glob(os.path.join(data_dir, '*rgb.tif')))
gt_files = sorted(glob(os.path.join(data_dir, '*comb.tif')))
files = [rgb_files, gt_files]
files = ersa_utils.rotate_list(files)

print(files)

code_list = ['02', '6', '46', '6', '0', '8']
save_dir = r'/media/ei-edl01/data/uab_datasets/aemo_comb/data/Original_Tiles'

for i in range(6):
    test_file = files[i]
    print('processing {}'.format(test_file[0]))
    rgb = ersa_utils.load_file(test_file[0])
    gt = ersa_utils.load_file(test_file[1])
    rgb_new, gt_new = makeup_aemo_img(rgb, gt, code_list[i])
    save_name = os.path.join(save_dir, os.path.basename(test_file[0]))
    ersa_utils.save_file(save_name, rgb_new)
    save_name = os.path.join(save_dir, os.path.basename(test_file[1]))
    ersa_utils.save_file(save_name, (gt_new/255).astype(np.uint8))'''

data_dir = r'/media/ei-edl01/data/uab_datasets/aemo_comb/data/Original_Tiles'
rgb_files = sorted(glob(os.path.join(data_dir, '*rgb.tif')))
gt_files = sorted(glob(os.path.join(data_dir, '*comb.tif')))
files = [rgb_files, gt_files]
files = ersa_utils.rotate_list(files)

from visualize import visualize_utils
for i in range(6):
    test_file = files[i]
    print('processing {}'.format(test_file[0]))
    rgb = ersa_utils.load_file(test_file[0])
    gt = ersa_utils.load_file(test_file[1])
    print(np.unique(gt))
    visualize_utils.compare_two_figure(rgb, gt)
