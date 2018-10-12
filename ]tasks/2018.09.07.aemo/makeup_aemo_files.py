import numpy as np
import matplotlib.pyplot as plt
import utils
import ersa_utils
from preprocess import histMatching
from collection import collectionMaker, collectionEditor


def get_blank_regions(img):
    img_tmp = img.astype(np.float32)
    img_tmp = np.sum(img_tmp, axis=2)
    blank_mask = (img_tmp < 0.1).astype(np.int)
    return blank_mask


def makeup_aemo_img(img):
    bm = get_blank_regions(img)

    # take care of horizontal makeup first



suffix = 'aemo'
np.random.seed(1004)
img_dir, task_dir = utils.get_task_img_folder()

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

aemo_files = cm.load_files(field_name='aus10,aus30,aus50', field_id='', field_ext='.*rgb')

test_file = aemo_files[0][0]
rgb = ersa_utils.load_file(test_file)

makeup_aemo_img(rgb)
