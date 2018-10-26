import os
import scipy.misc
import numpy as np
from glob import glob
from tqdm import tqdm
import ersa_utils
import processBlock
from nn import unet, deeplab, nn_utils
from preprocess import histMatching
from collection import collectionMaker, collectionEditor


class UpSampling(processBlock.BasicProcess):
    def __init__(self, up_size, path=None, ds_name=None, name='up_sampling'):
        self.up_size = up_size
        pe_name = '{}_h{}_w{}'.format(name, up_size[0], up_size[1])
        if path is None:
            assert ds_name is not None
            path = ersa_utils.get_block_dir('data', ['preprocess', ds_name, name])
        super().__init__(pe_name, path, func=self.process)

    def process(self, **kwargs):
        """
        Extract the patches
        :param kwargs:
            file_list: list of files
        :return:
        """
        pbar = tqdm(kwargs['file_list'])
        for files in pbar:
            pbar.set_description('Upsampling {} to {}*{}'.format(os.path.basename(files),
                                                                 self.up_size[0], self.up_size[1]))
            old_name = os.path.basename(files)
            tile_name = '{}_upsample{}.{}'.format(old_name.split('.')[0], self.up_size[0],
                                                  old_name.split('.')[1])
            tile_name = os.path.join(self.path, tile_name)
            old_file = ersa_utils.load_file(files)
            if len(old_file.shape) == 3:
                new_file = scipy.misc.imresize(old_file, self.up_size, 'bilinear')
            else:
                new_file = scipy.misc.imresize(old_file, self.up_size, 'nearest')
            ersa_utils.save_file(tile_name, new_file)

    def get_files(self):
        files = sorted(glob(os.path.join(self.path, '*.*')))
        files = [f for f in files if 'txt' not in f]
        return files

# settings
class_num = 2
tile_size = (5000, 5000)
suffix = 'aemo_pad'
bs = 5
gpu = 1
model_name = 'unet'

# define network
if model_name == 'unet':
    patch_size = (572, 572)
    unet = unet.UNet(class_num, patch_size, suffix=suffix, batch_size=bs)
else:
    patch_size = (321, 321)
    unet = deeplab.DeepLab(class_num, patch_size, suffix=suffix, batch_size=bs)
overlap = unet.get_overlap()

cm = collectionMaker.read_collection(raw_data_path=r'/home/lab/Documents/bohao/data/aemo/aemo_pad',
                                     field_name='aus10,aus30,aus50',
                                     field_id='',
                                     rgb_ext='.*rgb',
                                     gt_ext='.*gt',
                                     file_ext='tif',
                                     force_run=False,
                                     clc_name=suffix)
gt_d255 = collectionEditor.SingleChanMult(cm.clc_dir, 1/255, ['.*gt', 'gt_d255']).\
    run(force_run=False, file_ext='tif', d_type=np.uint8,)
cm.replace_channel(gt_d255.files, True, ['gt', 'gt_d255'])
# hist matching
ref_file = r'/media/ei-edl01/data/uab_datasets/spca/data/Original_Tiles/Fresno1_RGB.jpg'
ga = histMatching.HistMatching(ref_file, color_space='RGB', ds_name=suffix)
file_list = [f[0] for f in cm.meta_data['rgb_files']]
hist_match = ga.run(force_run=False, file_list=file_list)
cm.add_channel(hist_match.get_files(), '.*rgb_hist')

'''up_rgb = UpSampling(tile_size, name='up_rgb', ds_name=suffix)
up_gt = UpSampling(tile_size, name='up_gt', ds_name=suffix)
file_list_rgb = [f[0] for f in cm.meta_data['rgb_files']]
file_list_gt = [f[0] for f in cm.load_files(field_id='', field_ext='.*gt')]
ur = up_rgb.run(force_run=False, file_list=file_list_rgb)
ug = up_gt.run(force_run=False, file_list=file_list_gt)
cm.add_channel(ur.get_files(), '.*rgb_up')
cm.add_channel(ug.get_files(), '.*gt_up')'''
cm.print_meta_data()

file_list_train = cm.load_files(field_name='aus10,aus30', field_id='', field_ext='.*rgb_hist,.*gt_d255')
#file_list_valid = cm.load_files(field_name='aus50', field_id='', field_ext='.*rgb_hist,.*gt_d255')
file_list_valid = cm.load_files(field_name='aus50', field_id='', field_ext='.*rgb_hist,.*gt_d255')
chan_mean = cm.meta_data['chan_mean'][-3:]

nn_utils.tf_warn_level(3)
model_dir = r'/hdd6/Models/aemo/new3/unet_aemo_4_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1'
unet.evaluate(file_list_valid, patch_size, tile_size, bs, chan_mean, model_dir, gpu, save_result_parent_dir='aemo',
              sfn=32, force_run=True, score_results=True, split_char='.')
