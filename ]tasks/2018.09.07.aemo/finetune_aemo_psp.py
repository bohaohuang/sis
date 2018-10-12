import time
import numpy as np
import ersaPath
from nn import pspnet, hook, nn_utils
from preprocess import patchExtractor, histMatching
from reader import dataReaderSegmentation, reader_utils
from collection import collectionMaker, collectionEditor

# settings
class_num = 2
patch_size = (384, 384)
tile_size = (5000, 5000)
suffix = 'aemo_hist_rgb'
ds_name = 'aemo'
lr = 1e-3
ds = 60
dr = 0.1
epochs = 130
bs = 5
valid_mult = 5
gpu = 0
n_train = 785
n_valid = 500
verb_step = 50
save_epoch = 5
model_dir = r'/hdd6/Models/spca/psp101/pspnet_spca_PS(384, 384)_BS5_EP100_LR0.001_DS40_DR0.1'

nn_utils.set_gpu(gpu)

# define network
unet = pspnet.PSPNet(class_num, patch_size, suffix=suffix, learn_rate=lr, decay_step=ds, decay_rate=dr,
                     epochs=epochs, batch_size=bs, weight_decay=1e-3)
overlap = unet.get_overlap()

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

file_list_train = cm.load_files(field_name='aus10,aus30', field_id='', field_ext='.*rgb_hist,.*gt_d255')
file_list_valid = cm.load_files(field_name='aus50', field_id='', field_ext='.*rgb_hist,.*gt_d255')
chan_mean = cm.meta_data['chan_mean'][-3:]

patch_list_train = patchExtractor.PatchExtractor(patch_size, tile_size, ds_name+'_train', overlap, overlap//2).\
    run(file_list=file_list_train, file_exts=['jpg', 'png'], force_run=False).get_filelist()
patch_list_valid = patchExtractor.PatchExtractor(patch_size, tile_size, ds_name+'_valid', overlap, overlap//2).\
    run(file_list=file_list_valid, file_exts=['jpg', 'png'], force_run=False).get_filelist()

train_init_op, valid_init_op, reader_op = \
    dataReaderSegmentation.DataReaderSegmentationTrainValid(
        patch_size, patch_list_train, patch_list_valid, batch_size=bs, chan_mean=chan_mean,
        aug_func=[reader_utils.image_flipping, reader_utils.image_rotating],
        random=True, has_gt=True, gt_dim=1, include_gt=True, valid_mult=valid_mult).read_op()
feature, label = reader_op

unet.create_graph(feature)
unet.compile(feature, label, n_train, n_valid, patch_size, ersaPath.PATH['model'], par_dir=ds_name, loss_type='xent')
train_hook = hook.ValueSummaryHook(verb_step, [unet.loss, unet.lr_op], value_names=['train_loss', 'learning_rate'],
                                   print_val=[0])
model_save_hook = hook.ModelSaveHook(unet.get_epoch_step()*save_epoch, unet.ckdir)
valid_loss_hook = hook.ValueSummaryHook(unet.get_epoch_step(), [unet.loss, unet.loss_iou],
                                        value_names=['valid_loss', 'IoU'], log_time=True, run_time=unet.n_valid,
                                        iou_pos=1)
image_hook = hook.ImageValidSummaryHook(unet.input_size, unet.get_epoch_step(), feature, label, unet.pred,
                                        nn_utils.image_summary, img_mean=chan_mean)
start_time = time.time()
unet.load(model_dir)
unet.train(train_hooks=[train_hook, model_save_hook], valid_hooks=[valid_loss_hook, image_hook],
           train_init=train_init_op, valid_init=valid_init_op)
print('Duration: {:.3f}'.format((time.time() - start_time)/3600))
