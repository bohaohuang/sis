import time
import numpy as np
import tensorflow as tf
import ersaPath
from nn import unet, hook, nn_utils
from preprocess import patchExtractor, histMatching
from reader import dataReaderSegmentation, reader_utils
from collection import collectionMaker, collectionEditor

# settings
class_num = 2
patch_size = (572, 572)
tile_size = (5000, 5000)
ds_name = 'aemo_hist'
par_dir = 'aemo/new_xregion'
from_scratch = True
ds = 30
dr = 0.1
epochs = 80
bs = 5
valid_mult = 5
start_layer = 10
gpu = 1
n_train = 590
n_valid = 590
verb_step = 50
save_epoch = 20
model_dir = r'/hdd6/Models/spca/UnetCropWeighted_GridChipPretrained6Weighted4_PS(572, 572)_BS5_' \
            r'EP100_LR0.0001_DS50_DR0.1_SFN32'

nn_utils.set_gpu(gpu)
if start_layer >= 10:
    suffix_base = 'aemo'
else:
    suffix_base = 'aemo_up{}'.format(start_layer)
for lr in [1e-3, 1e-4]:
    for run_id in range(4):
        suffix = '{}_{}'.format(suffix_base, run_id)
        tf.reset_default_graph()

        np.random.seed(run_id)
        tf.set_random_seed(run_id)

        # define network
        model = unet.UNet(class_num, patch_size, suffix=suffix, learn_rate=lr, decay_step=ds, decay_rate=dr,
                         epochs=epochs, batch_size=bs)
        overlap = model.get_overlap()

        cm = collectionMaker.read_collection(raw_data_path=r'/home/lab/Documents/bohao/data/aemo/aemo_hist',
                                             field_name='aus10,aus30,aus50',
                                             field_id='',
                                             rgb_ext='.*rgb',
                                             gt_ext='.*gt',
                                             file_ext='tif',
                                             force_run=False,
                                             clc_name=ds_name)
        cm.print_meta_data()

        file_list = cm.load_files(field_name='aus10,aus30,aus50', field_id='', field_ext='.*rgb,.*gt')
        file_list_train = []
        file_list_valid = []
        for i in range(0, len(file_list), 2):
            file_list_train.append(file_list[i])
            file_list_valid.append(file_list[i + 1])

        patch_list_train = patchExtractor.PatchExtractor(patch_size, tile_size, ds_name+'_train_xregion',
                                                         overlap, overlap//2).\
            run(file_list=file_list_train, file_exts=['jpg', 'png'], force_run=False).get_filelist()
        patch_list_valid = patchExtractor.PatchExtractor(patch_size, tile_size, ds_name+'_valid_xregion',
                                                         overlap, overlap//2).\
            run(file_list=file_list_valid, file_exts=['jpg', 'png'], force_run=False).get_filelist()
        chan_mean = cm.meta_data['chan_mean']

        train_init_op, valid_init_op, reader_op = \
            dataReaderSegmentation.DataReaderSegmentationTrainValid(
                patch_size, patch_list_train, patch_list_valid, batch_size=bs, chan_mean=chan_mean,
                aug_func=[reader_utils.image_flipping, reader_utils.image_rotating],
                random=True, has_gt=True, gt_dim=1, include_gt=True, valid_mult=valid_mult).read_op()
        feature, label = reader_op

        model.create_graph(feature)
        if start_layer >= 10:
            model.compile(feature, label, n_train, n_valid, patch_size, ersaPath.PATH['model'], par_dir=par_dir,
                          loss_type='xent')
        else:
            model.compile(feature, label, n_train, n_valid, patch_size, ersaPath.PATH['model'], par_dir=par_dir,
                          loss_type='xent', train_var_filter=['layerup{}'.format(i) for i in range(start_layer, 10)])
        train_hook = hook.ValueSummaryHook(verb_step, [model.loss, model.lr_op], value_names=['train_loss', 'learning_rate'],
                                           print_val=[0])
        model_save_hook = hook.ModelSaveHook(model.get_epoch_step()*save_epoch, model.ckdir)
        valid_loss_hook = hook.ValueSummaryHookIters(model.get_epoch_step(), [model.loss_xent, model.loss_iou],
                                                     value_names=['valid_loss', 'IoU'], log_time=True, run_time=model.n_valid)
        image_hook = hook.ImageValidSummaryHook(model.input_size, model.get_epoch_step(), feature, label, model.pred,
                                                nn_utils.image_summary, img_mean=chan_mean)
        start_time = time.time()
        if not from_scratch:
            model.load(model_dir)
        model.train(train_hooks=[train_hook, model_save_hook], valid_hooks=[valid_loss_hook, image_hook],
                   train_init=train_init_op, valid_init=valid_init_op)
        print('Duration: {:.3f}'.format((time.time() - start_time)/3600))
