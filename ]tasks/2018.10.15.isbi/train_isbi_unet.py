import time
import numpy as np
import ersaPath
from nn import unet, hook, nn_utils
from preprocess import patchExtractor, histMatching
from reader import dataReaderSegmentation, reader_utils
from collection import collectionMaker, collectionEditor

# settings
class_num = 2
patch_size = (508, 508)
suffix = 'isbi'
ds_name = 'isbi'
lr = 1e-4
ds = 2
dr = 0.5
epochs = 20
bs = 5
valid_mult = 5
gpu = 1
n_train = 500
n_valid = 500
verb_step = 50
save_epoch = 5


def image_summary(image, truth, prediction, img_mean=np.array((0, 0, 0), dtype=np.float32), label_num=2):
    """
    Make a image summary where the format is image|truth|pred
    :param image: input rgb image
    :param truth: ground truth
    :param prediction: network prediction
    :param img_mean: image mean, need to add back here for visualization
    :param label_num: #distinct classes in ground truth
    :return:
    """
    truth_img = truth[:, :, :, 0] * 255
    prediction = nn_utils.pad_prediction(image, prediction)
    pred_img = np.argmax(prediction, axis=-1) * 255

    _, h, w, _ = image.shape
    image = image[:, :, :, 0]
    if w/h > 1.5:
        # concatenate image horizontally if it is too wide
        return np.repeat(np.expand_dims(np.concatenate([image+img_mean, truth_img, pred_img], axis=1), axis=-1), 3, axis=-1)
    else:
        return np.repeat(np.expand_dims(np.concatenate([image+img_mean, truth_img, pred_img], axis=2), axis=-1), 3, axis=-1)


nn_utils.set_gpu(gpu)

# define network
unet = unet.UNet(class_num, patch_size, suffix=suffix, learn_rate=lr, decay_step=ds, decay_rate=dr,
                 epochs=epochs, batch_size=bs)
overlap = unet.get_overlap()

cm = collectionMaker.read_collection(raw_data_path=r'/home/lab/Documents/bohao/data/isbi',
                                     field_name='train',
                                     field_id=','.join([str(i) for i in range(30)]),
                                     rgb_ext='volume',
                                     gt_ext='labels',
                                     file_ext='jpg,png',
                                     force_run=False,
                                     clc_name=ds_name)
cm.print_meta_data()

file_list_train = cm.load_files(field_name='train', field_id=','.join([str(i) for i in range(5, 30)]),
                                field_ext='volume,labels')
file_list_valid = cm.load_files(field_name='train', field_id=','.join([str(i) for i in range(5)]),
                                field_ext='volume,labels')
chan_mean = cm.meta_data['chan_mean'][:3]

train_init_op, valid_init_op, reader_op = \
    dataReaderSegmentation.DataReaderSegmentationTrainValid(
        patch_size, file_list_train, file_list_valid, batch_size=bs, chan_mean=chan_mean,
        aug_func=[reader_utils.image_flipping, reader_utils.image_rotating],
        random=True, has_gt=True, gt_dim=1, include_gt=True, valid_mult=valid_mult).read_op()
feature, label = reader_op

unet.create_graph(feature)
unet.compile(feature, label, n_train, n_valid, patch_size, ersaPath.PATH['model'], par_dir=ds_name, loss_type='xent')
train_hook = hook.ValueSummaryHook(verb_step, [unet.loss, unet.lr_op], value_names=['train_loss', 'learning_rate'],
                                   print_val=[0])
model_save_hook = hook.ModelSaveHook(unet.get_epoch_step()*save_epoch, unet.ckdir)
valid_loss_hook = hook.ValueSummaryHookIters(unet.get_epoch_step(), [unet.loss_xent, unet.loss_iou],
                                             value_names=['valid_loss', 'IoU'], log_time=True,
                                             run_time=unet.n_valid)
image_hook = hook.ImageValidSummaryHook(unet.input_size, unet.get_epoch_step(), feature, label, unet.pred,
                                        image_summary, img_mean=chan_mean)
start_time = time.time()
unet.train(train_hooks=[train_hook, model_save_hook], valid_hooks=[valid_loss_hook, image_hook],
           train_init=train_init_op, valid_init=valid_init_op)
print('Duration: {:.3f}'.format((time.time() - start_time)/3600))
