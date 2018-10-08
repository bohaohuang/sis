import time
import argparse
import ersaPath
from nn import pspnet, hook, nn_utils
from preprocess import patchExtractor
from reader import dataReaderSegmentation, reader_utils
from collection import collectionMaker


# define parameters
BATCH_SIZE = 5
DS_NAME = 'spca'
LEARNING_RATE = 1e-4
TILE_SIZE = (5000, 5000)
PATCH_SIZE = (713, 713)
EPOCHS = 60
NUM_CLASS = 2
PAR_DIR = DS_NAME+'/psp101'
SUFFIX = 'spca'
N_TRAIN = 8000
N_VALID = 1000
VAL_MULT = 5
GPU = 0
DECAY_STEP = 40
DECAY_RATE = 0.1
VERB_STEP = 200
SAVE_EPOCH = 5
DATA_DIR = r'/media/ei-edl01/data/uab_datasets/spca/data/Original_Tiles'
FORCE_RUN = False
RES_DIR = r'/hdd6/Models/pspnet101'
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9


def read_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (10)')
    parser.add_argument('--ds-name', default=DS_NAME, type=str, help='dataset name')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE, help='learning rate (1e-3)')
    parser.add_argument('--tile-size', default=TILE_SIZE, type=tuple, help='tile size 5000')
    parser.add_argument('--patch-size', default=PATCH_SIZE, type=tuple, help='patch size 731')
    parser.add_argument('--epochs', default=EPOCHS, type=int, help='# epochs (1)')
    parser.add_argument('--num-classes', type=int, default=NUM_CLASS, help='# classes (including background)')
    parser.add_argument('--model-par-dir', type=str, default=PAR_DIR, help='parent directory name to save the model')
    parser.add_argument('--model-suffix', type=str, default=SUFFIX, help='suffix in the model directory')
    parser.add_argument('--n-train', type=int, default=N_TRAIN, help='# samples per epoch')
    parser.add_argument('--n-valid', type=int, default=N_VALID, help='# patches to valid')
    parser.add_argument('--val-mult', type=int, default=VAL_MULT, help='validation_bs=val_mult*train_bs')
    parser.add_argument('--GPU', type=str, default=GPU, help="GPU used for computation.")
    parser.add_argument('--decay-step', type=int, default=DECAY_STEP, help='Learning rate decay step in number of epochs.')
    parser.add_argument('--decay-rate', type=float, default=DECAY_RATE, help='Learning rate decay rate')
    parser.add_argument('--verb-step', type=int, default=VERB_STEP, help='#steps between two verbose prints')
    parser.add_argument('--save-epoch', type=int, default=SAVE_EPOCH, help='#epochs between two model save')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR, help='root directory of cityscapes')
    parser.add_argument('--force-run', type=bool, default=FORCE_RUN, help='force run collection maker or not')
    parser.add_argument('--res-dir', type=str, default=RES_DIR, help='path to ckpt of Res101 model')
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY, help='decay for l2 loss')
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='momentum for optimizer')

    flags = parser.parse_args()
    return flags


def main(flags):
    nn_utils.set_gpu(GPU)

    # define network
    model = pspnet.PSPNet(flags.num_classes, flags.patch_size, suffix=flags.model_suffix, learn_rate=flags.learning_rate,
                          decay_step=flags.decay_step, decay_rate=flags.decay_rate, epochs=flags.epochs,
                          batch_size=flags.batch_size, weight_decay=flags.weight_decay, momentum=flags.momentum)
    overlap = model.get_overlap()

    cm = collectionMaker.read_collection(raw_data_path=flags.data_dir,
                                         field_name='Fresno,Modesto,Stockton',
                                         field_id=','.join(str(i) for i in range(663)),
                                         rgb_ext='RGB',
                                         gt_ext='GT',
                                         file_ext='jpg,png',
                                         force_run=False,
                                         clc_name=flags.ds_name)
    cm.print_meta_data()

    cm.print_meta_data()
    file_list_train = cm.load_files(field_id=','.join(str(i) for i in range(0, 250)), field_ext='RGB,GT')
    file_list_valid = cm.load_files(field_id=','.join(str(i) for i in range(250, 500)), field_ext='RGB,GT')
    chan_mean = cm.meta_data['chan_mean'][:3]

    patch_list_train = patchExtractor.PatchExtractor(flags.patch_size, flags.tile_size, flags.ds_name + '_train',
                                                     overlap, overlap // 2). \
        run(file_list=file_list_train, file_exts=['jpg', 'png'], force_run=False).get_filelist()
    patch_list_valid = patchExtractor.PatchExtractor(flags.patch_size, flags.tile_size, flags.ds_name + '_valid',
                                                     overlap, overlap // 2). \
        run(file_list=file_list_valid, file_exts=['jpg', 'png'], force_run=False).get_filelist()

    train_init_op, valid_init_op, reader_op = \
        dataReaderSegmentation.DataReaderSegmentationTrainValid(
            flags.patch_size, patch_list_train, patch_list_valid, batch_size=flags.batch_size, chan_mean=chan_mean,
            aug_func=[reader_utils.image_flipping, reader_utils.image_rotating],
            random=True, has_gt=True, gt_dim=1, include_gt=True, valid_mult=flags.val_mult).read_op()
    feature, label = reader_op

    model.create_graph(feature)
    model.load_resnet(flags.res_dir, keep_last=False)
    model.compile(feature, label, flags.n_train, flags.n_valid, flags.patch_size, ersaPath.PATH['model'],
                  par_dir=flags.model_par_dir, val_mult=flags.val_mult, loss_type='xent')
    train_hook = hook.ValueSummaryHook(flags.verb_step, [model.loss, model.lr_op],
                                       value_names=['train_loss', 'learning_rate'], print_val=[0])
    model_save_hook = hook.ModelSaveHook(model.get_epoch_step()*flags.save_epoch, model.ckdir)
    valid_loss_hook = hook.ValueSummaryHook(model.get_epoch_step(), [model.loss, model.loss_iou],
                                            value_names=['valid_loss', 'valid_mIoU'], log_time=True,
                                            run_time=model.n_valid, iou_pos=1)
    image_hook = hook.ImageValidSummaryHook(model.input_size, model.get_epoch_step(), feature, label, model.output,
                                            nn_utils.image_summary, img_mean=cm.meta_data['chan_mean'])
    start_time = time.time()
    model.train(train_hooks=[train_hook, model_save_hook], valid_hooks=[valid_loss_hook, image_hook],
                train_init=train_init_op, valid_init=valid_init_op)
    print('Duration: {:.3f}'.format((time.time() - start_time)/3600))


if __name__ == '__main__':
    flags = read_flag()
    main(flags)
