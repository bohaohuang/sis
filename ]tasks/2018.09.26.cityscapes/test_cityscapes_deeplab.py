import argparse
import skimage.transform
from nn import deeplab, nn_utils, nn_processor
from reader import dataReaderSegmentation
import cityscapes_reader, cityscapes_labels


# define parameters
BATCH_SIZE = 1
DS_NAME = 'cityscapes'
TILE_SIZE = (512, 1024)
NUM_CLASS = 19
PAR_DIR = DS_NAME+'/res101'
GPU = 0
DATA_DIR = r'/hdd/cityscapes'
RGB_TYPE = 'leftImg8bit'
GT_TYPE = 'gtFine'
RGB_EXT = RGB_TYPE
GT_EXT = 'labelTrainIds'
FORCE_RUN = False
RES_DIR = r'/hdd6/Models/cityscapes/res101/deeplab_test_PS(512, 1024)_BS1_EP40_LR0.0001_DS20.0_DR0.1'


def read_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (10)')
    parser.add_argument('--ds-name', default=DS_NAME, type=str, help='dataset name')
    parser.add_argument('--tile-size', default=TILE_SIZE, type=tuple, help='tile size 5000')
    parser.add_argument('--num-classes', type=int, default=NUM_CLASS, help='# classes (including background)')
    parser.add_argument('--model-par-dir', type=str, default=PAR_DIR, help='parent directory name to save the model')
    parser.add_argument('--GPU', type=str, default=GPU, help="GPU used for computation.")
    parser.add_argument('--data-dir', type=str, default=DATA_DIR, help='root directory of cityscapes')
    parser.add_argument('--rgb-type', type=str, default=RGB_TYPE, help='rgb type in cityscapes')
    parser.add_argument('--gt-type', type=str, default=GT_TYPE, help='gt type in cityscapes')
    parser.add_argument('--rgb-ext', type=str, default=RGB_EXT, help='rgb extension in their file names')
    parser.add_argument('--gt-ext', type=str, default=GT_EXT, help='gt extensions in their file names')
    parser.add_argument('--force-run', type=bool, default=FORCE_RUN, help='force run collection maker or not')
    parser.add_argument('--res-dir', type=str, default=RES_DIR, help='path to ckpt of Res101 model')

    flags = parser.parse_args()
    return flags


def main(flags):
    nn_utils.set_gpu(GPU)

    # define network
    model = deeplab.DeepLab(flags.num_classes, flags.tile_size, batch_size=flags.batch_size)

    cm_train = cityscapes_reader.CollectionMakerCityscapes(flags.data_dir, flags.rgb_type, flags.gt_type, 'train',
                                                           flags.rgb_ext, flags.gt_ext, ['png', 'png'],
                                                           clc_name='{}_train'.format(flags.ds_name),
                                                           force_run=flags.force_run)
    cm_test = cityscapes_reader.CollectionMakerCityscapes(flags.data_dir, flags.rgb_type, flags.gt_type, 'val',
                                                          flags.rgb_ext, flags.gt_ext, ['png', 'png'],
                                                          clc_name='{}_valid'.format(flags.ds_name),
                                                          force_run=flags.force_run)
    cm_test.print_meta_data()
    resize_func_train = lambda img: skimage.transform.resize(img, flags.tile_size, mode='reflect')
    resize_func_test = lambda img: skimage.transform.resize(img, cm_test.meta_data['tile_dim'], order=0,
                                                            preserve_range=True, mode='reflect')

    init_op, reader_op = dataReaderSegmentation.DataReaderSegmentation(
        flags.tile_size, cm_test.meta_data['file_list'], batch_size=flags.batch_size,
        chan_mean=cm_train.meta_data['chan_mean'], is_train=False, has_gt=True, gt_dim=1, include_gt=True,
        global_func=resize_func_train).read_op()
    estimator = nn_processor.NNEstimatorSegmentScene(
        model, cm_test.meta_data['file_list'], flags.res_dir, init_op, reader_op, ds_name='city_scapes',
        save_result_parent_dir='Cityscapes', gpu=flags.GPU, score_result=True, split_char='.',
        post_func=resize_func_test, save_func=cityscapes_labels.decode_labels, ignore_label=(-1, 255))
    estimator.run(force_run=flags.force_run)


if __name__ == '__main__':
    flags = read_flag()
    main(flags)
