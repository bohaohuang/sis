import os
from glob import glob
import tensorflow as tf
import ersa_utils
from reader import reader_utils
import processBlock
from collection import collectionMaker
from reader import dataReaderSegmentation


class CollectionMakerCityscapes(object):
    """
    Make a Cityscapes collection
    """
    def __init__(self, raw_data_path, rgb_type, gt_type, split, rgb_ext, gt_ext, file_ext=('png', 'png'),
                 clc_name=None, force_run=False):
        """
        Initialize the object
        :param raw_data_path: root to the dataset
        :param rgb_type: type of the rgb images, see https://github.com/mcordts/cityscapesScripts for more details
        :param gt_type: type of the ground truths, see https://github.com/mcordts/cityscapesScripts for more details
        :param split: train, valid or test
        :param rgb_ext: extension of rgb files, usually the same as rgb_type
        :param gt_ext: extension of gt files, this is used to determine object or instance level annotation to use
        :param file_ext: extension of the files, by default rgb and gt files are all png format in cityscapes
        :param clc_name: name of the collection
        :param force_run: force run even if the dataset already exists
        """
        self.raw_data_path = raw_data_path
        self.rgb_type = rgb_type
        self.gt_type = gt_type
        self.split = split
        self.rgb_ext = ersa_utils.str2list(rgb_ext, d_type=str)
        self.gt_ext = gt_ext
        if len(gt_ext) == 0:
            has_gt_ext = 0
        else:
            has_gt_ext = 1
        self.file_ext = ersa_utils.str2list(file_ext, d_type=str)
        # check if given file extensions matches the number of rgb and gt files
        assert len(self.file_ext) == 1 or len(self.file_ext) == len(self.rgb_ext) + has_gt_ext
        if len(self.file_ext) == 1:
            self.file_ext = [self.file_ext[0] for _ in range(len(self.rgb_ext) + has_gt_ext)]
        if clc_name is None:
            clc_name = os.path.basename(raw_data_path)
        self.clc_name = clc_name
        self.clc_dir = self.get_dir()
        self.force_run = force_run

        # make collection
        self.files = []
        self.clc_pb = processBlock.BasicProcess('collection_maker', self.clc_dir, self.make_collection)
        self.clc_pb.run(self.force_run)
        self.meta_data = self.read_meta_data()

    def get_dir(self):
        """
        Get or create directory of this collection
        :return: directory of the collection
        """
        return ersa_utils.get_block_dir('data', ['collection', self.clc_name])

    def make_collection(self):
        """
        Make meta data of the collection, including tile dimension, ground truth and rgb files list
        means of all channels in rgb files
        :return:
        """
        # collect files selection
        # get city names
        city_names = sorted([name for name in os.listdir(os.path.join(self.raw_data_path, self.rgb_type, self.split))])
        rgb_files = []
        gt_files = []
        for city_name in city_names:
            # add rgb files in each city
            files = sorted(glob(os.path.join(self.raw_data_path, self.rgb_type, self.split, city_name,
                                             '*{}.{}'.format(self.rgb_ext, self.file_ext[0]))))
            for file in files:
                rgb_files.append(file)

            # add gt files in each city
            files = sorted(glob(os.path.join(self.raw_data_path, self.gt_type, self.split, city_name,
                                             '*{}.{}'.format(self.gt_ext, self.file_ext[1]))))
            for file in files:
                gt_files.append(file)
        assert len(rgb_files) == len(gt_files)
        self.files = rgb_files + gt_files
        rgb_files = ersa_utils.rotate_list([rgb_files])

        # make meta_data
        tile_dim = ersa_utils.load_file(rgb_files[0][0]).shape[:2]
        channel_mean = collectionMaker.get_channel_mean(self.raw_data_path, rgb_files)

        meta_data = {'raw_data_path': self.raw_data_path,
                     'rgb_type': self.rgb_type,
                     'gt_type': self.gt_type,
                     'split': self.split,
                     'rgb_ext': self.rgb_ext,
                     'gt_ext': self.gt_ext,
                     'file_ext': self.file_ext,
                     'clc_name': self.clc_name,
                     'tile_dim': tile_dim,
                     'gt_files': gt_files,
                     'rgb_files': rgb_files,
                     'chan_mean': channel_mean,
                     'file_list': self.get_rgb_gt_file_list(rgb_files, gt_files),
                     }
        ersa_utils.save_file(os.path.join(self.clc_dir, 'meta.pkl'), meta_data)

    def read_meta_data(self):
        """
        Read meta data of the collection
        :return:
        """
        meta_data = ersa_utils.load_file(os.path.join(self.clc_dir, 'meta.pkl'))
        return meta_data

    def print_meta_data(self):
        """
        Print the meta data in a human readable format
        :return:
        """
        print(ersa_utils.make_center_string('=', 88, self.clc_name))
        skip_keys = ['gt_files', 'rgb_files', 'rgb_ext', 'gt_ext', 'file_ext', 'files']
        for key, val in self.meta_data.items():
            if key in skip_keys:
                continue
            if type(val) is list:
                if len(val) >= 10:
                    print('{}: [{}, {}, ..., {}]'.format(key, val[0], val[1], val[-1]))
                else:
                    print('{}: {}'.format(key, val))
            else:
                print('{}: {}'.format(key, val))
        print('Source file: {}'.format(' '.join(['*{}*.{}'.format(ext1, ext2)
                                                 for ext1, ext2 in zip(self.rgb_ext, self.file_ext)])))
        if len(self.gt_ext) > 0:
            print('GT file: {}'.format('*{}*.{}'.format(self.gt_ext, self.file_ext[-1])))
        print(ersa_utils.make_center_string('=', 88))

    @staticmethod
    def get_rgb_gt_file_list(rgb_files, gt_files):
        file_list = []
        for rgb, gt in zip(rgb_files, gt_files):
            file_list.append(rgb + [gt])
        return file_list


if __name__ == '__main__':
    root = r'/media/ei-edl01/data/remoteSensingDatasets/Cityscapes'
    rgb_type = 'leftImg8bit'
    gt_type = 'gtFine'
    rgb_ext = rgb_type
    gt_ext = 'labelTrainIds'
    batch_size = 5
    valid_mult = 5
    cm_train = CollectionMakerCityscapes(root, rgb_type, gt_type, 'train', rgb_ext, gt_ext, ['png', 'png'],
                                         clc_name='cityscapes_train', force_run=False)
    cm_valid = CollectionMakerCityscapes(root, rgb_type, gt_type, 'val', rgb_ext, gt_ext, ['png', 'png'],
                                         clc_name='cityscapes_valid', force_run=False)

    train_init_op, valid_init_op, reader_op = dataReaderSegmentation.DataReaderSegmentationTrainValid(
        cm_train.meta_data['tile_dim'], cm_train.meta_data['file_list'], cm_valid.meta_data['file_list'],
        batch_size, cm_train.meta_data['chan_mean'], aug_func=[reader_utils.image_flipping_hori],
        random=True, has_gt=True, gt_dim=1, include_gt=True, valid_mult=valid_mult).read_op()
    feature, label = reader_op

    with tf.Session() as sess:
        sess.run(train_init_op)
        img, lbl = sess.run([feature, label])

        from visualize import visualize_utils
        import numpy as np
        for i in range(5):
            visualize_utils.compare_two_figure((img[i, :, :, :]+cm_train.meta_data['chan_mean']).astype(np.uint8),
                                               lbl[i, :, :, 0])
