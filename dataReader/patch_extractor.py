import os
import re
import scipy.misc
import numpy as np
from tqdm import tqdm


def rotate_90(block):
    return np.rot90(block, 1, (0, 1))


def rotate_180(block):
    return np.rot90(block, 3, (0, 1))


def flip_horizontal(block):
    return block[:, ::-1, :]


def flip_vertical(block):
    return block[::-1, :, :]


def get_block(file_tuple):
    block = []
    for file in file_tuple:
        file_slice = scipy.misc.imread(file)
        if len(file_slice.shape) == 2:
            file_slice = np.expand_dims(file_slice, axis=2)
        block.append(file_slice)
    return np.dstack(block)


def crop_image(block, size, corner):
    return block[corner[0]:corner[0]+size[0],corner[1]:corner[1]+size[1],:]


def patchify(block, tile_dim, patch_size, overlap=0):
    max_h = tile_dim[0] - patch_size[0]
    max_w = tile_dim[1] - patch_size[1]
    h_step = np.ceil(tile_dim[0] / (patch_size[0] - overlap))
    w_step = np.ceil(tile_dim[1] / (patch_size[1] - overlap))
    patch_grid_h = np.floor(np.linspace(0, max_h, h_step)).astype(np.int32)
    patch_grid_w = np.floor(np.linspace(0, max_w, w_step)).astype(np.int32)
    for corner_h in patch_grid_h:
        for corner_w in patch_grid_w:
            yield crop_image(block, patch_size, (corner_h, corner_w))


class PatchExtractorInria(object):
    def __init__(self, file_list, patch_size, overlap=0, aug_funcs=None, appendix=''):
        self.file_list = file_list
        self.patch_size = patch_size
        self.overlap = overlap
        self.aug_funcs = aug_funcs
        self.appendix = appendix

        # load first tuple to see the dimension
        channel_recorder = []
        for file in self.file_list[0]:
            file_slice = scipy.misc.imread(file)
            self.tile_dim = file_slice.shape[:2]
            if len(file_slice.shape) == 2:
                file_slice = np.expand_dims(file_slice, axis=2)
            channel_recorder.append(file_slice.shape[2])
        self.channel_number = channel_recorder

        # make unique name
        func_name = ''
        if aug_funcs is not None:
            for func in aug_funcs:
                func_name += func.__name__.replace('_', '')

        self.name = 'PS_{}-OL_{}-AF_{}'.format(patch_size, overlap, func_name) + self.appendix

        # TODO check if extracted files already exists

    def save_img_label(self, patch, dest_dir, city_name, tile_id, cnt, appendix=None):
        patch_img, patch_label = patch[:, :, :3], patch[:, :, -1]
        cnt_str = '{0:05d}'.format(cnt)
        if appendix is None:
            image_name = '{}{}_img_{}.jpg'.format(city_name, tile_id, cnt_str)
            label_name = '{}{}_label_{}.jpg'.format(city_name, tile_id, cnt_str)
        else:
            image_name = '{}{}_img_{}_{}.jpg'.format(city_name, tile_id, appendix, cnt_str)
            label_name = '{}{}_label_{}_{}.jpg'.format(city_name, tile_id, appendix, cnt_str)
        file_name = os.path.join(dest_dir, self.name, image_name)
        scipy.misc.imsave(file_name, patch_img)
        file_name = os.path.join(dest_dir, self.name, label_name)
        scipy.misc.imsave(file_name, patch_label)
        with open(os.path.join(dest_dir, self.name, 'data_list.txt'), 'a') as file:
            file.write('{} {}\n'.format(image_name, label_name))

    def extract(self, dest_dir):
        if not os.path.exists(os.path.join(dest_dir, self.name)):
            os.makedirs(os.path.join(dest_dir, self.name))

        for file_tuple in tqdm(self.file_list):
            block = get_block(file_tuple)
            city_name = re.findall('[a-z\-]*(?=[0-9]+\.)', file_tuple[0])[0]
            tile_id = re.findall('[0-9]+(?=\.tif)', file_tuple[0])[0]

            for cnt, patch in enumerate(patchify(block, self.tile_dim,
                                                 self.patch_size, overlap=self.overlap)):
                self.save_img_label(patch, dest_dir, city_name, tile_id, cnt)

                if self.aug_funcs is not None:
                    for func in self.aug_funcs:
                        patch_aug = func(patch)
                        self.save_img_label(patch_aug, dest_dir, city_name, tile_id, cnt,
                                            appendix=func.__name__.replace('_', ''))


if __name__ == '__main__':
    from rsrClassData import rsrClassData
    Data = rsrClassData(r'/media/ei-edl01/data/remote_sensing_data')

    from random import shuffle

    (collect_files_train, meta_train) = Data.getCollectionByName('bohao_inria_train')
    shuffle(collect_files_train)
    pe = PatchExtractorInria(collect_files_train[:10], patch_size=(224, 224), appendix='train_toy')
    pe.extract(r'/media/ei-edl01/user/bh163/data/iai')

    (collect_files_valid, meta_valid) = Data.getCollectionByName('bohao_inria_valid')
    shuffle(collect_files_valid)
    pe = PatchExtractorInria(collect_files_valid[:10], patch_size=(224, 224), appendix='valid_toy')
    pe.extract(r'/media/ei-edl01/user/bh163/data/iai')
