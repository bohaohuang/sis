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


def pad_block(block, pad):
    padded_block = []
    _, _, c = block.shape
    for i in range(c):
        padded_block.append(np.pad(block[:, :, i],
                                   ((pad, pad), (pad, pad)), 'symmetric'))
    return np.dstack(padded_block)


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


def un_patchify(blocks, tile_dim, patch_size, overlap=0):
    _, _, _, c = blocks.shape
    image = np.zeros((tile_dim[0], tile_dim[1], c))
    max_h = tile_dim[0] - patch_size[0]
    max_w = tile_dim[1] - patch_size[1]
    h_step = np.ceil(tile_dim[0] / (patch_size[0] - overlap))
    w_step = np.ceil(tile_dim[1] / (patch_size[1] - overlap))
    patch_grid_h = np.floor(np.linspace(0, max_h, h_step)).astype(np.int32)
    patch_grid_w = np.floor(np.linspace(0, max_w, w_step)).astype(np.int32)

    cnt = 0
    for corner_h in patch_grid_h:
        for corner_w in patch_grid_w:
            cnt += 1
            image[corner_h:corner_h+patch_size[0], corner_w:corner_w+patch_size[1], :] += blocks[cnt-1, :, :, :]
    return image


def un_patchify_shrink(blocks, tile_dim, tile_dim_output, patch_size, patch_size_output, overlap=0):
    _, _, _, c = blocks.shape
    image = np.zeros((tile_dim_output[0], tile_dim_output[1], c))
    max_h = tile_dim[0] - patch_size[0]
    max_w = tile_dim[1] - patch_size[1]
    h_step = np.ceil(tile_dim[0] / (patch_size[0] - overlap))
    w_step = np.ceil(tile_dim[1] / (patch_size[1] - overlap))
    patch_grid_h = np.floor(np.linspace(0, max_h, h_step)).astype(np.int32)
    patch_grid_w = np.floor(np.linspace(0, max_w, w_step)).astype(np.int32)

    cnt = 0
    for corner_h in patch_grid_h:
        for corner_w in patch_grid_w:
            cnt += 1
            image[corner_h:corner_h+patch_size_output[0], corner_w:corner_w+patch_size_output[1], :] += blocks[cnt-1, :, :, :]
    return image


class PatchExtractorInria(object):
    def __init__(self, base_dir, file_list, patch_size, tile_dim, overlap=0, aug_funcs=None, appendix=''):
        self.base_dir = base_dir
        self.file_list = file_list
        self.patch_size = patch_size
        self.overlap = overlap
        self.aug_funcs = aug_funcs
        self.appendix = appendix
        self.tile_dim = tile_dim

        # make unique name
        func_name = ''
        if aug_funcs is not None:
            for func in aug_funcs:
                func_name += func.__name__.replace('_', '')

        self.name = 'PS_{}-OL_{}-AF_{}'.format(patch_size, overlap, func_name) + self.appendix

        # TODO check if extracted files already exists

    def save_img_label(self, patch, dest_dir, city_name, tile_id, cnt, appendix=None):
        patch_img, patch_label = patch[:, :, :3], patch[:, :, -1]
        '''patch_label = scipy.misc.toimage(patch_label,
                                         high=255,
                                         low=1,
                                         mode='I')'''
        cnt_str = '{0:05d}'.format(cnt)
        if appendix is None:
            image_name = '{}{}_img_{}.jpg'.format(city_name, tile_id, cnt_str)
            label_name = '{}{}_label_{}.png'.format(city_name, tile_id, cnt_str)
        else:
            image_name = '{}{}_img_{}_{}.jpg'.format(city_name, tile_id, appendix, cnt_str)
            label_name = '{}{}_label_{}_{}.png'.format(city_name, tile_id, appendix, cnt_str)
        file_name = os.path.join(dest_dir, self.name, image_name)
        scipy.misc.imsave(file_name, patch_img)
        file_name = os.path.join(dest_dir, self.name, label_name)
        scipy.misc.imsave(file_name, patch_label)
        #patch_label.save(file_name)
        with open(os.path.join(dest_dir, self.name, 'data_list.txt'), 'a') as file:
            file.write('{} {}\n'.format(image_name, label_name))

    def extract(self, dest_dir, pad=0):
        if not os.path.exists(os.path.join(dest_dir, self.name)):
            os.makedirs(os.path.join(dest_dir, self.name))
        else:
            print('{} already exists, only return the path'.format(
                os.path.join(dest_dir, self.name)
            ))
            return os.path.join(dest_dir, self.name)

        for file_tuple in tqdm(self.file_list):
            file_tuple = [os.path.join(self.base_dir, file) for file in file_tuple]
            block = get_block(file_tuple)
            block = pad_block(block, pad)
            self.tile_dim = block.shape[:2]
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
        return os.path.join(dest_dir, self.name)


class PatchExtractorUrbanMapper(object):
    def __init__(self, base_dir, file_list, patch_size, tile_dim, overlap=0, aug_funcs=None, appendix=''):
        self.base_dir = base_dir
        self.file_list = file_list
        self.patch_size = patch_size
        self.overlap = overlap
        self.aug_funcs = aug_funcs
        self.appendix = appendix
        self.tile_dim = tile_dim

        # make unique name
        func_name = ''
        if aug_funcs is not None:
            for func in aug_funcs:
                func_name += func.__name__.replace('_', '')

        self.name = 'PS_{}-OL_{}-AF_{}'.format(patch_size, overlap, func_name) + self.appendix

        # TODO check if extracted files already exists

    def save_img_label(self, patch, dest_dir, city_name, tile_id, cnt, appendix=None, label_min=0, label_max=1):
        patch_img, patch_label = patch[:, :, :3], patch[:, :, -1]
        '''patch_label = scipy.misc.toimage(patch_label,
                                         high=label_max,
                                         low=label_min,
                                         mode='I')'''
        cnt_str = '{0:05d}'.format(cnt)
        if appendix is None:
            image_name = '{}{}_img_{}.jpg'.format(city_name, tile_id, cnt_str)
            label_name = '{}{}_label_{}.png'.format(city_name, tile_id, cnt_str)
        else:
            image_name = '{}{}_img_{}_{}.jpg'.format(city_name, tile_id, appendix, cnt_str)
            label_name = '{}{}_label_{}_{}.png'.format(city_name, tile_id, appendix, cnt_str)
        file_name = os.path.join(dest_dir, self.name, image_name)
        scipy.misc.imsave(file_name, patch_img)
        file_name = os.path.join(dest_dir, self.name, label_name)
        scipy.misc.imsave(file_name, patch_label)
        #patch_label.save(file_name)
        with open(os.path.join(dest_dir, self.name, 'data_list.txt'), 'a') as file:
            file.write('{} {}\n'.format(image_name, label_name))

    def extract(self, dest_dir):
        if not os.path.exists(os.path.join(dest_dir, self.name)):
            os.makedirs(os.path.join(dest_dir, self.name))
        else:
            print('{} already exists, only return the path'.format(
                os.path.join(dest_dir, self.name)
            ))
            return os.path.join(dest_dir, self.name)

        for file_tuple in tqdm(self.file_list):
            file_tuple = [os.path.join(self.base_dir, file) for file in file_tuple]
            block = get_block(file_tuple)
            city_name = os.path.basename(file_tuple[0])[:3]
            tile_id = os.path.basename(file_tuple[0])[9:12]

            label_max = np.max(block[:, :, -1].flatten())
            label_min = np.min(block[:, :, -1].flatten())

            for cnt, patch in enumerate(patchify(block, self.tile_dim,
                                                 self.patch_size, overlap=self.overlap)):
                self.save_img_label(patch, dest_dir, city_name, tile_id, cnt,
                                    label_min=label_min, label_max=label_max)

                if self.aug_funcs is not None:
                    for func in self.aug_funcs:
                        patch_aug = func(patch)
                        self.save_img_label(patch_aug, dest_dir, city_name, tile_id, cnt,
                                            appendix=func.__name__.replace('_', ''))
        return os.path.join(dest_dir, self.name)


class PatchExtractorUrbanMapperHeight(PatchExtractorUrbanMapper):
    def save_img_label(self, patch, dest_dir, city_name, tile_id, cnt, appendix=None, label_min=0, label_max=255):
        assert patch.shape[-1] == 6
        patch_img, patch_dsm, patch_dtm, patch_label = patch[:, :, :3], \
                                                       patch[:, :, 3], \
                                                       patch[:, :, 4], \
                                                       patch[:, :, -1]
        '''patch_label = scipy.misc.toimage(patch_label,
                                         high=label_max,
                                         low=label_min,
                                         mode='I')'''
        cnt_str = '{0:05d}'.format(cnt)
        if appendix is None:
            image_name = '{}{}_img_{}.jpg'.format(city_name, tile_id, cnt_str)
            dsm_name = '{}{}_dsm_{}.npy'.format(city_name, tile_id, cnt_str)
            dtm_name = '{}{}_dtm_{}.npy'.format(city_name, tile_id, cnt_str)
            label_name = '{}{}_label_{}.png'.format(city_name, tile_id, cnt_str)
        else:
            image_name = '{}{}_img_{}_{}.jpg'.format(city_name, tile_id, appendix, cnt_str)
            dsm_name = '{}{}_dsm_{}_{}.npy'.format(city_name, tile_id, appendix, cnt_str)
            dtm_name = '{}{}_dtm_{}_{}.npy'.format(city_name, tile_id, appendix, cnt_str)
            label_name = '{}{}_label_{}_{}.png'.format(city_name, tile_id, appendix, cnt_str)
        file_name = os.path.join(dest_dir, self.name, image_name)
        scipy.misc.imsave(file_name, patch_img)
        file_name = os.path.join(dest_dir, self.name, dsm_name)
        #scipy.misc.imsave(file_name, patch_dsm)
        np.save(file_name, patch_dsm)
        file_name = os.path.join(dest_dir, self.name, dtm_name)
        #scipy.misc.imsave(file_name, patch_dtm)
        np.save(file_name, patch_dtm)
        file_name = os.path.join(dest_dir, self.name, label_name)
        scipy.misc.imsave(file_name, patch_label)
        #patch_label.save(file_name)
        with open(os.path.join(dest_dir, self.name, 'data_list.txt'), 'a') as file:
            file.write('{} {} {} {}\n'.format(image_name, dsm_name, dtm_name, label_name))


class PatchExtractorUrbanMapperHeightFmap(PatchExtractorUrbanMapper):
    def save_img_label(self, patch, dest_dir, city_name, tile_id, cnt, appendix=None, label_min=0, label_max=1):
        assert patch.shape[-1] == 9
        patch_img, patch_dsm, patch_dtm, patch_f, patch_label = patch[:, :, :3], \
                                                       patch[:, :, 3], \
                                                       patch[:, :, 4], \
                                                       patch[:, :, 5:8], \
                                                       patch[:, :, -1]
        patch_label = scipy.misc.toimage(patch_label,
                                         high=label_max,
                                         low=label_min,
                                         mode='I')
        cnt_str = '{0:05d}'.format(cnt)
        if appendix is None:
            image_name = '{}{}_img_{}.jpg'.format(city_name, tile_id, cnt_str)
            dsm_name = '{}{}_dsm_{}.npy'.format(city_name, tile_id, cnt_str)
            dtm_name = '{}{}_dtm_{}.npy'.format(city_name, tile_id, cnt_str)
            f_name = '{}{}_f_{}.png'.format(city_name, tile_id, cnt_str)
            label_name = '{}{}_label_{}.png'.format(city_name, tile_id, cnt_str)
        else:
            image_name = '{}{}_img_{}_{}.jpg'.format(city_name, tile_id, appendix, cnt_str)
            dsm_name = '{}{}_dsm_{}_{}.npy'.format(city_name, tile_id, appendix, cnt_str)
            dtm_name = '{}{}_dtm_{}_{}.npy'.format(city_name, tile_id, appendix, cnt_str)
            f_name = '{}{}_f_{}_{}.png'.format(city_name, tile_id, appendix, cnt_str)
            label_name = '{}{}_label_{}_{}.png'.format(city_name, tile_id, appendix, cnt_str)
        file_name = os.path.join(dest_dir, self.name, image_name)
        scipy.misc.imsave(file_name, patch_img)
        file_name = os.path.join(dest_dir, self.name, dsm_name)
        #scipy.misc.imsave(file_name, patch_dsm)
        np.save(file_name, patch_dsm)
        file_name = os.path.join(dest_dir, self.name, dtm_name)
        #scipy.misc.imsave(file_name, patch_dtm)
        np.save(file_name, patch_dtm)
        file_name = os.path.join(dest_dir, self.name, f_name)
        scipy.misc.imsave(file_name, patch_f)
        file_name = os.path.join(dest_dir, self.name, label_name)
        #scipy.misc.imsave(file_name, patch_label)
        patch_label.save(file_name)
        with open(os.path.join(dest_dir, self.name, 'data_list.txt'), 'a') as file:
            file.write('{} {} {} {} {}\n'.format(image_name, dsm_name, dtm_name, f_name, label_name))


class PatchExtractorUrbanMapperHeightWeight(PatchExtractorUrbanMapper):
    def save_img_label(self, patch, dest_dir, city_name, tile_id, cnt, appendix=None, label_min=0, label_max=1):
        assert patch.shape[-1] == 7
        patch_img, patch_dsm, patch_dtm, patch_f, patch_label = patch[:, :, :3], \
                                                       patch[:, :, 3], \
                                                       patch[:, :, 4], \
                                                       patch[:, :, 5], \
                                                       patch[:, :, -1]
        patch_label = scipy.misc.toimage(patch_label,
                                         high=label_max,
                                         low=label_min,
                                         mode='I')
        cnt_str = '{0:05d}'.format(cnt)
        if appendix is None:
            image_name = '{}{}_img_{}.jpg'.format(city_name, tile_id, cnt_str)
            dsm_name = '{}{}_dsm_{}.npy'.format(city_name, tile_id, cnt_str)
            dtm_name = '{}{}_dtm_{}.npy'.format(city_name, tile_id, cnt_str)
            f_name = '{}{}_w_{}.png'.format(city_name, tile_id, cnt_str)
            label_name = '{}{}_label_{}.png'.format(city_name, tile_id, cnt_str)
        else:
            image_name = '{}{}_img_{}_{}.jpg'.format(city_name, tile_id, appendix, cnt_str)
            dsm_name = '{}{}_dsm_{}_{}.npy'.format(city_name, tile_id, appendix, cnt_str)
            dtm_name = '{}{}_dtm_{}_{}.npy'.format(city_name, tile_id, appendix, cnt_str)
            f_name = '{}{}_w_{}_{}.png'.format(city_name, tile_id, appendix, cnt_str)
            label_name = '{}{}_label_{}_{}.png'.format(city_name, tile_id, appendix, cnt_str)
        file_name = os.path.join(dest_dir, self.name, image_name)
        scipy.misc.imsave(file_name, patch_img)
        file_name = os.path.join(dest_dir, self.name, dsm_name)
        #scipy.misc.imsave(file_name, patch_dsm)
        np.save(file_name, patch_dsm)
        file_name = os.path.join(dest_dir, self.name, dtm_name)
        #scipy.misc.imsave(file_name, patch_dtm)
        np.save(file_name, patch_dtm)
        file_name = os.path.join(dest_dir, self.name, f_name)
        scipy.misc.imsave(file_name, patch_f)
        file_name = os.path.join(dest_dir, self.name, label_name)
        #scipy.misc.imsave(file_name, patch_label)
        patch_label.save(file_name)
        with open(os.path.join(dest_dir, self.name, 'data_list.txt'), 'a') as file:
            file.write('{} {} {} {} {}\n'.format(image_name, dsm_name, dtm_name, f_name, label_name))


if __name__ == '__main__':
    '''from rsrClassData import rsrClassData
    Data = rsrClassData(r'/media/ei-edl01/data/remote_sensing_data')

    from random import shuffle

    (collect_files_train, meta_train) = Data.getCollectionByName('dcc_inria_train')
    #shuffle(collect_files_train)
    pe = PatchExtractorInria(r'/media/ei-edl01/data/remote_sensing_data',
                             collect_files_train, patch_size=(224, 224),
                             tile_dim=(5000, 5000), appendix='train_noaug_dcc')
    pe.extract(r'/media/ei-edl01/user/bh163/data/iai')

    (collect_files_valid, meta_valid) = Data.getCollectionByName('dcc_inria_valid')
    #shuffle(collect_files_valid)
    pe = PatchExtractorInria(r'/media/ei-edl01/data/remote_sensing_data',
                             collect_files_valid, patch_size=(224, 224),
                             tile_dim=(5000, 5000), appendix='valid_noaug_dcc')
    pe.extract(r'/media/ei-edl01/user/bh163/data/iai')'''

    img = scipy.misc.imread(r'/media/ei-edl01/data/remote_sensing_data/inria/image/austin1.tif')
    img = pad_block(img, 0)

    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()
