import os
import re
import scipy.misc
import numpy as np
import tensorflow as tf
from dataReader import patch_extractor


def block_flipping(block):
    return tf.image.random_flip_left_right(tf.image.random_flip_up_down(block))


def block_rotating(block):
    random_times = tf.to_int32(tf.random_uniform([1], minval=0, maxval=4))[0]
    return tf.image.rot90(block, random_times)


def image_flipping(img, label):
    """
    randomly flips images left-right and up-down
    :param img:
    :param label:
    :return:flipped images
    """
    label = tf.cast(label, dtype=tf.float32)
    temp = tf.concat([img, label], axis=2)
    temp_flipped = block_flipping(temp)
    img = tf.slice(temp_flipped, [0, 0, 0], [-1, -1, 3])
    label = tf.slice(temp_flipped, [0, 0, 3], [-1, -1, 1])
    return img, label


def image_rotating(img, label):
    """
    randomly rotate images by 0/90/180/270 degrees
    :param img:
    :param label:
    :return:rotated images
    """
    temp = tf.concat([img, label], axis=2)
    temp_rotated = block_rotating(temp)
    img = tf.slice(temp_rotated, [0, 0, 0], [-1, -1, 3])
    label = tf.slice(temp_rotated, [0, 0, 3], [-1, -1, 1])
    return img, label


def read_images_labels_from_disk(input_queue, input_size, data_aug='', image_mean=0):
    image_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])

    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.cast(image, dtype=tf.float32)
    image -= image_mean
    # adhoc decoding for labels
    label = tf.image.decode_png(label_contents, channels=1)/255

    image = tf.image.resize_images(image, input_size)
    label = tf.image.resize_images(label, input_size)

    if 'flip' in data_aug:
        image, label = image_flipping(image, label)
    if 'rotate' in data_aug:
        image, label = image_rotating(image, label)

    # this is necessary for dcc, which uses older version of tf
    image = tf.image.resize_images(image, input_size)
    label = tf.image.resize_images(label, input_size)

    return image, label


def read_images_heights_labels_from_disk(input_queue, input_size, data_aug=''):
    image_contents = tf.read_file(input_queue[0])
    dsm_contents = tf.read_file(input_queue[1])
    dtm_contents = tf.read_file(input_queue[2])
    label_contents = tf.read_file(input_queue[3])

    image = tf.image.decode_jpeg(image_contents, channels=3)
    dsm = tf.image.decode_png(dsm_contents, channels=1)
    dtm = tf.image.decode_png(dtm_contents, channels=1)
    # adhoc decoding for labels
    label = tf.image.decode_png(label_contents, channels=1)/255

    image = tf.image.resize_images(image, input_size)
    dsm = tf.image.resize_images(dsm, input_size)
    dtm = tf.image.resize_images(dtm, input_size)
    label = tf.image.resize_images(label, input_size)

    if 'flip' in data_aug:
        label = tf.cast(label, dtype=tf.float32)
        temp = tf.concat([image, dsm, dtm, label], axis=2)
        temp_flipped = block_flipping(temp)
        image = tf.slice(temp_flipped, [0, 0, 0], [-1, -1, 3])
        dsm = tf.slice(temp_flipped, [0, 0, 3], [-1, -1, 1])
        dtm = tf.slice(temp_flipped, [0, 0, 4], [-1, -1, 1])
        label = tf.slice(temp_flipped, [0, 0, 5], [-1, -1, 1])
    if 'rotate' in data_aug:
        temp = tf.concat([image, dsm, dtm, label], axis=2)
        temp_rotated = block_rotating(temp)
        image = tf.slice(temp_rotated, [0, 0, 0], [-1, -1, 3])
        dsm = tf.slice(temp_rotated, [0, 0, 3], [-1, -1, 1])
        dtm = tf.slice(temp_rotated, [0, 0, 4], [-1, -1, 1])
        label = tf.slice(temp_rotated, [0, 0, 5], [-1, -1, 1])

    # this is necessary for dcc, which uses older version of tf
    image = tf.image.resize_images(image, input_size)
    dsm = tf.image.resize_images(dsm, input_size)
    dtm = tf.image.resize_images(dtm, input_size)
    label = tf.image.resize_images(label, input_size)

    return image, dsm, dtm, label


def image_label_iterator(image_dir, batch_size, tile_dim, patch_size, overlap, padding=0, image_mean=0):
    # this is a iterator for test
    block = scipy.misc.imread(image_dir)
    if padding > 0:
        block = patch_extractor.pad_block(block, padding)
        tile_dim = (tile_dim[0]+padding*2, tile_dim[1]+padding*2)
    cnt = 0
    image_batch = np.zeros((batch_size, patch_size[0], patch_size[1], 3))
    for patch in patch_extractor.patchify(block, tile_dim, patch_size, overlap=overlap):
        cnt += 1
        image_batch[cnt-1, :, :, :] = patch
        if cnt == batch_size:
            cnt = 0
            yield image_batch - image_mean
    # yield the last chunck
    if cnt > 0:
        yield image_batch[:cnt, :, :, :] - image_mean


def image_height_label_iterator(image_dir, batch_size, tile_dim, patch_size, overlap, padding=0, height_mode='subtract'):
    # this is a iterator for test
    block = []
    for file in image_dir:
        if file[-3:] != 'npy':
            img = scipy.misc.imread(file)
        else:
            img = np.load(file)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        block.append(img)
    if height_mode == 'all':
        block = np.dstack(block)
        image_batch = np.zeros((batch_size, patch_size[0], patch_size[1], 5))
    elif height_mode == 'subtract':
        res = 5*(block[1]-block[2]) + 50
        #res[np.where(res < -1)] = -1
        #res = 35 * np.log(res + 2) + 40

        block = np.dstack([block[0], res])
        image_batch = np.zeros((batch_size, patch_size[0], patch_size[1], 4))
    else:
        block = np.dstack([block[0], block[1], block[2], block[1]-block[2]])
        image_batch = np.zeros((batch_size, patch_size[0], patch_size[1], 6))

    if padding > 0:
        block = patch_extractor.pad_block(block, padding)
        tile_dim = (tile_dim[0]+padding*2, tile_dim[1]+padding*2)
    cnt = 0
    for patch in patch_extractor.patchify(block, tile_dim, patch_size, overlap=overlap):
        cnt += 1
        image_batch[cnt-1, :, :, :] = patch
        if cnt == batch_size:
            cnt = 0
            yield image_batch
    # yield the last chunck
    if cnt > 0:
        yield image_batch[:cnt, :, :, :]


def read_batch_from_list(file_list, batch_idx):
    block = []
    for idx in batch_idx:
        if file_list[idx][-3:] != 'npy':
            img = scipy.misc.imread(file_list[idx])
        else:
            img = np.load(file_list[idx])
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        block.append(img)
    return np.stack(block, axis=0)


class ImageLabelReader(object):
    def __init__(self, data_dir, input_size, coord, city_list, tile_list,
                 data_list='data_list.txt', random=True, ds_name='inria',
                 data_aug='', image_mean=0):
        self.original_dir = ''
        self.data_dir = data_dir
        self.data_list = data_list
        self.ds_name = ds_name
        self.data_aug = data_aug
        self.image_mean = image_mean
        self.input_size = input_size
        self.coord = coord
        self.city_list = city_list
        self.tile_list = tile_list
        self.image_list, self.label_list = self.read_image_label_list()
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.labels], shuffle=random)
        self.image, self.label = read_images_labels_from_disk(self.queue, self.input_size, self.data_aug, self.image_mean)

    def read_image_label_list(self):
        with open(os.path.join(self.data_dir, self.data_list), 'r') as file:
            files = file.readlines()
        image_list = []
        label_list = []
        for file in files:
            file_tuple = file.strip('\n').split(' ')
            if self.ds_name == 'inria':
                city_name = re.findall('^[a-z\-]*', file_tuple[0])[0]
                tile_id = re.findall('[0-9]+(?=_img)', file_tuple[0])[0]
            else:
                city_name = file_tuple[0][:3]
                tile_id = file_tuple[0][3:6].lstrip('0')
            if city_name in self.city_list and tile_id in self.tile_list:
                image_list.append(os.path.join(self.data_dir, file_tuple[0]))
                label_list.append(os.path.join(self.data_dir, file_tuple[1]))
        if len(image_list) == 0:
            raise ValueError
        return image_list, label_list

    def set_original_image_label_dir(self, origin_dir):
        self.original_dir = origin_dir

    def dequeue(self, num_elements):
        image_batch, label_batch = tf.train.batch([self.image, self.label], num_elements)
        return image_batch, label_batch


class ImageLabelReaderHeight(object):
    def __init__(self, data_dir, input_size, coord, city_list, tile_list,
                 data_list='data_list.txt', random=True, ds_name='inria',
                 data_aug='', height_mode='all'):
        self.original_dir = ''
        self.data_dir = data_dir
        self.data_list = data_list
        self.ds_name = ds_name
        self.data_aug = data_aug
        self.height_mode = height_mode
        self.input_size = input_size
        self.coord = coord
        self.city_list = city_list
        self.tile_list = tile_list
        self.image_list, self.dsm_list, self.dtm_list, self.label_list = \
            self.read_image_label_list()
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.dsms = tf.convert_to_tensor(self.dsm_list, dtype=tf.string)
        self.dtms = tf.convert_to_tensor(self.dtm_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.dsms, self.dtms, self.labels],
                                                   shuffle=random)
        self.image, self.dsm, self.dtm, self.label = \
            read_images_heights_labels_from_disk(self.queue, self.input_size, self.data_aug)

    def read_image_label_list(self):
        with open(os.path.join(self.data_dir, self.data_list), 'r') as file:
            files = file.readlines()
        image_list = []
        dsm_list = []
        dtm_list = []
        label_list = []
        for file in files:
            file_tuple = file.strip('\n').split(' ')
            if self.ds_name == 'inria':
                city_name = re.findall('^[a-z\-]*', file_tuple[0])[0]
                tile_id = re.findall('[0-9]+(?=_img)', file_tuple[0])[0]
            else:
                city_name = file_tuple[0][:3]
                tile_id = file_tuple[0][3:6].lstrip('0')
            if city_name in self.city_list and tile_id in self.tile_list:
                image_list.append(os.path.join(self.data_dir, file_tuple[0]))
                dsm_list.append(os.path.join(self.data_dir, file_tuple[1]))
                dtm_list.append(os.path.join(self.data_dir, file_tuple[2]))
                label_list.append(os.path.join(self.data_dir, file_tuple[3]))
        if len(image_list) == 0:
            raise ValueError
        return image_list, dsm_list, dtm_list, label_list

    def set_original_image_label_dir(self, origin_dir):
        self.original_dir = origin_dir

    def dequeue(self, num_elements):
        image_batch, dsm_batch, dtm_batch, label_batch = \
            tf.train.batch([self.image, self.dsm, self.dtm, self.label], num_elements)
        if self.height_mode == 'all':
            return tf.concat([image_batch, dsm_batch, dtm_batch], axis=3), label_batch
        elif self.height_mode == 'subtract':
            return tf.concat([image_batch, dsm_batch-dtm_batch], axis=3), label_batch
        elif self.height_mode == 'subtract_all':
            return tf.concat([image_batch, dsm_batch, dtm_batch,
                              dsm_batch-dtm_batch], axis=3), label_batch

    def image_height_label_iterator(self, batch_size):
        assert len(self.image_list) == len(self.dsm_list) == len(self.dtm_list) == len(self.label_list)
        image_num = len(self.image_list)
        while True:
            idx = np.random.permutation(image_num)
            for i in range(0, image_num, batch_size):
                batch_idx = idx[i:i+batch_size]
                if len(batch_idx) < batch_size:
                    continue
                image_batch = read_batch_from_list(self.image_list, batch_idx)
                dsm_batch = read_batch_from_list(self.dsm_list, batch_idx)
                dtm_batch = read_batch_from_list(self.dtm_list, batch_idx)
                label_batch = read_batch_from_list(self.label_list, batch_idx)
                if self.height_mode == 'all':
                    yield np.concatenate([image_batch, dsm_batch, dtm_batch], axis=3), label_batch
                elif self.height_mode == 'subtract':
                    res = 5 * (dsm_batch - dtm_batch) + 50
                    #res[np.where(res < -1)] = -1
                    #res = 35 * np.log(res + 2) + 40
                    yield np.concatenate([image_batch, res], axis=3), label_batch
                elif self.height_mode == 'subtract_all':
                    yield np.concatenate([image_batch, dsm_batch, dtm_batch, dsm_batch-dtm_batch], axis=3), label_batch


if __name__ == '__main__':
    '''os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    data_dir = r'/home/lab/Documents/bohao/data/urban_mapper/PS_(572, 572)-OL_0-AF_valid_augfr_um'
    input_size = (572, 572)
    coord = tf.train.Coordinator()
    city_list = ['JAX', 'TAM']
    tile_list = ['6', '7', '8']
    reader = ImageLabelReaderHeight(data_dir, input_size, coord, city_list, tile_list, ds_name='urban_mapper')
    iterator = reader.image_height_label_iterator(5)
    x_batch, label_batch = next(iterator)
    print(x_batch.shape)
    print(label_batch.shape)'''

    data_dir = r'/media/ei-edl01/data/remote_sensing_data/urban_mapper/image'
    tile_id = 4
    img_file = 'JAX_Tile_{:03d}_RGB.tif'.format(tile_id)
    dsm_file = 'JAX_Tile_{:03d}_DSM.tif'.format(tile_id)
    dtm_file = 'JAX_Tile_{:03d}_DTM.tif'.format(tile_id)

    img = scipy.misc.imread(os.path.join(data_dir, img_file))
    dsm = scipy.misc.imread(os.path.join(data_dir, dsm_file))
    dtm = scipy.misc.imread(os.path.join(data_dir, dtm_file))

    res = 5*(dsm-dtm)+50
    '''res[np.where(res<-2)] = -2
    res = 35 * np.sqrt(res + 2) + 20'''
    print(res.shape)

    import matplotlib.pyplot as plt
    plt.subplot(211)
    plt.hist(img.flatten())
    plt.xlim(0, 255)
    plt.subplot(212)
    plt.hist(res.flatten())
    plt.xlim(0, 255)
    plt.show()
