import os
import re
import scipy.misc
import numpy as np
import tensorflow as tf
from dataReader import patch_extractor


def read_images_labels_from_disk(input_queue, input_size):
    image_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])

    image = tf.image.decode_jpeg(image_contents, channels=3)
    # adhoc decoding for labels
    label = tf.image.decode_png(label_contents, channels=1)/255

    image = tf.image.resize_images(image, input_size)
    label = tf.image.resize_images(label, input_size)

    return image, label


def image_label_iterator(image_dir, batch_size, tile_dim, patch_size, overlap, padding=0):
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
            yield image_batch
    # yield the last chunck
    if cnt > 0:
        yield image_batch[:cnt, :, :, :]


class ImageLabelReader(object):
    def __init__(self, data_dir, input_size, coord, city_list, tile_list, data_list='data_list.txt', random=True, ds_name='inria'):
        self.original_dir = ''
        self.data_dir = data_dir
        self.data_list = data_list
        self.ds_name = ds_name
        self.input_size = input_size
        self.coord = coord
        self.city_list = city_list
        self.tile_list = tile_list
        self.image_list, self.label_list = self.read_image_label_list()
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.labels], shuffle=random)
        self.image, self.label = read_images_labels_from_disk(self.queue, self.input_size)

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


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    data_dir = r'/media/ei-edl01/user/bh163/data/iai/PS_(224, 224)-OL_0-AF_train_noaug'
    input_size = (224, 224)
    coord = tf.train.Coordinator()
    city_list = ['vienna', 'chicago']
    tile_list = ['6', '7', '8']
    reader = ImageLabelReader(data_dir, input_size, coord, city_list, tile_list)

    X_batch_op, y_batch_op = reader.dequeue(5)

    '''with tf.Session() as sess:
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        X_batch, y_batch = sess.run([X_batch_op, y_batch_op])
        coord.request_stop()
        coord.join(threads)

    print(reader.data_dir)

    print(X_batch.shape)
    print(y_batch.shape)

    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.imshow(X_batch[2, :, :, :])
    plt.subplot(122)
    plt.imshow(y_batch[2, :, :, 0])
    plt.show()'''

    image_dir = r'/media/ei-edl01/data/remote_sensing_data/inria/image'
    reader.set_original_image_label_dir(image_dir)
    iterator = reader.image_label_iterator('austin1.tif', 4, (5000, 5000), (500, 500), 250)
    for i in range(4):
        image = next(iterator)
        print(image.shape)

        import matplotlib.pyplot as plt
        plt.subplot(221)
        plt.imshow(image[0, :, :, :])
        plt.subplot(222)
        plt.imshow(image[1, :, :, :])
        plt.subplot(223)
        plt.imshow(image[2, :, :, :])
        plt.subplot(224)
        plt.imshow(image[3, :, :, :])
        plt.show()
