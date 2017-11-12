import os
import re
import tensorflow as tf


def read_images_from_disk(input_queue, input_size):
    image_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])

    image = tf.image.decode_jpeg(image_contents, channels=3)
    # adhoc decoding for labels
    label = tf.image.decode_png(label_contents, channels=1)/255

    image = tf.image.resize_images(image, input_size)
    label = tf.image.resize_images(label, input_size)

    return image, label


class ImageReader(object):
    def __init__(self, data_dir, input_size, coord, city_list, tile_list, data_list='data_list.txt', random=True):
        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size
        self.coord = coord
        self.city_list = city_list
        self.tile_list = tile_list
        self.image_list, self.label_list = self.read_image_label_list()
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.labels], shuffle=random)
        self.image, self.label = read_images_from_disk(self.queue, self.input_size)

    def read_image_label_list(self):
        with open(os.path.join(self.data_dir, self.data_list), 'r') as file:
            files = file.readlines()
        image_list = []
        label_list = []
        for file in files:
            file_tuple = file.strip('\n').split(' ')
            city_name = re.findall('^[a-z\-]*', file_tuple[0])[0]
            tile_id = re.findall('[0-9]+(?=_img)', file_tuple[0])[0]
            if city_name in self.city_list and tile_id in self.tile_list:
                image_list.append(os.path.join(self.data_dir, file_tuple[0]))
                label_list.append(os.path.join(self.data_dir, file_tuple[1]))
        if len(image_list) == 0:
            raise ValueError
        return image_list, label_list

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
    reader = ImageReader(data_dir, input_size, coord, city_list, tile_list)

    X_batch_op, y_batch_op = reader.dequeue(5)

    with tf.Session() as sess:
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        X_batch, y_batch = sess.run([X_batch_op, y_batch_op])
        coord.request_stop()
        coord.join(threads)

    print(X_batch.shape)
    print(y_batch.shape)

    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.imshow(X_batch[2, :, :, :])
    plt.subplot(122)
    plt.imshow(y_batch[2, :, :, 0])
    plt.show()
