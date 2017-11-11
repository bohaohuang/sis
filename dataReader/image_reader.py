import os
import tensorflow as tf


def read_image_label_list(data_dir, data_list):
    with open(os.path.join(data_dir, data_list), 'r') as file:
        files = file.readlines()
    image_list = []
    label_list = []
    for file in files:
        file_tuple = file.strip('\n').split(' ')
        image_list.append(os.path.join(data_dir, file_tuple[0]))
        label_list.append(os.path.join(data_dir, file_tuple[1]))
    return image_list, label_list


def read_images_from_disk(input_queue, input_size):
    image_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])

    image = tf.image.decode_jpeg(image_contents, channels=3)
    # TODO fix this, combine with metadata information
    label = tf.image.decode_png(label_contents, channels=1)/255
    image = tf.image.resize_images(image, input_size)
    label = tf.image.resize_images(label, input_size)

    return image, label


class ImageReader(object):
    def __init__(self, data_dir, input_size, coord, data_list='data_list.txt', random=True):
        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size
        self.coord = coord
        self.image_list, self.label_list  = \
            read_image_label_list(self.data_dir, self.data_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.labels], shuffle=random)
        self.image, self.label = read_images_from_disk(self.queue, self.input_size)

    def dequeue(self, num_elements):
        image_batch, label_batch = tf.train.batch([self.image, self.label], num_elements)
        return image_batch, label_batch


if __name__ == '__main__':
    data_dir = r'/media/ei-edl01/user/bh163/data/iai/PS_(224, 224)-OL_0-AF_train'
    input_size = (224, 224)
    coord = tf.train.Coordinator()
    reader = ImageReader(data_dir, input_size, coord)

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
