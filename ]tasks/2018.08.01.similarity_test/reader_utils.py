import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def check_reader(reader, bs, img_mean, iters=10):
    with tf.Session() as sess:
        for _ in range(iters):
            plt.figure(figsize=(12, 5))
            X_batch, y_batch = reader.readerAction(sess)
            for i in range(bs):
                plt.subplot(2, bs, i+1)
                plt.imshow((X_batch[i, :, :, :]+img_mean).astype(np.uint8))
                plt.axis('off')
                plt.subplot(2, bs, i+bs+1)
                plt.imshow(y_batch[i, :, :, 0])
                plt.axis('off')
            plt.tight_layout()
            plt.show()
