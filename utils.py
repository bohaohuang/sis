import numpy as np


def decode_labels(label, num_images=10):
    n, h, w, c = label.shape
    outputs = np.zeros((n, h, w, 3), dtype=np.uint8)
    label_colors = {0: (255, 255, 255), 1: (0, 0, 255)}
    for i in range(num_images):
        pixels = np.zeros((h, w, 3), dtype=np.uint8)
        for j in range(h):
            for k in range(w):
                pixels[j, k] = label_colors[np.int(label[i, j, k, 0])]
        outputs[i] = pixels
    return outputs


def decode_labels_binary(label, colormap, num_images=None):
    label_binary = label[:, :, :, 0]
    n, h, w = label_binary.shape
    if num_images is not None:
        n = num_images
    outputs = np.zeros((n, h, w), dtype=np.uint8)
    encode_func = np.vectorize(lambda x, y: y[x])

    for i in range(n):
        outputs[i, :, :] = encode_func(label_binary[i, :, :], colormap)

    return outputs


def get_pred_labels(pred):
    n, h, w, c = pred.shape
    outputs = np.zeros((n, h, w, 1), dtype=np.uint8)
    for i in range(n):
        outputs[i] = np.expand_dims(np.argmax(pred[i,:,:,:], axis=2), axis=2)
    return outputs


def image_summary(image, truth, prediction):
    truth_img = decode_labels(truth, 10)
    pred_labels = get_pred_labels(prediction)
    pred_img = decode_labels(pred_labels, 10)
    return np.concatenate([image, truth_img, pred_img], axis=2)