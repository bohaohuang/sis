import os
import numpy as np
import matplotlib.pyplot as plt
from dataReader import patch_extractor


def make_output_file(label, colormap):
    encode_func = np.vectorize(lambda x, y: y[x])
    return encode_func(label, colormap)


def decode_labels(label, num_images=10):
    n, h, w, c = label.shape
    outputs = np.zeros((n, h, w, 3), dtype=np.uint8)
    label_colors = {0: (255, 255, 255), 1: (0, 0, 255)}
    for i in range(n):
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
    if len(pred.shape) == 4:
        n, h, w, c = pred.shape
        outputs = np.zeros((n, h, w, 1), dtype=np.uint8)
        for i in range(n):
            outputs[i] = np.expand_dims(np.argmax(pred[i,:,:,:], axis=2), axis=2)
        return outputs
    elif len(pred.shape) == 3:
        outputs = np.argmax(pred, axis=2)
        return outputs


def image_summary(image, truth, prediction):
    truth_img = decode_labels(truth, 10)
    pred_labels = get_pred_labels(prediction)
    pred_img = decode_labels(pred_labels, 10)
    return np.concatenate([image, truth_img, pred_img], axis=2)


def get_output_label(result, image_dim, input_size, colormap):
    image_pred = patch_extractor.un_patchify(result, image_dim, input_size)
    labels_pred = get_pred_labels(image_pred)
    output_pred = make_output_file(labels_pred, colormap)
    return output_pred


def iou_metric(truth, pred, truth_val=255):
    truth = truth / truth_val
    pred = pred / truth_val
    truth = truth.flatten()
    pred = pred.flatten()
    intersect = truth*pred
    return sum(intersect == 1) / \
           (sum(truth == 1)+sum(pred == 1)-sum(intersect == 1))


def set_full_screen_img():
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()


def make_task_img_folder(parent_dir):
    task_fold_name = os.path.basename(os.getcwd())
    if not os.path.exists(os.path.join(parent_dir, task_fold_name)):
        os.makedirs(os.path.join(parent_dir, task_fold_name))
    return os.path.join(parent_dir, task_fold_name)
