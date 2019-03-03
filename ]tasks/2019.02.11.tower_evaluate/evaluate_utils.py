import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from object_detection.utils import ops as utils_ops


def extract_grids(img, patch_size_h, patch_size_w):
    """
    Get patch grids for given image
    :param img:
    :param patch_size_h:
    :param patch_size_w:
    :return:
    """
    h, w, _ = img.shape
    if h % patch_size_h == 0:
        h_steps = np.arange(0, h, patch_size_h).astype(int)
    else:
        h_steps = np.append(np.arange(0, h-patch_size_h, patch_size_h).astype(int), h-patch_size_h)
    if w % patch_size_w == 0:
        w_steps = np.arange(0, w, patch_size_w).astype(int)
    else:
        w_steps = np.append(np.arange(0, w-patch_size_w, patch_size_w).astype(int), w-patch_size_w)
    return h_steps, w_steps


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def get_predict_info(output_dict, patch_size, offset, category_index=('T',), th=0.5):
    for db, dc, ds in zip(output_dict['detection_boxes'], output_dict['detection_classes'],
                          output_dict['detection_scores']):
        left = int(db[1] * patch_size[1])
        top = int(db[0] * patch_size[0])
        right = int(db[3] * patch_size[1])
        bottom = int(db[2] * patch_size[0])
        confidence = ds
        class_name = category_index[dc-1]

        left += offset[1]
        top += offset[0]
        right += offset[1]
        bottom += offset[0]

        if confidence > th:
            yield left, top, right, bottom, confidence, class_name


def get_center_point(ymin, xmin, ymax, xmax):
    return ((ymin+ymax)/2, (xmin+xmax)/2)


def parse_result(output_line):
    info = output_line.strip().split(' ')
    class_name = str(info[0])
    confidence = float(info[1])
    left = int(info[2])
    top = int(info[3])
    right = int(info[4])
    bottom = int(info[5])
    return class_name, confidence, left, top, right, bottom


def local_maxima_suppression(preds, th=20):
    center_list = []
    conf_list = []
    for line in preds:
        class_name, confidence, left, top, right, bottom = parse_result(line)
        y, x = get_center_point(top, left, bottom, right)
        center_list.append([y, x])
        conf_list.append(confidence)

    center_list = np.array(center_list)
    n_samples = center_list.shape[0]
    dist_mat = np.inf * np.ones((n_samples, n_samples))
    merge_list = []
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            dist_mat[i, j] = np.sqrt(np.sum(np.square(center_list[i, :] - center_list[j, :])))
        merge_dist = dist_mat[i, :]
        merge_candidate = np.where(merge_dist < th)
        if merge_candidate[0].shape[0] > 0:
            merge_list.append({i: merge_candidate[0].tolist()})

    remove_idx = []
    for merge_item in merge_list:
        center_points = []
        conf_idx = []
        for k in merge_item.keys():
            center_points.append(center_list[k, :])
            conf_idx.append(k)
            for v in merge_item[k]:
                center_points.append(center_list[v, :])
                conf_idx.append(v)
        center_points = np.mean(center_points, axis=0)

        confs = [conf_list[a] for a in conf_idx]
        keep_idx = int(np.argmax(confs))
        remove_idx.extend([conf_idx[a] for a in range(len(confs)) if a != keep_idx])
        center_list[keep_idx, :] = center_points
        conf_list[keep_idx] = max(confs)

    center_list = [center_list[a] for a in range(n_samples) if not a in remove_idx]
    conf_list = [conf_list[a] for a in range(n_samples) if not a in remove_idx]

    return center_list, conf_list, remove_idx


def overlay_rectangle(img, bbox, color='r'):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for coord in bbox:
        rect = patches.Rectangle((coord[0], coord[1]), coord[2]-coord[0], coord[3]-coord[1],
                                 linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import sis_utils
    import ersa_utils

    img_dir, task_dir = sis_utils.get_task_img_folder()
    city_name = 'AZ_Tucson'
    tile_id = 3
    data_dir = r'/home/lab/Documents/bohao/data/transmission_line'
    info_dir = os.path.join(data_dir, 'info')
    raw_dir = os.path.join(data_dir, 'raw')

    pred_file_name = os.path.join(task_dir, 'USA_{}_{}.txt'.format(city_name, tile_id))
    preds = ersa_utils.load_file(pred_file_name)
    raw_rgb = ersa_utils.load_file(os.path.join(raw_dir, 'USA_{}_{}.tif'.format(city_name, tile_id)))
    csv_file_name = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_name, tile_id))

    center_list, conf_list, remove_idx = local_maxima_suppression(preds)

    fig, ax = plt.subplots(1)
    ax.imshow(raw_rgb)
    for cnt, line in enumerate(preds):
        class_name, confidence, left, top, right, bottom = parse_result(line)
        y, x = get_center_point(top, left, bottom, right)
        if cnt in remove_idx:
            rect = patches.Rectangle((left, top), right - left, bottom - top, linewidth=2, edgecolor='r',
                                     facecolor='none')
        else:
            rect = patches.Rectangle((left, top), right - left, bottom - top, linewidth=2, edgecolor='g',
                                     facecolor='none')
        ax.add_patch(rect)

    plt.tight_layout()
    plt.show()
