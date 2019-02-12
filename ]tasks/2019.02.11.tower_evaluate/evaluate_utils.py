import numpy as np
import tensorflow as tf
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
