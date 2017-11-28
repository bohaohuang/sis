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


def pad_prediction(image, prediction):
    _, img_w, img_h, _ = image.shape
    n, pred_img_w, pred_img_h, c = prediction.shape

    if img_w > pred_img_w and img_h > pred_img_h:
        pad_w = int((img_w - pred_img_w) / 2)
        pad_h = int((img_h - pred_img_h) / 2)
        prediction_padded = np.zeros((n, img_w, img_h, c))
        pad_dim = ((pad_w, pad_w),
                   (pad_h, pad_h))

        for batch_id in range(n):
            for channel_id in range(c):
                prediction_padded[batch_id, :, :, channel_id] = \
                    np.pad(prediction[batch_id, :, :, channel_id], pad_dim, 'constant')
        prediction = prediction_padded
        return prediction
    else:
        return prediction


def image_summary(image, truth, prediction):
    truth_img = decode_labels(truth, 10)

    prediction = pad_prediction(image, prediction)

    pred_labels = get_pred_labels(prediction)
    pred_img = decode_labels(pred_labels, 10)

    return np.concatenate([image, truth_img, pred_img], axis=2)


def get_output_label(result, image_dim, input_size, colormap, overlap=0,
                     output_image_dim=None, output_patch_size=None):
    if output_image_dim is not None and output_patch_size is not None:
        image_pred = patch_extractor.un_patchify_shrink(result, image_dim, output_image_dim,
                                                        input_size, output_patch_size, overlap=overlap)
    else:
        image_pred = patch_extractor.un_patchify(result, image_dim, input_size, overlap=overlap)
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


def get_task_img_folder():
    IMG_DIR = r'/media/ei-edl01/user/bh163/figs'
    TASK_DIR = r'/media/ei-edl01/user/bh163/tasks'
    return make_task_img_folder(IMG_DIR), make_task_img_folder(TASK_DIR)


def barplot_autolabel(ax, rects, margin=0.05):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2, (1+margin)*height,
                '{:.3f}'.format(height), ha='center', va='bottom')


def test_unet(rsr_data_dir,
              test_data_dir,
              input_size,
              model_name,
              num_classes,
              ckdir,
              city,
              batch_size,
              ds_name='inria',
              GPU='0',
              random_seed=1234):
    import re
    import scipy.misc
    import tensorflow as tf
    from network import unet
    from dataReader import image_reader
    from rsrClassData import rsrClassData

    def name_has_city(city_list, name):
        for city in city_list:
            if city in name:
                return True
        return False

    city = city.split(',')

    # set gpu
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    tf.reset_default_graph()
    # environment settings
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)

    # data prepare step
    Data = rsrClassData(rsr_data_dir)
    (collect_files_test, meta_test) = Data.getCollectionByName(test_data_dir)

    # image reader
    coord = tf.train.Coordinator()

    # define place holder
    X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')

    # initialize model
    model = unet.UnetModel({'X': X, 'Y': y}, trainable=mode, model_name=model_name, input_size=input_size)
    model.create_graph('X', num_classes)
    model.make_update_ops('X', 'Y')
    # set ckdir
    model.make_ckdir(ckdir)
    # set up graph and initialize
    config = tf.ConfigProto()

    result_dict = {}

    # run training
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)

        if os.path.exists(model.ckdir) and tf.train.get_checkpoint_state(model.ckdir):
            latest_check_point = tf.train.latest_checkpoint(model.ckdir)
            saver.restore(sess, latest_check_point)
            print('loaded model from {}'.format(latest_check_point))

        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        for (image_name, label_name) in collect_files_test:
            #if city in image_name:
            if name_has_city(city, image_name):
                if ds_name == 'inria':
                    city_name = re.findall('[a-z\-]*(?=[0-9]+\.)', image_name)[0]
                    tile_id = re.findall('[0-9]+(?=\.tif)', image_name)[0]
                else:
                    city_name = os.path.basename(image_name)[:3]
                    tile_id = os.path.basename(image_name)[9:12]

                # load reader
                iterator_test = image_reader.image_label_iterator(
                    os.path.join(rsr_data_dir, image_name),
                    batch_size=batch_size,
                    tile_dim=meta_test['dim_image'][:2],
                    patch_size=input_size,
                    overlap=0)
                # run
                result = model.test('X', sess, iterator_test)
                pred_label_img = get_output_label(result, meta_test['dim_image'],
                                                  input_size, meta_test['colormap'])
                # evaluate
                truth_label_img = scipy.misc.imread(os.path.join(rsr_data_dir, label_name))
                iou = iou_metric(truth_label_img, pred_label_img)
                result_dict['{}{}'.format(city_name, tile_id)] = iou
            coord.request_stop()
            coord.join(threads)
    return result_dict


def test_authentic_unet(rsr_data_dir,
              test_data_dir,
              input_size,
              model_name,
              num_classes,
              ckdir,
              city,
              batch_size,
              ds_name='inria',
              GPU='0',
              random_seed=1234):
    import re
    import scipy.misc
    import tensorflow as tf
    from network import unet
    from dataReader import image_reader
    from rsrClassData import rsrClassData

    def name_has_city(city_list, name):
        for city in city_list:
            if city in name:
                return True
        return False

    city = city.split(',')

    # set gpu
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    tf.reset_default_graph()
    # environment settings
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)

    # data prepare step
    Data = rsrClassData(rsr_data_dir)
    (collect_files_test, meta_test) = Data.getCollectionByName(test_data_dir)

    # image reader
    coord = tf.train.Coordinator()

    # define place holder
    X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')

    # initialize model
    model = unet.UnetModel_Origin({'X': X, 'Y': y}, trainable=mode, model_name=model_name, input_size=input_size)
    model.create_graph('X', num_classes)
    model.make_update_ops('X', 'Y')
    # set ckdir
    model.make_ckdir(ckdir)
    # set up graph and initialize
    config = tf.ConfigProto()

    result_dict = {}

    # run training
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)

        if os.path.exists(model.ckdir) and tf.train.get_checkpoint_state(model.ckdir):
            latest_check_point = tf.train.latest_checkpoint(model.ckdir)
            saver.restore(sess, latest_check_point)
            print('loaded model from {}'.format(latest_check_point))

        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        for (image_name, label_name) in collect_files_test:
            #if city in image_name:
            if name_has_city(city, image_name):
                if ds_name == 'inria':
                    city_name = re.findall('[a-z\-]*(?=[0-9]+\.)', image_name)[0]
                    tile_id = re.findall('[0-9]+(?=\.tif)', image_name)[0]
                else:
                    city_name = os.path.basename(image_name)[:3]
                    tile_id = os.path.basename(image_name)[9:12]

                # load reader
                iterator_test = image_reader.image_label_iterator(
                    os.path.join(rsr_data_dir, image_name),
                    batch_size=batch_size,
                    tile_dim=meta_test['dim_image'][:2],
                    patch_size=input_size,
                    overlap=184, padding=92)
                # run
                result = model.test('X', sess, iterator_test)
                pred_label_img = get_output_label(result,
                                                  (meta_test['dim_image'][0]+184, meta_test['dim_image'][1]+184),
                                                  input_size,
                                                  meta_test['colormap'], overlap=184,
                                                  output_image_dim=meta_test['dim_image'],
                                                  output_patch_size=(input_size[0]-184, input_size[1]-184))
                # evaluate
                truth_label_img = scipy.misc.imread(os.path.join(rsr_data_dir, label_name))
                iou = iou_metric(truth_label_img, pred_label_img)
                result_dict['{}{}'.format(city_name, tile_id)] = iou
            coord.request_stop()
            coord.join(threads)
    return result_dict


def test_authentic_unet_height(rsr_data_dir,
                               test_data_dir,
                               input_size,
                               model_name,
                               num_classes,
                               ckdir,
                               city,
                               batch_size,
                               ds_name='inria',
                               GPU='0',
                               random_seed=1234,
                               height_mode='subtract'):
    import re
    import scipy.misc
    import tensorflow as tf
    from network import unet
    from dataReader import image_reader
    from rsrClassData import rsrClassData

    def name_has_city(city_list, name):
        for city in city_list:
            if city in name:
                return True
        return False

    city = city.split(',')

    # set gpu
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    tf.reset_default_graph()
    # environment settings
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)

    # data prepare step
    Data = rsrClassData(rsr_data_dir)
    (collect_files_test, meta_test) = Data.getCollectionByName(test_data_dir)

    # image reader
    coord = tf.train.Coordinator()

    # define place holder
    if height_mode == 'all':
        X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 5], name='X')
    elif height_mode == 'subtract':
        X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 4], name='X')
    else:
        X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 6], name='X')
    y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')

    # initialize model
    model = unet.UnetModel_Origin({'X': X, 'Y': y}, trainable=mode, model_name=model_name, input_size=input_size)
    model.create_graph('X', num_classes)
    model.make_update_ops('X', 'Y')
    # set ckdir
    model.make_ckdir(ckdir)
    # set up graph and initialize
    config = tf.ConfigProto()

    result_dict = {}

    # run training
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)

        if os.path.exists(model.ckdir) and tf.train.get_checkpoint_state(model.ckdir):
            latest_check_point = tf.train.latest_checkpoint(model.ckdir)
            saver.restore(sess, latest_check_point)
            print('loaded model from {}'.format(latest_check_point))

        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        for (image_name, dsm_name, dtm_name, label_name) in collect_files_test:
            #if city in image_name:
            if name_has_city(city, image_name):
                if ds_name == 'inria':
                    city_name = re.findall('[a-z\-]*(?=[0-9]+\.)', image_name)[0]
                    tile_id = re.findall('[0-9]+(?=\.tif)', image_name)[0]
                else:
                    city_name = os.path.basename(image_name)[:3]
                    tile_id = os.path.basename(image_name)[9:12]

                iter_file_list = [os.path.join(rsr_data_dir, image_name),
                                  os.path.join(rsr_data_dir, dsm_name),
                                  os.path.join(rsr_data_dir, dtm_name)]

                # load reader
                iterator_test = image_reader.image_height_label_iterator(
                    iter_file_list,
                    batch_size=batch_size,
                    tile_dim=meta_test['dim_image'][:2],
                    patch_size=input_size,
                    overlap=184, padding=92, height_mode=height_mode)
                # run
                result = model.test('X', sess, iterator_test)
                pred_label_img = get_output_label(result,
                                                  (meta_test['dim_image'][0]+184, meta_test['dim_image'][1]+184),
                                                  input_size,
                                                  meta_test['colormap'], overlap=184,
                                                  output_image_dim=meta_test['dim_image'],
                                                  output_patch_size=(input_size[0]-184, input_size[1]-184))
                # evaluate
                truth_label_img = scipy.misc.imread(os.path.join(rsr_data_dir, label_name))

                import matplotlib.pyplot as plt
                plt.figure(figsize=(10,6))
                plt.subplot(121)
                plt.imshow(pred_label_img)
                plt.subplot(122)
                plt.imshow(truth_label_img)
                plt.show()

                iou = iou_metric(truth_label_img, pred_label_img)
                result_dict['{}{}'.format(city_name, tile_id)] = iou
            coord.request_stop()
            coord.join(threads)
    return result_dict


def get_unet_first_layer_weight(model_dir, additional_num, input_size=(572, 572)):
    import tensorflow as tf
    from network import unet

    X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')

    model = unet.UnetModel_Origin({'X': X, 'Y': y}, trainable=mode, model_name='test_unet',
                                  input_size=input_size)
    model.create_graph('X', 2)
    trainable_vars = tf.trainable_variables()
    model.ckdir = model_dir

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        initialize_var = sess.run(trainable_vars[0])

        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)

        assert os.path.exists(model.ckdir) and tf.train.get_checkpoint_state(model.ckdir)
        latest_check_point = tf.train.latest_checkpoint(model.ckdir)
        saver.restore(sess, latest_check_point)

        pretrained_var = sess.run(trainable_vars[0])
        h, w, c, o = initialize_var.shape

        filter_kernel = np.zeros((h, w, c+additional_num, o))
        filter_kernel[:, :, :3, :] = pretrained_var
        for i in range(additional_num):
            filter_kernel[:, :, 3+i%c, :] = initialize_var[:, :, i%c, :]

        return filter_kernel
