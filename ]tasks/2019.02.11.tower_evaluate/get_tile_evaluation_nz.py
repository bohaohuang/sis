import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import tensorflow as tf
from glob import glob
from tqdm import tqdm
from natsort import natsorted
import sis_utils
import ersa_utils
from nn import nn_utils
from evaluate_utils import extract_grids, run_inference_for_single_image, get_predict_info

# settings
nn_utils.tf_warn_level(3)
img_dir, task_dir = sis_utils.get_task_img_folder()
task_fold = r'/media/ei-edl01/user/bh163/tasks/2018.11.16.transmission_line'
data_dir = r'/home/lab/Documents/bohao/data/transmission_line'
info_dir = os.path.join(data_dir, 'info')
raw_dir = os.path.join(data_dir, 'raw2')

city_list = ['Dunedin', 'Gisborne', 'Palmerston North', 'Rotorua', 'Tauranga']
patch_size = (500, 500)

for city_id in range(len(city_list)):
    # model info
    # model_name = 'faster_rcnn_NZ_2019-07-01_18-33-39'
    # model_name = 'faster_rcnn_res101_NZ_2019-07-03_11-00-34'
    model_name = 'faster_rcnn_res50_NZ_2019-07-02_19-13-37'

    model_id = 25000
    pred_dir = 'predicted{}'.format(model_name)
    graph_path = r'/hdd6/Models/transmission_line/' \
                 r'export_model/{}/{}/frozen_inference_graph.pb'.format(model_name, model_id)
    th = 0.5

    # load frozen tf model into memory
    tf.reset_default_graph()
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    for tile_id in [1, 2, 3]:
        # get prediction
        try:
            raw_rgb = ersa_utils.load_file(os.path.join(raw_dir, 'NZ_{}_{}_resize.tif'.format(city_list[city_id], tile_id)))
        except OSError:
            print('No file found for {}'.format(os.path.join(raw_dir, 'NZ_{}_{}_resize.tif'.format(city_list[city_id], tile_id))))
            continue
        # pred_files = natsorted(glob(os.path.join(task_fold, pred_dir, '*_{}.txt'.format(tile_id))))
        npy_file_name = os.path.join(info_dir, 'NZ_{}_{}_resize.npy'.format(city_list[city_id], tile_id))
        coords = ersa_utils.load_file(npy_file_name)
        text_save_dir = os.path.join(task_dir, 'faster_rcnn_res50_v2')
        ersa_utils.make_dir_if_not_exist(text_save_dir)
        text_file_name = os.path.join(text_save_dir, 'NZ_{}_{}_resize.txt'.format(city_list[city_id], tile_id))

        h_steps, w_steps = extract_grids(raw_rgb, patch_size[0], patch_size[1])
        with open(text_file_name, 'w+') as f:
            patch_cnt = 0
            for h_cnt, h in enumerate(tqdm(h_steps)):
                for w_cnt, w in enumerate(tqdm(w_steps)):
                    image_patch = raw_rgb[h:h+patch_size[0], w:w+patch_size[1], :3]
                    image_np_expanded = np.expand_dims(image_patch, axis=0)
                    output_dict = run_inference_for_single_image(image_patch, detection_graph)
                    for left, top, right, bottom, confidence, class_name in get_predict_info(
                            output_dict, patch_size, (h, w), ('T', ), th):
                        f.write('{} {} {} {} {} {}\n'.format(class_name, confidence, left, top, right, bottom))
