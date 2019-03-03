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
raw_dir = os.path.join(data_dir, 'raw')
for city_id in range(4):
    city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']
    patch_size = (500, 500)

    # model info
    model_name = ['faster_rcnn_res50_2019-02-13_16-30-28',
                  'faster_rcnn_res50_2019-02-13_16-32-30',
                  'faster_rcnn_res50_2019-02-13_16-33-24',
                  'faster_rcnn_res50_2019-02-13_16-34-12']
    '''model_name = ['faster_rcnn_2019-02-05_19-20-08',
                  'faster_rcnn_2019-02-05_19-24-35',
                  'faster_rcnn_2019-02-05_19-49-39',
                  'faster_rcnn_2019-02-05_20-00-56']'''
    model_id = 25000
    pred_dir = 'predicted{}'.format(model_name[city_id])
    graph_path = r'/hdd6/Models/transmission_line/' \
                 r'export_model/{}/{}/frozen_inference_graph.pb'.format(model_name[city_id], model_id)
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
        raw_rgb = ersa_utils.load_file(os.path.join(raw_dir, 'USA_{}_{}.tif'.format(city_list[city_id], tile_id)))
        pred_files = natsorted(glob(os.path.join(task_fold, pred_dir, '*_{}.txt'.format(tile_id))))
        npy_file_name = os.path.join(info_dir, 'USA_{}_{}.npy'.format(city_list[city_id], tile_id))
        coords = ersa_utils.load_file(npy_file_name)
        text_save_dir = os.path.join(task_dir, 'faster_rcnn_res50')
        ersa_utils.make_dir_if_not_exist(text_save_dir)
        text_file_name = os.path.join(text_save_dir, 'USA_{}_{}.txt'.format(city_list[city_id], tile_id))

        h_steps, w_steps = extract_grids(raw_rgb, patch_size[0], patch_size[1])
        with open(text_file_name, 'w+') as f:
            patch_cnt = 0
            for h_cnt, h in enumerate(tqdm(h_steps)):
                for w_cnt, w in enumerate(tqdm(w_steps)):
                    image_patch = raw_rgb[h:h+patch_size[0], w:w+patch_size[1], :]
                    image_np_expanded = np.expand_dims(image_patch, axis=0)
                    output_dict = run_inference_for_single_image(image_patch, detection_graph)
                    for left, top, right, bottom, confidence, class_name in get_predict_info(
                            output_dict, patch_size, (h, w), ('T', ), th):
                        f.write('{} {} {} {} {} {}\n'.format(class_name, confidence, left, top, right, bottom))
