import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import utils
import ersa_utils
import processBlock
from preprocess import patchExtractor
from collection import collectionMaker


def crop_center(img, cropx, cropy):
    y,x = img.shape[:2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    if len(img.shape) == 3:
        return img[starty:starty+cropy, startx:startx+cropx, :]
    else:
        return img[starty:starty+cropy, startx:startx+cropx]


def data_reader(file_list, chan_mean, th=1e-2):
    for cnt, (rgb_file, gt_file) in enumerate(file_list):
        rgb = crop_center(ersa_utils.load_file(rgb_file), 224, 224)
        gt = crop_center(ersa_utils.load_file(gt_file), 224, 224)

        if np.sum(gt)/(224*224) > th:
            yield rgb - chan_mean, rgb_file


def make_res_features(data_reader, save_dir, gpu=0):
    # set gpu
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu)
    import keras

    # make file names
    feature_file_name = os.path.join(save_dir, 'res50_feature.csv')
    patch_file_name = os.path.join(save_dir, 'res50_patches.txt')

    res50 = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet')
    fc2048 = keras.models.Model(inputs=res50.input, outputs=res50.get_layer('flatten_1').output)
    with open(feature_file_name, 'w+') as f:
        with open(patch_file_name, 'w+') as f2:
            for patch, patch_name in data_reader:
                patch = np.expand_dims(patch, axis=0)
                fc1000 = fc2048.predict(patch).reshape((-1,)).tolist()
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(['{}'.format(x) for x in fc1000])
                f2.write('{}\n'.format(patch_name))


if __name__ == '__main__':
    # settings
    patch_size = (572, 572)
    tile_size = (5000, 5000)
    np.random.seed(1004)
    gpu = 0
    img_dir, task_dir = utils.get_task_img_folder()
    use_hist = False
    cm = collectionMaker.read_collection('spca')
    cm.print_meta_data()
    chan_mean = cm.meta_data['chan_mean'][:3]

    file_list = cm.load_files(field_id=','.join(str(i) for i in range(0, 663)), field_ext='RGB,GT')
    patch_list_train = patchExtractor.PatchExtractor(patch_size, tile_size, 'spca_all',
                                                     184, 184 // 2). \
        run(file_list=file_list, file_exts=['jpg', 'png'], force_run=False).get_filelist()


    feature_dir = os.path.join(task_dir, 'spca_patches')
    dr = data_reader(patch_list_train, chan_mean)
    ersa_utils.make_dir_if_not_exist(feature_dir)
    processBlock.BasicProcess('make_feature', feature_dir,
                              func=lambda: make_res_features(dr, feature_dir, gpu)).run()
