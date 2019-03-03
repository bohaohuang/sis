import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import sis_utils
import ersa_utils
import processBlock
from preprocess import patchExtractor
from collection import collectionMaker


def make_patches(files, patch_size, save_dir, overlap=0):
    for f in tqdm(files):
        tile_name = '_'.join(os.path.basename(f).split('_')[:2])
        rgb = ersa_utils.load_file(f)
        h, w, _ = rgb.shape
        grid = patchExtractor.make_grid((h, w), patch_size, overlap)
        file_list = os.path.join(save_dir, 'file_list.txt')
        with open(file_list, 'w+') as f:
            for cnt, patch in enumerate(patchExtractor.patch_block(rgb, overlap//2, grid, patch_size)):
                file_name = '{}_{:04d}.jpg'.format(tile_name, cnt)
                ersa_utils.save_file(os.path.join(save_dir, file_name), patch)
                f.write('{}\n'.format(os.path.join(save_dir, file_name)))


def data_reader(save_dir, chan_mean):
    file_list = os.path.join(save_dir, 'file_list.txt')
    with open(file_list, 'r') as f:
        files = f.readlines()
    for f in files:
        patch = ersa_utils.load_file(f.strip())
        yield patch - chan_mean, os.path.basename(f.strip())


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
    patch_size = (224, 224)
    np.random.seed(1004)
    gpu = 0
    img_dir, task_dir = sis_utils.get_task_img_folder()
    use_hist = True
    cm = collectionMaker.read_collection('aemo_pad')
    cm.print_meta_data()

    if not use_hist:
        aemo_dir = r'/home/lab/Documents/bohao/data/aemo/aemo_pad'
        aemo_files = glob(os.path.join(aemo_dir, '*rgb.tif'))
        chan_mean = cm.meta_data['chan_mean'][:3]
        save_dir = os.path.join(img_dir, 'aemo_patches')
        feature_dir = os.path.join(task_dir, 'aemo_patches')
    else:
        aemohist_dir = r'/hdd/ersa/preprocess/aemo_pad/hist_matching'
        aemo_files = glob(os.path.join(aemohist_dir, '*histRGB.tif'))
        chan_mean = cm.meta_data['chan_mean'][-3:]
        save_dir = os.path.join(img_dir, 'aemo_hist_patches')
        feature_dir = os.path.join(task_dir, 'aemo_hist_patches')

    ersa_utils.make_dir_if_not_exist(save_dir)
    processBlock.BasicProcess('patch_extract', save_dir,
                              func=lambda: make_patches(aemo_files, patch_size, save_dir, 0)).run()

    dr = data_reader(save_dir, chan_mean)
    ersa_utils.make_dir_if_not_exist(feature_dir)
    processBlock.BasicProcess('make_feature', feature_dir,
                              func=lambda: make_res_features(dr, feature_dir, gpu)).run()
