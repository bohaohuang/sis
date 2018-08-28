import os
import csv
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils
from run_tsne import run_tsne
from make_res50_features import crop_center


def get_category_id(path_list):
    cat_dict = dict()
    id_list = []
    id_cnt = 0
    for file in path_list:
        par_dir = file.split('/')[-2]
        if par_dir not in cat_dict:
            cat_dict[par_dir] = id_cnt
            id_cnt += 1
        id_list.append(cat_dict[par_dir])
    return id_list, cat_dict


def get_image_list(img_dir):
    file_list = []
    for path, subdirs, files in os.walk(img_dir):
        for name in files:
            file_list.append(os.path.join(path, name))
    file_list = sorted(file_list)
    return file_list


def get_image_mean(file_list):
    img_mean = np.zeros(3)
    for file in tqdm(file_list):
        img = imageio.imread(file)
        for cnt in range(3):
            img_mean[cnt] += np.mean(img[:, :, cnt])
    img_mean = img_mean / len(file_list)
    return img_mean


def plot_tsne(feature_encode, id_list, cat_dict):
    id_list = np.array(id_list)
    cat_list = ['' for i in range(len(cat_dict))]
    for key, val in cat_dict.items():
        cat_list[val] = key

    cmap = plt.get_cmap('Set1').colors
    marker_list = ['s', 'v', 'D']
    plt.figure(figsize=(15, 8))
    for i in range(len(cat_dict)):
        plt.scatter(feature_encode[id_list == i, 0], feature_encode[id_list == i, 1], color=cmap[i % 7],
                    marker=marker_list[i // 7], label=cat_list[i], edgecolors='k')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('TSNE Projection Result')
    plt.legend(ncol=3)
    plt.tight_layout()


if __name__ == '__main__':
    img_dir = r'/home/lab/Documents/bohao/data/UCMerced_LandUse/Images'
    file_list = get_image_list(img_dir)
    id_list, cat_dict = get_category_id(file_list)
    img_mean = get_image_mean(file_list)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import keras

    input_size_fit = (224, 224)
    img_dir, task_dir = utils.get_task_img_folder()
    feature_file_name = os.path.join(task_dir, 'ucmerced_res50_inria.csv')
    patch_file_name = os.path.join(task_dir, 'ucmerced_res50_inria.txt')
    res50 = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet')
    fc2048 = keras.models.Model(inputs=res50.input, outputs=res50.get_layer('flatten_1').output)
    if not os.path.exists(feature_file_name) and not os.path.exists(os.path.join(patch_file_name)):
        with open(feature_file_name, 'w+') as f, open(patch_file_name, 'w+') as f2:
            for file_line in tqdm(file_list):
                img = imageio.imread(file_line)
                img = np.expand_dims(crop_center(img, input_size_fit[0], input_size_fit[1]), axis=0)

                fc1000 = fc2048.predict(img).reshape((-1,)).tolist()
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(['{}'.format(x) for x in fc1000])
                f2.write('{}\n'.format(file_line.split('/')[-1]))

    feature = pd.read_csv(feature_file_name, sep=',', header=None).values
    with open(patch_file_name, 'r') as f:
        patch_names = f.readlines()

    perplex = 40
    file_name = os.path.join(task_dir, 'land_inria_p{}.npy'.format(perplex))
    feature_encode = run_tsne(feature, file_name, perplex=perplex, force_run=False)
    plot_tsne(feature_encode, id_list, cat_dict)
    plt.savefig(os.path.join(img_dir, 'land_use_tsne_p{}.png'.format(perplex)))
    plt.show()
