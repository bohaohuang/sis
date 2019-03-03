import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from natsort import natsorted
from skimage.draw import polygon
import sis_utils
import ersa_utils


def read_polygon_csv_data(csv_file, encoder):
    label_order = ['SS', 'OT', 'DT', 'TT', 'OL', 'DL', 'TL']
    df = pd.read_csv(csv_file)
    df['temp_label'] = pd.Categorical(df['Label'], categories=label_order, ordered=True)
    df.sort_values('temp_label', inplace=True, kind='mergesort')

    for name, group in df.groupby('Object', sort=False):
        label = group['Label'].values[0]
        if group['Type'].values[0] == 'Polygon' and label in encoder:
            x, y = polygon(group['X'].values, group['Y'].values)
            y, x = check_bounds(x, y, df['width'][0] - 1, df['height'][0] - 1)
            yield label, x, y


def check_bounds(cc, rr, size_x, size_y):
    cc = np.maximum(np.minimum(cc, size_x), 0)
    rr = np.maximum(np.minimum(rr, size_y), 0)
    return cc, rr


def get_bounding_box(y, x):
    y_min = int(np.min(y))
    x_min = int(np.min(x))
    y_max = int(np.max(y))
    x_max = int(np.max(x))
    return y_max - y_min, x_max - x_min


def get_height_width_list(rgb_files, csv_files, save_dir):
    for rgb_file, csv_file in zip(rgb_files, csv_files):
        city_name = os.path.basename(rgb_file[:-4]).split('_')[2]
        city_id = os.path.basename(rgb_file[:-4]).split('_')[-1]
        print('Processing {}_{} ...'.format(city_name, city_id), end='')

        height_list = []
        width_list = []

        for label, y, x in read_polygon_csv_data(csv_file, encoder):
            height, width = get_bounding_box(y, x)
            height_list.append(height)
            width_list.append(width)

        height_width_stats = np.stack([height_list, width_list])
        ersa_utils.save_file(os.path.join(save_dir, 'boxes_height_width_stats_{}{}.npy'.format(city_name, city_id)),
                             height_width_stats)
        print('Done!')


if __name__ == '__main__':
    data_dir = r'/home/lab/Documents/bohao/data/transmission_line'
    save_dir = os.path.join(data_dir, 'info')
    img_dir, task_dir = sis_utils.get_task_img_folder()

    rgb_files = natsorted([a for a in glob(os.path.join(data_dir, 'raw', '*.tif'))
                               if 'multiclass' not in a])
    csv_files = natsorted(glob(os.path.join(data_dir, 'raw', '*.csv')))
    patch_size = 500
    city_list = ['Tucson', 'Colwich', 'Clyde', 'Wilmington']
    encoder = {'DT': 1, 'TT': 2}

    # get stats
    # get_height_width_list(rgb_files, csv_files, task_dir)

    # read stats
    stats_files = natsorted(glob(os.path.join(task_dir, 'boxes_height_width_stats_*.npy')))
    height_list_all = []
    width_list_all = []

    for city_name in city_list:
        height_list = []
        width_list = []
        for stat_file in stats_files:
            if city_name in stat_file:
                stats = ersa_utils.load_file(stat_file)
                height_list.append(stats[0, :])
                width_list.append(stats[1, :])

        height_list = np.concatenate(height_list)
        width_list = np.concatenate(width_list)

        height_list_all.append(height_list)
        width_list_all.append(width_list)

        # plot stats
        plt.figure(figsize=(8, 8))
        ax1 = plt.subplot(211)
        plt.hist(height_list, bins=100)
        plt.xlabel('Height Distribution')
        plt.ylabel('Counts')
        plt.title(city_name)

        ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
        plt.hist(width_list, bins=100)
        plt.xlabel('Width Distribution')
        plt.ylabel('Counts')

        plt.tight_layout()
        #plt.savefig(os.path.join(img_dir, 'tower_size_distribution_{}.png'.format(city_name)))
        plt.close()

    H, x_edges, y_edges = np.histogram2d(np.concatenate(height_list_all), np.concatenate(width_list_all),
                                         bins=(50, 50), range=[[0, 200], [0, 200]])
    H = H.T
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, H)
    plt.colorbar()
    plt.xlabel('Height')
    plt.ylabel('Width')
    plt.title('Tower Size Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'towers_size_dist_2dhist.png'))
    plt.show()

    H, x_edges, y_edges = np.histogram2d(np.concatenate(height_list_all), np.concatenate(width_list_all),
                                         bins=(25, 25), range=[[0, 75], [0, 75]])
    H = H.T
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, H)
    plt.colorbar()
    plt.xlabel('Height')
    plt.ylabel('Width')
    plt.title('Tower Size Distribution (Zoom in)')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'towers_size_dist_2dhist_zoomin.png'))
    plt.show()

