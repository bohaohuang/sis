import os
import cv2
import csv
from glob import glob
from natsort import natsorted
import ersa_utils
import numpy as np
import uab_collectionFunctions
from reader import reader_utils
from visualize import visualize_utils


def down_sample(img, mult_num):
    h, w = img.shape[:2]
    new_h = h // mult_num
    new_w = w // mult_num
    img = reader_utils.resize_image(img, (new_h, new_w), True)
    return img


def read_polygon_info(file_name, city_name):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            if city_name in line[0]:
                line_num = int(line[2])
                line_record = []
                for i in range(line_num - 1):
                    try:
                        p_pair = list()
                        x = line[3 + i*2]
                        y = line[3 + i*2 + 1]
                        if str(x) != 'NaN' and str(y) != 'NaN':
                            p_pair.append(float(x))
                            p_pair.append(float(y))
                            line_record.append(p_pair)
                        else:
                            line_record_temp = line_record
                            line_record = []
                            yield line_record+line_record_temp
                    except IndexError:
                        print('Warining: index error at {}!'.format(file_name))
                yield line_record


def add_polygons(img, reader):
    for coords in reader:
        vrx = np.array(coords, dtype='int32')
        img = cv2.fillPoly(img, [vrx], (1, 0, 0))
    return img[:, :, 0]


def add_lines(img, reader):
    for coords in reader:
        vrx = np.array(coords, dtype='int32')
        nodes = vrx.shape[0]
        for i in range(nodes - 1):
            img = cv2.line(img, tuple(vrx[i]), tuple(vrx[i+1]), (1, 0, 0), thickness=15)
    return img[:, :, 0]


if __name__ == '__main__':
    '''resize_dict = {'Arlington': 1, 'Atlanta': 2, 'Austin': 2, 'DC': 2, 'NewHaven': 1,
                   'NewYork': 2, 'Norfolk': 1, 'SanFrancisco': 1, 'Seekonk': 1}'''
    resize_dict = {'Norfolk': 1}
    check_result = False
    make_uab = True

    for city_name, resize_factor in resize_dict.items():
        data_dir = r'/home/lab/Documents/bohao/data/figshare/{}'.format(city_name)
        save_dir = r'/media/ei-edl01/data/uab_datasets/{}/data/Original_Tiles'.format(city_name)
        ersa_utils.make_dir_if_not_exist(save_dir)

        rgb_files = natsorted(glob(os.path.join(data_dir, city_name + '_0*.tif')))
        building_files = natsorted(glob(os.path.join(data_dir, '{}*buildingCoord.csv'.format(city_name))))
        road_files = natsorted(glob(os.path.join(data_dir, '{}*roadCoord.csv'.format(city_name))))

        for city_id, (rgb_file, building_file, road_file) in enumerate(zip(rgb_files, building_files, road_files)):
            rgb = ersa_utils.load_file(rgb_file)[:, :, :3]

            building_reader = read_polygon_info(building_file, city_name)
            building = np.zeros_like(rgb, np.uint8)
            building = add_polygons(building, building_reader)

            road_reader = read_polygon_info(road_file, city_name)
            road = np.zeros_like(rgb, np.uint8)
            road = add_lines(road, road_reader)

            rgb = down_sample(rgb, resize_factor).astype(np.uint8)
            building = down_sample(building, resize_factor)
            road = down_sample(road, resize_factor)

            rgb_save_name = os.path.join(save_dir, '{}{}_RGB.tif'.format(city_name, city_id+1))
            building_save_name = os.path.join(save_dir, '{}{}_BGT.png'.format(city_name, city_id+1))
            road_save_name = os.path.join(save_dir, '{}{}_RGT.png'.format(city_name, city_id+1))

            ersa_utils.save_file(rgb_save_name, rgb.astype(np.uint8))
            ersa_utils.save_file(building_save_name, building.astype(np.uint8))
            ersa_utils.save_file(road_save_name, road.astype(np.uint8))

            if check_result:
                visualize_utils.compare_figures([rgb, building, road], (1, 3), fig_size=(12, 4))

        if make_uab:
            blCol = uab_collectionFunctions.uabCollection(city_name)
            blCol.readMetadata()
            img_mean = blCol.getChannelMeans([0, 1, 2])

    '''save_dir = r'/media/ei-edl01/data/uab_datasets/AIOI/data/Original_Tiles'
    rgb_files = natsorted(glob(os.path.join(save_dir, '*_RGB.tif')))
    bui_files = natsorted(glob(os.path.join(save_dir, '*_BGT.png')))
    roa_files = natsorted(glob(os.path.join(save_dir, '*_RGT.png')))

    for i, b, r in zip(rgb_files, bui_files, roa_files):
        im = ersa_utils.load_file(i)
        bm = ersa_utils.load_file(b)
        rm = ersa_utils.load_file(r)

        visualize_utils.compare_figures([im, bm, rm], (1, 3), fig_size=(12, 4))'''
