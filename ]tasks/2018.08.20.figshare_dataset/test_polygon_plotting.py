import os
import csv
import imageio
import scipy.misc
import cv2 as cv
import util_functions
import numpy as np
import matplotlib.pyplot as plt


def down_sample(img, mult_num, dim=3):
    if dim == 3:
        h, w, c = img.shape
    else:
        h, w = img.shape
    new_h = h // mult_num
    new_w = w // mult_num
    img = scipy.misc.imresize(img, (new_h, new_w))
    return img


def read_polygon_info(file_name, city_name):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            if city_name in line[0]:
                line_num = int(line[2])
                line_record = []
                for i in range(line_num - 1):
                    p_pair = list()
                    x = line[3 + i*2]
                    y = line[3 + i*2 + 1]
                    if str(x) != 'NaN' and str(y) != 'NaN':
                        p_pair.append(float(x))
                        p_pair.append(float(y))
                        line_record.append(p_pair)
                    else:
                        pass
                        line_record_temp = line_record
                        line_record = [line_record_temp[-1]]
                        print(line_record_temp)
                        yield line_record_temp
                #yield line_record


def add_polygons(img, reader):
    for coords in reader:
        vrx = np.array(coords, dtype='int32')
        # img = cv.fillPoly(img, [vrx], (1, 0, 0))
        img = cv.polylines(img, [vrx], False, (1, 0, 0), thickness=20)
    return img[:, :, 0]


if __name__ == '__main__':
    city_name = 'Arlington'
    img_dir = r'/home/lab/Documents/bohao/data/arlington'
    check_result = True
    for img_id in [1, 2, 3]:
        img_name = '{}_0{}.tif'.format(city_name, img_id)
        gt_file_name = '{}_0{}_roadCoord.csv'.format(city_name, img_id)

        info_reader = read_polygon_info(os.path.join(img_dir, gt_file_name), city_name)
        img = imageio.imread(os.path.join(img_dir, img_name))

        gt_img = np.zeros(img.shape, np.uint8)
        gt_img = add_polygons(gt_img, info_reader)

        img = down_sample(img, 2)
        gt_img = down_sample(gt_img, 2, dim=2)

        new_img_name = '{}{}_RGB.tif'.format(city_name, img_id)
        new_gt_name = '{}{}_GT.png'.format(city_name, img_id)

        imageio.imsave(os.path.join(img_dir, new_img_name), img)
        imageio.imsave(os.path.join(img_dir, new_gt_name), gt_img)

        if check_result:
            plt.figure(figsize=(10, 8))
            # ax1 = plt.subplot(211)
            img = util_functions.add_mask(img, gt_img, [255, 0, 0], mask_1=1)
            plt.imshow(img)
            plt.axis('off')
            '''ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
            plt.imshow(gt_img)
            plt.axis('off')'''
            plt.tight_layout()
            plt.show()
