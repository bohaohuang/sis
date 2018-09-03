import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
from collection import collectionMaker as cm


def get_contour(image, contour_length=5):
    _, contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, contours, -1, 1, contour_length)
    return mask


if __name__ == '__main__':
    ds = cm.read_collection('Inria')
    ds.print_meta_data()
    gt_file = ds.load_files(field_name='austin', field_id='1', field_ext='gt_d255')

    gt = imageio.imread(gt_file)
    #plt.imshow(gt)
    #plt.show()

    mask = get_contour(gt)
    plt.imshow(mask)
    plt.show()
