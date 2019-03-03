import os
import cv2
import time
import scipy.spatial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from skimage import measure
from PIL import Image, ImageDraw
from sklearn.metrics import precision_recall_curve
import sis_utils
import ersa_utils


def get_blank_regions(img):
    img_tmp = img.astype(np.float32)
    img_tmp = np.sum(img_tmp, axis=2)
    blank_mask = (img_tmp < 0.1).astype(np.int)
    return blank_mask


class ConfMapObjectScoring:
    """
    Take ground truth and confidence map, do object based scoring
    """
    def __init__(self, min_region=5, max_region=None, thresh_min=0.5, thresh_max=0.5,
                 epsilon=2, link_radius=55):
        self.min_region = min_region
        self.max_region = max_region
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        self.epsilon = epsilon
        self.link_radius = link_radius

        self.group_density = 0.1
        self.group_radius = 10

    def add_object_wise_info(self, conf_map):
        """
        Get object wise information with given confidence map
        :param conf_map: confidence map, should be normalized
        :return:
        """
        binary_map = (conf_map >= self.thresh_min).astype(np.int)           # binarize the conf map
        object_lbl = measure.label(binary_map)                              # get connected components
        rps = measure.regionprops(object_lbl, conf_map)                     # get info of each component

        object_structure = pd.DataFrame(
            columns=['iLocation', 'jLocation', 'pixelList', 'confidence', 'area', 'maxIntensity', 'isCommercial'])
        object_structure.pixelList = object_structure.pixelList.astype(object)

        polygon_list = []
        for rp in rps:
            if rp.area >= self.min_region and rp.max_intensity >= self.thresh_max:
                # get contour info
                _, contours, _ = cv2.findContours(rp.convex_image.astype(np.uint8),
                                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if len(contours) > 0:
                    assert len(contours) == 1
                    # Douglas-Peucker
                    contours = measure.approximate_polygon(np.squeeze(contours[0], axis=1), self.epsilon)
                    contours[:, 1] += rp.bbox[0]
                    contours[:, 0] += rp.bbox[1]
                    polygon_list.append(contours)
                else:
                    polygon_list.append(np.array([]))

                temp = [*[int(c) for c in rp.centroid], rp.coords, rp.mean_intensity, rp.area, rp.max_intensity, 0]
                object_structure = object_structure.append(
                    dict(zip(
                        ['iLocation', 'jLocation', 'pixelList', 'confidence', 'area', 'maxIntensity', 'isCommercial'],
                        temp)), ignore_index=True)

        object_structure['polygon'] = pd.Series(polygon_list, index=object_structure.index).astype(object)
        return object_structure

    def group_objects(self, obj_struct, orig_map):
        neighbor_filter = np.ones([2 * self.group_radius + 1] * 2)
        temp_img = np.zeros(orig_map.shape)
        pixel_list = np.vstack(np.array(obj_struct['pixelList'])).transpose()
        temp_img[pixel_list[0], pixel_list[1]] = 1

        neighbor_cnt = np.real(
            np.fft.ifft2(np.fft.fft2(temp_img) * np.fft.fft2(neighbor_filter, temp_img.shape)))  # imfilt
        cnt_th = int(self.group_density * np.prod(neighbor_filter.shape))
        group_map = neighbor_cnt > cnt_th
        group_map = ((group_map + orig_map) > 0).astype(np.int)

        object_lbl = measure.label(group_map)
        rps = measure.regionprops((object_lbl * orig_map).astype(np.int), orig_map)

        object_structure = pd.DataFrame(
            columns=['iLocation', 'jLocation', 'pixelList', 'confidence', 'area', 'maxIntensity', 'isCommercial'])
        object_structure.pixelList = object_structure.pixelList.astype(object)

        polygon_list = []
        for rp in rps:
            if rp.area >= self.min_region and rp.max_intensity >= self.thresh_max:
                # get contour info
                _, contours, _ = cv2.findContours(rp.convex_image.astype(np.uint8),
                                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if len(contours) > 0:
                    assert len(contours) == 1
                    # Douglas-Peucker
                    contours = measure.approximate_polygon(np.squeeze(contours[0], axis=1), self.epsilon)
                    contours[:, 1] += rp.bbox[0]
                    contours[:, 0] += rp.bbox[1]
                    polygon_list.append(contours)
                else:
                    polygon_list.append(np.array([]))

                temp = [*[int(c) for c in rp.centroid], rp.coords, rp.mean_intensity, rp.area, rp.max_intensity, 0]
                object_structure = object_structure.append(
                    dict(zip(
                        ['iLocation', 'jLocation', 'pixelList', 'confidence', 'area', 'maxIntensity', 'isCommercial'],
                        temp)), ignore_index=True)

        object_structure['polygon'] = pd.Series(polygon_list, index=object_structure.index).astype(object)
        return object_structure

    def link_objects2(self, obj_struct_gt, obj_struct_cm):
        """
        Link objects in gt_map and conf_map, return the objects with information
        :param gt_map:
        :param conf_map:
        :return:
        """
        i_coords = np.array(obj_struct_gt['iLocation'])
        j_coords = np.array(obj_struct_gt['jLocation'])
        pts_gt = np.stack([i_coords, j_coords], axis=1)
        house_ids = np.arange(pts_gt.shape[0])

        i_coords = np.array(obj_struct_cm['iLocation'])
        j_coords = np.array(obj_struct_cm['jLocation'])
        pts_cm = np.stack([i_coords, j_coords], axis=1)

        Mdl = scipy.spatial.KDTree(pts_gt)
        ids = Mdl.query_ball_point(pts_cm, r=self.link_radius)

        houseId = np.zeros_like(ids)
        for cnt, i in enumerate(ids):
            if len(i) > 0:
                houseId[cnt] = [house_ids[a] for a in i]
            else:
                houseId[cnt] = [-1]

        obj_struct_cm['iOut'] = pd.Series(houseId)

        return obj_struct_cm

    def link_objects(self, obj_struct_gt, obj_struct_cm):
        """
        Link objects in gt_map and conf_map, return the objects with information
        :param gt_map:
        :param conf_map:
        :return:
        """
        i_coords = np.array(obj_struct_gt['iLocation'])
        j_coords = np.array(obj_struct_gt['jLocation'])
        pts_gt = np.stack([i_coords, j_coords], axis=1)
        house_ids = np.arange(pts_gt.shape[0])

        i_coords = np.array(obj_struct_cm['iLocation'])
        j_coords = np.array(obj_struct_cm['jLocation'])
        pts_cm = np.stack([i_coords, j_coords], axis=1)

        Mdl = scipy.spatial.KDTree(pts_gt)

        distance, ids = Mdl.query(pts_cm, distance_upper_bound=self.link_radius)
        houseId = np.zeros_like(distance)
        for cnt, (d, i) in enumerate(zip(distance, ids)):
            if d <= self.link_radius:
                houseId[cnt] = house_ids[ids[cnt]]
            else:
                houseId[cnt] = -1

        obj_struct_cm['iOut'] = pd.Series(houseId)

        return obj_struct_cm

    def get_info_summary(self):
        # TODO print object information summary
        pass

    @staticmethod
    def get_intersection(A, B):
        inter = np.array([x for x in set(tuple(x) for x in A) & set(tuple(x) for x in B)])
        union = np.array([x for x in set(tuple(x) for x in A) | set(tuple(x) for x in B)])
        return inter, union

    def scoring_func(self, obj_struct_gt, obj_struct_cm, iou_th=0.5):
        conf = []
        true = []
        area = []
        ious = []

        pl_gt = np.array(obj_struct_gt['pixelList'])
        area_gt = np.array(obj_struct_gt['area'])
        area_pp = np.array(obj_struct_cm['area'])
        pl_pp = np.array(obj_struct_cm['pixelList'])
        pl_pp_cf = np.array(obj_struct_cm['confidence'])

        panel_num = pl_gt.shape[0]
        pp_house_id = obj_struct_cm['iOut'].tolist()
        for i in range(panel_num):
            if i in pp_house_id:
                pp_is = [index for index, value in enumerate(pp_house_id) if value == i]
                pts_pp = []
                for pp_i in pp_is:
                    pts_pp.append(pl_pp[pp_i])
                pts_pp = np.concatenate(pts_pp, axis=0)
                inter, union = self.get_intersection(pl_gt[i], pts_pp)
                iou = inter.shape[0] / union.shape[0]
                if iou >= iou_th:
                    # tp
                    conf.append(pl_pp_cf[pp_i])
                    true.append(1)
                else:
                    # fn
                    conf.append(pl_pp_cf[pp_i])
                    true.append(0)

                area.append(area_gt[i])
                ious.append(iou)
            else:
                conf.append(-1000)
                true.append(1)

                area.append(area_gt[i])
                ious.append(0)
        for i in range(len(pp_house_id)):
            if pp_house_id[i] == -1:
                true.append(0)
                conf.append(pl_pp_cf[i])

                area.append(-1)
                ious.append(-1)

        return np.array(conf), np.array(true), np.array(area), np.array(ious)

    @staticmethod
    def get_image(tile_size, obj_struct):  # polygon cords in form of xy
        polygon_im = np.zeros(tile_size)
        for poly in obj_struct['polygon']:
            if len(poly) > 0:
                img = Image.new('L', tile_size, 0)
                ImageDraw.Draw(img).polygon(poly.ravel().tolist(), outline=1, fill=None)
                polygon_im += np.array(img, dtype=bool)

        for pixel_list in obj_struct['pixelList']:
            polygon_im[pixel_list[:, 0], pixel_list[:, 1]] = 2

        return polygon_im


if __name__ == '__main__':
    start_time = time.time()

    img_dir, task_dir = sis_utils.get_task_img_folder()

    model_dir = ['confmap_uab_UnetCrop_aemo_comb_xfold0_1_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32',
                 'confmap_uab_UnetCrop_aemo_comb_xfold1_1_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32',
                 'confmap_uab_UnetCrop_aemo_comb_xfold2_1_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32',
                 ]
    model_name = ['Fold 0', 'Fold 1', 'Fold 2']

    true_agg = []
    conf_agg = []

    for md, mn in zip(model_dir, model_name):
        print('Scoring {}...'.format(mn))

        gt_dir = r'/home/lab/Documents/bohao/data/aemo/aemo_union'
        rgb_dir = r'/home/lab/Documents/bohao/data/aemo'
        conf_dir = os.path.join(task_dir, md)
        conf_files = glob(os.path.join(conf_dir, '*.npy'))
        true_all = []
        conf_all = []
        area_all = []
        ious_all = []

        for i_name in conf_files:
            print('\tEvaluating {}...'.format(os.path.basename(i_name)[:-8]))

            conf_im = ersa_utils.load_file(i_name)
            gt_file = os.path.join(gt_dir, '{}comb.tif'.format(os.path.basename(i_name)[:-8]))
            gt = ersa_utils.load_file(gt_file)

            rgb_file = os.path.join(rgb_dir, '{}_rgb.tif'.format(os.path.basename(i_name)[:-12]))
            rgb = ersa_utils.load_file(rgb_file)
            bm = 1 - get_blank_regions(rgb)

            conf_im = bm * conf_im
            gt = bm * gt

            cmos = ConfMapObjectScoring()
            gt_obj = cmos.add_object_wise_info(gt)
            gt_obj = cmos.group_objects(gt_obj, gt)
            cm_obj = cmos.add_object_wise_info(conf_im)

            cm_obj = cmos.link_objects(gt_obj, cm_obj)

            conf, true, area, ious = cmos.scoring_func(gt_obj, cm_obj)
            conf_all.append(conf)
            true_all.append(true)
            area_all.append(area)
            ious_all.append(ious)

            conf_agg.append(conf)
            true_agg.append(true)

        true_fold = np.concatenate(true_all)
        conf_fold = np.concatenate(conf_all)
        area_fold = np.concatenate(area_all)
        ious_fold = np.concatenate(ious_all)

        ersa_utils.save_file(os.path.join(task_dir, '{}_true.npy'.format(mn)), true_fold)
        ersa_utils.save_file(os.path.join(task_dir, '{}_conf.npy'.format(mn)), conf_fold)
        ersa_utils.save_file(os.path.join(task_dir, '{}_area.npy'.format(mn)), area_fold)
        ersa_utils.save_file(os.path.join(task_dir, '{}_ious.npy'.format(mn)), ious_fold)

        p, r, _ = precision_recall_curve(true_fold, conf_fold)
        plt.plot(r[1:], p[1:], linewidth=3, label=mn + ' largest recall={:.3f}'.format(r[1]))

    p, r, _ = precision_recall_curve(np.concatenate(true_agg), np.concatenate(conf_agg))
    plt.plot(r[1:], p[1:], '--', linewidth=3, label='Aggregate' + ' largest recall={:.3f}'.format(r[1]))

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Object-wise PR Curve Comparison')
    plt.legend()
    plt.tight_layout()
    #plt.savefig(os.path.join(img_dir, 'pr_cmp_uab_xfold_cust.png'))
    plt.show()
