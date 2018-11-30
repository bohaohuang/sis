import os
import cv2
import time
import scipy.spatial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, confusion_matrix
from PIL import Image, ImageDraw
from skimage import measure
import utils
import ersa_utils


class spClass_confMapToPolygonStructure_v2:
    version = 1
    # ------------ panel params ------------
    minRegion = 5  # any detected regions must be at least this large
    minThreshold = 0.5
    # ------------ polygon params ------------
    epsilon = 2
    linkingRadius = 55
    # ------------ commercial params ------------

    def __init__(self, mt, cm_min, cm_max):
        self.maxThreshold = mt
        self.commercialAreaThreshold_min = cm_min
        self.commercialAreaThreshold_max = cm_max
        self.objectStructure = pd.DataFrame(
            columns=['iLocation', 'jLocation', 'pixelList', 'confidence', 'area', 'maxIntensity', 'isCommercial'])
        self.objectStructure.pixelList = self.objectStructure.pixelList.astype(object)

    def getConfigs(self):
        return "minR{:d}-maxT{:f}-minT{:f}-Com-min{:d}-Com-max{:d}".format(
            self.minRegion, self.maxThreshold, self.minThreshold, self.commercialAreaThreshold_min,
            self.commercialAreaThreshold_max)

    """  MAP THE POLYGONS ONTO AN IMAGE FOR DISPLAY """
    def polygonStructureToImage(self, confidenceImage):  # polygon cords in form of xy
        H, W = confidenceImage.shape
        polygonImage = np.zeros((H, W))
        for poly in self.objectStructure.polygon:
            img = Image.new('L', (W, H), 0)
            ImageDraw.Draw(img).polygon(poly.ravel().tolist(), outline=1, fill=1)
            polygonImage += np.array(img, dtype=bool)
        return polygonImage

    """  CREATE REGIONS FROM CONFIDENCE MAPS """
    def confidenceImageToObjectStructure(self, confidenceImage):
        imThresh = confidenceImage >= self.minThreshold
        imLabel = measure.label(imThresh)
        regProps = measure.regionprops(imLabel, confidenceImage)
        for rp in regProps:
            if rp.area >= self.minRegion and rp.max_intensity >= self.maxThreshold:
                temp = [*[int(c) for c in rp.centroid], rp.coords, rp.mean_intensity, rp.area, rp.max_intensity, 0]
                self.objectStructure = self.objectStructure.append(
                    dict(zip(
                        ['iLocation', 'jLocation', 'pixelList', 'confidence', 'area', 'maxIntensity', 'isCommercial'],
                        temp)), ignore_index=True)

    def addCommercialLabelToObjectStructure(self, confidenceImage, commercial=True, clear_coords=False):
        if not self.objectStructure.empty:
            """ IDENTIFY USING CONNECTED COMPONENT SIZE """
            objAreas = self.objectStructure['area'].values
            isCommercialSize = []
            for area in objAreas:
                isCommercialSize.append(self.commercialAreaThreshold_min <= area < self.commercialAreaThreshold_max)

            self.objectStructure.isCommercial = isCommercialSize

            if clear_coords:
                self.objectStructure = self.objectStructure[self.objectStructure['isCommercial'] == commercial].\
                    reset_index(drop=True)

    def addPolygonToObjectStructure(self, predIm):
        polygons = [list() for _ in range(self.objectStructure.shape[0])]
        for r, reg in self.objectStructure.iterrows():
            dummyImage = np.zeros(predIm.shape)
            pixl = reg['pixelList'].transpose()
            dummyImage[pixl[0], pixl[1]] = 1
            _, contours, _ = cv2.findContours(dummyImage.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(np.array(contours).shape) == 1:
                contours = np.expand_dims(np.concatenate(contours, axis=0), axis=0)
            polygons[r] = measure.approximate_polygon(np.squeeze(contours), self.epsilon)  # Douglas Peucker
        self.objectStructure['polygon'] = pd.Series(polygons, index=self.objectStructure.index).astype(object)

    """ LINK EACH HOUSE WITH ONE (OR MORE) DETECTED PANELS """
    def linkHousesToObjects(self, housePixelCoordinates, houseIdList):
        i_coords = np.array(self.objectStructure['iLocation'])
        j_coords = np.array(self.objectStructure['jLocation'])

        ptsDet = np.stack([i_coords, j_coords], axis=1)
        Mdl = scipy.spatial.KDTree(housePixelCoordinates)

        distance, ids = Mdl.query(ptsDet, distance_upper_bound=self.linkingRadius)
        houseId = np.zeros_like(distance)
        for cnt, (d, i) in enumerate(zip(distance, ids)):
            if d <= self.linkingRadius:
                houseId[cnt] = houseIdList[ids[cnt]]
            else:
                houseId[cnt] = -1
        self.objectStructure['iOut'] = pd.Series(houseId)


def get_intersection(A, B):
    inter = np.array([x for x in set(tuple(x) for x in A) & set(tuple(x) for x in B)])
    union = np.array([x for x in set(tuple(x) for x in A) | set(tuple(x) for x in B)])
    return inter, union


def get_blank_regions(img):
    img_tmp = img.astype(np.float32)
    img_tmp = np.sum(img_tmp, axis=2)
    blank_mask = (img_tmp < 0.1).astype(np.int)
    return blank_mask


def scoring_func2(gtObj, ppObj, iou_th=0.5):
    conf = []
    true = []

    cm_idc = np.array(gtObj.objectStructure['isCommercial'])
    pl_gt = np.array(gtObj.objectStructure['pixelList'])
    pl_pp = np.array(ppObj.objectStructure['pixelList'])
    pl_pp_cf = np.array(ppObj.objectStructure['confidence'])

    panel_num = pl_gt.shape[0]
    pp_house_id = ppObj.objectStructure['iOut'].tolist()
    for i in range(panel_num):
        if i in pp_house_id:
            pp_i = pp_house_id.index(i)
            inter, union = get_intersection(pl_gt[i], pl_pp[pp_i])
            iou = inter.shape[0] / union.shape[0]
            if iou >= iou_th:
                conf.append(pl_pp_cf[pp_i])
                true.append(1)
            else:
                conf.append(pl_pp_cf[pp_i])
                true.append(0)
        else:
            conf.append(-1000)
            true.append(1)
    for i in range(len(pp_house_id)):
        if pp_house_id[i] == -1:
            if i < len(cm_idc):
                true.append(0)
                conf.append(pl_pp_cf[i])
    return np.array(conf), np.array(true)


if __name__ == '__main__':
    start_time = time.time()
    img_dir, task_dir = utils.get_task_img_folder()

    model_dir = ['confmap_uab_UnetCrop_aemo_comb_xfold0_1_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32',
                 'confmap_uab_UnetCrop_aemo_comb_xfold1_1_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32',
                 'confmap_uab_UnetCrop_aemo_comb_xfold2_1_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32',
                 ]
    model_name = ['Fold 0', 'Fold 1', 'Fold 2']

    true_agg = []
    conf_agg = []

    iou_mt = 0.5
    pbar = tqdm(range(0, 1000, 40))
    for min_th in pbar:
        conf_mat = []

        max_th = min_th + 40
        for md, mn in zip(model_dir, model_name):
            conf_dir = os.path.join(task_dir, md)
            gt_dir = r'/home/lab/Documents/bohao/data/aemo/aemo_union'
            rgb_dir = r'/home/lab/Documents/bohao/data/aemo'
            conf_files = glob(os.path.join(conf_dir, '*.npy'))
            true_all = []
            conf_all = []

            for i_name in conf_files:
                pbar.set_description('Scoring {}...'.format(os.path.basename(i_name)[:-8]))
                conf_im = ersa_utils.load_file(i_name)
                gt_file = os.path.join(gt_dir, '{}comb.tif'.format(os.path.basename(i_name)[:-8]))
                gt = ersa_utils.load_file(gt_file)

                rgb_file = os.path.join(rgb_dir, '{}_rgb.tif'.format(os.path.basename(i_name)[:-12]))
                rgb = ersa_utils.load_file(rgb_file)
                bm = 1 - get_blank_regions(rgb)

                conf_im = bm * conf_im
                gt = bm * gt

                # Instantiate the class
                ppObj = spClass_confMapToPolygonStructure_v2(iou_mt, min_th, max_th)
                # Map tp objects
                ppObj.confidenceImageToObjectStructure(conf_im)
                # APPROXIMATE EACH OBJECT WITH POLYGON
                ppObj.addPolygonToObjectStructure(conf_im)
                ppObj.addCommercialLabelToObjectStructure(conf_im, clear_coords=True)

                gtObj = spClass_confMapToPolygonStructure_v2(iou_mt, min_th, max_th)
                gtObj.confidenceImageToObjectStructure(gt)
                gtObj.addPolygonToObjectStructure(gt)
                gtObj.addCommercialLabelToObjectStructure(gt, clear_coords=True)

                # link the house
                i_coords = np.array(gtObj.objectStructure['iLocation'])
                j_coords = np.array(gtObj.objectStructure['jLocation'])
                housePixelCoordinates = np.stack([i_coords, j_coords], axis=1)
                houseId = np.arange(housePixelCoordinates.shape[0])
                ppObj.linkHousesToObjects(housePixelCoordinates, houseId)

                conf, true = scoring_func2(gtObj, ppObj)
                conf_all.append(conf)
                true_all.append(true)

                conf_agg.append(conf)
                true_agg.append(true)

            p, r, th = precision_recall_curve(np.concatenate(true_all), np.concatenate(conf_all))
            conf_mat.append(confusion_matrix(np.concatenate(true_all),
                                             (np.concatenate(conf_all) > th[1]).astype(int)).ravel())
            plt.plot(r[1:], p[1:], linewidth=3, label=mn + ' largest recall={:.3f}'.format(r[1]))

            true_pred = np.stack([np.concatenate(true_all), np.concatenate(conf_all)])
            pr_save_name = os.path.join(task_dir, '{}_{}-{}_true_pred.npy'.format(md, min_th, max_th))
            ersa_utils.save_file(pr_save_name, true_pred)

        p, r, th = precision_recall_curve(np.concatenate(true_agg), np.concatenate(conf_agg))
        plt.plot(r[1:], p[1:], '--', linewidth=3, label='Aggregate' + ' largest recall={:.3f}'.format(r[1]))
        conf_mat.append(confusion_matrix(np.concatenate(true_agg),
                                         (np.concatenate(conf_agg) > th[1]).astype(int)).ravel())

        save_file_name = os.path.join(task_dir, 'confmat_comb_{}-{}.npy'.format(min_th, max_th))
        ersa_utils.save_file(save_file_name, conf_mat)

        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title('Object-wise PR Curve Comparison ({}<=Panel Size<{})'.format(min_th, max_th))

        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, 'pr_cmp_uab_xfold_comb_{}-{}.png'.format(min_th, max_th)))

        plt.close()
