import os
import cv2
import time
import scipy.spatial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
from PIL import Image, ImageDraw
from scipy.spatial import KDTree
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
    commercialAreaThreshold = 1500
    commercialPanelDensityThreshold = 0.2  # Any panel that is within a region with panel density above this threshold is labeled as commercial
    commercialNeighborhoodRadius = 50

    def __init__(self, mt):
        self.maxThreshold = mt
        self.objectStructure = pd.DataFrame(
            columns=['iLocation', 'jLocation', 'pixelList', 'confidence', 'area', 'maxIntensity', 'isCommercial'])
        self.objectStructure.pixelList = self.objectStructure.pixelList.astype(object)

    def getConfigs(self):
        return "minR{:d}-maxT{:f}-minT{:f}-ComA{:d}-DenT{:f}-Rad{:d}".format(
            self.minRegion, self.maxThreshold, self.minThreshold, self.commercialAreaThreshold,
            self.commercialPanelDensityThreshold, self.commercialNeighborhoodRadius)

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

    def addCommercialLabelToObjectStructure(self, confidenceImage, return_sum=False):
        if not self.objectStructure.empty:
            """ IDENTIFY USING CONNECTED COMPONENT SIZE """
            objAreas = self.objectStructure['area']
            isCommercialSize = objAreas >= self.commercialAreaThreshold

            """ IDENTIFY USING PANEL PIXEL DENSITY """
            neighborhoodFilter = np.ones([2 * self.commercialNeighborhoodRadius + 1] * 2)
            dummyImage = np.zeros(confidenceImage.shape)
            pixelList = np.vstack(np.array(self.objectStructure['pixelList'])).transpose()
            dummyImage[pixelList[0], pixelList[1]] = 1

            panelNeighborhoodCount = np.real(
                np.fft.ifft2(np.fft.fft2(dummyImage) * np.fft.fft2(neighborhoodFilter, dummyImage.shape)))  # imfilt
            countThreshold = int(self.commercialPanelDensityThreshold * np.prod(neighborhoodFilter.shape))
            commercialPanelMap = panelNeighborhoodCount > countThreshold
            commercialCenters = np.vstack(np.nonzero(commercialPanelMap)).transpose()

            if commercialCenters.any():
                panelCenters = np.vstack(
                    (self.objectStructure['iLocation'], self.objectStructure['jLocation'])).transpose()
                Mdl = KDTree(commercialCenters)
                # Search for panels with neighborhood radius of a commercial center
                out = Mdl.query_ball_point(panelCenters, self.commercialNeighborhoodRadius)
                isCommercialDensity = [bool(i) for i in out]
            else:
                isCommercialDensity = np.zeros(objAreas.shape, dtype=bool)

            self.objectStructure.isCommercial = isCommercialSize & isCommercialDensity

            if return_sum:
                non_commercial = self.objectStructure.loc[self.objectStructure['isCommercial'] is False]

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
            true.append(0)
            conf.append(pl_pp_cf[i])
    return np.array(conf), np.array(true)


if __name__ == '__main__':
    start_time = time.time()

    img_dir, task_dir = utils.get_task_img_folder()

    model_dir = ['confmap_uab_UnetCrop_aemo_ft_1_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32',
                 'confmap_uab_UnetCrop_aemo_sc_0_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32',
                 'confmap_uab_UnetCrop_aemo_ft_0_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32',
                 'confmap_uab_UnetCrop_aemo_hd_0_PS(572, 572)_BS5_EP20_LR1e-05_DS10_DR0.1_SFN32'
                 ]
    model_name = ['Raw Finetune 1e-3', 'Raw Scratch 1e-3', 'Hist Finetune 1e-3', 'Hard Sample']

    f2s = []
    largest_recall = []

    for iou_mt in tqdm([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        try:
            f2s_m = []
            largest_recall_m = []
            for md, mn in zip(model_dir, model_name):
                conf_dir = os.path.join(task_dir, md)
                gt_dir = r'/home/lab/Documents/bohao/data/aemo/aemo_hist/'
                rgb_dir = r'/home/lab/Documents/bohao/data/aemo'
                conf_files = glob(os.path.join(conf_dir, '*.npy'))
                true_all = []
                conf_all = []

                for i_name in conf_files:
                    conf_im = ersa_utils.load_file(i_name)
                    gt_file = os.path.join(gt_dir, '{}.tif'.format(os.path.basename(i_name)[:-4]))
                    gt = ersa_utils.load_file(gt_file)

                    rgb_file = os.path.join(rgb_dir, '{}_rgb.tif'.format(os.path.basename(i_name)[:-12]))
                    rgb = ersa_utils.load_file(rgb_file)
                    bm = 1 - get_blank_regions(rgb)

                    conf_im = bm * conf_im
                    gt = bm * gt

                    # Instantiate the class
                    ppObj = spClass_confMapToPolygonStructure_v2(iou_mt)
                    # Map tp objects
                    ppObj.confidenceImageToObjectStructure(conf_im)
                    # APPROXIMATE EACH OBJECT WITH POLYGON
                    ppObj.addPolygonToObjectStructure(conf_im)
                    gtObj = spClass_confMapToPolygonStructure_v2(iou_mt)
                    gtObj.confidenceImageToObjectStructure(gt)
                    gtObj.addPolygonToObjectStructure(gt)

                    # link the house
                    i_coords = np.array(gtObj.objectStructure['iLocation'])
                    j_coords = np.array(gtObj.objectStructure['jLocation'])
                    housePixelCoordinates = np.stack([i_coords, j_coords], axis=1)
                    houseId = np.arange(housePixelCoordinates.shape[0])
                    ppObj.linkHousesToObjects(housePixelCoordinates, houseId)

                    conf, true = scoring_func2(gtObj, ppObj)
                    conf_all.append(conf)
                    true_all.append(true)

                p, r, _ = precision_recall_curve(np.concatenate(true_all), np.concatenate(conf_all))
                plt.plot(r[1:], p[1:], linewidth=3, label=mn)

                f2s_m.append(5 * p[1] * r[1] / (4 * p[1] + r[1]))
                largest_recall_m.append(r[1])

            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.title('Object-wise PR Curve Comparison')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(img_dir, 'pr_cmp_uab_iou_mt{}_temp.png'.format(iou_mt)))
            plt.close()

            f2s.append(f2s_m)
            largest_recall.append(largest_recall_m)

            # print('duration = {}'.format(time.time() - start_time))
        except IndexError:
            continue

    f2_name = os.path.join(task_dir, 'f2s.npy')
    lr_name = os.path.join(task_dir, 'lrs.npy')
    ersa_utils.save_file(f2_name, f2s)
    ersa_utils.save_file(lr_name, largest_recall)
