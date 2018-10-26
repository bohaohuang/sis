import os
import cv2
import time
import scipy.spatial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image, ImageDraw
from scipy.spatial import KDTree
from sklearn.metrics import precision_recall_curve
from skimage import measure
import utils
import ersa_utils
from nn import nn_utils


class spClass_confMapToPolygonStructure_v2:
    version = 1
    # ------------ panel params ------------
    minRegion = 5  # any detected regions must be at least this large
    maxThreshold = 0.5 # 50
    minThreshold = 0.2 # 20
    # ------------ polygon params ------------
    epsilon = 2
    linkingRadius = 55
    # ------------ commercial params ------------
    commercialAreaThreshold = 1500  # Any panel above this threshold (but below maxRegion threshold) is labeled as commercial
    commercialPanelDensityThreshold = 0.2  # Any panel that is within a region with panel density above this threshold is labeled as commercial
    commercialNeighborhoodRadius = 50

    def __init__(self):
        self.objectStructure = pd.DataFrame(
            columns=['iLocation', 'jLocation', 'pixelList', 'confidence', 'area', 'maxIntensity', 'isCommercial'])
        self.objectStructure.pixelList = self.objectStructure.pixelList.astype(object)

    def dropStructures(self):
        self.objectStructure = self.objectStructure.iloc[0:0]
        self.__init__()

    def getConfigs(self):
        return "minR{:d}-maxT{:d}-minT{:d}-ComA{:d}-DenT{:f}-Rad{:d}".format(
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
            else:
                conf.append(0)
        else:
            conf.append(0)
        true.append(1)
    for i in range(len(pp_house_id)):
        if pp_house_id[i] == -1:
            true.append(0)
            conf.append(pl_pp_cf[i])
    return np.array(conf), np.array(true)


if __name__ == '__main__':
    start_time = time.time()

    img_dir, task_dir = utils.get_task_img_folder()

    model_dir = ['conf_map_unet_aemo_hist_0_hist_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1',
                 'conf_map_unet_aemo_1_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1',
                 'conf_map_unet_aemo_scratch_3_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1']
    model_name = ['naive', 'panel', 'tile']

    for md, mn in zip(model_dir, model_name):
        conf_dir = os.path.join(task_dir, md)
        gt_dir = r'/home/lab/Documents/bohao/data/aemo/aemo_hist/'
        conf_files = glob(os.path.join(conf_dir, '*.npy'))
        true_all = []
        conf_all = []

        for i_name in conf_files:
            conf_im = ersa_utils.load_file(i_name)
            gt_file = os.path.join(gt_dir, '{}.tif'.format(os.path.basename(i_name)[:-4]))
            gt = ersa_utils.load_file(gt_file)

            # Instantiate the class
            ppObj = spClass_confMapToPolygonStructure_v2()
            # Map tp objects
            ppObj.confidenceImageToObjectStructure(conf_im)
            # APPROXIMATE EACH OBJECT WITH POLYGON
            ppObj.addPolygonToObjectStructure(conf_im)
            gtObj = spClass_confMapToPolygonStructure_v2()
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
        plt.plot(r, p, linewidth=3, label=mn)

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Object-wise PR Curve Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'pr_cmp2.png'))
    plt.show()
    
    print('duration = {}'.format(time.time() - start_time))
