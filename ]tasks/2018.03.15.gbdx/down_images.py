import os
import skimage
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from gbdxtools import Interface
from gbdxtools import TmsImage
import utils


bboxes = [(-71.99158530555556,
           41.57871111111111,
           -71.98236708333333,
           41.58551122222222),
          (-72.200405, 41.73770069444445, -72.19118611111111, 41.74451727777778),
          (-72.20057972222223, 41.71711886111111, -72.19136377777778, 41.7239355),
          (-72.46543647222222, 41.87587191666667, -72.4562258888889, 41.88270938888889),
          (-72.46561775, 41.83470894444444, -72.45641302777778, 41.84154652777778),
          (-72.47484972222223,
           41.82101005555556,
           -72.46564791666667,
           41.82784841666667),
          (-72.55716933333333, 41.89664275, -72.5479655, 41.9034875),
          (-72.61221977777777, 41.91730275, -72.60301883333332, 41.92415183333333),
          (-72.61220513888888,
           41.924163138888886,
           -72.60300319444444,
           41.93101222222222),
          (-72.65867072222223,
           41.54686894444444,
           -72.64952716666667,
           41.55372233333333),
          (-72.6678036388889, 41.54687583333333, -72.65866105555556, 41.55372994444444),
          (-72.71343736111112,
           41.60178719444445,
           -72.70429186111112,
           41.60864486111111)]

imstocat = ['10400100279E1C00',
            '1040010021A63E00',
            '10400100279E1C00',
            '1040010021200B00',
            '1040010027460D00',
            '1040010021A63E00',
            '1040010021200B00',
            '1040010021A63E00',
            '1040010033CCDF00',
            '1040010021200B00',
            '1040010033CCDF00',
            '1040010033CCDF00']

imnames = ['205770_ne',
           '150830_sw',
           '150820_nw',
           '075880_se',
           '075865_se',
           '075860_sw',
           '050885_ne',
           '035895_se',
           '035895_ne',
           '025760_sw',
           '020760_se',
           '010780_sw']

gbdx = Interface()

[gbdx.ordering.order(imid) for imid in imstocat]  # ORDER IMAGES IF NEEDED

img_dir, task_dir = utils.get_task_img_folder()

for i in tqdm(range(len(imstocat))):
    imprel = TmsImage(imstocat[i])  # READ IMAGE
    aoi = imprel.aoi(bbox=list(bboxes[i]))
    print(type(aoi.read()))
    '''l = skimage.img_as_ubyte(imprel.aoi(bbox=list(bboxes[i])).read())  # CROP TO BBOX AND CONVERT TO RGB
    img_name = os.path.join(r'/media/ei-edl01/user/bh163/figs/2018.03.15.gbdx/gbdx_geotiff',
                            imnames[i] + '.tif')
    imprel.geotiff(path=img_name, proj='EPSG:4326', bands=[4,2,1])'''
    #Image.fromarray(l).save(os.path.join(img_dir, imnames[i] + '.jpg'))
