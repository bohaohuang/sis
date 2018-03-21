import os
import imageio
from glob import glob
from tqdm import tqdm
import util_functions

pred_dir = r'/media/batcave/personal/huang.bohao/CA'
gt_dir = r'/media/ei-edl01/data/uab_datasets/sp/CA/data/test'
pred_imgs = sorted(glob(os.path.join(pred_dir, '*.png')))

A = 0
B = 0
for pred_img in tqdm(pred_imgs):
    img_name = os.path.basename(pred_img)
    tile_id = img_name.split('_')[0]
    gt_img_name = '{}_GT.png'.format(tile_id)
    pred = imageio.imread(pred_img)/255
    gt = imageio.imread(os.path.join(gt_dir, gt_img_name))
    a, b = util_functions.iou_metric(gt, pred, truth_val=1, divide_flag=True)
    A += a
    B += b
print(A/B)
