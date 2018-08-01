import os
import imageio
from tqdm import tqdm
from shutil import copy2
import uabRepoPaths
import bohaoCustom.uabPreprocClasses as bPreproc
import uabPreprocClasses
import uab_collectionFunctions
import uab_DataHandlerFunctions


model_name = 'unet'
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
rgb_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles'

if model_name == 'unet':
    pred_dir = r'/hdd/Results/domain_loo_all/UnetCrop_inria_aug_leave_{}_0_PS(572, 572)_BS5_EP100_LR0.0001_' \
               r'DS60_DR0.1_SFN32/inria/pred'
    new_ds_dir = os.path.join(uabRepoPaths.dataPath, 'inria_unet_retrain', 'data', 'Original_Tiles')
    if not os.path.exists(new_ds_dir):
        os.makedirs(new_ds_dir)

        for city_cnt in range(len(city_list)):
            pbar = tqdm(range(1, 37))
            for img_cnt in pbar:
                pbar.set_description('Copying city: {}'.format(city_list[city_cnt]))
                pred_file = os.path.join(pred_dir.format(city_cnt), '{}{}.png'.format(city_list[city_cnt], img_cnt))
                copy2(pred_file, os.path.join(new_ds_dir, '{}{}_GT.png'.format(city_list[city_cnt], img_cnt)))

                rgb_file = os.path.join(rgb_dir, '{}{}_RGB.tif'.format(city_list[city_cnt], img_cnt))
                rgb = imageio.imread(rgb_file)
                imageio.imsave(os.path.join(new_ds_dir, '{}{}_RGB.png'.format(city_list[city_cnt], img_cnt)), rgb)

    blCol = uab_collectionFunctions.uabCollection('inria_unet_retrain')
    opDetObj = bPreproc.uabOperTileDivide(255)  # inria GT has value 0 and 255, we map it back to 0 and 1
    # [3] is the channel id of GT
    rescObj = uabPreprocClasses.uabPreprocMultChanOp([], 'GT_Divide.tif', 'Map GT to (0, 1)', [0], opDetObj)
    blCol.readMetadata()
    rescObj.run(blCol)
    blCol.readMetadata()
    img_mean = blCol.getChannelMeans([1, 2, 3])  # get mean of rgb info

    # extract patches
    extrObj = uab_DataHandlerFunctions.uabPatchExtr([1, 2, 3, 4],
                                                    cSize=(572, 572),
                                                    numPixOverlap=184,
                                                    extSave=['jpg', 'jpg', 'jpg', 'png'],
                                                    isTrain=True,
                                                    gtInd=3,
                                                    pad=92)
    patchDir = extrObj.run(blCol)
