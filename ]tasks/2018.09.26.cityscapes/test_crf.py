import os
import numpy as np
from glob import glob
from tqdm import tqdm
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_gaussian, create_pairwise_bilateral
import ersa_utils

pred_dir = r'/home/lab/Documents/bohao/data/deeplab_model/vis/raw_segmentation_results'
pred_files = sorted(glob(os.path.join(pred_dir, '*.png')))
rgb_dir = r'/home/lab/Documents/bohao/data/deeplab_model/vis/segmentation_results'
rgb_files = sorted(glob(os.path.join(rgb_dir, '*_prediction.png')))

for g_window in [10]:
    for b_sdim in range(60, 121, 20):
        for b_schan in range(5, 22, 4):
            print('g_window={}, b_s_dim={}, b_schan={}'.format(g_window, b_sdim, b_schan))

            save_dir = r'/home/lab/Documents/bohao/data/deeplab_model/post_{}_{}_{}'.format(g_window, b_sdim, b_schan)
            ersa_utils.make_dir_if_not_exist(save_dir)

            for p_file, r_file in tqdm(zip(pred_files, rgb_files), total=len(pred_files)):
                # take one sample file
                pred = ersa_utils.load_file(p_file)
                rgb = ersa_utils.load_file(r_file)

                # define 2d class
                d = dcrf.DenseCRF2D(pred.shape[0], pred.shape[1], 33)

                # get unary
                unary = unary_from_labels(pred, 33, 0.5)
                d.setUnaryEnergy(unary)

                # get pairwise potentials
                feats = create_pairwise_gaussian(sdims=(g_window, g_window), shape=pred.shape[:2])
                d.addPairwiseEnergy(feats, compat=3,
                                    kernel=dcrf.DIAG_KERNEL,
                                    normalization=dcrf.NORMALIZE_SYMMETRIC)
                feats = create_pairwise_bilateral(sdims=(b_sdim, b_sdim), schan=(13, 13, 13),
                                                  img=rgb, chdim=2)
                d.addPairwiseEnergy(feats, compat=10,
                                    kernel=dcrf.DIAG_KERNEL,
                                    normalization=dcrf.NORMALIZE_SYMMETRIC)

                Q = d.inference(5)

                res = np.argmax(Q, axis=0).reshape((pred.shape[0], pred.shape[1])) + 1

                #if len(res) == len(pred):
                #    assert np.all(np.unique(res) == np.unique(pred))

                pred_name = os.path.join(save_dir, os.path.basename(p_file))
                ersa_utils.save_file(pred_name, res.astype(np.uint8))
