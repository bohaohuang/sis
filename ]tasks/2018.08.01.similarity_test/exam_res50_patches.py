import os
import pandas as pd
import matplotlib.pyplot as plt
import utils
from make_res50_features import make_res50_features
from run_tsne import run_tsne, plot_tsne, view_encoded_patches

model_name = 'deeplab'
perplex = 25
do_tsne = False

# 1. make features
img_dir, task_dir = utils.get_task_img_folder()
feature_file_name, patch_file_name, ps, patchDir, idx = make_res50_features(model_name, task_dir, GPU=0,
                                                                            force_run=False)
feature = pd.read_csv(feature_file_name, sep=',', header=None).values
with open(patch_file_name, 'r') as f:
    patch_names = f.readlines()

# 2. do tsne
if do_tsne:
    file_name = os.path.join(task_dir, '{}_inria_p{}.npy'.format(model_name, perplex))
    feature_encode = run_tsne(feature, file_name, perplex=perplex, force_run=False)
    plot_tsne(feature_encode, patch_names, rand_percent=1, show_id=True)
    #plt.savefig(os.path.join(img_dir, 'tsne_unet_inria_n{}_all'.format(perplex)))
    plt.show()

# view patches
img_ids = [17098, 31759, 23864, 43949, 30788, 28843]
view_encoded_patches(img_ids, patch_names, patchDir, model_name, img_dir, 'tsne_encoded')
