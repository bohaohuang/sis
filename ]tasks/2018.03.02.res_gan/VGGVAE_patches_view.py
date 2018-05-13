import os
import csv
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
import utils

run_clustering = True
model_dir = r'/hdd6/Models/VGGVAE/VGGVAE_inria_z500_0_PS(256, 256)_BS5_EP400_LR1e-05_DS200.0_DR0.5_SFN32'
img_dir, task_dir = utils.get_task_img_folder()

for perplexity in [20, 40, 60]:
    for learning_rate in [200, 400, 600, 800]:
        npy_file_name = os.path.join(task_dir, '{}_p{}_l{}.npy'.
                                     format(model_dir.split('/')[-1], perplexity, learning_rate))
        city_order = {'aus': 0, 'chi': 1, 'kit': 2, 'tyr': 3, 'vie': 4}
        city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
        np.random.seed(1004)

        if run_clustering:
            file_name = os.path.join(task_dir, '{}_inria.csv'.format(model_dir.split('/')[-1]))
            features = []
            with open(file_name, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    features.append(row)

            feature_encode = TSNE(n_components=2, perplexity=20, learning_rate=600, verbose=True).fit_transform(features)
            np.save(npy_file_name, feature_encode)
        else:
            feature_encode = np.load(npy_file_name)
        feature_encode = np.array(feature_encode)

        random_idx = np.random.binomial(1, 0.2, feature_encode.shape[0])
        patch_ids = np.arange(feature_encode.shape[0])
        feature_encode = feature_encode[random_idx == 1, :]
        patch_ids = patch_ids[random_idx == 1]

        patch_name_fname = os.path.join(task_dir, '{}_inria.txt'.format(model_dir.split('/')[-1]))
        with open(patch_name_fname, 'r') as f:
            patch_name_list = f.readlines()

        '''patch_percent_list = [patch_name_list[i] for i in range(len(patch_name_list)) if random_idx[i] == 1]
        patch_percent = np.zeros(len(patch_percent_list))
        patchDir = r'/hdd/uab_datasets/Results/PatchExtr/inria/chipExtrReg_cSz321x321_pad0'
        for cnt, patch_name in enumerate(tqdm(patch_percent_list)):
            gt = imageio.imread(os.path.join(patchDir, '{}_GT_Divide.png'.format(patch_name.strip())))
            patch_percent[cnt] = np.sum(gt)/(gt.shape[0] * gt.shape[1])'''

        patch_name_list = [city_order[a[:3]] for a in patch_name_list]

        patch_name_code = []
        for i in patch_name_list:
            patch_name_code.append(i)
        patch_name_code = np.array(patch_name_code)
        patch_name_code = patch_name_code[random_idx == 1]

        cmap = plt.get_cmap('Set1').colors
        patch_id = np.arange(feature_encode.shape[0])
        fig = plt.figure(figsize=(15, 8))
        for i in range(5):
            plt.scatter(feature_encode[patch_name_code == i, 0], feature_encode[patch_name_code == i, 1], color=cmap[i],
                        label=city_list[i], edgecolors='k')
        #plt.scatter(feature_encode[:, 0], feature_encode[:, 1], c=patch_percent, cmap=plt.get_cmap('bwr'))
        #plt.colorbar()

        #for i in range(feature_encode.shape[0]):
        #    plt.text(feature_encode[i, 0], feature_encode[i, 1], patch_ids[i])
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('TSNE Projection Result')
        plt.legend()
        #plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, 'vgg_vae_view_p{}_l{}.png'.format(perplexity, learning_rate)))
        #plt.show()
        plt.close(fig)
