import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
import utils
import uabCrossValMaker

# settings
random_seed = 0
img_dir, task_dir = utils.get_task_img_folder()
file_name = os.path.join(task_dir, 'res50_fc1000_inria.csv')
input_size = 321
patchDir = r'/hdd/uab_datasets/Results/PatchExtr/inria/chipExtrReg_cSz321x321_pad0'

features = []
with open(file_name, 'r') as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        features.append(row)
features = np.array(features)

labels = KMeans(n_clusters=5, random_state=random_seed).fit_predict(features)
plt.hist(labels)
plt.show()
