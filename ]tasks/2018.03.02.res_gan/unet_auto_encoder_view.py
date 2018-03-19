import os
import csv
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

run_clustering = False
file_name = os.path.join(r'/hdd6/temp', 'encoded_unet.npy')
cmap = plt.get_cmap('Set1').colors

if run_clustering:
    file_name = os.path.join(r'/hdd6/temp', 'encoded_unet.csv')
    features = []
    with open(file_name, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            features.append(row)

    feature_encode = TSNE(n_components=2, perplexity=50).fit_transform(features)
    np.save(file_name, feature_encode)
else:
    feature_encode = np.load(file_name)
    for i in range(1000):
        plt.scatter(feature_encode[i, 0], feature_encode[i, 1], edgecolors=cmap[i%5])
    plt.xlim([-50, 50])
    plt.ylim([-50, 50])
    plt.show()
