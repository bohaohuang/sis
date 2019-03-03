import pandas as pd
from tqdm import tqdm
import sis_utils
from make_res50_features import make_res50_features
from kernel_distance_matching import distance_matching
from city_building_truth import make_building_truth, make_city_truth
from gmm_cluster import *


model_name = 'unet'
top_cnt = None
force_run = False
xregion = False

if model_name == 'unet':
    base_iou = np.array([55.7, 63.4, 56.9, 53.6, 72.6])
    mmd_iou = np.array([55.8, 64.8, 58.2, 55.7, 71.9])
else:
    base_iou = np.array([63.1, 66.3, 59.9, 54.4, 74.5])
    mmd_iou = np.array([65.2, 65.0, 61.9, 61.8, 74.5])

city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
distance_list = np.zeros(len(city_list))
delta_list = np.zeros(len(city_list))

for target_city in tqdm(range(5)):
    img_dir, task_dir = sis_utils.get_task_img_folder()
    feature_file_name, patch_file_name, ps, patchDir, idx = make_res50_features(model_name, task_dir, GPU=0,
                                                                                force_run=False)
    feature = pd.read_csv(feature_file_name, sep=',', header=None).values
    with open(patch_file_name, 'r') as f:
        patch_names = f.readlines()

    # 2. make city and building truth
    truth_city = make_city_truth(task_dir, model_name, patch_names, force_run=False)
    truth_building = make_building_truth(ps, task_dir, model_name, patchDir, patch_names, force_run=False)
    truth_building = np.ones_like(truth_building)

    # 3. do feature mapping
    if xregion:
        source_feature = select_feature(feature, np.array(idx) >= 6, truth_city, truth_building,
                                        [i for i in range(5)], True)
    else:
        source_feature = select_feature(feature, np.array(idx) >= 6, truth_city, truth_building,
                                        [i for i in range(5) if i != target_city], True)
    target_feature = select_feature(feature, np.array(idx) < 6, truth_city, truth_building, [target_city], False)

    _, dist_record = distance_matching(target_feature, source_feature, top_cnt=top_cnt)

    distance_list[target_city] = np.mean(dist_record)
    delta_list[target_city] = (mmd_iou[target_city] - base_iou[target_city]) / base_iou[target_city]

X = np.arange(len(city_list))
plt.figure(figsize=(8, 6))
plt.subplot(211)
plt.plot(X, distance_list)
plt.xticks(X, city_list)
plt.subplot(212)
plt.plot(X, delta_list)
plt.xticks(X, city_list)
plt.tight_layout()
plt.show()
