import os
import json
from glob import glob
from shutil import copyfile

orig_train_dir = r'/media/ei-edl01/data/uab_datasets/sp/CA/data/Original_Tiles'
orig_test_dir = r'/media/ei-edl01/data/uab_datasets/sp/CA/data/test'
dest_dir = r'/media/ei-edl01/data/uab_datasets/spca_sub/data/Original_Tiles'

json_file = r'/media/ei-edl01/data/uab_datasets/sp/CA/fold1info.json'
with open(json_file, 'r') as f:
    ds = json.load(f)

idx_dict = {'Fresno': {}, 'Modesto': {}, 'Stockton': {}}
panel_dict = {'Fresno': {}, 'Modesto': {}, 'Stockton': {}}
cnt = {'Fresno': 1, 'Modesto': 1, 'Stockton': 1}

for key in ds.keys():
    for item in ds[key]:
        panel_dict[item['city_name']][item['image_prefix']] = item['n_panel_objects']

# train files
rgb_files = [path.split('/')[-1] for path in sorted(glob(os.path.join(orig_train_dir, '*.jpg')))]
gt_files = [path.split('/')[-1] for path in sorted(glob(os.path.join(orig_train_dir, '*.png')))]

for rgb_name in rgb_files:
    city_name = ''.join(i for i in rgb_name.split('-')[0] if not i.isdigit())
    idx = rgb_name.split('-')[-1].split('_')[0]
    if idx not in idx_dict[city_name]:
        idx_dict[city_name][idx] = cnt[city_name]
        cnt[city_name] += 1

for rgb_name, gt_name in zip(rgb_files, gt_files):
    idx_rgb = rgb_name.split('-')[-1].split('_')[0]
    idx_gt = gt_name.split('-')[-1].split('_')[0]
    assert idx_rgb == idx_gt
    city_name = ''.join(i for i in rgb_name.split('-')[0] if not i.isdigit())
    new_idx = idx_dict[city_name][idx_rgb]
    new_rgb_name = ''.join(i for i in rgb_name.split('-')[0] if not i.isdigit()) + str(new_idx) + '_' + rgb_name.split('_')[-1]
    new_gt_name = ''.join(i for i in gt_name.split('-')[0] if not i.isdigit()) + str(new_idx) + '_' + gt_name.split('_')[-1]
    if panel_dict[city_name][idx_rgb] > 0:
        print(new_rgb_name, new_gt_name)
        copyfile(os.path.join(orig_train_dir, rgb_name), os.path.join(dest_dir, new_rgb_name))
        copyfile(os.path.join(orig_train_dir, gt_name), os.path.join(dest_dir, new_gt_name))

cnt['Fresno'] += 250
cnt['Modesto'] += 250
cnt['Stockton'] += 250

# test files
rgb_files = [path.split('/')[-1] for path in sorted(glob(os.path.join(orig_test_dir, '*.jpg')))]
gt_files = [path.split('/')[-1] for path in sorted(glob(os.path.join(orig_test_dir, '*.png')))]

for rgb_name in rgb_files:
    city_name = ''.join(i for i in rgb_name.split('-')[0] if not i.isdigit())
    idx = rgb_name.split('-')[-1].split('_')[0]
    if idx not in idx_dict[city_name]:
        idx_dict[city_name][idx] = cnt[city_name]
        cnt[city_name] += 1

for rgb_name, gt_name in zip(rgb_files, gt_files):
    idx_rgb = rgb_name.split('-')[-1].split('_')[0]
    idx_gt = gt_name.split('-')[-1].split('_')[0]
    assert idx_rgb == idx_gt
    city_name = ''.join(i for i in rgb_name.split('-')[0] if not i.isdigit())
    new_idx = idx_dict[city_name][idx_rgb]
    new_rgb_name = ''.join(i for i in rgb_name.split('-')[0] if not i.isdigit()) + str(new_idx) + '_' + rgb_name.split('_')[-1]
    new_gt_name = ''.join(i for i in gt_name.split('-')[0] if not i.isdigit()) + str(new_idx) + '_' + gt_name.split('_')[-1]
    if panel_dict[city_name][idx_rgb] > 0:
        print(new_rgb_name, new_gt_name)
        copyfile(os.path.join(orig_test_dir, rgb_name), os.path.join(dest_dir, new_rgb_name))
        copyfile(os.path.join(orig_test_dir, gt_name), os.path.join(dest_dir, new_gt_name))
