import os
import json

json_file = r'/media/ei-edl01/data/uab_datasets/sp/CA/fold1info.json'
with open(json_file, 'r') as f:
    ds = json.load(f)

cnt = 0
for item in ds['validationImages']:
    if item['n_panel_objects'] > 0:
       cnt += 1
print(cnt/len(ds['validationImages']))
