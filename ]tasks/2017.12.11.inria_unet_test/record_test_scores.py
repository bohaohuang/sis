import os
import operator
from glob import glob
import numpy as np
import sis_utils

if __name__ == '__main__':
    img_dir, task_dir = sis_utils.get_task_img_folder()
    record_files = glob(os.path.join(task_dir, '*.npy'))
    record_files = sorted(record_files)
    record_files = [a for a in record_files if 'CT' not in a]
    iou_record = {}

    for file in record_files:
        model_name = file.split('/')[-1].split('.npy')[0]
        iou = dict(np.load(file).tolist())
        iou_mean = []
        for key, val in iou.items():
            iou_mean.append(val)
        iou_record[model_name] = np.mean(iou_mean)

    sorted_iou_record = sorted(iou_record.items(), key=operator.itemgetter(1), reverse=True)
    txt_file = os.path.join(task_dir, 'exp_record.txt')
    with open(txt_file, 'w') as file:
        for item in sorted_iou_record:
            str2write = '{}: {}\n'.format(item[0], item[1])
            print(str2write.strip())
            file.write(str2write)
