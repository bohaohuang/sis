import os
from glob import glob
import numpy as np
import utils

if __name__ == '__main__':
    img_dir, task_dir = utils.get_task_img_folder()
    record_files = glob(os.path.join(task_dir, '*.npy'))

    txt_file = os.path.join(task_dir, 'exp_record.txt')
    with open(txt_file, 'w'):
        pass

    for file in record_files:
        model_name = file.split('/')[-1].split('.npy')[0]
        iou = dict(np.load(file).tolist())
        iou_mean = []
        for key, val in iou.items():
            iou_mean.append(val)
        str2write = '{}: {}\n'.format(model_name, np.mean(iou_mean))
        print(str2write.strip())
        with open(txt_file, 'a') as file:
            file.write(str2write)
