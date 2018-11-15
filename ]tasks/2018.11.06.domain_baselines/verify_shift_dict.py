import os
import numpy as np
import matplotlib.pyplot as plt
import utils
import ersa_utils

img_dir, task_dir = utils.get_task_img_folder()
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']

plt.figure(figsize=(8, 8))
for city_id in range(5):
    path_to_save = os.path.join(task_dir, 'dtda', city_list[city_id], 'shift_dict.pkl')
    shift_dict = ersa_utils.load_file(path_to_save)

    vals = []
    for n, v in shift_dict.items():
        vals.append(v)
    vals = np.array(vals)

    for i in range(3):
        plt.subplot(311 + i)
        plt.plot(vals[:, i], label=city_list[city_id])

plt.legend()
plt.tight_layout()
plt.show()
