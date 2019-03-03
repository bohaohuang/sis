import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sis_utils
import ersa_utils

img_dir, task_dir = sis_utils.get_task_img_folder()

beta_vals = np.arange(0.5, 1.1, 0.1)
alpha_vals = np.arange(-20, 21, 1)
gamma_vals = np.arange(0.5, 1.6, 0.1)

record_file = os.path.join(task_dir, 'cust_func_record.txt')
records = ersa_utils.load_file(record_file)
record_df = pd.DataFrame(columns=['beta', 'alpha', 'gamma', 'iou'])

for cnt, line in enumerate(records):
    record_df.loc[cnt] = [float(a) for a in line.split(' ')]
record_df.sort_values(['beta', 'alpha', 'gamma'], ascending=[True, True, True])

for beta in beta_vals:
    plt.figure(figsize=(8, 5))
    df_beta = record_df[record_df['beta'] == beta]
    for alpha in alpha_vals:
        df_alpha = df_beta[df_beta['alpha'] == alpha]
        plt.plot(df_alpha['gamma'], df_alpha['iou'], label=alpha)
    plt.legend(ncol=4)
    plt.title('Beta={}'.format(beta))
    plt.xlabel('Alpha')
    plt.ylabel('Gamma')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'pretrained_cust_beta{}.png'.format(beta)))
    plt.close()
