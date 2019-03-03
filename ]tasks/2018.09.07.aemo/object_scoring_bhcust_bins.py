import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import sis_utils
import ersa_utils

if __name__ == '__main__':
    img_dir, task_dir = sis_utils.get_task_img_folder()

    model_name = ['Fold 0', 'Fold 1', 'Fold 2']

    size_ths = [0, 200, 400, 600, 800, 1000, np.inf]
    area_all = []
    true_all = [[] for i in range(len(size_ths) - 1)]
    conf_all = [[] for i in range(len(size_ths) - 1)]
    cnt_list_all = np.zeros(len(size_ths) - 1)

    for size_cnt in range(len(size_ths[:-1])):
        plt.figure()
        for plt_cnt, mn in enumerate(model_name):
            cnt_list = np.zeros(len(size_ths) - 1)
            true_list = [[] for i in range(len(size_ths) - 1)]
            conf_list = [[] for i in range(len(size_ths) - 1)]

            area_fold = ersa_utils.load_file(os.path.join(task_dir, '{}_area.npy'.format(mn)))
            ious_fold = ersa_utils.load_file(os.path.join(task_dir, '{}_ious.npy'.format(mn)))
            conf_fold = ersa_utils.load_file(os.path.join(task_dir, '{}_conf.npy'.format(mn)))
            true_fold = ersa_utils.load_file(os.path.join(task_dir, '{}_true.npy'.format(mn)))

            area_all.append(area_fold)
            min_th = size_ths[size_cnt]
            max_th = size_ths[size_cnt + 1]

            for cnt, area in enumerate(area_fold):
                if min_th <= area < max_th:
                    cnt_list[size_cnt] += 1
                    cnt_list_all[size_cnt] += 1
                    true_list[size_cnt].append(true_fold[cnt])
                    conf_list[size_cnt].append(conf_fold[cnt])

                    true_all[size_cnt].append(true_fold[cnt])
                    conf_all[size_cnt].append(conf_fold[cnt])

            p, r, _ = precision_recall_curve(np.array(true_list[size_cnt]), np.array(conf_list[size_cnt]))
            plt.plot(r[1:], p[1:], linewidth=3, label=mn + ' largest recall={:.3f}'.format(r[1]))

        # aggregate results
        p, r, _ = precision_recall_curve(np.array(true_all[size_cnt]), np.array(conf_all[size_cnt]))
        plt.plot(r[1:], p[1:], '--', linewidth=3, label='Aggregate largest recall={:.3f}'.format(r[1]))

        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title('Object-wise PR Curve Comparison {:.0f}~{:.0f}'.format(size_ths[size_cnt] * 0.09,
                                                                         size_ths[size_cnt + 1] * 0.09))
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, 'precision_recall_{}.png'.format(size_ths[size_cnt])))
        plt.close()
