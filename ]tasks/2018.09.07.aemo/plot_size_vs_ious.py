import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import utils
import ersa_utils

if __name__ == '__main__':
    img_dir, task_dir = utils.get_task_img_folder()

    model_name = ['Fold 0', 'Fold 1', 'Fold 2']

    size_ths = list(range(0, 1000, 50))
    ious_all = []
    area_all = []
    true_all = [[] for i in range(len(size_ths) - 1)]
    conf_all = [[] for i in range(len(size_ths) - 1)]
    cnt_list_all = np.zeros(len(size_ths) - 1)

    for plt_cnt, mn in enumerate(model_name):
        cnt_list = np.zeros(len(size_ths) - 1)
        mean_ious = [[] for i in range(len(size_ths) - 1)]
        true_list = [[] for i in range(len(size_ths) - 1)]
        conf_list = [[] for i in range(len(size_ths) - 1)]

        area_fold = ersa_utils.load_file(os.path.join(task_dir, '{}_area.npy'.format(mn)))
        ious_fold = ersa_utils.load_file(os.path.join(task_dir, '{}_ious.npy'.format(mn)))
        conf_fold = ersa_utils.load_file(os.path.join(task_dir, '{}_conf.npy'.format(mn)))
        true_fold = ersa_utils.load_file(os.path.join(task_dir, '{}_true.npy'.format(mn)))

        ious_all.append(ious_fold)
        area_all.append(area_fold)

        for size_cnt in range(len(size_ths[:-1])):
            min_th = size_ths[size_cnt]
            max_th = size_ths[size_cnt + 1]

            for cnt, area in enumerate(area_fold):
                if min_th <= area < max_th:
                    mean_ious[size_cnt].append(ious_fold[cnt])
                    cnt_list[size_cnt] += 1
                    cnt_list_all[size_cnt] += 1
                    true_list[size_cnt].append(true_fold[cnt])
                    conf_list[size_cnt].append(conf_fold[cnt])

                    true_all[size_cnt].append(true_fold[cnt])
                    conf_all[size_cnt].append(conf_fold[cnt])

        mean_ious = [np.mean(a) for a in mean_ious]
        X = np.arange(len(size_ths) - 1)
        plt.figure(figsize=(12, 5))
        plt.bar(X, mean_ious)
        plt.xticks(X, ['{:.0f}~{:.0f}'.format(size_ths[a] * 0.09, size_ths[a + 1] * 0.09)
                       for a in range(len(size_ths[:-1]))])
        plt.ylim([0, 1])
        for i, panel_cnt in enumerate(cnt_list):
            plt.text(X[i]-0.2, mean_ious[i]+0.01, int(panel_cnt))
        plt.xlabel('Area:$m^2$')
        plt.ylabel('mean IoU')
        plt.title('Region {}'.format(plt_cnt))
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, '{}_mean_iou_bars_thinner.png'.format(mn)))
        plt.show()

        '''tp_list = []
        fp_list = []
        fn_list = []
        for size_cnt in range(len(size_ths[:-1])):
            tn, fp, fn, tp = confusion_matrix(np.array(true_list[size_cnt]),
                                             np.array(conf_list[size_cnt]) > 0.5).ravel()
            tn, fp, fn, tp = np.array([tn, fp, fn, tp]) / np.sum([tn, fp, fn, tp])
            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)
        plt.figure(figsize=(8, 5))
        X = np.arange(len(size_ths) - 1)
        width = 0.3
        plt.bar(X, tp_list, width=width, label='TP')
        plt.bar(X+width, fp_list, width=width, label='FP')
        plt.bar(X+2*width, fn_list, width=width, label='FN')
        plt.xticks(X+width, ['{:.0f}~{:.0f}'.format(size_ths[a] * 0.09, size_ths[a + 1] * 0.09)
                             for a in range(len(size_ths[:-1]))])
        plt.xlabel('Area:$m^2$')
        plt.ylim([0, 1])
        plt.legend()
        plt.title('Region {}'.format(plt_cnt))
        plt.tight_layout()
        #plt.savefig(os.path.join(img_dir, '{}_tp_tn_fp_fn.png'.format(mn)))
        plt.show()'''

    # aggregate results
    ious_fold = np.concatenate(ious_all)
    area_fold = np.concatenate(area_all)
    mean_ious = [[] for i in range(len(size_ths) - 1)]
    for size_cnt in range(len(size_ths[:-1])):
        min_th = size_ths[size_cnt]
        max_th = size_ths[size_cnt + 1]

        for cnt, area in enumerate(area_fold):
            if min_th <= area < max_th:
                mean_ious[size_cnt].append(ious_fold[cnt])

    mean_ious = [np.mean(a) for a in mean_ious]
    X = np.arange(len(size_ths) - 1)
    plt.figure(figsize=(12, 5))
    plt.bar(X, mean_ious)
    plt.xticks(X, ['{:.0f}~{:.0f}'.format(size_ths[a] * 0.09, size_ths[a + 1] * 0.09)
                   for a in range(len(size_ths[:-1]))])
    plt.ylim([0, 1])
    for i, panel_cnt in enumerate(cnt_list_all):
        plt.text(X[i] - 0.2, mean_ious[i] + 0.01, int(panel_cnt))
    plt.xlabel('Area:$m^2$')
    plt.ylabel('mean IoU')
    plt.title('Aggregate')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'agg_mean_iou_bars_thinner.png'))
    plt.show()

    '''tp_list = []
    fp_list = []
    fn_list = []
    for size_cnt in range(len(size_ths[:-1])):
        tn, fp, fn, tp = confusion_matrix(np.array(true_all[size_cnt]),
                                          np.array(conf_all[size_cnt]) > 0.5).ravel()
        tn, fp, fn, tp = np.array([tn, fp, fn, tp]) / np.sum([tn, fp, fn, tp])
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)
    plt.figure(figsize=(8, 5))
    X = np.arange(len(size_ths) - 1)
    width = 0.3
    plt.bar(X, tp_list, width=width, label='TP')
    plt.bar(X + width, fp_list, width=width, label='FP')
    plt.bar(X + 2 * width, fn_list, width=width, label='FN')
    plt.xticks(X + width, ['{:.0f}~{:.0f}'.format(size_ths[a] * 0.09, size_ths[a + 1] * 0.09)
                           for a in range(len(size_ths[:-1]))])
    plt.xlabel('Area:$m^2$')
    plt.ylim([0, 1])
    plt.legend()
    plt.title('Aggregate')
    plt.tight_layout()
    #plt.savefig(os.path.join(img_dir, 'agg_tp_tn_fp_fn.png'))
    plt.show()'''
